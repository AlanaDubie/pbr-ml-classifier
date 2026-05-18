# ── SceneScanner.py ──────────────────────────────────────────
# Handles all Maya-side logic for the PBR ML Classifier.
#
# Responsibilities:
#   1. Mesh collection  — find objects in the scene or selection
#   2. Albedo extraction — find texture file paths from shading networks
#   3. Classification   — call classifier.predict() for each texture
#   4. Metadata writing — stamp results onto shader nodes in the scene
#
# This file only deals with Maya — no ML code lives here.
# All inference is handled by classifier.py.
# ─────────────────────────────────────────────────────────────

import os
import maya.cmds as cmds

# Import the predict function from classifier.py.
# The model will not load until predict() is first called —
# so importing this here does not slow down Maya startup.
from classifier import predict

class SceneScanner:
    def __init__(self):
        self.objects = []   # list of transform node paths to scan
        self.results = {}   # filled by scan_and_classify()

    # ── Mesh collection ──────────────────────────────────────

    def get_selected_meshes(self):
        """
        Return only the mesh transforms the artist has selected
        in the Maya viewport.
        """

        # ls() with selection=True returns everything currently selected
        selection = cmds.ls(selection=True, long=True, type="transform")

        meshes = []
        for obj in selection:
            # Transforms can hold cameras, lights, locators etc.
            # We only want transforms that have a mesh shape as a child.
            shapes = cmds.listRelatives(obj, shapes=True, fullPath=True) or []
            if shapes:
                meshes.append(obj)

        self.objects = meshes
        return meshes

    def get_all_scene_meshes(self):
        """
        Return every mesh transform in the active Maya scene.
        """

        # ls(type="mesh") returns mesh shape nodes — not their transforms.
        # We need the transforms (the parent nodes) for the rest of the pipeline,
        # so we go up one level with listRelatives(parent=True).
        all_meshes = cmds.ls(type="mesh", long=True)
        transforms = cmds.listRelatives(all_meshes, parent=True, fullPath=True) or []

        # A single transform can have multiple mesh shapes under it —
        # set() removes any duplicates that would cause double scanning.
        self.objects = list(set(transforms))
        return self.objects

    # ── Albedo extraction ────────────────────────────────────

    def get_albedo_path(self, transform):
        """
        Follow the shading network from a mesh transform back to
        its albedo (colour) texture and return the file path.
        Returns None if no valid texture file is found.

        How Maya shading works:
          mesh shape
            └── shading engine  (groups geometry with a material)
                  └── surface shader  (the actual material node)
                        └── file texture node  (points to the image on disk)

        We walk this chain and return the first valid file path found.
        """

        # Get shape nodes under this transform
        shapes = cmds.listRelatives(transform, shapes=True, fullPath=True) or []
        if not shapes:
            return None

        # Find shading engines connected to the shape.
        # The shading engine is what links a mesh to its material.
        shading_engines = cmds.listConnections(shapes, type="shadingEngine") or []
        if not shading_engines:
            return None

        for sg in shading_engines:

            # Get the surface shader plugged into this shading engine.
            # surfaceShader is the standard input port for all shader types.
            shaders = cmds.listConnections(f"{sg}.surfaceShader") or []

            for shader in shaders:

                # Different shader types store the base colour in different attributes:
                #   aiStandardSurface (Arnold) → baseColor
                #   Lambert, Blinn, Phong      → color
                #   Other PBR shaders          → baseColorMap, diffuseColor
                # We try each one and skip any that don't exist on this shader.
                for attr in ["baseColor", "color", "baseColorMap", "diffuseColor"]:

                    if not cmds.attributeQuery(attr, node=shader, exists=True):
                        continue

                    # Check if a file texture node is connected to this attribute
                    connected_files = cmds.listConnections(
                        f"{shader}.{attr}", type="file"
                    ) or []

                    for file_node in connected_files:
                        path = cmds.getAttr(f"{file_node}.fileTextureName")
                        # Confirm the file actually exists on disk before returning
                        if path and os.path.exists(path):
                            return path

                # Fallback — if no named albedo attribute matched, grab the
                # first file texture connected anywhere on the shader
                all_file_nodes = cmds.listConnections(shader, type="file") or []
                for file_node in all_file_nodes:
                    path = cmds.getAttr(f"{file_node}.fileTextureName")
                    if path and os.path.exists(path):
                        return path

        return None

    # ── Metadata writing ─────────────────────────────────────

    def write_metadata(self, transform, label, confidence):
        """
        Write the predicted material type and confidence score onto
        the shader node as custom attributes in the Maya scene.

        Why write to the shader? The data travels with the .mb/.ma file.
        Any artist who opens the scene later can read what the classifier
        predicted without needing to re-run the tool.

        Attributes written:
          materialType  (string) — e.g. "wood"
          mlConfidence  (float)  — e.g. 0.9943

        Returns the shader node name so the caller does not need to
        traverse the shading network again just to get the name.
        """

        shapes = cmds.listRelatives(transform, shapes=True, fullPath=True) or []
        if not shapes:
            return None

        shader_name = None
        shading_engines = cmds.listConnections(shapes, type="shadingEngine") or []

        for sg in shading_engines:
            shaders = cmds.listConnections(f"{sg}.surfaceShader") or []

            for shader in shaders:

                # Only add the attribute if it does not already exist.
                # Without this check, running the tool twice on the same
                # scene would throw an error because the attribute exists.
                if not cmds.attributeQuery("materialType", node=shader, exists=True):
                    cmds.addAttr(shader, longName="materialType",
                                 dataType="string", keyable=False)
                cmds.setAttr(f"{shader}.materialType", label, type="string")

                if not cmds.attributeQuery("mlConfidence", node=shader, exists=True):
                    cmds.addAttr(shader, longName="mlConfidence",
                                 attributeType="float", keyable=False)
                cmds.setAttr(f"{shader}.mlConfidence", confidence)

                # Capture the shader name here so we can return it —
                # this avoids traversing the shading network a second time
                shader_name = shader
                print(f"[SceneScanner] Tagged {shader} → {label} ({confidence*100:.1f}%)")

        return shader_name

    # ── Full scan pipeline ───────────────────────────────────

    def scan_and_classify(self, progress_callback=None):
        """
        The main pipeline. Runs all steps across every object in self.objects.

        Split into three phases:
          Phase 1 — collect albedo texture paths for all objects
          Phase 2 — classify every texture using classifier.predict()
          Phase 3 — write metadata to shader nodes and store results for the UI

        Why collect all paths first?
        classifier.predict() loads the model on the very first call (~2s).
        After that every prediction is fast (~50-100ms).
        By collecting paths first we can give the artist clear progress feedback
        and avoid loading the model for objects that have no texture at all.

        progress_callback(current, total, name) is optional.
        The UI passes one in so the status bar updates during long scans.
        """

        self.results = {}
        total = len(self.objects)

        # ── Phase 1: collect albedo paths ─────────────────────
        # Walk the shading network for every object and store
        # the texture path (or None if no texture is connected).

        if progress_callback:
            progress_callback(0, total, "Collecting textures...")

        albedo_map = {}
        for transform in self.objects:
            albedo_map[transform] = self.get_albedo_path(transform)

        # Split into two groups so we can handle them separately
        classifiable   = {t: p for t, p in albedo_map.items() if p}
        unclassifiable = {t: p for t, p in albedo_map.items() if not p}

        # Objects with no texture go straight to results as unknown
        for transform in unclassifiable:
            short = transform.split("|")[-1]
            print(f"[SceneScanner] No albedo found for {short} — skipping")
            self.results[transform] = {
                "label":       "unknown",
                "confidence":  0.0,
                "albedo_path": None,
                "all_scores":  {},
                "shader":      None,
            }

        if not classifiable:
            # No objects had textures — nothing left to do
            return self.results

        # ── Phase 2: classify all textures ────────────────────
        # Call predict() for each texture path.
        # The model loads on the first call and stays in memory —
        # so this is fast for every call after the first.

        if progress_callback:
            progress_callback(0, total, "Loading model...")

        transforms_list = list(classifiable.keys())
        paths_list      = list(classifiable.values())
        predictions     = [predict(path) for path in paths_list]

        # ── Phase 3: write metadata and store results ─────────
        # Loop through predictions, stamp the shader nodes,
        # and store everything for the UI to display.

        for i, transform in enumerate(transforms_list):
            short       = transform.split("|")[-1]
            albedo_path = paths_list[i]
            prediction  = predictions[i]

            if progress_callback:
                progress_callback(i + 1, total, short)

            # Handle any prediction that came back as an error
            if not prediction or "error" in prediction:
                print(f"[SceneScanner] Inference failed for {short}")
                self.results[transform] = {
                    "label":       "error",
                    "confidence":  0.0,
                    "albedo_path": albedo_path,
                    "all_scores":  {},
                    "shader":      None,
                }
                continue

            label      = prediction["label"]
            confidence = prediction["confidence"]
            all_scores = prediction.get("all_scores", {})

            # Write materialType + mlConfidence to the shader node.
            # write_metadata returns the shader name so we don't
            # need to traverse the shading network again.
            shader = self.write_metadata(transform, label, confidence)

            # Store the full result for the UI
            self.results[transform] = {
                "label":       label,
                "confidence":  confidence,
                "albedo_path": albedo_path,
                "all_scores":  all_scores,
                "shader":      shader,
            }

            print(f"[SceneScanner] {short} → {label} ({confidence*100:.1f}%)")

        return self.results