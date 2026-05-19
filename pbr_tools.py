# ── pbr_tools.py ──────────────────────────────────────────
# Handles all Maya-side logic for the PBR ML Classifier.
#
# Responsibilities:
#   1. Mesh collection   — find objects in the scene or selection
#   2. Albedo extraction — find texture file paths from shading networks
#   3. Classification    — call classifier.predict() for each texture
#   4. Metadata writing  — stamp results onto shader nodes in the scene
#   5. File organization — move textures into category subfolders on disk
#                          and update Maya's file nodes to point to the new paths
#
# This file only deals with Maya — no ML code lives here.
# All inference is handled by classifier.py.
# ─────────────────────────────────────────────────────────────

import os
import shutil
import maya.cmds as cmds

# Import the predict function from classifier.py.
# The model will not load until predict() is first called —
# so importing this here does not slow down Maya startup.
from classifier import predict


class PBRTools:
    def __init__(self):
        self.objects = []   # list of transform node paths to scan
        self.results = {}   # filled by scan_and_classify(), read by organize_textures()

    # ── Mesh collection ──────────────────────────────────────

    def get_selected_meshes(self):
        """
        Return all mesh transforms found in the current selection,
        including meshes inside groups and deeply nested hierarchies.

        Why recursive?
        When an artist selects a group, Maya only returns the group
        transform — not the meshes inside it. A simple shapes check
        on the group itself finds nothing because the meshes are
        children of children. We recurse all the way down the
        hierarchy to find every mesh regardless of nesting depth.

        Examples that all work:
          - Selecting a mesh directly       → finds it
          - Selecting a group with meshes   → finds all meshes inside
          - Selecting a nested group        → finds all meshes at any depth
          - Mixed selection of groups + meshes → finds everything
        """

        # ls() returns everything currently selected in the viewport
        selection = cmds.ls(selection=True, long=True)

        # Collect all mesh transforms found under the selection
        meshes = []

        for obj in selection:
            # _collect_meshes_recursive walks the full hierarchy
            # under obj and appends any mesh transforms it finds
            self._collect_meshes_recursive(obj, meshes)

        # Remove duplicates — selecting both a group and a mesh inside
        # it would otherwise add that mesh twice
        seen = set()
        unique_meshes = []
        for m in meshes:
            if m not in seen:
                seen.add(m)
                unique_meshes.append(m)

        self.objects = unique_meshes
        return unique_meshes

    def _collect_meshes_recursive(self, transform, results):
        """
        Walk down the hierarchy from transform and add any mesh
        transforms found to the results list.

        Called recursively so it handles any depth of nesting:
          group1
            group2
              mesh1   ← found
              mesh2   ← found
            mesh3     ← found
        """

        # Check if this transform directly has a mesh shape under it
        shapes = cmds.listRelatives(transform, shapes=True, fullPath=True) or []
        has_mesh = any(
            cmds.nodeType(s) == "mesh" for s in shapes
        )
        if has_mesh:
            results.append(transform)

        # Recurse into any child transforms (groups, nested meshes etc.)
        children = cmds.listRelatives(
            transform, children=True, fullPath=True, type="transform"
        ) or []
        for child in children:
            self._collect_meshes_recursive(child, results)

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
                # scene would throw an error because the attribute already exists.
                if not cmds.attributeQuery("materialType", node=shader, exists=True):
                    cmds.addAttr(shader, longName="materialType",
                                 dataType="string", keyable=False)
                cmds.setAttr(f"{shader}.materialType", label, type="string")

                if not cmds.attributeQuery("mlConfidence", node=shader, exists=True):
                    cmds.addAttr(shader, longName="mlConfidence",
                                 attributeType="float", keyable=False)
                cmds.setAttr(f"{shader}.mlConfidence", confidence)

                # Capture the shader name so we can return it —
                # this avoids traversing the shading network a second time
                shader_name = shader
                print(f"[pbr_tools] tagged {shader} → {label} ({confidence*100:.1f}%)")

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
            print(f"[pbr_tools] No albedo found for {short} — skipping")
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
                print(f"[pbr_tools] Inference failed for {short}")
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

            print(f"[pbr_tools] {short} → {label} ({confidence*100:.1f}%)")

        return self.results

    # ── File organization ────────────────────────────────────

    def organize_textures(self, output_dir, progress_callback=None):
        """
        Move each classified texture into a category subfolder on disk,
        then update Maya's file texture nodes to point to the new location.

        This should only be called AFTER scan_and_classify() has been run,
        because it reads from self.results to know which label each texture got.

        output_dir is the root folder chosen by the user. Category subfolders
        are created inside it automatically:
            <output_dir>/
                wood/
                    oak_planks_color.png
                metal/
                    rust_01_color.png
                rock/  ground/  fabric/  ...

        Why update Maya's file nodes?
        If we just move the file and don't tell Maya, every texture in the
        scene goes missing — the red X problem. We call cmds.setAttr() on
        every file texture node that referenced the old path so the scene
        stays intact after the move.

        Why deduplicate?
        Multiple objects can share the same texture file. We only move each
        unique file once, then update all nodes that referenced it together.

        Returns a summary dict:
            {
                "moved":   5,   — number of files successfully moved
                "skipped": 1,   — files that were already in the right place
                "failed":  0,   — files that could not be moved (permissions etc.)
            }
        """

        # ── Step 1: validate the output directory ────────────────────

        if not output_dir:
            print("[pbr_tools] No output directory specified.")
            return {"moved": 0, "skipped": 0, "failed": 0}

        textures_root = os.path.normpath(output_dir)

        print(f"[pbr_tools] Organizing textures into: {textures_root}")

        # ── Step 2: collect unique texture paths and their labels ─────

        # Build a mapping of  texture_path → label
        # from the results that scan_and_classify() already stored.
        #
        # We use a dict here (not a list) so that if two objects share
        # the same texture file, we only store that path once.
        # The label will be the same either way since the same image
        # would have been classified the same way.
        path_to_label = {}

        for transform, data in self.results.items():
            albedo_path = data.get("albedo_path")
            label       = data.get("label")

            # Skip objects that had no texture or failed classification
            if not albedo_path or label in (None, "unknown", "error"):
                continue

            # os.path.normpath normalises slashes so the same file referenced
            # with different slash styles doesn't appear as two separate entries
            normalized_path = os.path.normpath(albedo_path)
            path_to_label[normalized_path] = label

        if not path_to_label:
            print("[pbr_tools] No classified textures found to organize.")
            return {"moved": 0, "skipped": 0, "failed": 0}

        total   = len(path_to_label)
        moved   = 0
        skipped = 0
        failed  = 0

        # ── Step 3: move each unique texture into its category folder ──

        # old_to_new stores the path remapping so we can update Maya's
        # file nodes in one pass after all files have been moved.
        # Format:  { old_path_str: new_path_str }
        old_to_new = {}

        for index, (src_path, label) in enumerate(path_to_label.items()):

            if progress_callback:
                filename = os.path.basename(src_path)
                progress_callback(index + 1, total, filename)

            # Build the destination folder: textures/wood/, textures/metal/, etc.
            dest_folder = os.path.join(textures_root, label)

            # os.makedirs creates the full folder path including any missing
            # parent folders. exist_ok=True means no error if it already exists.
            os.makedirs(dest_folder, exist_ok=True)

            # The destination file path keeps the original filename intact
            dest_path = os.path.join(dest_folder, os.path.basename(src_path))

            # Normalise the destination path for consistent comparisons
            dest_path = os.path.normpath(dest_path)

            # Check if the file is already sitting in the right category folder.
            # This happens if organize has been run before on the same scene.
            if src_path == dest_path:
                print(f"[pbr_tools] Already organized: {os.path.basename(src_path)}")
                skipped += 1
                continue

            # Check if a file with the same name already exists at the destination.
            # This can happen when two textures from different folders have the
            # same filename. We skip rather than silently overwrite.
            if os.path.exists(dest_path):
                print(f"[pbr_tools] Skipping — file already exists at destination: {dest_path}")
                skipped += 1
                continue

            # Move the file from its current location to the category folder.
            # shutil.move handles files across different drives correctly,
            # unlike os.rename which only works on the same filesystem.
            try:
                shutil.move(src_path, dest_path)
                old_to_new[src_path] = dest_path
                moved += 1
                print(f"[pbr_tools] Moved → {label}/{os.path.basename(src_path)}")

            except Exception as error:
                # If the move fails (e.g. file is locked, permissions issue),
                # log the error and continue rather than stopping the whole batch.
                print(f"[pbr_tools] Failed to move {os.path.basename(src_path)}: {error}")
                failed += 1

        # ── Step 4: update Maya's file texture nodes ──────────────────

        # After moving files, every file texture node in the scene that
        # referenced an old path must be updated to the new path.
        # Without this step, Maya shows the red X missing texture warning.
        #
        # We query ALL file nodes in the scene, not just the ones from the scan,
        # because multiple file nodes can point to the same texture file.

        if old_to_new:
            print("[pbr_tools] Updating Maya file texture paths...")

            # Get every file texture node currently in the scene
            all_file_nodes = cmds.ls(type="file") or []

            for file_node in all_file_nodes:
                current_path = cmds.getAttr(f"{file_node}.fileTextureName")

                if not current_path:
                    continue

                # Normalise the path so the comparison works regardless of
                # forward/back slashes or trailing separators
                normalized_current = os.path.normpath(current_path)

                # If this node was pointing at a file we moved, update it
                if normalized_current in old_to_new:
                    new_path = old_to_new[normalized_current]

                    # setAttr updates the file node's path inside Maya.
                    # Maya will immediately try to load the texture from the new
                    # location, so the viewport updates as soon as this runs.
                    cmds.setAttr(f"{file_node}.fileTextureName", new_path, type="string")
                    print(f"[pbr_tools] Updated file node '{file_node}' → {new_path}")

            # Also update self.results so the UI reflects the new paths
            # without needing a full re-scan.
            for transform, data in self.results.items():
                old_path = data.get("albedo_path")
                if old_path:
                    normalized_old = os.path.normpath(old_path)
                    if normalized_old in old_to_new:
                        self.results[transform]["albedo_path"] = old_to_new[normalized_old]

        # ── Step 5: report results ────────────────────────────────────

        print(
            f"[pbr_tools] Organization complete — "
            f"{moved} moved, {skipped} skipped, {failed} failed"
        )

        return {"moved": moved, "skipped": skipped, "failed": failed}