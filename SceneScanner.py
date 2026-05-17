# ── SceneScanner.py ──────────────────────────────────────────
# Core Maya pipeline class for the PBR ML Classifier.
# Handles three responsibilities:
#   1. Mesh collection — traverse scene or selection
#   2. Albedo extraction — find texture paths from shading networks
#   3. Classification — call infer.py via subprocess, write metadata
#
# Maya cannot import PyTorch directly so inference runs in a
# separate mayapy.exe child process and returns JSON to stdout.
# ─────────────────────────────────────────────────────────────

import os
import json
import subprocess
import maya.cmds as cmds

from config import INFER_SCRIPT, PYTHON_EXE


class SceneScanner:
    def __init__(self):
        self.objects = []
        self.results = {}  # {transform_name: {label, confidence, albedo_path}}

    # ── Mesh collection ──────────────────────────────────────

    def get_selected_meshes(self):
        """Return mesh transforms from the current Maya selection"""
        selection = cmds.ls(selection=True, long=True, type="transform")
        meshes = []
        for obj in selection:
            # Only include transforms that have a mesh shape child
            shapes = cmds.listRelatives(obj, shapes=True, fullPath=True) or []
            if shapes:
                meshes.append(obj)
        self.objects = meshes
        return meshes

    def get_all_scene_meshes(self):
        """Return all mesh transforms in the active scene"""
        all_meshes = cmds.ls(type="mesh", long=True)
        transforms = cmds.listRelatives(all_meshes, parent=True, fullPath=True) or []
        # Deduplicate — multiple shapes can share a transform
        self.objects = list(set(transforms))
        return self.objects

    # ── Albedo extraction ────────────────────────────────────

    def get_albedo_path(self, transform):
        """
        Walk the shading network attached to a mesh transform and
        return the file path of its albedo/color texture.
        Returns None if no valid texture is found.

        Supports Arnold (aiStandardSurface), Lambert, Blinn,
        and any shader with a color or baseColor attribute.
        """
        shapes = cmds.listRelatives(transform, shapes=True, fullPath=True) or []
        if not shapes:
            return None

        # Get shading engines (material groups) connected to this mesh
        shading_engines = cmds.listConnections(shapes, type="shadingEngine") or []
        if not shading_engines:
            return None

        for sg in shading_engines:
            # Get the surface shader plugged into this shading engine
            shaders = cmds.listConnections(f"{sg}.surfaceShader") or []
            for shader in shaders:

                # Try known albedo attribute names across common shader types:
                # baseColor — aiStandardSurface (Arnold), UsdPreviewSurface
                # color     — Lambert, Blinn, Phong
                # baseColorMap, diffuseColor — other PBR shaders
                for attr in ["baseColor", "color", "baseColorMap", "diffuseColor"]:
                    # Skip attributes that don't exist on this shader type
                    if not cmds.attributeQuery(attr, node=shader, exists=True):
                        continue
                    connections = cmds.listConnections(
                        f"{shader}.{attr}", type="file"
                    ) or []
                    for file_node in connections:
                        path = cmds.getAttr(f"{file_node}.fileTextureName")
                        if path and os.path.exists(path):
                            return path

                # Fallback: grab the first valid file texture connected
                # to the shader regardless of which attribute it's on
                all_file_nodes = cmds.listConnections(shader, type="file") or []
                for file_node in all_file_nodes:
                    path = cmds.getAttr(f"{file_node}.fileTextureName")
                    if path and os.path.exists(path):
                        return path

        return None

    # ── Inference ────────────────────────────────────────────

    def run_inference(self, image_path):
        """
        Spawn infer.py as a mayapy child process, pass the texture
        path as an argument, and parse the JSON result from stdout.
        Returns None if the process errors or times out.
        """
        try:
            result = subprocess.run(
                [PYTHON_EXE, INFER_SCRIPT, image_path],
                capture_output=True,
                text=True,
                timeout=30  # 30s hard cap per texture
            )
            if result.returncode != 0:
                print(f"[SceneScanner] infer.py error: {result.stderr}")
                return None
            return json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            print(f"[SceneScanner] Inference timed out for {image_path}")
            return None
        except Exception as e:
            print(f"[SceneScanner] Subprocess error: {e}")
            return None

    # ── Metadata writing ─────────────────────────────────────

    def write_metadata(self, transform, label, confidence):
        """
        Write the predicted material category and confidence score
        as custom attributes on the shader node in the Maya scene.

        Adds two attributes if they don't already exist:
          materialType  (string) — e.g. "wood"
          mlConfidence  (float)  — e.g. 0.9943
        """
        shapes = cmds.listRelatives(transform, shapes=True, fullPath=True) or []
        if not shapes:
            return

        shading_engines = cmds.listConnections(shapes, type="shadingEngine") or []
        for sg in shading_engines:
            shaders = cmds.listConnections(f"{sg}.surfaceShader") or []
            for shader in shaders:

                # materialType — human readable predicted category
                if not cmds.attributeQuery("materialType", node=shader, exists=True):
                    cmds.addAttr(shader, longName="materialType",
                                 dataType="string", keyable=False)
                cmds.setAttr(f"{shader}.materialType", label, type="string")

                # mlConfidence — model's confidence score for the prediction
                if not cmds.attributeQuery("mlConfidence", node=shader, exists=True):
                    cmds.addAttr(shader, longName="mlConfidence",
                                 attributeType="float", keyable=False)
                cmds.setAttr(f"{shader}.mlConfidence", confidence)

                print(f"[SceneScanner] Tagged {shader} → {label} ({confidence*100:.1f}%)")

    # ── Shader lookup ─────────────────────────────────────────

    def get_shader_name(self, transform):
        """Return the name of the surface shader connected to this mesh, or None."""
        shapes = cmds.listRelatives(transform, shapes=True, fullPath=True) or []
        if not shapes:
            return None
        shading_engines = cmds.listConnections(shapes, type="shadingEngine") or []
        for sg in shading_engines:
            shaders = cmds.listConnections(f"{sg}.surfaceShader") or []
            if shaders:
                return shaders[0]
        return None

    # ── Full scan pipeline ───────────────────────────────────

    def scan_and_classify(self, progress_callback=None):
        """
        Run the full classification pipeline across all collected objects:
          1. Extract albedo texture path from shading network
          2. Run inference via infer.py subprocess
          3. Write materialType and mlConfidence to the shader node
          4. Return results dict for the UI to display

        progress_callback(current, total, object_name) is an optional
        hook so the UI can show a progress indicator during long scans.
        """
        self.results = {}
        total = len(self.objects)

        for i, transform in enumerate(self.objects):
            short = transform.split("|")[-1]

            if progress_callback:
                progress_callback(i + 1, total, short)

            # Step 1 — find the albedo texture connected to this mesh
            albedo_path = self.get_albedo_path(transform)
            if not albedo_path:
                print(f"[SceneScanner] No albedo found for {short} — skipping")
                self.results[transform] = {
                    "label":       "unknown",
                    "confidence":  0.0,
                    "albedo_path": None,
                }
                continue

            # Step 2 — classify the texture via infer.py
            prediction = self.run_inference(albedo_path)
            if not prediction or "error" in prediction:
                print(f"[SceneScanner] Inference failed for {short}")
                self.results[transform] = {
                    "label":       "error",
                    "confidence":  0.0,
                    "albedo_path": albedo_path,
                }
                continue

            label      = prediction["label"]
            confidence = prediction["confidence"]
            all_scores = prediction.get("all_scores", {})

            # Step 3 — tag the shader node with the predicted category
            self.write_metadata(transform, label, confidence)

            # Step 4 — store result for UI display, including shader and scores
            self.results[transform] = {
                "label":       label,
                "confidence":  confidence,
                "albedo_path": albedo_path,
                "all_scores":  all_scores,
                "shader":      self.get_shader_name(transform),
            }

            print(f"[SceneScanner] {short} → {label} ({confidence*100:.1f}%)")

        return self.results