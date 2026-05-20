# ── pbr_tools.py ──────────────────────────────────────────────
# Handles all Maya-side logic for the PBR ML Classifier.
#
# Responsibilities:
#   1. Mesh collection   — find objects in the scene or selection
#   2. Albedo extraction — find texture file paths from shading networks
#   3. Classification    — predict material categories (scan only, no side effects)
#   4. Apply approved    — write metadata + organize files for approved items only
#
# Intentional design decision:
#   scan_and_classify() does NOT write metadata or move any files.
#   It only runs the ML model and stores predictions.
#   This gives the artist a chance to review, override, and approve
#   results in the UI before anything in the scene or on disk is touched.
#   The actual writes happen in apply_approved() only.
# ─────────────────────────────────────────────────────────────

import os
import shutil
import maya.cmds as cmds

# The model will not load until predict() is first called —
# importing here does not slow down Maya startup.
from classifier import predict

# Material categories the model was trained on.
# Must match the order used during training.
CLASSES = ["fabric", "ground", "metal", "rock", "wood"]


class PBRTools:
    def __init__(self):
        self.objects = []   # list of transform node paths collected by a scan
        self.results = {}   # predictions stored by scan_and_classify()

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
        """

        selection = cmds.ls(selection=True, long=True)
        meshes    = []

        for obj in selection:
            self._collect_meshes_recursive(obj, meshes)

        # Remove duplicates — selecting both a group and a child mesh
        # would otherwise add that mesh twice.
        seen          = set()
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

        Handles any depth of nesting:
          group1
            group2
              mesh1   ← found
              mesh2   ← found
            mesh3     ← found
        """

        shapes   = cmds.listRelatives(transform, shapes=True, fullPath=True) or []
        has_mesh = any(cmds.nodeType(s) == "mesh" for s in shapes)
        if has_mesh:
            results.append(transform)

        children = cmds.listRelatives(
            transform, children=True, fullPath=True, type="transform"
        ) or []
        for child in children:
            self._collect_meshes_recursive(child, results)

    def get_all_scene_meshes(self):
        """Return every mesh transform in the active Maya scene."""

        all_meshes = cmds.ls(type="mesh", long=True)
        transforms = cmds.listRelatives(all_meshes, parent=True, fullPath=True) or []

        # set() removes duplicates from transforms with multiple mesh shapes.
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
        """

        shapes = cmds.listRelatives(transform, shapes=True, fullPath=True) or []
        if not shapes:
            return None

        shading_engines = cmds.listConnections(shapes, type="shadingEngine") or []
        if not shading_engines:
            return None

        for sg in shading_engines:
            shaders = cmds.listConnections(f"{sg}.surfaceShader") or []

            for shader in shaders:

                # Different shader types store the base colour in different attributes:
                #   aiStandardSurface (Arnold) → baseColor
                #   Lambert, Blinn, Phong      → color
                #   Other PBR shaders          → baseColorMap, diffuseColor
                for attr in ["baseColor", "color", "baseColorMap", "diffuseColor"]:

                    if not cmds.attributeQuery(attr, node=shader, exists=True):
                        continue

                    connected_files = cmds.listConnections(
                        f"{shader}.{attr}", type="file"
                    ) or []

                    for file_node in connected_files:
                        path = cmds.getAttr(f"{file_node}.fileTextureName")
                        if path and os.path.exists(path):
                            return path

        return None

    # ── Metadata writing ─────────────────────────────────────

    def write_metadata(self, transform, label, confidence):
        """
        Write the predicted material type and confidence score onto
        the shader node as custom attributes in the Maya scene.

        Only called from apply_approved() — never during scanning.
        This ensures the scene is not modified until the artist
        has reviewed and approved the predictions.

        Attributes written:
          materialType  (string) — e.g. "wood"
          mlConfidence  (float)  — e.g. 0.9943

        Returns the shader node name.
        """

        shapes = cmds.listRelatives(transform, shapes=True, fullPath=True) or []
        if not shapes:
            return None

        shader_name     = None
        shading_engines = cmds.listConnections(shapes, type="shadingEngine") or []

        for sg in shading_engines:
            shaders = cmds.listConnections(f"{sg}.surfaceShader") or []

            for shader in shaders:

                # addAttr only if the attribute doesn't already exist —
                # running the tool twice would error otherwise.
                if not cmds.attributeQuery("materialType", node=shader, exists=True):
                    cmds.addAttr(shader, longName="materialType",
                                 dataType="string", keyable=False)
                cmds.setAttr(f"{shader}.materialType", label, type="string")

                if not cmds.attributeQuery("mlConfidence", node=shader, exists=True):
                    cmds.addAttr(shader, longName="mlConfidence",
                                 attributeType="float", keyable=False)
                cmds.setAttr(f"{shader}.mlConfidence", confidence)

                shader_name = shader
                print(f"[pbr_tools] Tagged {shader} → {label} ({confidence*100:.1f}%)")

        return shader_name

    # ── Scan pipeline (predict only — no side effects) ────────

    def scan_and_classify(self, progress_callback=None):
        """
        Scan every object in self.objects, run the ML model, and store
        predictions in self.results.

        This method intentionally does NOT:
          - write any attributes to shader nodes
          - move any files on disk
          - modify the Maya scene in any way

        All of that happens later in apply_approved(), only after the
        artist has reviewed and approved the predictions in the UI.

        Phase 1 — collect albedo texture paths for all objects
        Phase 2 — classify every texture using classifier.predict()
        Phase 3 — store results (no writes)

        progress_callback(current, total, name) is optional.
        """

        self.results = {}
        total        = len(self.objects)

        # ── Phase 1: collect albedo paths ────────────────────

        if progress_callback:
            progress_callback(0, total, "Collecting textures...")

        albedo_map = {}
        for transform in self.objects:
            albedo_map[transform] = self.get_albedo_path(transform)

        classifiable   = {t: p for t, p in albedo_map.items() if p}
        unclassifiable = {t: p for t, p in albedo_map.items() if not p}

        # Objects with no texture are stored immediately as unknown
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
            return self.results

        # ── Phase 2: classify all textures ───────────────────
        # predict() loads the model on the first call and keeps it in memory.
        # Every call after the first is fast (~50-100ms).

        transforms_list = list(classifiable.keys())
        paths_list      = list(classifiable.values())
        predictions     = [predict(path) for path in paths_list]

        # ── Phase 3: store results — no metadata writes yet ──
        # write_metadata() is deliberately NOT called here.
        # The artist reviews predictions in the UI first.

        for i, transform in enumerate(transforms_list):
            short       = transform.split("|")[-1]
            albedo_path = paths_list[i]
            prediction  = predictions[i]

            if progress_callback:
                progress_callback(i + 1, total, short)

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

            self.results[transform] = {
                "label":       prediction["label"],
                "confidence":  prediction["confidence"],
                "albedo_path": albedo_path,
                "all_scores":  prediction.get("all_scores", {}),
                "shader":      None,   # filled in by write_metadata() during apply
            }

            print(f"[pbr_tools] {short} → {prediction['label']} ({prediction['confidence']*100:.1f}%)")

        return self.results

    # ── Apply approved ────────────────────────────────────────

    def apply_approved(self, review_queue, output_dir, dry_run=False, progress_callback=None):
        """
        Apply the artist-approved predictions — write metadata to shader
        nodes and move texture files on disk.

        Only items with status == "accepted" are processed.
        Items with status == "rejected" or "pending" are skipped entirely.

        If the artist set an override label on a row, that label is used
        instead of the original ML prediction when writing metadata and
        deciding which category subfolder to move the texture into.

        dry_run=True logs every action to the Script Editor but makes
        no changes to the scene or the file system. Use this to preview
        exactly what will happen before committing.

        review_queue is the list of entry dicts built by the UI:
          [
            {
              "transform":  "|pPlane1",
              "short":      "pPlane1",
              "label":      "wood",         — original ML prediction
              "confidence": 0.994,
              "override":   None,           — or "rock" if artist changed it
              "status":     "accepted",     — "accepted" | "rejected" | "pending"
              "albedo_path": "C:/tex/...",
            },
            ...
          ]

        Returns:
          {
            "metadata_written": int,   — shader nodes tagged
            "files_moved":      int,   — texture files moved on disk
            "skipped":          int,   — already in right place or name conflict
            "failed":           int,   — errors during move
          }
        """

        prefix = "[DRY RUN]" if dry_run else "[pbr_tools]"

        # ── Step 1: filter to accepted items only ─────────────

        accepted = [
            entry for entry in review_queue
            if entry.get("status") == "accepted"
            and entry.get("albedo_path")
            and entry.get("label") not in (None, "unknown", "error")
        ]

        if not accepted:
            print(f"{prefix} No accepted items to apply.")
            return {"metadata_written": 0, "files_moved": 0, "skipped": 0, "failed": 0}

        total            = len(accepted)
        metadata_written = 0
        files_moved      = 0
        skipped          = 0
        failed           = 0

        # Tracks old → new path for updating Maya's file nodes after all moves.
        # Format: { normalized_old_path: new_path }
        old_to_new = {}

        # Tracks paths already processed this run to avoid moving the same
        # texture twice when multiple objects share the same file.
        processed_paths = set()

        for i, entry in enumerate(accepted):
            transform   = entry["transform"]
            short       = entry["short"]
            confidence  = entry["confidence"]
            albedo_path = entry["albedo_path"]

            # Use the override label if the artist changed it,
            # otherwise use the original ML prediction.
            label = entry.get("override") or entry["label"]

            if progress_callback:
                progress_callback(i + 1, total, short)

            # ── Write metadata to the shader node ────────────

            if dry_run:
                print(f"{prefix} Would tag shader on '{short}' → materialType={label}, mlConfidence={confidence}")
            else:
                shader = self.write_metadata(transform, label, confidence)
                # Update self.results so the detail panel reflects the
                # final applied label (which may differ from the prediction
                # if the artist used an override).
                if transform in self.results:
                    self.results[transform]["shader"] = shader
                    self.results[transform]["label"]  = label
                metadata_written += 1

            # ── Move the texture file ─────────────────────────

            if not output_dir:
                continue

            normalized_src = os.path.normpath(albedo_path)

            # Skip if we already moved this exact file earlier in the loop.
            # This handles the case where multiple objects share one texture.
            if normalized_src in processed_paths:
                continue
            processed_paths.add(normalized_src)

            dest_folder = os.path.join(os.path.normpath(output_dir), label)
            dest_path   = os.path.normpath(
                os.path.join(dest_folder, os.path.basename(albedo_path))
            )

            # Already in the right place — nothing to do
            if normalized_src == dest_path:
                print(f"{prefix} Already organized: {os.path.basename(albedo_path)}")
                skipped += 1
                continue

            # A different file with the same name already exists — skip
            if os.path.exists(dest_path):
                print(f"{prefix} Skipping — file already exists at destination: {dest_path}")
                skipped += 1
                continue

            if dry_run:
                print(f"{prefix} Would move: {albedo_path}  →  {label}/{os.path.basename(albedo_path)}")
                files_moved += 1   # count what would be moved for the dry run summary
            else:
                try:
                    os.makedirs(dest_folder, exist_ok=True)
                    # shutil.move works across drives; os.rename does not.
                    shutil.move(albedo_path, dest_path)
                    old_to_new[normalized_src] = dest_path
                    files_moved += 1
                    print(f"{prefix} Moved → {label}/{os.path.basename(albedo_path)}")

                except Exception as error:
                    print(f"{prefix} Failed to move {os.path.basename(albedo_path)}: {error}")
                    failed += 1

        # ── Step 2: update Maya's file texture nodes ──────────
        # After moving files, every file node that referenced an old path
        # must be updated to the new path — otherwise Maya shows the
        # red X missing texture warning in the viewport.
        # Skipped entirely during a dry run.

        if old_to_new and not dry_run:
            print(f"[pbr_tools] Updating Maya file texture paths...")

            for file_node in cmds.ls(type="file") or []:
                current_path = cmds.getAttr(f"{file_node}.fileTextureName")
                if not current_path:
                    continue

                normalized_current = os.path.normpath(current_path)
                if normalized_current in old_to_new:
                    new_path = old_to_new[normalized_current]
                    cmds.setAttr(
                        f"{file_node}.fileTextureName", new_path, type="string"
                    )
                    print(f"[pbr_tools] Updated '{file_node}' → {new_path}")

            # Update self.results so the UI detail panel shows the new paths
            # without needing a full re-scan after applying.
            for transform, data in self.results.items():
                old_path = data.get("albedo_path")
                if old_path:
                    normalized = os.path.normpath(old_path)
                    if normalized in old_to_new:
                        self.results[transform]["albedo_path"] = old_to_new[normalized]

        print(
            f"{prefix} Apply complete — "
            f"{metadata_written} tagged, {files_moved} moved, "
            f"{skipped} skipped, {failed} failed"
        )

        return {
            "metadata_written": metadata_written,
            "files_moved":      files_moved,
            "skipped":          skipped,
            "failed":           failed,
        }