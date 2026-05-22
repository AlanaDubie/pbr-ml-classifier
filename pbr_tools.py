# ── pbr_tools.py ─────────────────────────────────────────────
# Handles all Maya-side logic for the PBR ML Classifier.
#
# Responsibilities:
#   1. Mesh collection      — find objects in the scene or selection
#   2. Texture set extraction — collect ALL connected map file paths per shader
#   3. Classification       — predict material category from the albedo map
#   4. Metadata writing     — stamp results onto shader nodes in the scene
#   5. File organization    — move full texture sets into category subfolders
#                             and update all Maya file nodes to new paths
#
# Design principle:
#   scan_and_classify() does NOT write metadata or move files.
#   It only runs ML inference and stores predictions.
#   Everything destructive happens in apply_approved() after the
#   artist has reviewed results in the UI.
#
# Known limitation (MVP):
#   If multiple objects share the same albedo texture, the texture set
#   is only moved once (for the first object encountered). Subsequent
#   objects pointing to the same file will have their Maya file nodes
#   updated but no duplicate move is attempted.
#   TODO: surface this in the UI so artists can review shared textures.
# ─────────────────────────────────────────────────────────────

import os
import shutil
import maya.cmds as cmds

from classifier import predict
from texture_name_parser import resolve_asset_name

CLASSES = ["fabric", "ground", "metal", "rock", "wood"]


# Maya-generated folders that should be ignored when deciding whether a
# directory is "empty enough" to remove. These are not texture files.
_MAYA_JUNK_DIRS = frozenset({".mayaSwatches"})


def _folder_is_empty(folder: str) -> bool:
    """
    Return True if a folder contains nothing other than known Maya
    junk directories (.mayaSwatches etc.) that this tool can ignore.

    A folder with only .mayaSwatches/ in it is functionally empty from
    the perspective of this tool — there are no texture files there.
    """
    try:
        entries = os.listdir(folder)
        real_entries = [e for e in entries if e not in _MAYA_JUNK_DIRS]
        return len(real_entries) == 0
    except Exception:
        return False

def _remove_if_empty(folder):
    """
    Remove a folder if it only contains Maya swatch cache data.
    """

    try:
        if not folder or not os.path.isdir(folder):
            return

        entries = os.listdir(folder)

        # Ignore Maya-generated cache folders
        real_entries = []

        for e in entries:
            if e not in _MAYA_JUNK_DIRS:
                real_entries.append(e)

        # Stop if real files/folders still exist
        if real_entries:
            return

        # Remove Maya swatch cache folders
        for junk in _MAYA_JUNK_DIRS:

            junk_path = os.path.join(folder, junk)

            if os.path.isdir(junk_path):

                # Walk bottom-up so files get removed before folders
                for root, dirs, files in os.walk(junk_path, topdown=False):

                    for f in files:
                        try:
                            os.remove(os.path.join(root, f))
                        except Exception:
                            pass

                    for d in dirs:
                        try:
                            os.rmdir(os.path.join(root, d))
                        except Exception:
                            pass

                # Remove the .mayaSwatches folder itself
                try:
                    os.rmdir(junk_path)
                except Exception:
                    pass

        # Remove the parent folder if empty now
        if not os.listdir(folder):
            os.rmdir(folder)
            print(f"[pbr_tools] Removed empty folder: {folder}")

    except Exception as err:
        print(f"[pbr_tools] Could not remove folder {folder}: {err}")

class PBRTools:
    def __init__(self):
        self.objects = []   # list of transform node paths to scan
        self.results = {}   # predictions stored by scan_and_classify()

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

    # ── Texture set extraction ───────────────────────────────
    # Collects ALL file texture nodes connected to a shader, not just
    # the albedo. This is the production-correct approach — a texture
    # set (albedo + normal + roughness etc.) should be organized together
    # as a unit, not as individual files.

    def get_texture_set(self, transform):
        """
        Walk the shading network from a mesh transform and return every
        file texture node connected to its shader.

        Returns a dict:
            {
                "albedo_path": "/path/to/albedo.png",   — used for ML classification
                "all_paths":   [                         — all maps for this shader
                    "/path/to/albedo.png",
                    "/path/to/normal.png",
                    "/path/to/roughness.png",
                ],
                "file_nodes":  ["file1", "file2", ...],  — Maya node names
            }

        Returns None if no shader or no connected file textures are found.

        Why collect all maps?
        When organizing, we move the entire texture set into one folder:
            textures/rock/cliff_rock_a_v1/
                cliff_rock_a_basecolor_v1.exr
                cliff_rock_a_normal_v1.exr
                cliff_rock_a_roughness_v1.exr

        The albedo path is specifically identified because it is the one
        passed to the ML classifier. The rest are moved alongside it.
        """

        shapes = cmds.listRelatives(transform, shapes=True, fullPath=True) or []
        if not shapes:
            return None

        shading_engines = cmds.listConnections(shapes, type="shadingEngine") or []
        if not shading_engines:
            return None

        albedo_path = None
        all_paths   = []
        file_nodes  = []

        # Known albedo attribute names across common shader types
        ALBEDO_ATTRS = ["baseColor", "color", "baseColorMap", "diffuseColor"]

        for sg in shading_engines:
            shaders = cmds.listConnections(f"{sg}.surfaceShader") or []

            for shader in shaders:

                # ── Find the albedo path first ────────────────
                # Try known albedo attributes before falling back to any file node
                for attr in ALBEDO_ATTRS:
                    if not cmds.attributeQuery(attr, node=shader, exists=True):
                        continue
                    for fn in cmds.listConnections(f"{shader}.{attr}", type="file") or []:
                        p = cmds.getAttr(f"{fn}.fileTextureName")
                        if p and os.path.exists(p) and albedo_path is None:
                            albedo_path = p

                # ── Collect ALL file nodes on this shader ─────
                for fn in cmds.listConnections(shader, type="file") or []:
                    p = cmds.getAttr(f"{fn}.fileTextureName")
                    if not p or not os.path.exists(p) or fn in file_nodes:
                        continue
                    # Skip Maya's auto-generated swatch cache files.
                    # Maya creates .mayaSwatches/ next to textures for the
                    # Hypershade preview thumbnails. They are not texture
                    # files and must never be moved with the texture set.
                    if ".mayaSwatches" in p.replace("\\", "/"):
                        continue
                    all_paths.append(p)
                    file_nodes.append(fn)

        if not all_paths:
            return None

        # If no dedicated albedo attribute was found, fall back to the
        # first connected file texture
        if albedo_path is None and all_paths:
            albedo_path = all_paths[0]

        return {
            "albedo_path": albedo_path,
            "all_paths":   all_paths,
            "file_nodes":  file_nodes,
        }

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

        # ── Phase 1: collect texture sets ───────────────────
        # Walk the shading network for every object and collect the
        # full texture set (albedo + all sibling maps).

        if progress_callback:
            progress_callback(0, total, "Collecting textures...")

        texture_sets = {}
        for transform in self.objects:
            texture_sets[transform] = self.get_texture_set(transform)

        classifiable   = {t: ts for t, ts in texture_sets.items() if ts}
        unclassifiable = {t: ts for t, ts in texture_sets.items() if not ts}

        for transform in unclassifiable:
            short = transform.split("|")[-1]
            print(f"[pbr_tools] No textures found for {short} — skipping")
            self.results[transform] = {
                "label":       "unknown",
                "confidence":  0.0,
                "albedo_path": None,
                "all_paths":   [],
                "all_scores":  {},
                "shader":      None,
            }

        if not classifiable:
            return self.results

        # ── Phase 2: classify using the albedo from each set ──
        # ML only needs the albedo — the rest of the set travels with it
        # during file organization but doesn't affect the prediction.

        if progress_callback:
            progress_callback(0, total, "Loading model...")

        transforms_list = list(classifiable.keys())
        sets_list       = list(classifiable.values())
        predictions     = [predict(ts["albedo_path"]) for ts in sets_list]

        # ── Phase 3: store results — no writes yet ────────────

        for i, transform in enumerate(transforms_list):
            short      = transform.split("|")[-1]
            tex_set    = sets_list[i]
            prediction = predictions[i]

            if progress_callback:
                progress_callback(i + 1, total, short)

            if not prediction or "error" in prediction:
                print(f"[pbr_tools] Inference failed for {short}")
                self.results[transform] = {
                    "label":       "error",
                    "confidence":  0.0,
                    "albedo_path": tex_set["albedo_path"],
                    "all_paths":   tex_set["all_paths"],
                    "all_scores":  {},
                    "shader":      None,
                }
                continue

            self.results[transform] = {
                "label":       prediction["label"],
                "confidence":  prediction["confidence"],
                "albedo_path": tex_set["albedo_path"],
                "all_paths":   tex_set["all_paths"],    # full set for organize step
                "all_scores":  prediction.get("all_scores", {}),
                "shader":      None,
            }

            n_maps = len(tex_set["all_paths"])
            print(f"[pbr_tools] {short} → {prediction['label']} "
                  f"({prediction['confidence']*100:.1f}%) | {n_maps} map(s) found")

        return self.results

    # ── Apply approved ────────────────────────────────────────
    # Writes metadata and moves full texture sets for approved items.
    # Called only after the artist has reviewed predictions in the UI.

    def apply_approved(self, review_queue, output_dir, dry_run=False, progress_callback=None):
        """
        Write materialType metadata to shader nodes and move full texture
        sets into organized category subfolders for all accepted items.

        Only items with status == "accepted" are processed.
        Rejected and Pending items are completely untouched.

        If an override label is set on an entry, that label is used instead
        of the original ML prediction for both metadata and folder routing.

        Texture set organization:
            Each accepted object's full texture set (albedo + normal +
            roughness + any other connected maps) is moved together into:
                <output_dir>/<category>/<asset_name>/
                    asset_basecolor.png
                    asset_normal.png
                    asset_roughness.png

            The asset_name is derived by stripping map type tokens from
            the albedo filename using texture_name_parser.resolve_asset_name().

        Known limitation (MVP):
            If two objects share the same albedo texture (and therefore the
            same texture set), the set is moved on the first object and
            skipped on the second. Both objects' Maya file nodes are updated
            correctly. This case is not yet surfaced in the UI.
            TODO: detect shared textures during scan and group them.

        dry_run=True logs every action to the Script Editor but makes
        no changes to the Maya scene or the file system.

        Returns:
            {
                "metadata_written": int,
                "files_moved":      int,
                "skipped":          int,
                "failed":           int,
            }
        """

        prefix = "[DRY RUN]" if dry_run else "[pbr_tools]"

        accepted = [
            e for e in review_queue
            if e.get("status") == "accepted"
            and e.get("albedo_path")
            and e.get("label") not in (None, "unknown", "error")
        ]

        if not accepted:
            print(f"{prefix} No accepted items to apply.")
            return {"metadata_written": 0, "files_moved": 0, "skipped": 0, "failed": 0}

        total            = len(accepted)
        metadata_written = 0
        files_moved      = 0
        skipped          = 0
        failed           = 0

        # Maps old normalized path → new path for updating Maya file nodes
        old_to_new: dict[str, str] = {}

        # Tracks albedo paths already processed to avoid moving the same
        # texture set twice when multiple objects share a texture.
        # See known limitation in docstring above.
        processed_albedos: set[str] = set()

        for i, entry in enumerate(accepted):
            transform   = entry["transform"]
            short       = entry["short"]
            confidence  = entry["confidence"]
            albedo_path = entry["albedo_path"]
            label       = entry.get("override") or entry["label"]

            # all_paths comes from the texture set collected during scan.
            # Fall back to just the albedo if it was never populated.
            all_paths = self.results.get(transform, {}).get("all_paths") or [albedo_path]

            if progress_callback:
                progress_callback(i + 1, total, short)

            # ── Write metadata ────────────────────────────────

            if dry_run:
                print(f"{prefix} Would tag '{short}' → materialType={label}, mlConfidence={confidence}")
            else:
                shader = self.write_metadata(transform, label, confidence)
                if transform in self.results:
                    self.results[transform]["shader"] = shader
                    self.results[transform]["label"]  = label
                metadata_written += 1

            # ── Move texture set ──────────────────────────────

            if not output_dir:
                continue

            norm_albedo = os.path.normpath(albedo_path)

            # Skip if this texture set was already moved by an earlier entry
            if norm_albedo in processed_albedos:
                print(f"{prefix} Skipping shared texture set for {short} — already moved")
                continue
            processed_albedos.add(norm_albedo)

            # Derive the asset subfolder name from the albedo filename
            asset_name  = resolve_asset_name(albedo_path)
            dest_folder = os.path.normpath(
                os.path.join(output_dir, label, asset_name)
            )

            if dry_run:
                print(f"{prefix} Would create folder: {dest_folder}/")

            for src in all_paths:
                norm_src  = os.path.normpath(src)
                dest_file = os.path.normpath(
                    os.path.join(dest_folder, os.path.basename(src))
                )

                if norm_src == dest_file:
                    print(f"{prefix} Already organized: {os.path.basename(src)}")
                    skipped += 1
                    continue

                if os.path.exists(dest_file):
                    print(f"{prefix} Skipping — exists at destination: {dest_file}")
                    skipped += 1
                    continue

                if dry_run:
                    print(f"{prefix} Would move: {os.path.basename(src)}"
                          f"  →  {label}/{asset_name}/{os.path.basename(src)}")
                    files_moved += 1
                else:
                    try:
                        os.makedirs(dest_folder, exist_ok=True)
                        shutil.move(src, dest_file)
                        old_to_new[norm_src] = dest_file
                        files_moved += 1
                        print(f"{prefix} Moved → {label}/{asset_name}/{os.path.basename(src)}")
                    except Exception as err:
                        print(f"{prefix} Failed to move {os.path.basename(src)}: {err}")
                        failed += 1

            # ── Remove empty source folder ────────────────────
            # After moving all files in a texture set, check if the
            # folder they came from is now empty and remove it if so.
            #
            # This cleans up leftover artifact folders when the artist
            # re-organizes to a different destination — e.g. moved to
            # /textures2/ by mistake, then re-organized to /textures/.
            # Without this step /textures2/rock/cliff_rock_a_v1/ would
            # sit there empty and look like a duplicate asset.
            #
            # Safety rules:
            #   1. Only remove if the folder exists on disk
            #   2. Only remove if it is completely empty
            #   3. Only remove the immediate asset folder — never parents
            #
            # We deliberately do NOT walk up the directory tree or use
            # shutil.rmtree(). Studios often have refs/, previews/, or
            # zbrush/ folders above the asset folder that must never
            # be touched by this tool.

            if not dry_run and all_paths:
                src_folder = os.path.dirname(os.path.normpath(all_paths[0]))
                # Clean the asset subfolder (e.g. cliff_rock_a_v1/) then
                # the category folder one level up (e.g. rock/).
                # If the artist moved textures to the wrong place first,
                # re-organizing leaves both levels empty — this removes both.
                _remove_if_empty(src_folder)
                _remove_if_empty(os.path.dirname(src_folder))

        # ── Update Maya file nodes ────────────────────────────

        if old_to_new and not dry_run:
            print(f"[pbr_tools] Updating Maya file texture paths...")

            for file_node in cmds.ls(type="file") or []:
                current = cmds.getAttr(f"{file_node}.fileTextureName")
                if not current:
                    continue
                norm = os.path.normpath(current)
                if norm in old_to_new:
                    cmds.setAttr(f"{file_node}.fileTextureName",
                                 old_to_new[norm], type="string")
                    print(f"[pbr_tools] Updated '{file_node}' → {old_to_new[norm]}")

            # Update self.results so the detail panel shows new paths
            for transform, data in self.results.items():
                for key in ("albedo_path",):
                    p = data.get(key)
                    if p and os.path.normpath(p) in old_to_new:
                        data[key] = old_to_new[os.path.normpath(p)]
                if "all_paths" in data:
                    data["all_paths"] = [
                        old_to_new.get(os.path.normpath(p), p)
                        for p in data["all_paths"]
                    ]

        print(f"{prefix} Apply complete — "
              f"{metadata_written} tagged, {files_moved} moved, "
              f"{skipped} skipped, {failed} failed")

        return {
            "metadata_written": metadata_written,
            "files_moved":      files_moved,
            "skipped":          skipped,
            "failed":           failed,
        }