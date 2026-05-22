# ── pbr_tools.py ─────────────────────────────────────────────
# Handles all Maya-side logic for the PBR ML Classifier.
#
# Responsibilities:
#   1. Mesh collection        — find objects in the scene or selection
#   2. Texture set extraction — collect ALL connected map file paths per shader
#   3. Shader lookup          — read shader name at scan time (read-only)
#   4. Classification         — predict material category from the albedo map
#   5. Metadata writing       — stamp results onto shader nodes in the scene
#   6. File organization      — move full texture sets into category subfolders
#                               and update all Maya file nodes to new paths
#
# Design principle:
#   scan_and_classify() does NOT write metadata or move files.
#   It only runs ML inference and stores predictions.
#   Everything destructive happens in apply_approved() after the
#   artist has reviewed results in the UI.
# ─────────────────────────────────────────────────────────────

import os
import shutil
import maya.cmds as cmds

from classifier import predict

try:
    from texture_name_parser import resolve_asset_name
except ImportError:
    def resolve_asset_name(path):
        return os.path.splitext(os.path.basename(path))[0] if path else "unknown"

CLASSES = ["fabric", "ground", "metal", "rock", "wood"]

# Maya-generated folders that should not prevent cleanup
_MAYA_JUNK_DIRS = frozenset({".mayaSwatches"})


def _folder_is_empty(folder: str):
    """
    Return True if the folder contains nothing except Maya junk folders.
    """

    try:
        entries = os.listdir(folder)
        real_entries = [e for e in entries if e not in _MAYA_JUNK_DIRS]
        return len(real_entries) == 0
    except Exception:
        return False


def _remove_if_empty(folder):
    """
    Remove a folder if it only contains Maya-generated junk folders.

    This safely removes leftover empty asset folders after texture sets
    are moved to a new destination root.

    Example:
        old_textures/rock/cliff_rock_a_v1/

    Maya may leave:
        .mayaSwatches/

    behind, which prevents normal os.rmdir() cleanup.
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

        # Remove Maya junk folders recursively
        for junk in _MAYA_JUNK_DIRS:

            junk_path = os.path.join(folder, junk)

            if os.path.isdir(junk_path):

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

                try:
                    os.rmdir(junk_path)
                except Exception:
                    pass

        # Remove parent folder if empty now
        if not os.listdir(folder):
            os.rmdir(folder)
            print(f"[pbr_tools] Removed empty folder: {folder}")

    except Exception as err:
        print(f"[pbr_tools] Could not remove folder {folder}: {err}")


class PBRTools:

    def __init__(self):
        self.objects = []
        self.results = {}

    # ── Mesh collection ──────────────────────────────────────

    def get_selected_meshes(self):

        selection = cmds.ls(selection=True, long=True, type="transform")

        meshes = []

        for obj in selection:
            shapes = cmds.listRelatives(obj, shapes=True, fullPath=True) or []

            if shapes:
                meshes.append(obj)

        self.objects = meshes
        return meshes

    def get_all_scene_meshes(self):

        all_meshes = cmds.ls(type="mesh", long=True)
        transforms = cmds.listRelatives(all_meshes, parent=True, fullPath=True) or []

        self.objects = list(set(transforms))
        return self.objects

    # ── Shader lookup ─────────────────────────────────────────

    def get_shader_name(self, transform):

        shapes = cmds.listRelatives(transform, shapes=True, fullPath=True) or []

        if not shapes:
            return None

        shading_engines = cmds.listConnections(shapes, type="shadingEngine") or []

        for sg in shading_engines:
            shaders = cmds.listConnections(f"{sg}.surfaceShader") or []

            if shaders:
                return shaders[0]

        return None

    # ── Texture set extraction ───────────────────────────────

    def get_texture_set(self, transform):

        shapes = cmds.listRelatives(transform, shapes=True, fullPath=True) or []

        if not shapes:
            return None

        shading_engines = cmds.listConnections(shapes, type="shadingEngine") or []

        if not shading_engines:
            return None

        albedo_path = None
        all_paths   = []
        file_nodes  = []

        ALBEDO_ATTRS = ["baseColor", "color", "baseColorMap", "diffuseColor"]

        for sg in shading_engines:

            shaders = cmds.listConnections(f"{sg}.surfaceShader") or []

            for shader in shaders:

                # ── Find albedo first ────────────────────────

                for attr in ALBEDO_ATTRS:

                    if not cmds.attributeQuery(attr, node=shader, exists=True):
                        continue

                    for fn in cmds.listConnections(f"{shader}.{attr}", type="file") or []:

                        p = cmds.getAttr(f"{fn}.fileTextureName")

                        if p and os.path.exists(p) and albedo_path is None:
                            albedo_path = p

                # ── Collect ALL file nodes ───────────────────

                for fn in cmds.listConnections(shader, type="file") or []:

                    p = cmds.getAttr(f"{fn}.fileTextureName")

                    if not p or not os.path.exists(p) or fn in file_nodes:
                        continue

                    if ".mayaSwatches" in p.replace("\\", "/"):
                        continue

                    all_paths.append(p)
                    file_nodes.append(fn)

        if not all_paths:
            return None

        if albedo_path is None:
            albedo_path = all_paths[0]

        return {
            "albedo_path": albedo_path,
            "all_paths":   all_paths,
            "file_nodes":  file_nodes,
        }

    # ── Metadata writing ─────────────────────────────────────

    def write_metadata(self, transform, label, confidence):

        shapes = cmds.listRelatives(transform, shapes=True, fullPath=True) or []

        if not shapes:
            return None

        shader_name = None

        shading_engines = cmds.listConnections(shapes, type="shadingEngine") or []

        for sg in shading_engines:

            shaders = cmds.listConnections(f"{sg}.surfaceShader") or []

            for shader in shaders:

                if not cmds.attributeQuery("materialType", node=shader, exists=True):
                    cmds.addAttr(
                        shader,
                        longName="materialType",
                        dataType="string",
                        keyable=False
                    )

                cmds.setAttr(
                    f"{shader}.materialType",
                    label,
                    type="string"
                )

                if not cmds.attributeQuery("mlConfidence", node=shader, exists=True):
                    cmds.addAttr(
                        shader,
                        longName="mlConfidence",
                        attributeType="float",
                        keyable=False
                    )

                cmds.setAttr(
                    f"{shader}.mlConfidence",
                    confidence
                )

                shader_name = shader

                print(
                    f"[pbr_tools] Tagged {shader} → "
                    f"{label} ({confidence*100:.1f}%)"
                )

        return shader_name

    # ── Scan pipeline ────────────────────────────────────────

    def scan_and_classify(self, progress_callback=None):

        self.results = {}
        total = len(self.objects)

        # ── Phase 1: collect texture sets + shader names ────

        if progress_callback:
            progress_callback(0, total, "Collecting textures...")

        texture_sets = {}
        shader_names = {}

        for transform in self.objects:

            texture_sets[transform] = self.get_texture_set(transform)
            shader_names[transform] = self.get_shader_name(transform)

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
                "shader":      shader_names.get(transform),
            }

        if not classifiable:
            return self.results

        # ── Phase 2: classify ────────────────────────────────

        transforms_list = list(classifiable.keys())
        sets_list       = list(classifiable.values())

        predictions = [
            predict(ts["albedo_path"])
            for ts in sets_list
        ]

        # ── Phase 3: store results ───────────────────────────

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
                    "shader":      shader_names.get(transform),
                }

                continue

            self.results[transform] = {
                "label":       prediction["label"],
                "confidence":  prediction["confidence"],
                "albedo_path": tex_set["albedo_path"],
                "all_paths":   tex_set["all_paths"],
                "all_scores":  prediction.get("all_scores", {}),
                "shader":      shader_names.get(transform),
            }

            n_maps = len(tex_set["all_paths"])

            print(
                f"[pbr_tools] {short} → {prediction['label']} "
                f"({prediction['confidence']*100:.1f}%) | "
                f"{n_maps} map(s) found"
            )

        return self.results

    # ── Apply approved ────────────────────────────────────────

    def apply_approved(
        self,
        review_queue,
        output_dir,
        dry_run=False,
        progress_callback=None
    ):

        prefix = "[DRY RUN]" if dry_run else "[pbr_tools]"

        accepted = [
            e for e in review_queue
            if e.get("status") == "accepted"
            and e.get("albedo_path")
            and e.get("label") not in (None, "unknown", "error")
        ]

        if not accepted:

            print(f"{prefix} No accepted items to apply.")

            return {
                "metadata_written": 0,
                "files_moved": 0,
                "skipped": 0,
                "failed": 0,
            }

        total            = len(accepted)
        metadata_written = 0
        files_moved      = 0
        skipped          = 0
        failed           = 0

        old_to_new = {}

        processed_albedos = set()

        # ── Step 1: move texture sets + write metadata ──────

        for i, entry in enumerate(accepted):

            transform   = entry["transform"]
            short       = entry["short"]
            confidence  = entry["confidence"]
            albedo_path = entry["albedo_path"]
            label       = entry.get("override") or entry["label"]

            all_paths = (
                self.results.get(transform, {}).get("all_paths")
                or [albedo_path]
            )

            if progress_callback:
                progress_callback(i + 1, total, short)

            # ── Write metadata ───────────────────────────────

            if dry_run:

                print(
                    f"{prefix} Would tag '{short}' → "
                    f"materialType={label}, mlConfidence={confidence}"
                )

            else:

                shader = self.write_metadata(
                    transform,
                    label,
                    confidence
                )

                if transform in self.results:
                    self.results[transform]["shader"] = shader
                    self.results[transform]["label"]  = label

                metadata_written += 1

            # ── Move texture set ─────────────────────────────

            if not output_dir:
                continue

            norm_albedo = os.path.normpath(albedo_path)

            if norm_albedo in processed_albedos:

                print(
                    f"{prefix} Skipping shared texture set for "
                    f"{short} — already moved"
                )

                continue

            processed_albedos.add(norm_albedo)

            asset_name = resolve_asset_name(albedo_path)

            dest_folder = os.path.normpath(
                os.path.join(output_dir, label, asset_name)
            )

            if dry_run:
                print(f"{prefix} Would create: {dest_folder}/")

            for src in all_paths:

                norm_src = os.path.normpath(src)

                dest_file = os.path.normpath(
                    os.path.join(
                        dest_folder,
                        os.path.basename(src)
                    )
                )

                if norm_src == dest_file:

                    print(
                        f"{prefix} Already organized: "
                        f"{os.path.basename(src)}"
                    )

                    skipped += 1
                    continue

                if os.path.exists(dest_file):

                    print(
                        f"{prefix} Skipping — exists at destination: "
                        f"{dest_file}"
                    )

                    skipped += 1
                    continue

                if dry_run:

                    print(
                        f"{prefix} Would move: "
                        f"{os.path.basename(src)}  →  "
                        f"{label}/{asset_name}/{os.path.basename(src)}"
                    )

                    files_moved += 1

                else:

                    try:
                        os.makedirs(dest_folder, exist_ok=True)

                        shutil.move(src, dest_file)

                        old_to_new[norm_src] = dest_file

                        files_moved += 1

                        print(
                            f"{prefix} Moved → "
                            f"{label}/{asset_name}/"
                            f"{os.path.basename(src)}"
                        )

                    except Exception as err:

                        print(
                            f"{prefix} Failed to move "
                            f"{os.path.basename(src)}: {err}"
                        )

                        failed += 1

            # ── Cleanup old source folders ───────────────────
            #
            # Remove:
            #   old_root/category/asset_name/
            # and possibly:
            #   old_root/category/
            #
            # if they became empty after moving files.
            #
            # Uses _remove_if_empty() so Maya swatch cache folders
            # don't block cleanup.

            if not dry_run and all_paths:

                src_folder = os.path.dirname(
                    os.path.normpath(all_paths[0])
                )

                _remove_if_empty(src_folder)
                _remove_if_empty(os.path.dirname(src_folder))

        # ── Step 2: update Maya file texture nodes ───────────

        if old_to_new and not dry_run:

            print("[pbr_tools] Updating Maya file texture paths...")

            for file_node in cmds.ls(type="file") or []:

                current = cmds.getAttr(
                    f"{file_node}.fileTextureName"
                )

                if not current:
                    continue

                norm = os.path.normpath(current)

                if norm in old_to_new:

                    cmds.setAttr(
                        f"{file_node}.fileTextureName",
                        old_to_new[norm],
                        type="string"
                    )

                    print(
                        f"[pbr_tools] Updated "
                        f"'{file_node}' → {old_to_new[norm]}"
                    )

            # Refresh stored results so the detail panel reflects
            # new locations without a re-scan

            for transform, data in self.results.items():

                p = data.get("albedo_path")

                if p and os.path.normpath(p) in old_to_new:
                    data["albedo_path"] = old_to_new[os.path.normpath(p)]

                if "all_paths" in data:

                    data["all_paths"] = [
                        old_to_new.get(os.path.normpath(p), p)
                        for p in data["all_paths"]
                    ]

        print(
            f"{prefix} Apply complete — "
            f"{metadata_written} tagged, "
            f"{files_moved} moved, "
            f"{skipped} skipped, "
            f"{failed} failed"
        )

        return {
            "metadata_written": metadata_written,
            "files_moved":      files_moved,
            "skipped":          skipped,
            "failed":           failed,
        }