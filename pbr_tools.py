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
#
# Texture set collection strategy:
#   get_texture_set() walks listHistory on the shading engine — not just
#   direct connections to the shader — so file nodes routed through bump2d,
#   aiNormalMap, displacementShader, colorCorrect, etc. are all captured.
#
# File update strategy:
#   apply_approved() moves every file node in the scene whose path lives in
#   the same source folder as the albedo, not just the ones found by the node
#   graph walk. This catches maps wired into slots we don't enumerate
#   (coat, sheen, anisotropy, etc.) and updates their Maya paths correctly.
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

# Maya-generated folders that should never block folder cleanup
_MAYA_JUNK_DIRS = frozenset({".mayaSwatches"})


def _remove_if_empty(folder):
    """
    Remove a folder if it contains nothing except Maya-generated junk.

    After moving a texture set, Maya sometimes leaves .mayaSwatches/
    behind which prevents a plain os.rmdir(). This walks those junk
    folders out first, then removes the parent if truly empty.
    """
    try:
        if not folder or not os.path.isdir(folder):
            return

        real_entries = [e for e in os.listdir(folder) if e not in _MAYA_JUNK_DIRS]
        if real_entries:
            return

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
        """Return transform nodes for the current selection."""
        selection = cmds.ls(selection=True, long=True, type="transform")
        meshes = [
            obj for obj in selection
            if cmds.listRelatives(obj, shapes=True, fullPath=True)
        ]
        self.objects = meshes
        return meshes

    def get_all_scene_meshes(self):
        """Return transform nodes for every mesh in the scene."""
        all_meshes = cmds.ls(type="mesh", long=True)
        transforms = cmds.listRelatives(all_meshes, parent=True, fullPath=True) or []
        self.objects = list(set(transforms))
        return self.objects

    # ── Shader lookup ─────────────────────────────────────────

    def get_shader_name(self, transform):
        """Return the surface shader node name for a transform, or None."""
        shapes = cmds.listRelatives(transform, shapes=True, fullPath=True) or []
        if not shapes:
            return None
        for sg in cmds.listConnections(shapes, type="shadingEngine") or []:
            shaders = cmds.listConnections(f"{sg}.surfaceShader") or []
            if shaders:
                return shaders[0]
        return None

    # ── Texture set extraction ───────────────────────────────

    def get_texture_set(self, transform):
        """
        Return a dict with albedo_path, all_paths, and file_nodes for the
        texture set associated with this transform's shader.

        Two-phase approach:
          1. Node graph — walk listHistory on the shading engine to find every
             file node actually connected in Maya, regardless of intermediate
             nodes (bump2d, aiNormalMap, displacementShader, etc.).
          2. Disk siblings — once we have the albedo path, scan its parent
             folder for image files not wired as file nodes in the scene.
             This is common when artists connect only the diffuse/color map
             and leave the rest of the PBR set on disk but unconnected.

        Returns None if no albedo can be found at all.
        """
        shapes = cmds.listRelatives(transform, shapes=True, fullPath=True) or []
        if not shapes:
            return None

        shading_engines = cmds.listConnections(shapes, type="shadingEngine") or []
        if not shading_engines:
            return None

        ALBEDO_ATTRS = ["baseColor", "color", "baseColorMap", "diffuseColor"]

        IMAGE_EXTS = {
            ".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff",
            ".exr", ".hdr", ".bmp", ".psd", ".tx", ".rat",
        }

        albedo_path = None
        all_paths   = []
        file_nodes  = []

        for sg in shading_engines:

            shaders = cmds.listConnections(f"{sg}.surfaceShader") or []

            # ── Phase 1a: find albedo via known color attributes ──
            for shader in shaders:
                for attr in ALBEDO_ATTRS:
                    if not cmds.attributeQuery(attr, node=shader, exists=True):
                        continue
                    for fn in cmds.listConnections(f"{shader}.{attr}", type="file") or []:
                        p = cmds.getAttr(f"{fn}.fileTextureName")
                        if p and os.path.exists(p) and albedo_path is None:
                            albedo_path = p

            # ── Phase 1b: collect all connected file nodes ────────
            # Walk listHistory on the SHADER nodes (not the shading engine).
            # listHistory(sg) is too broad — it can return file nodes from
            # other materials that happen to share graph history. Scoping to
            # each shader keeps results strictly to this material's network.
            # We also explicitly walk the sg's displacement slot because that
            # connection doesn't go through surfaceShader.
            nodes_to_walk = list(shaders)
            disp = cmds.listConnections(f"{sg}.displacementShader") or []
            nodes_to_walk.extend(disp)

            for root_node in nodes_to_walk:
                for fn in cmds.ls(cmds.listHistory(root_node) or [], type="file"):
                    if fn in file_nodes:
                        continue
                    p = cmds.getAttr(f"{fn}.fileTextureName")
                    if not p or not os.path.exists(p):
                        continue
                    if ".mayaSwatches" in p.replace("\\", "/"):
                        continue
                    all_paths.append(p)
                    file_nodes.append(fn)

        if not all_paths and not albedo_path:
            return None

        if albedo_path is None:
            albedo_path = all_paths[0]

        # ── Phase 2: disk sibling scan ────────────────────────────
        # Many PBR workflows only wire the diffuse map and leave the rest on
        # disk unconnected. Scan the albedo folder for sibling image files so
        # the full set is shown in the UI and moved correctly during organize.
        src_dir    = os.path.normpath(os.path.dirname(albedo_path))
        known_norm = {os.path.normpath(p) for p in all_paths}

        try:
            all_files = os.listdir(src_dir)
            for fname in sorted(all_files):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in IMAGE_EXTS:
                    continue
                full = os.path.normpath(os.path.join(src_dir, fname))
                if full in known_norm:
                    continue
                if ".mayaSwatches" in full.replace("\\", "/"):
                    continue
                all_paths.append(full)
                known_norm.add(full)
        except OSError as e:
            print(f"[pbr_tools] Could not scan texture folder: {e}")

        return {
            "albedo_path": albedo_path,
            "all_paths":   all_paths,
            "file_nodes":  file_nodes,
        }

    # ── Metadata writing ─────────────────────────────────────

    def write_metadata(self, transform, label, confidence):
        """
        Stamp materialType and mlConfidence onto the surface shader.
        Adds the attributes if they don't already exist.
        Returns the shader node name, or None if nothing was written.
        """
        shapes = cmds.listRelatives(transform, shapes=True, fullPath=True) or []
        if not shapes:
            return None

        shader_name = None

        for sg in cmds.listConnections(shapes, type="shadingEngine") or []:
            for shader in cmds.listConnections(f"{sg}.surfaceShader") or []:

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

    # ── Scan pipeline ────────────────────────────────────────

    def scan_and_classify(self, progress_callback=None):
        """
        Classify every object in self.objects.

        Phase 1 — collect texture sets and shader names (read-only Maya queries).
        Phase 2 — run ML inference on each albedo path.
        Phase 3 — store results; nothing is written to the scene.
        """
        self.results = {}
        total = len(self.objects)

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

        transforms_list = list(classifiable.keys())
        sets_list       = list(classifiable.values())

        predictions = [predict(ts["albedo_path"]) for ts in sets_list]

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

            print(
                f"[pbr_tools] {short} → {prediction['label']} "
                f"({prediction['confidence']*100:.1f}%) | "
                f"{len(tex_set['all_paths'])} map(s) found"
            )

        return self.results

    # ── Apply approved ────────────────────────────────────────

    def apply_approved(self, review_queue, output_dir,
                       dry_run=False, progress_callback=None):
        """
        For every accepted entry:
          1. Write materialType + mlConfidence to its shader node.
          2. Move its texture set to output_dir/<label>/<asset>/.
          3. Update every Maya file node whose path has moved.

        File collection strategy:
          all_paths from the scan captures what the node graph walk found.
          To also catch maps that are connected in Maya but were missed
          (e.g. wired through unusual intermediate nodes), we additionally
          scan every file node in the scene and include any whose path sits
          in the same source folder as the albedo.

        Only accepted items are touched. Rejected and pending are left alone.
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
            return {"metadata_written": 0, "files_moved": 0,
                    "skipped": 0, "failed": 0}

        total            = len(accepted)
        metadata_written = 0
        files_moved      = 0
        skipped          = 0
        failed           = 0
        old_to_new       = {}
        processed_albedos = set()

        for i, entry in enumerate(accepted):

            transform   = entry["transform"]
            short       = entry["short"]
            confidence  = entry["confidence"]
            albedo_path = entry["albedo_path"]
            label       = entry.get("override") or entry["label"]

            # Prefer live results (paths may have changed from a prior organize)
            all_paths = (
                self.results.get(transform, {}).get("all_paths") or [albedo_path]
            )

            if progress_callback:
                progress_callback(i + 1, total, short)

            # ── Write metadata ────────────────────────────────
            if dry_run:
                print(f"{prefix} Would tag '{short}' → "
                      f"materialType={label}, mlConfidence={confidence}")
            else:
                shader = self.write_metadata(transform, label, confidence)
                if transform in self.results:
                    self.results[transform]["shader"] = shader
                    self.results[transform]["label"]  = label
                metadata_written += 1

            if not output_dir:
                continue

            norm_albedo = os.path.normpath(albedo_path)
            if norm_albedo in processed_albedos:
                print(f"{prefix} Skipping shared texture set for "
                      f"{short} — already moved")
                continue
            processed_albedos.add(norm_albedo)

            asset_name  = resolve_asset_name(albedo_path)
            dest_folder = os.path.normpath(
                os.path.join(output_dir, label, asset_name)
            )
            src_dir = os.path.normpath(os.path.dirname(albedo_path))

            if dry_run:
                print(f"{prefix} Would create: {dest_folder}/")

            # ── Build candidate list ──────────────────────────
            # Start with what the node graph walk found, then add any
            # file node in the scene whose path lives in the same folder.
            # This catches maps connected through slots we don't enumerate
            # and ensures their Maya paths are updated after the move.
            known_norm = {os.path.normpath(p) for p in all_paths}
            candidate_paths = list(all_paths)

            for fn in cmds.ls(type="file") or []:
                p = cmds.getAttr(f"{fn}.fileTextureName")
                if not p:
                    continue
                norm_p = os.path.normpath(p)
                if (
                    os.path.normpath(os.path.dirname(norm_p)) == src_dir
                    and norm_p not in known_norm
                    and os.path.exists(norm_p)
                    and ".mayaSwatches" not in norm_p.replace("\\", "/")
                ):
                    candidate_paths.append(p)
                    known_norm.add(norm_p)

            # ── Move files ────────────────────────────────────
            for src in candidate_paths:
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
                    print(f"{prefix} Would move: {os.path.basename(src)}  →  "
                          f"{label}/{asset_name}/{os.path.basename(src)}")
                    files_moved += 1
                else:
                    try:
                        os.makedirs(dest_folder, exist_ok=True)
                        shutil.move(src, dest_file)
                        old_to_new[norm_src] = dest_file
                        files_moved += 1
                        print(f"{prefix} Moved → {label}/{asset_name}/"
                              f"{os.path.basename(src)}")
                    except Exception as err:
                        print(f"{prefix} Failed to move "
                              f"{os.path.basename(src)}: {err}")
                        failed += 1

            # ── Cleanup empty source folders ──────────────────
            if not dry_run and candidate_paths:
                _remove_if_empty(src_dir)
                _remove_if_empty(os.path.dirname(src_dir))

        # ── Update all Maya file texture nodes ────────────────
        # old_to_new now contains every moved file, including sibling maps
        # that weren't in the original all_paths list. Step through every
        # file node in the scene and redirect any whose path has moved.
        if old_to_new and not dry_run:
            print("[pbr_tools] Updating Maya file texture paths...")

            for file_node in cmds.ls(type="file") or []:
                current = cmds.getAttr(f"{file_node}.fileTextureName")
                if not current:
                    continue
                norm = os.path.normpath(current)
                if norm in old_to_new:
                    cmds.setAttr(f"{file_node}.fileTextureName",
                                 old_to_new[norm], type="string")
                    print(f"[pbr_tools] Updated '{file_node}' → {old_to_new[norm]}")

            # Refresh stored results so the detail panel shows new paths
            for transform, data in self.results.items():
                p = data.get("albedo_path")
                if p and os.path.normpath(p) in old_to_new:
                    data["albedo_path"] = old_to_new[os.path.normpath(p)]
                if "all_paths" in data:
                    data["all_paths"] = [
                        old_to_new.get(os.path.normpath(p), p)
                        for p in data["all_paths"]
                    ]

        print(f"{prefix} Apply complete — {metadata_written} tagged, "
              f"{files_moved} moved, {skipped} skipped, {failed} failed")

        return {
            "metadata_written": metadata_written,
            "files_moved":      files_moved,
            "skipped":          skipped,
            "failed":           failed,
        }