# ── texture_name_parser.py ────────────────────────────────────
# Extracts a clean asset identifier from a texture filename by
# stripping known map type tokens.
#
# Used by pbr_tools.py to determine the subfolder name when
# organizing a texture set on disk:
#
#   cliff_rock_a_basecolor_v1.exr  →  cliff_rock_a_v1
#   muddy_ground_02_albedo.png     →  muddy_ground_02
#   old_wood_planks_a_col.tif      →  old_wood_planks_a
#
# The asset name becomes the folder that holds all sibling maps:
#   textures/rock/cliff_rock_a_v1/
#       cliff_rock_a_basecolor_v1.exr
#       cliff_rock_a_normal_v1.exr
#       cliff_rock_a_roughness_v1.exr
# ─────────────────────────────────────────────────────────────

import os
import re


# Known texture map channel tokens across common DCCs and export conventions
# (Substance Painter, Maya, Unreal, Houdini, Marmoset, etc.)
MAP_TOKENS = frozenset({
    # albedo / base colour variants
    "basecolor", "base_color", "albedo", "diffuse", "diff",
    "col", "color", "colour",
    # normal map variants
    "normal", "nrm", "nor", "normaldx", "normalgl",
    # roughness variants
    "roughness", "rough", "rgh",
    # metallic variants
    "metallic", "metal", "metalness", "met",
    # other common maps
    "height", "displacement", "disp",
    "ambientocclusion", "ao",
    "emissive", "emission", "emit",
    "opacity", "alpha", "mask",
    "specular", "spec",
})


def resolve_asset_name(file_path: str) -> str:
    """
    Extract a clean asset identifier from a texture file path by
    stripping known map type tokens.

    The result is used as the subfolder name for the texture set.

    Examples:
        cliff_rock_a_basecolor_v1.exr  →  cliff_rock_a_v1
        muddy_ground_02_albedo.png     →  muddy_ground_02
        old_wood_planks_a_col.tif      →  old_wood_planks_a
        rust_metal_rough.png           →  rust_metal

    Behaviour:
        - Strips known map tokens (case-insensitive)
        - Keeps everything else including version tags (v1, v02 etc.)
        - Preserves original token ordering
        - Does NOT enforce or validate naming conventions
        - Returns the raw filename stem (no extension) if no tokens
          could be stripped — better to create a flat folder than fail
    """
    filename  = os.path.basename(file_path)
    stem      = os.path.splitext(filename)[0]

    # Split on common separators, preserve case for the rebuilt name
    tokens = re.split(r"[_\-. ]+", stem)

    kept = [t for t in tokens if t and t.lower() not in MAP_TOKENS]

    # If every token was stripped (unlikely but possible), fall back to
    # the full stem so we still get a valid folder name
    return "_".join(kept) if kept else stem


def get_map_type(file_path: str) -> str | None:
    """
    Identify which map type a texture file represents, or return None
    if no known token is found.

    Used for labelling file nodes in the UI and debug output.

    Examples:
        cliff_rock_a_basecolor_v1.exr  →  "basecolor"
        muddy_ground_02_normal.png     →  "normal"
        old_wood_planks_a_rough.tif    →  "roughness"
        some_random_file.png           →  None
    """
    stem   = os.path.splitext(os.path.basename(file_path))[0]
    tokens = [t.lower() for t in re.split(r"[_\-. ]+", stem) if t]

    # Normalise known aliases to a canonical name
    ALIASES = {
        "basecolor": "basecolor", "base_color": "basecolor",
        "albedo":    "basecolor", "diffuse":    "basecolor",
        "diff":      "basecolor", "col":        "basecolor",
        "color":     "basecolor", "colour":     "basecolor",
        "normal":    "normal",    "nrm":        "normal",
        "nor":       "normal",    "normaldx":   "normal",
        "normalgl":  "normal",
        "roughness": "roughness", "rough":      "roughness",
        "rgh":       "roughness",
        "metallic":  "metallic",  "metal":      "metallic",
        "metalness": "metallic",  "met":        "metallic",
        "height":    "height",    "displacement":"height",
        "disp":      "height",
        "ao":        "ao",        "ambientocclusion": "ao",
    }

    for token in tokens:
        if token in ALIASES:
            return ALIASES[token]

    return None


def debug_parse(file_path: str) -> dict:
    """
    Return a full breakdown of parsing decisions for a texture path.
    Useful for UI tooltips, pipeline validation, and debugging.

    Returns:
        {
            "filename":         "cliff_rock_a_basecolor_v1.exr",
            "asset_name":       "cliff_rock_a_v1",
            "map_type":         "basecolor",
            "map_tokens_found": ["basecolor"],
            "kept_tokens":      ["cliff", "rock", "a", "v1"],
            "all_tokens":       ["cliff", "rock", "a", "basecolor", "v1"],
        }
    """
    filename = os.path.basename(file_path)
    stem     = os.path.splitext(filename)[0]
    tokens   = [t for t in re.split(r"[_\-. ]+", stem) if t]

    map_tokens_found = [t for t in tokens if t.lower() in MAP_TOKENS]
    kept_tokens      = [t for t in tokens if t.lower() not in MAP_TOKENS]

    return {
        "filename":         filename,
        "asset_name":       "_".join(kept_tokens) if kept_tokens else stem,
        "map_type":         get_map_type(file_path),
        "map_tokens_found": map_tokens_found,
        "kept_tokens":      kept_tokens,
        "all_tokens":       tokens,
    }