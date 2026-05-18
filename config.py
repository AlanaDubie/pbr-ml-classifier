# ── config.py ────────────────────────────────────────────────
# Central config for the PBR Material Classifier.
# The only file that needs updating if you move the project
# to a new machine or Maya version.
#
# Imported by:
#   classifier.py  — needs CKPT_PATH, CLASSES, IMAGE_SIZE
#   pbr_tools.py   — no direct import, but PYTHON_EXE is documented here
# ─────────────────────────────────────────────────────────────

import os
import sys
import platform

# Folder this file lives in — all other paths are relative to it
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Trained model checkpoint exported from Colab
CKPT_PATH = os.path.join(SCRIPTS_DIR, "pbr_classifier.pth")

# Maya's own Python executable.
# Executable name differs by OS: mayapy.exe on Windows, mayapy on macOS.
# sys.executable points to maya.exe / maya when running inside Maya,
# so swapping the filename gives us mayapy in the same bin folder.
_mayapy_exe = "mayapy.exe" if platform.system() == "Windows" else "mayapy"
MAYA_BIN    = os.path.dirname(sys.executable)
PYTHON_EXE  = os.path.join(MAYA_BIN, _mayapy_exe)

# Material categories — order must match the training class order
CLASSES = ["fabric", "ground", "metal", "rock", "wood"]

# Image size the model expects (set during training)
IMAGE_SIZE = 224