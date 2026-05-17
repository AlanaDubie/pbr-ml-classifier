import os
import sys

# ── config.py ────────────────────────────────────────────────
# Central configuration for the PBR ML Classifier pipeline.
# All paths are resolved relative to this file's location so
# the entire pbr-ml-classifier folder can be dropped anywhere
# on any machine without any manual path setup required.
# ─────────────────────────────────────────────────────────────

# Root folder — resolves to wherever this config.py lives
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the trained ResNet-18 checkpoint exported from Colab
CKPT_PATH = os.path.join(SCRIPTS_DIR, "pbr_classifier.pth")

# Path to the inference script called by Maya via subprocess
INFER_SCRIPT = os.path.join(SCRIPTS_DIR, "infer.py")

# mayapy.exe always lives in the same bin folder as maya.exe
# Using sys.executable from inside Maya gives us maya.exe,
# so we just swap the filename to get mayapy.exe
MAYA_BIN   = os.path.dirname(sys.executable)
PYTHON_EXE = os.path.join(MAYA_BIN, "mayapy.exe")

# Material categories — must match the class order used during training
CLASSES = ["fabric", "ground", "metal", "rock", "wood"]

# Input image size expected by the model (matches training transforms)
IMAGE_SIZE = 224