# ── config.py ────────────────────────────────────────────────
# Central configuration for the PBR ML Classifier.
#
# This is the only file that needs to change if you move the
# project to a different machine or folder. Everything else
# imports from here so there is no path hardcoding anywhere else.
#
# Imported by:
#   classifier.py   — needs CKPT_PATH, CLASSES, IMAGE_SIZE
#   SceneScanner.py — needs INFER_SCRIPT, PYTHON_EXE
# ─────────────────────────────────────────────────────────────

import os
import sys

# ── Folder location ───────────────────────────────────────────
# __file__ is the path to this config.py file.
# dirname gives us the folder it lives in.
# All other paths are built relative to this folder so the
# project works on any machine without manual path setup.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ── File paths ────────────────────────────────────────────────

# The trained model weights exported from Colab
CKPT_PATH = os.path.join(SCRIPTS_DIR, "pbr_classifier.pth")

# The command line inference script — used for testing outside Maya
INFER_SCRIPT = os.path.join(SCRIPTS_DIR, "infer.py")

# ── Maya Python executable ────────────────────────────────────
# mayapy.exe lives in the same bin folder as maya.exe.
# sys.executable gives us the path to maya.exe when running
# inside Maya, so we just swap the filename to get mayapy.exe.
MAYA_BIN   = os.path.dirname(sys.executable)
PYTHON_EXE = os.path.join(MAYA_BIN, "mayapy.exe")

# ── Model settings ────────────────────────────────────────────

# The five material categories the model was trained to recognise.
# Order must match the class order used during training.
CLASSES = ["fabric", "ground", "metal", "rock", "wood"]

# Input image size the model expects — set during training
IMAGE_SIZE = 224