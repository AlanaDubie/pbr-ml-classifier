# ── classifier.py ────────────────────────────────────────────
# Handles model loading and inference for the PBR ML Classifier.
#
# The model is loaded LAZILY — meaning it only loads from disk
# the first time predict() is called, not when the file is imported.
# This keeps Maya from freezing when the tool window opens.
#
# After the first call, the model stays in memory for the entire
# Maya session. Every scan after the first is fast because the
# model is already loaded and ready.
#
# Imported by:
#   SceneScanner.py — calls predict() during scan_and_classify()
#   infer.py        — also uses these functions for CLI testing
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image as PILImage

from config import CKPT_PATH, CLASSES, IMAGE_SIZE

# Always run on CPU — this runs locally inside Maya, no GPU available
DEVICE = torch.device("cpu")

# ── Preprocessing transform ───────────────────────────────────
# Prepares an image for the model:
#   1. Resize to 224x224 (what the model expects)
#   2. Convert pixels to a tensor (numbers the model can process)
#   3. Normalise using ImageNet mean and std — required because
#      ResNet-18 was originally trained on ImageNet images
#
# This must exactly match val_transforms used during training.
_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225]),   # ImageNet std
])

# ── Model cache ───────────────────────────────────────────────
# Starts as None. Gets filled the first time predict() is called.
# Stays loaded in memory for all future calls in the same session.
_model = None


def _load_model():
    """
    Load the trained ResNet-18 model from the checkpoint file.
    This is called automatically on the first predict() call.
    Takes ~2s on first run, then the result is cached in _model.
    """

    print("[Classifier] Loading model from checkpoint...")

    # Recreate the same ResNet-18 architecture used during training.
    # weights=None means we are NOT loading ImageNet weights —
    # we will load our own trained weights from the checkpoint instead.
    model = models.resnet18(weights=None)

    # The original ResNet-18 outputs 1000 ImageNet classes.
    # We replace the final layer to output our 5 PBR material classes.
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    # Load the weights saved at the end of Colab training
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # eval() switches off dropout and batch normalisation training behaviour
    # — must be called before running any predictions
    model.eval()

    print("[Classifier] Model ready.")
    return model


def predict(image_path: str) -> dict:
    """
    Classify a single texture image and return the predicted material.

    On the first call this loads the model from disk (~2s).
    On every call after that the model is already in memory (~50-100ms).

    Returns a dict:
      {
        "label":      "wood",       — predicted material category
        "confidence": 0.9999,       — how confident the model is (0-1)
        "all_scores": {             — probability for every category
            "fabric": 0.0,
            "ground": 0.0,
            "metal":  0.0,
            "rock":   0.0,
            "wood":   0.9999
        }
      }

    Returns {"error": "..."} if the image cannot be opened.
    """

    # Load the model on the first call, reuse it on all future calls
    global _model
    if _model is None:
        _model = _load_model()

    # Open the texture and convert to RGB.
    # Some textures are saved as RGBA (with alpha) or greyscale —
    # convert("RGB") forces 3 colour channels so the model always
    # gets the same format regardless of how the file was saved.
    try:
        img = PILImage.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Could not open image: {e}"}

    # Preprocess the image into a tensor the model can read.
    # unsqueeze(0) adds a batch dimension:
    #   [3, 224, 224]  →  [1, 3, 224, 224]
    # The model expects a batch even when classifying a single image.
    tensor = _transform(img).unsqueeze(0).to(DEVICE)

    # Run the forward pass through the model.
    # torch.no_grad() tells PyTorch we are not training — this saves
    # memory and makes inference faster.
    with torch.no_grad():
        logits = _model(tensor)                      # raw output scores [1, 5]
        probs  = torch.softmax(logits, dim=1)[0]     # convert to probabilities [5]

    # The class with the highest probability is our prediction
    top_idx = probs.argmax().item()

    return {
        "label":      CLASSES[top_idx],
        "confidence": round(probs[top_idx].item(), 4),
        # Include all scores so the UI can show the full breakdown
        "all_scores": {c: round(probs[i].item(), 4) for i, c in enumerate(CLASSES)},
    }