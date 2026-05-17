# ── infer.py ─────────────────────────────────────────────────
# Standalone inference script called by Maya via subprocess.
# Takes a texture image path as a command line argument and
# prints a JSON result to stdout:
#   {"label": "wood", "confidence": 0.99, "all_scores": {...}}
#
# Maya never imports this directly — it calls it as a child
# process through SceneScanner.run_inference() and reads back
# the JSON response from stdout.
# ─────────────────────────────────────────────────────────────

import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image as PILImage

from config import CKPT_PATH, CLASSES, IMAGE_SIZE

# Always run on CPU — this script is called locally by Maya,
# not inside Colab, so no GPU is available or needed
DEVICE = torch.device("cpu")

# ── Preprocessing transform ───────────────────────────────────
# Must exactly match val_transforms used during training
# so the model sees the same input distribution it was trained on
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225]),   # ImageNet std
])


# ── Model loader ──────────────────────────────────────────────
def load_model():
    # Rebuild the same ResNet-18 architecture used during training
    # with the fc layer replaced to output 5 PBR classes
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    # Load the checkpoint weights exported from Colab
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────
def predict(image_path: str) -> dict:
    # Open and validate the texture image
    try:
        img = PILImage.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Could not open image: {e}"}

    model  = load_model()
    tensor = transform(img).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]  # [num_classes]

    top_idx = probs.argmax().item()
    return {
        "label":      CLASSES[top_idx],
        "confidence": round(probs[top_idx].item(), 4),
        # All class probabilities — useful for UI confidence display
        "all_scores": {c: round(probs[i].item(), 4) for i, c in enumerate(CLASSES)},
    }


# ── Entry point ───────────────────────────────────────────────
# Called by SceneScanner via:
#   subprocess.run([mayapy, infer.py, texture_path])
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    result = predict(sys.argv[1])
    print(json.dumps(result))