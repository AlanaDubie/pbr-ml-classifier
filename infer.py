import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image as PILImage

# ── Config ───────────────────────────────────────────────────
CLASSES    = ["fabric", "ground", "metal", "rock", "wood"]
IMAGE_SIZE = 224
CKPT_PATH  = r"pbr_classifier.pth"   
DEVICE     = torch.device("cpu")  # CPU only — Maya use

# ── Transform ────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Load model ───────────────────────────────────────────────
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

# ── Inference ────────────────────────────────────────────────
def predict(image_path: str) -> dict:
    try:
        img = PILImage.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Could not open image: {e}"}

    model  = load_model()
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    top_idx = probs.argmax().item()
    return {
        "label":      CLASSES[top_idx],
        "confidence": round(probs[top_idx].item(), 4),
        "all_scores": {c: round(probs[i].item(), 4) for i, c in enumerate(CLASSES)},
    }

# ── Entry point — called by Maya via subprocess ───────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    result = predict(sys.argv[1])
    print(json.dumps(result))