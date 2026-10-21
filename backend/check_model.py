from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

# Locate the trained model
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "mobilenetv3_species_baseline.pth"
)

print("BAMBOOX AI MODEL CHECK")
print("---------------------------")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

# Load our trusted Colab training checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

class_names = checkpoint["class_names"]

# Recreate the training architecture
model = mobilenet_v3_small(weights=None)

in_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    in_features,
    len(class_names)
)

# Load trained weights
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully!")
print("Architecture:", checkpoint["architecture"])
print("Number of classes:", len(class_names))
print("Classes:", class_names)
print("Device: CPU")
print("AI model verification PASSED!")
