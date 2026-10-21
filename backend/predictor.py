from pathlib import Path
from functools import lru_cache
import time

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" /
    "mobilenetv3_species_baseline.pth"
)

# Same preprocessing used during Colab evaluation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@lru_cache(maxsize=1)
def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    class_names = checkpoint["class_names"]

    model = mobilenet_v3_small(weights=None)

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, class_names


def predict_species(image):
    model, class_names = load_model()

    image = image.convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    inference_time = (
        time.perf_counter() - start
    ) * 1000

    confidence, predicted_index = torch.max(
        probabilities, dim=0
    )

    top_values, top_indices = torch.topk(
        probabilities,
        k=min(3, len(class_names))
    )

    top_predictions = [
        {
            "species": class_names[idx.item()],
            "confidence_percent": round(
                value.item() * 100, 2
            )
        }
        for value, idx in zip(top_values, top_indices)
    ]

    return {
        "predicted_species": class_names[
            predicted_index.item()
        ],
        "confidence_percent": round(
            confidence.item() * 100, 2
        ),
        "inference_time_ms": round(
            inference_time, 2
        ),
        "top_predictions": top_predictions,
        "model": "MobileNetV3-Small",
        "classification_type": "species"
    }
