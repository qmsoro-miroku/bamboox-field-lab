"""
BambooX 3.0 — Growth-Stage Detection API

YOLOv8n object detection:
    0 = matured
    1 = old
    2 = young

This is separate from the existing species classifier.
No database writes are performed by this endpoint.
"""

from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO


router = APIRouter(tags=["Growth-Stage Detection"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolov8n_growth_stage_best.pt"
)

EXPECTED_CLASSES = {
    0: "matured",
    1: "old",
    2: "young",
}

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

CONFIDENCE_THRESHOLD = 0.25

_model = None
_model_lock = Lock()


def _load_model():
    """Load the trained detector once per server process."""
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Growth-stage model file was not found.",
        )

    try:
        model = YOLO(str(MODEL_PATH))

        if model.task != "detect":
            raise ValueError("Expected a detection model.")

        if model.names != EXPECTED_CLASSES:
            raise ValueError(
                "Unexpected growth-stage class mapping."
            )

        _model = model
        return _model

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Growth-stage model could not be loaded.",
        ) from exc


def _perform_detection(image):
    """Run local CPU inference and build a JSON response."""

    # Serialize inference because this model instance is shared
    # between requests.
    with _model_lock:

        model = _load_model()

        start_time = perf_counter()

        try:
            results = model.predict(
                source=image,
                imgsz=640,
                conf=CONFIDENCE_THRESHOLD,
                device="cpu",
                verbose=False,
            )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Growth-stage inference failed.",
            ) from exc

        elapsed_ms = (
            perf_counter() - start_time
        ) * 1000

        result = results[0]

        detections = []
        class_counts = {
            "matured": 0,
            "old": 0,
            "young": 0,
        }

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls.item())

                if class_id not in EXPECTED_CLASSES:
                    continue

                class_name = EXPECTED_CLASSES[class_id]

                confidence = float(
                    box.conf.item()
                )

                x1, y1, x2, y2 = (
                    box.xyxy[0].cpu().tolist()
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,

                    "confidence": round(
                        confidence, 6
                    ),

                    "confidence_percent": round(
                        confidence * 100, 2
                    ),

                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                })

                class_counts[class_name] += 1

        detections.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        return {
            "status": "success",
            "task": "growth_stage_detection",

            "model": "YOLOv8n",
            "model_file": MODEL_PATH.name,

            "image_width": image.width,
            "image_height": image.height,

            "confidence_threshold": CONFIDENCE_THRESHOLD,

            "detection_count": len(detections),
            "class_counts": class_counts,

            "detections": detections,

            "inference_time_ms": round(
                elapsed_ms, 2
            ),

            "disclaimer": (
                "Predictions represent dataset-defined "
                "growth-stage categories and do not "
                "independently establish biological age "
                "or structural safety."
            ),
        }


@router.post("/api/detect-growth-stage")
async def detect_growth_stage(
    file: UploadFile = File(...)
):
    """
    Accept JPG, PNG, or WEBP.
    Return YOLO bounding boxes and growth-stage labels.
    """

    try:
        content = await file.read(
            MAX_FILE_SIZE + 1
        )

    finally:
        await file.close()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB limit.",
        )

    try:
        with Image.open(BytesIO(content)) as source:

            if source.format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Only JPG, PNG, and WEBP "
                        "images are supported."
                    ),
                )

            width, height = source.size

            if (
                width <= 0
                or height <= 0
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions are invalid or too large.",
                )

            image = ImageOps.exif_transpose(
                source
            ).convert("RGB")

            image.load()

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image.",
        ) from exc

    return await run_in_threadpool(
        _perform_detection,
        image,
    )
