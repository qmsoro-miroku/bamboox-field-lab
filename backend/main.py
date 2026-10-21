
"""BambooX Field Lab — FastAPI application.

Version 3.0 Phase A:
- Welcome screen
- Mission Hub
- Camera preparation page
- Existing species AI and SQLite endpoints preserved
"""

from io import BytesIO
from pathlib import Path
from backend.growth_detection_api import router as growth_detection_router

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.staticfiles import StaticFiles

from PIL import Image, ImageOps, UnidentifiedImageError

from backend.ui import FRONTEND_DIR, render_page
from backend.predictor import predict_species
from backend.records_api import router as records_router
from backend.dashboard_api import router as dashboard_router
from backend.labeling_api import router as labeling_router
from backend.settings_api import router as settings_router


app = FastAPI(
    title="BambooX Field Lab API",
    description="Offline-capable bamboo computer vision workspace",
    version="3.0.0",
)

app.mount(
    "/assets",
    StaticFiles(directory=str(FRONTEND_DIR / "assets")),
    name="assets",
)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


# ============================================================
# FIELD LAB PAGES
# ============================================================

@app.get("/", include_in_schema=False)
def welcome_screen(request: Request):
    return render_page(request, "intro.html", "home")


@app.get("/app/hub", include_in_schema=False)
def mission_hub(request: Request):
    return render_page(request, "hub.html", "hub")


@app.get("/app/camera", include_in_schema=False)
def live_camera_page(request: Request):
    return render_page(request, "camera.html", "camera")


# ============================================================
# EXISTING PAGES
# ============================================================

@app.get("/app", include_in_schema=False)
def bamboo_dashboard(request: Request):
    return render_page(request, "index.html", "dashboard")


@app.get("/app/inspection", include_in_schema=False)
def new_inspection_page(request: Request):
    return render_page(request, "inspection.html", "inspection")


@app.get("/app/records", include_in_schema=False)
def show_records_page(request: Request):
    return render_page(request, "records.html", "records")


@app.get("/app/settings", include_in_schema=False)
def settings_page(request: Request):
    return render_page(request, "settings.html", "settings")


# ============================================================
# SYSTEM HEALTH
# ============================================================

@app.get("/health")
def health_check():
    # Preserve previous compatibility response.
    # This is not a live model readiness test.
    return {
        "system": "BambooX",
        "backend": "operational",
        "ai_model": "not_connected",
        "database": "not_connected",
    }


# ============================================================
# EXISTING IMAGE VALIDATION
# ============================================================

async def _read_image(file: UploadFile):
    try:
        image_bytes = await file.read(MAX_UPLOAD_SIZE + 1)
    finally:
        await file.close()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB limit.",
        )

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = image.format
            width, height = image.size

            if image_format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail="Only JPG, PNG, and WEBP are supported.",
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions are too large.",
                )

            image.verify()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image.",
        )

    return image_bytes, image_format, width, height


@app.post("/api/upload-image")
async def upload_bamboo_image(
    file: UploadFile = File(...),
    sample_id: str = Form(...),
    species: str = Form(""),
    origin: str = Form(""),
    notes: str = Form(""),
):
    sample_id = sample_id.strip()

    if not sample_id or len(sample_id) > 80:
        raise HTTPException(
            status_code=400,
            detail="Sample ID must contain 1-80 characters.",
        )

    image_bytes, image_format, width, height = await _read_image(file)

    return {
        "status": "validated",
        "message": "Bamboo image received successfully!",
        "sample_id": sample_id,
        "species": species,
        "origin": origin,
        "notes": notes,
        "filename": Path(file.filename or "image").name,
        "image_format": image_format,
        "image_width": width,
        "image_height": height,
        "image_size_kb": round(len(image_bytes) / 1024, 2),
        "ai_model": "not_connected",
        "saved_to_database": False,
    }


# ============================================================
# EXISTING SPECIES PREDICTION
# ============================================================

@app.post("/api/predict-species")
async def predict_bamboo_species(
    file: UploadFile = File(...),
    sample_id: str = Form(...),
):
    sample_id = sample_id.strip()

    if not sample_id or len(sample_id) > 80:
        raise HTTPException(
            status_code=400,
            detail="Sample ID must contain 1-80 characters.",
        )

    image_bytes, _, _, _ = await _read_image(file)

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image.",
        )

    try:
        result = predict_species(image)

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Trained AI model file not found.",
        )

    return {
        "status": "success",
        "sample_id": sample_id,
        "filename": Path(file.filename or "image").name,
        **result,
        "saved_to_database": False,
    }


# ============================================================
# PRESERVE ALL EXISTING API ROUTERS
# ============================================================

app.include_router(records_router)
app.include_router(dashboard_router)
app.include_router(labeling_router)
app.include_router(settings_router)
app.include_router(growth_detection_router)

