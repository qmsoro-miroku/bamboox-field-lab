"""Read-only local system diagnostics for BambooX Settings.

This endpoint never creates, modifies, backs up, or deletes the user's data.
Presence of a model checkpoint does not prove that ML inference will succeed.
"""
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from fastapi import APIRouter

from backend.database import DATABASE_PATH
from backend.ui import PROJECT_DIR
from backend.labeling_api import growth_dataset_summary

router = APIRouter(prefix="/api/settings", tags=["Settings"])

MODEL_PATH = PROJECT_DIR / "models" / "mobilenetv3_species_baseline.pth"
UPLOADS_DIR = PROJECT_DIR / "uploads"
DATASET_DIR = PROJECT_DIR / "dataset" / "quality"
LABEL_FOLDERS = ("good", "usable", "poor", "needs_review", "unlabeled")
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _folder_stats(folder: Path):
    """Count regular files only, without recursing or following symlinks."""
    count = 0
    total_bytes = 0
    if not folder.is_dir():
        return {"count": count, "bytes": total_bytes}
    try:
        for image in folder.iterdir():
            if image.is_file() and not image.is_symlink() and image.suffix.lower() in IMAGE_TYPES:
                count += 1
                total_bytes += _file_size(image)
    except OSError:
        pass
    return {"count": count, "bytes": total_bytes}


def _database_stats():
    """Open an existing DB read-only; never create a new empty database."""
    exists = DATABASE_PATH.is_file()
    result = {
        "exists": exists, "status": "missing" if not exists else "ready",
        "record_count": 0, "size_bytes": _file_size(DATABASE_PATH),
        "path": str(DATABASE_PATH), "relative_path": "database/bamboox.db",
        "error": None,
    }
    if not exists:
        return result
    try:
        # Use read-only URI for safety. SQLite handles Windows and POSIX paths.
        uri = f"file:{DATABASE_PATH.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=3) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'inspections'"
            ).fetchone()
            if row:
                result["record_count"] = conn.execute(
                    "SELECT COUNT(*) FROM inspections"
                ).fetchone()[0]
            else:
                result["status"] = "missing_table"
    except (sqlite3.Error, OSError):
        result["status"] = "error"
        result["error"] = "SQLite file could not be read. Check the server logs."
    return result


@router.get("/overview")
def settings_overview():
    labels = {label: _folder_stats(DATASET_DIR / label) for label in LABEL_FOLDERS}
    uploads = _folder_stats(UPLOADS_DIR)
    checkpoint_present = MODEL_PATH.is_file()
    return {
        "application": "BambooX",
        "version": "3.0 research prototype",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": {"status": "running", "execution": "local FastAPI", "external_services_required": False},
        "model": {
            "name": "MobileNetV3-Small", "task": "Bamboo species recognition",
            "class_count": 9, "checkpoint_present": checkpoint_present,
            "checkpoint_size_bytes": _file_size(MODEL_PATH),
            "relative_path": "models/mobilenetv3_species_baseline.pth",
            "note": "Checkpoint presence is not an inference health check. Growth-stage detection uses a separate YOLO model.",
        },
        "growth_model": {
            "name": "YOLOv8n", "task": "Growth-stage object detection",
            "classes": ["matured", "old", "young"],
            "checkpoint_present": (PROJECT_DIR / "models" / "yolov8n_growth_stage_best.pt").is_file(),
            "checkpoint_size_bytes": _file_size(PROJECT_DIR / "models" / "yolov8n_growth_stage_best.pt"),
        },
        "database": _database_stats(),
        "storage": {
            "uploads": uploads,
            "growth_dataset": growth_dataset_summary(),
            "labels": labels,
            "total_dataset_images": sum(info["count"] for info in labels.values()),
            "total_dataset_bytes": sum(info["bytes"] for info in labels.values()),
        },
    }
