"""Dashboard summary and safe local inspection thumbnails."""
import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.database import DATABASE_PATH, init_database
from backend.ui import PROJECT_DIR
from backend.labeling_api import growth_dataset_summary

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])
UPLOAD_DIR = (PROJECT_DIR / "uploads").resolve()
LABEL_ROOT = PROJECT_DIR / "dataset" / "quality"
LABELS = ("good", "usable", "poor", "needs_review")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _image_column_exists(connection):
    return any(row[1] == "image_path" for row in connection.execute("PRAGMA table_info(inspections)"))


def _safe_stored_image(image_path):
    if not image_path or not isinstance(image_path, str):
        return None
    relative = Path(image_path.replace("\\", "/"))
    if len(relative.parts) != 2 or relative.parts[0] != "uploads":
        return None
    if relative.suffix.lower() not in EXTENSIONS:
        return None
    destination = (UPLOAD_DIR / relative.parts[1]).resolve()
    if destination.parent != UPLOAD_DIR or not destination.is_file():
        return None
    return destination


@router.get("/summary")
def dashboard_summary():
    init_database()
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        stats = conn.execute("""
            SELECT COUNT(*) AS total_inspections,
              COUNT(DISTINCT CASE WHEN classification_type = 'species'
                THEN predicted_species END) AS species_identified,
              ROUND(AVG(CASE WHEN classification_type = 'species'
                THEN confidence_score END), 2) AS average_confidence,
              ROUND(AVG(CASE WHEN classification_type = 'species'
                THEN inference_time_ms END), 2) AS average_inference_ms
            FROM inspections
        """).fetchone()
        species = conn.execute("""
            SELECT predicted_species AS species, COUNT(*) AS total
            FROM inspections
            WHERE classification_type = 'species' AND predicted_species IS NOT NULL
            GROUP BY predicted_species ORDER BY total DESC, predicted_species
        """).fetchall()
        image_field = "image_path" if _image_column_exists(conn) else "NULL AS image_path"
        recent = conn.execute(f"""
            SELECT id, sample_id, predicted_species, confidence_score,
                   origin, inspection_date, {image_field}
            FROM inspections ORDER BY id DESC LIMIT 5
        """).fetchall()

    label_counts = {}
    for label in ("unlabeled", *LABELS):
        folder = LABEL_ROOT / label
        label_counts[label] = sum(
            1 for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ) if folder.is_dir() else 0
    label_total = sum(label_counts.values())
    labeled = label_total - label_counts["unlabeled"]

    return {
        "status": "success",
        "total_inspections": stats["total_inspections"],
        "species_identified": stats["species_identified"],
        "average_confidence": stats["average_confidence"],
        "average_inference_ms": stats["average_inference_ms"],
        "species_distribution": [dict(row) for row in species],
        "recent_inspections": [
            {
                **{key: row[key] for key in row.keys() if key != "image_path"},
                "image_saved": _safe_stored_image(row["image_path"]) is not None,
            }
            for row in recent
        ],
        "labeling": {
            "counts": label_counts,
            "total": label_total,
            "processed": labeled,
            "progress_percent": round(labeled * 100 / label_total, 1) if label_total else 0,
        },
        "growth_dataset": growth_dataset_summary(),
        "classification_type": "species_demo",
    }


@router.get("/image/{inspection_id}")
def dashboard_inspection_image(inspection_id: int):
    init_database()
    with sqlite3.connect(DATABASE_PATH) as conn:
        if not _image_column_exists(conn):
            raise HTTPException(status_code=404, detail="No inspection images stored yet.")
        row = conn.execute(
            "SELECT image_path FROM inspections WHERE id = ?", (inspection_id,)
        ).fetchone()
    image = _safe_stored_image(row[0]) if row else None
    if image is None:
        raise HTTPException(status_code=404, detail="Inspection image unavailable.")
    return FileResponse(image)
