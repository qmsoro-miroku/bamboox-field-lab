"""BambooX local inspection API: species demo, records, filters, export, delete.

Compatible with the existing inspections SQLite table. The only existing additive
migration is image_path TEXT, already used by the Phase 1 implementation.
"""
import csv
from io import BytesIO, StringIO
from pathlib import Path
import re
import sqlite3
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.database import DATABASE_PATH, init_database
from backend.predictor import predict_species

router = APIRouter(prefix="/api", tags=["Inspection Records"])
PROJECT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = (PROJECT_DIR / "uploads").resolve()

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_FORMATS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp"
}

SortChoice = Literal[
    "date_desc",
    "date_asc",
    "confidence_desc",
    "confidence_asc"
]

SORT_SQL = {
    "date_desc": "inspection_date DESC, id DESC",
    "date_asc": "inspection_date ASC, id ASC",
    "confidence_desc": "confidence_score DESC, id DESC",
    "confidence_asc": "confidence_score ASC, id DESC",
}

FIELDS = (
    "id", "sample_id", "species", "origin", "notes", "filename",
    "predicted_species", "confidence_score", "inference_time_ms",
    "model_name", "classification_type", "inspection_date"
)


def _prepare_database():
    init_database()

    with sqlite3.connect(DATABASE_PATH) as conn:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(inspections)")
        }

        if "image_path" not in columns:
            conn.execute(
                "ALTER TABLE inspections ADD COLUMN image_path TEXT"
            )


def _connect():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_upload_path(relative_path):
    """Never use a stored path outside the local uploads directory."""
    if not isinstance(relative_path, str):
        return None

    relative = Path(relative_path.replace("\\", "/"))

    if (
        len(relative.parts) != 2
        or relative.parts[0] != "uploads"
        or relative.suffix.lower()
        not in {".jpg", ".png", ".webp", ".jpeg"}
    ):
        return None

    resolved = (UPLOAD_DIR / relative.name).resolve()

    return resolved if resolved.parent == UPLOAD_DIR else None


def _serialize(row):
    data = {name: row[name] for name in FIELDS}

    image = _safe_upload_path(row["image_path"])

    data["image_saved"] = bool(image and image.is_file())

    data["image_url"] = (
        f"/api/inspections/{row['id']}/image"
        if data["image_saved"]
        else None
    )

    return data


def _filters(q, species_filter):
    clauses, params = [], []

    if q.strip():
        clauses.append(
            "(sample_id LIKE ? OR predicted_species LIKE ? OR origin LIKE ?)"
        )
        pattern = "%" + q.strip() + "%"
        params.extend([pattern, pattern, pattern])

    if species_filter.strip():
        clauses.append("predicted_species = ?")
        params.append(species_filter.strip())

    return (
        (" WHERE " + " AND ".join(clauses)) if clauses else "",
        params
    )


def _csv_safe(value):
    """Avoid formula execution when opened in spreadsheet applications."""
    value = "" if value is None else str(value)

    if value and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value

    return value


@router.post("/inspections", status_code=201)
async def save_inspection(
    file: UploadFile = File(...),
    sample_id: str = Form(...),
    species: str = Form(""),
    origin: str = Form(""),
    notes: str = Form(""),
):
    """Verify, rerun server-side AI, then save the file and prediction."""
    sample_id = sample_id.strip()

    if not 1 <= len(sample_id) <= 80:
        raise HTTPException(
            400,
            "Sample ID must be 1 to 80 characters."
        )

    if len(species) > 100 or len(origin) > 250 or len(notes) > 2000:
        raise HTTPException(
            400,
            "One or more text fields are too long."
        )

    try:
        image_bytes = await file.read(MAX_UPLOAD_SIZE + 1)
    finally:
        await file.close()

    if not image_bytes:
        raise HTTPException(400, "Select a non-empty image.")

    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, "Image exceeds the 10 MB limit.")

    try:
        with Image.open(BytesIO(image_bytes)) as check:
            image_format = check.format
            width, height = check.size

            if image_format not in ALLOWED_FORMATS:
                raise HTTPException(
                    400,
                    "Only JPG, PNG, and WEBP are supported."
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    400,
                    "Image dimensions are too large."
                )

            check.verify()

        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError
    ):
        raise HTTPException(400, "Image is invalid or corrupted.")

    try:
        prediction = predict_species(image)

    except FileNotFoundError:
        raise HTTPException(
            503,
            "The trained species model was not found."
        )

    _prepare_database()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    stored_file = UPLOAD_DIR / (
        uuid4().hex + ALLOWED_FORMATS[image_format]
    )

    stored_file.write_bytes(image_bytes)

    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.execute(
                """
                INSERT INTO inspections (
                    sample_id,
                    species,
                    origin,
                    notes,
                    filename,
                    predicted_species,
                    confidence_score,
                    inference_time_ms,
                    model_name,
                    classification_type,
                    image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    species.strip(),
                    origin.strip(),
                    notes.strip(),
                    Path(file.filename or "image").name,
                    prediction["predicted_species"],
                    prediction["confidence_percent"],
                    prediction["inference_time_ms"],
                    prediction["model"],
                    "species",
                    "uploads/" + stored_file.name
                ),
            )

            inspection_id = cursor.lastrowid

    except Exception:
        stored_file.unlink(missing_ok=True)
        raise

    return {
        "status": "saved",
        "inspection_id": inspection_id,
        "sample_id": sample_id,
        "predicted_species": prediction["predicted_species"],
        "confidence_percent": prediction["confidence_percent"],
        "classification_type": "species",
        "image_saved": True,
        "note": "Species AI demo only; this is not a bamboo quality grade.",
    }


@router.get("/inspections/next-id")
def suggest_next_sample_id():
    """Suggest a value, not a reservation."""
    _prepare_database()

    with _connect() as conn:
        values = conn.execute(
            "SELECT sample_id FROM inspections WHERE sample_id LIKE 'BAM-%'"
        ).fetchall()

    used = [
        int(match.group(1))
        for row in values
        if (
            match := re.fullmatch(
                r"BAM-(\d+)",
                row["sample_id"],
                re.I
            )
        )
    ]

    next_number = max(used, default=0) + 1

    return {
        "suggested_id": f"BAM-{next_number:04d}",
        "reserved": False
    }


@router.get("/inspections/species")
def inspection_species_options():
    _prepare_database()

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT predicted_species
            FROM inspections
            WHERE predicted_species IS NOT NULL
              AND TRIM(predicted_species) != ''
            ORDER BY predicted_species COLLATE NOCASE
            """
        ).fetchall()

    return {"species": [row[0] for row in rows]}


@router.get("/inspections/export.csv")
def export_inspections_csv(
    q: str = Query("", max_length=120),
    species_filter: str = Query("", max_length=100),
    sort: SortChoice = "date_desc",
):
    _prepare_database()

    where, params = _filters(q, species_filter)

    with _connect() as conn:
        rows = conn.execute(
            f"SELECT {', '.join(FIELDS)} FROM inspections "
            f"{where} ORDER BY {SORT_SQL[sort]}",
            params,
        ).fetchall()

    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(FIELDS)

    for row in rows:
        writer.writerow([
            _csv_safe(row[field])
            for field in FIELDS
        ])

    return Response(
        content="\ufeff" + output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition":
                'attachment; filename="bamboox_inspections.csv"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/inspections")
def list_inspections(
    q: str = Query("", max_length=120),
    species_filter: str = Query("", max_length=100),
    sort: SortChoice = "date_desc",
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Return saved records with filtering and pagination."""
    _prepare_database()

    where, params = _filters(q, species_filter)

    with _connect() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM inspections {where}",
            params
        ).fetchone()[0]

        rows = conn.execute(
            f"SELECT {', '.join(FIELDS)}, image_path "
            f"FROM inspections {where} "
            f"ORDER BY {SORT_SQL[sort]} LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()

    result = [_serialize(row) for row in rows]

    return {
        "count": len(result),
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(result) < total,
        "records": result,
    }


@router.get(
    "/inspections/{inspection_id}/image",
    include_in_schema=False
)
def inspection_image(inspection_id: int):
    """Serve images through the Records API.

    Rows without valid image_path return 404.
    """
    _prepare_database()

    with _connect() as conn:
        row = conn.execute(
            "SELECT image_path FROM inspections WHERE id = ?",
            (inspection_id,)
        ).fetchone()

    if row is None:
        raise HTTPException(
            404,
            "Inspection record not found."
        )

    image_path = _safe_upload_path(row["image_path"])

    if image_path is None or not image_path.is_file():
        raise HTTPException(
            404,
            "No stored image available for this record."
        )

    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }[image_path.suffix.lower()]

    return FileResponse(
        image_path,
        media_type=media_type,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/inspections/{inspection_id}")
def inspection_details(inspection_id: int):
    _prepare_database()

    with _connect() as conn:
        row = conn.execute(
            f"SELECT {', '.join(FIELDS)}, image_path "
            "FROM inspections WHERE id = ?",
            (inspection_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(
            404,
            "Inspection record not found."
        )

    return _serialize(row)


@router.delete("/inspections/{inspection_id}")
def delete_inspection(inspection_id: int):
    _prepare_database()

    with _connect() as conn:
        row = conn.execute(
            "SELECT image_path FROM inspections WHERE id = ?",
            (inspection_id,)
        ).fetchone()

        if row is None:
            raise HTTPException(
                404,
                "Inspection record not found."
            )

        conn.execute(
            "DELETE FROM inspections WHERE id = ?",
            (inspection_id,)
        )

        shared_image = bool(
            row["image_path"]
            and conn.execute(
                """
                SELECT 1 FROM inspections
                WHERE image_path = ?
                LIMIT 1
                """,
                (row["image_path"],)
            ).fetchone()
        )

    image = (
        None
        if shared_image
        else _safe_upload_path(row["image_path"])
    )

    image_removed = False

    if image is not None and image.is_file():
        try:
            image.unlink()
            image_removed = True
        except OSError:
            # Database deletion succeeded;
            # locked image may require manual cleanup.
            pass

    return {
        "status": "deleted",
        "inspection_id": inspection_id,
        "image_removed": image_removed,
    }