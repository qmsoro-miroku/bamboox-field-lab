"""BambooX pilot dataset labeling, review, and CSV export.

Folder name = provisional label, NOT an expert-verified quality grade.
Existing Phase 1 endpoints are kept for compatibility.
"""
import csv
import io
import os
import math
from time import monotonic
from threading import Lock
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from backend.ui import PROJECT_DIR, render_page

router = APIRouter(tags=["Image Labeling"])
DATASET_DIR = PROJECT_DIR / "dataset" / "quality"
LABELS = ("good", "usable", "poor", "needs_review")
FOLDERS = ("unlabeled", *LABELS)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LabelName = Literal["unlabeled", "good", "usable", "poor", "needs_review"]


def ensure_folders():
    for label in FOLDERS:
        (DATASET_DIR / label).mkdir(parents=True, exist_ok=True)


def safe_filename(filename: str):
    if (
        not filename
        or filename != Path(filename).name
        or "/" in filename
        or "\\" in filename
        or filename.startswith(".")
        or Path(filename).suffix.lower() not in EXTENSIONS
    ):
        raise HTTPException(status_code=400, detail="Invalid image filename.")
    return filename


def list_images(folder: Path):
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: p.name.lower(),
    )


def label_counts():
    ensure_folders()
    return {label: len(list_images(DATASET_DIR / label)) for label in FOLDERS}


def image_location(label: LabelName, filename: str):
    filename = safe_filename(filename)
    folder = (DATASET_DIR / label).resolve()
    path = folder / filename
    # Block symlinks to files outside the expected dataset folder.
    if path.resolve().parent != folder:
        raise HTTPException(status_code=400, detail="Invalid image location.")
    return path


class LabelRequest(BaseModel):
    filename: str
    label: Literal["good", "usable", "poor", "needs_review"]


class RelabelRequest(BaseModel):
    filename: str
    source_label: LabelName
    target_label: LabelName


def move_label(filename: str, source_label: LabelName, target_label: LabelName):
    if source_label == target_label:
        raise HTTPException(status_code=400, detail="Choose a different label.")
    source = image_location(source_label, filename)
    destination = image_location(target_label, filename)
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Image no longer exists in its source folder. Refresh the page.")
    if destination.exists():
        raise HTTPException(status_code=409, detail="An image with that filename already exists in the destination folder.")
    os.replace(source, destination)
    return {
        "status": "labeled",
        "filename": filename,
        "previous_label": source_label,
        "label": target_label,
    }


@router.get("/api/labeling/current")
def get_current_image():
    counts = label_counts()
    files = list_images(DATASET_DIR / "unlabeled")
    return {"filename": files[0].name if files else None, "remaining": counts["unlabeled"],
            "counts": {label: counts[label] for label in LABELS}}


@router.get("/api/labeling/images")
def get_labeling_images():
    counts = label_counts()
    images = [
        {"filename": path.name, "label": label}
        for label in FOLDERS
        for path in list_images(DATASET_DIR / label)
    ]
    total = sum(counts.values())
    return {
        "images": images,
        "counts": counts,
        "total": total,
        "processed": total - counts["unlabeled"],
        "progress_percent": round((total - counts["unlabeled"]) * 100 / total, 1) if total else 0,
    }


@router.get("/api/labeling/image/{label}/{filename}")
def view_labeled_image(label: LabelName, filename: str):
    path = image_location(label, filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path)


@router.get("/api/labeling/image/{filename}")
def view_labeling_image(filename: str):
    """Compatibility endpoint: original unlabeled queue image URL."""
    return view_labeled_image("unlabeled", filename)


@router.post("/api/labeling/assign")
def assign_label(request: LabelRequest):
    ensure_folders()
    return move_label(request.filename, "unlabeled", request.label)


@router.post("/api/labeling/relabel")
def relabel_image(request: RelabelRequest):
    ensure_folders()
    return move_label(request.filename, request.source_label, request.target_label)


def safe_csv_field(value: str):
    """Keep filenames from being interpreted as formulas in spreadsheet apps."""
    if value and value[0] in ("=", "+", "-", "@", "\t", "\r", "\n"):
        return "'" + value
    return value


@router.get("/api/labeling/export")
def export_labels():
    ensure_folders()
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["filename", "label", "relative_path", "defect_type", "review_status"])
    for label in FOLDERS:
        for path in list_images(DATASET_DIR / label):
            writer.writerow([
                safe_csv_field(path.name),
                label,
                safe_csv_field(f"dataset/quality/{label}/{path.name}"),
                "unverified",
                "pending_expert_verification",
            ])
    return Response(
        content="\ufeff" + output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="bamboox_labels_export.csv"'},
    )


@router.get("/app/labeling", include_in_schema=False)
def labeling_page(request: Request):
    return render_page(request, "labeling.html", "labeling")


# Growth annotation review is separate from the preserved legacy quality API.
# Existing quality images and YOLO annotations are never moved or rewritten.
GROWTH_CLASSES = ("matured", "old", "young")
GROWTH_SPLITS = ("train", "valid", "test")
GrowthSplit = Literal["all", "train", "valid", "test"]
GrowthStage = Literal["all", "matured", "old", "young", "needs_review"]


def growth_dataset_dir():
    for name in ("growth_stage_clean", "growth_stage"):
        candidate = PROJECT_DIR / "dataset" / name
        if (candidate / "data.yaml").is_file():
            return candidate.resolve()
    return (PROJECT_DIR / "dataset" / "growth_stage_clean").resolve()


def _growth_annotations(path: Path):
    objects, invalid = [], 0
    if not path.is_file() or path.is_symlink():
        return objects, "missing_label", invalid
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        return objects, "invalid_label", 1
    for line in lines:
        if not line.strip():
            continue
        try:
            values = line.split()
            if len(values) != 5:
                raise ValueError("Expected a YOLO bounding box")
            class_id = int(values[0])
            x, y, width, height = map(float, values[1:])
            if not 0 <= class_id < len(GROWTH_CLASSES):
                raise ValueError("Unknown class")
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, width, height)) or width <= 0 or height <= 0:
                raise ValueError("Invalid normalized coordinates")
            objects.append({"class_id": class_id, "class_name": GROWTH_CLASSES[class_id],
                            "x_center": x, "y_center": y, "width": width, "height": height})
        except (ValueError, OverflowError):
            invalid += 1
    return objects, "invalid_label" if invalid else "annotated" if objects else "no_objects", invalid


def _scan_growth_dataset(root):
    result = {"dataset": root.name, "available": (root / "data.yaml").is_file(),
              "counts": dict.fromkeys(GROWTH_CLASSES, 0), "splits": dict.fromkeys(GROWTH_SPLITS, 0),
              "total_images": 0, "annotated_images": 0, "object_count": 0,
              "needs_review": 0, "empty_images": 0, "invalid_rows": 0, "total_bytes": 0,
              "images": []}
    for split in GROWTH_SPLITS:
        folder = root / split / "images"
        if not folder.is_dir() or not folder.resolve().is_relative_to(root):
            continue
        labels_folder = root / split / "labels"
        labels_safe = labels_folder.resolve().is_relative_to(root)
        with os.scandir(folder) as entries:
            image_entries = sorted(entries, key=lambda item: item.name.lower())
        for entry in image_entries:
            image = Path(entry.path)
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False) or image.suffix.lower() not in EXTENSIONS:
                continue
            label = labels_folder / (image.stem + ".txt")
            if not labels_safe:
                objects, status, invalid = [], "invalid_label", 1
            else:
                objects, status, invalid = _growth_annotations(label)
            result["images"].append({"filename": image.name, "split": split,
                                     "objects": objects, "annotation_status": status})
            result["total_images"] += 1
            result["splits"][split] += 1
            result["total_bytes"] += entry.stat(follow_symlinks=False).st_size
            result["annotated_images"] += int(status == "annotated")
            result["empty_images"] += int(status == "no_objects")
            result["needs_review"] += int(status in ("missing_label", "invalid_label"))
            result["invalid_rows"] += invalid
            for box in objects:
                result["counts"][box["class_name"]] += 1
                result["object_count"] += 1
    return result


_growth_cache = None
_growth_cache_root = None
_growth_cache_until = 0.0
_growth_cache_lock = Lock()


def growth_dataset_snapshot():
    """Share a short-lived read-only index across dataset, Dashboard and Settings."""
    global _growth_cache, _growth_cache_root, _growth_cache_until
    root = growth_dataset_dir()
    def changed_at(path):
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return None
    signature = (root, changed_at(root / "data.yaml"),
                 *(changed_at(root / split / folder) for split in GROWTH_SPLITS for folder in ("images", "labels")))
    with _growth_cache_lock:
        if _growth_cache is None or _growth_cache_root != signature or monotonic() >= _growth_cache_until:
            _growth_cache = _scan_growth_dataset(root)
            _growth_cache_root = signature
            _growth_cache_until = monotonic() + 10
        # Callers may remove/filter top-level fields without modifying the shared index.
        return dict(_growth_cache)


@router.get("/api/growth-dataset/summary")
def growth_dataset_summary():
    data = growth_dataset_snapshot()
    data.pop("images")
    return data


@router.get("/api/growth-dataset/images")
def growth_dataset_images(split: GrowthSplit = "all", stage: GrowthStage = "all"):
    data = growth_dataset_snapshot()
    data["images"] = [item for item in data["images"]
                      if (split == "all" or item["split"] == split)
                      and (stage == "all"
                           or stage == "needs_review" and item["annotation_status"] in ("missing_label", "invalid_label")
                           or stage in {box["class_name"] for box in item["objects"]})]
    return data


@router.get("/api/growth-dataset/image/{split}/{filename}")
def growth_dataset_image(split: Literal["train", "valid", "test"], filename: str):
    filename = safe_filename(filename)
    root = growth_dataset_dir()
    folder = (root / split / "images").resolve()
    path = folder / filename
    if not folder.is_relative_to(root) or path.is_symlink() or path.resolve().parent != folder:
        raise HTTPException(status_code=400, detail="Invalid dataset image location.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Dataset image not found.")
    return FileResponse(path)


@router.get("/api/growth-dataset/export")
def growth_dataset_export():
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["filename", "split", "class_id", "class_name", "x_center", "y_center", "width", "height", "annotation_status"])
    for item in growth_dataset_snapshot()["images"]:
        for box in item["objects"] or [None]:
            values = [box[key] for key in ("class_id", "class_name", "x_center", "y_center", "width", "height")] if box else [""] * 6
            writer.writerow([safe_csv_field(item["filename"]), item["split"], *values, item["annotation_status"]])
    return Response(content="\ufeff" + output.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="bamboox_growth_annotations.csv"'})
