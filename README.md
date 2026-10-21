# BambooX Field Lab

Bamboo species recognition and growth-stage object detection for an academic research prototype.

**Public website:** https://qmsoro-miroku.github.io/bamboox-field-lab/

The public website runs both trained models inside the visitor's browser using ONNX Runtime Web. It does not require a paid Python hosting service or a running development computer.

## Features

- Forest-green/lime Home and Mission Hub, responsive layouts, and light/dark appearance.
- Image Scanner with sample ID, origin/location, inspector notes, and separate species/growth tabs.
- MobileNetV3-Small classification for nine BambooBD species.
- YOLOv8n bounding boxes for **Matured**, **Old**, and **Young** objects.
- Automatic Live YOLO growth boxes, explicit start/stop, captured-frame scans, species analysis, and local clip downloads.
- Clearly labeled public validation sample to test the detector without granting webcam access.
- Private browser Field Journal with saved photos, filtering, deletion, CSV export, and printable reports.
- Growth dataset viewer with **7,050 images** and **12,177 annotations**, split/class filters, and CSV export.
- Dashboard and Settings that describe the public browser runtime accurately.

## Public demo limits and privacy

The first scan downloads its ONNX model and the WebAssembly runtime. Inference speed depends on the visitor's device. WebAssembly is used with one CPU thread for compatibility with GitHub Pages.

Uploaded inspection photos are processed on the visitor's device. Saved species inspections use **IndexedDB in that browser**. They are not sent to a shared database, synchronized between devices, or visible to other visitors. Clearing site data removes that browser's journal. Use CSV export and printable reports for records you want to keep.

Growth detections remain preview-only. Predictions are research outputs and do not establish verified botanical identity, biological age, structural safety, or construction suitability.

The hosted application and AI models are downloaded over HTTPS. The public reference dataset is separate from visitors' inspection photos. No existing team inspection database, private uploads, credentials, or local backup folders are included in this repository.

## Repository layout

| Path | Purpose |
| --- | --- |
| `docs/` | Ready-to-serve GitHub Pages website, models, runtime, and public dataset assets |
| `frontend/` | Original FastAPI/Jinja templates and shared styling |
| `backend/` | Original FastAPI APIs, species/growth inference, and SQLite record handling |
| `models/` | Original trained PyTorch checkpoints |
| `scripts/web-api.js` | Browser inference, IndexedDB journal, and public-demo compatibility adapter |
| `scripts/build_public.py` | Render the eight public pages from the Jinja templates |

## Run the original Python application

Use Python 3.11. From this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Keep that terminal open and press **Ctrl+C** to stop it. The local application creates its own SQLite database and upload directory when needed. Its journal is separate from the public browser demo.

The optional full local dataset belongs in `dataset/growth_stage_clean/` with `train`, `valid`, and `test` image/label directories. Public static annotations are included under `docs/data/`.

## Update the public website

After editing the templates or the browser adapter:

```powershell
python -m pip install Jinja2
python scripts/build_public.py
git add frontend backend scripts docs
git commit -m "Update BambooX website"
git push
```

GitHub Pages publishes `main` → `/docs`. Do not add `.venv`, database files, personal inspection uploads, secrets, or backup folders.

## Model and dataset notes

The deployed ONNX graphs were exported from the original trained checkpoints. Numerical output parity was checked against PyTorch. The growth graph supports dynamic rectangular inputs. Browser preprocessing matches the original Ultralytics single-image resize and stride-32 letterbox instead of always padding to a 640-square image. Canvas resizing and inference hardware can still cause score differences. The confidence threshold remains 25% with class-aware NMS at IoU 0.7. Historical model benchmarks are not a new evaluation of the browser deployment. Re-filming a phone screen can add blur, glare, and unrelated background; upload the original image for a reproducible comparison. No detections does not prove bamboo is absent.

Live mode scans one new frame after each completed inference; it does not build a queue. It pauses on capture, clip recording, camera stop, and hidden tabs. Its speed depends on the device. Dataset-sample mode is explicitly labeled and uses a public validation photo; it is a functionality check, not independent accuracy evidence.

Re-export growth after replacing its weights with `python scripts/export_growth_onnx.py` in an environment with PyTorch, Ultralytics, ONNX, and ONNX Runtime. The export checks six rectangular/square input shapes against PyTorch. Run controller regressions with `node --test tests/camera_growth.test.cjs`.

Growth dataset source: [Bamboo Detection and Classification v6](https://universe.roboflow.com/bamboo-6gkce/bamboo-detection-and-classification/dataset/6), licensed **CC BY 4.0** according to its source metadata. The public copy uses the team's cleaned splits and retains the original annotations. See [ATTRIBUTION.md](ATTRIBUTION.md).

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the licenses of the inference runtime and model tooling. Original team-authored project code is made available here for academic review; no separate blanket license grant is added for it.
