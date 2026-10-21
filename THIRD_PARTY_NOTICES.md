# Third-party components

Third-party projects and data keep their own licenses. Public availability of this repository does not replace those terms.

- **ONNX Runtime Web 1.30.0** — Microsoft, MIT license. The JavaScript/WebAssembly runtime is included in `docs/assets/ort/`, with its MIT license.
- **Ultralytics YOLO** — Ultralytics, AGPL-3.0 / commercial licensing. The original YOLO checkpoint, derived ONNX model, and the local Python dependency are subject to the applicable Ultralytics license. Source and terms: https://github.com/ultralytics/ultralytics and https://www.ultralytics.com/license . This project makes the application and model-serving code available for inspection in this repository; it does not grant an Ultralytics commercial license.
- **PyTorch and Torchvision** — PyTorch contributors, BSD-style licenses. Used by the original Python application and the model conversion pipeline. https://github.com/pytorch/pytorch and https://github.com/pytorch/vision .
- **FastAPI**, **Starlette**, **Jinja2**, **Pillow**, and their dependencies retain their respective upstream licenses. The public static site is rendered from the team's Jinja templates.
- **Growth-stage images and annotations** — CC BY 4.0; see `ATTRIBUTION.md`.

Review the applicable upstream terms before using this research prototype in a commercial or otherwise restricted deployment.
