"""Export the trained YOLO detector with dynamic stride-aligned rectangular inputs.

Requires the original Python environment plus torch, onnx, onnxruntime, and numpy.
Run from the repository: python scripts/export_growth_onnx.py
Species weights and export remain unchanged.
"""
from pathlib import Path
import copy
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
output = REPO / "docs/models"
torch.set_num_threads(2)
model = copy.deepcopy(YOLO(str(REPO / "models/yolov8n_growth_stage_best.pt")).model).cpu().float().eval()
model.fuse()
for module in model.modules():
    if module.__class__.__name__ == "Detect":
        module.export = True
        module.format = "onnx"
        module.dynamic = True


class ExportModel(torch.nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, images):
        prediction = self.detector(images)
        return prediction[0] if isinstance(prediction, tuple) else prediction


export = ExportModel(model).eval()
path = output / "growth.onnx"
torch.onnx.export(export, torch.zeros(1, 3, 640, 640), str(path), opset_version=17,
                  input_names=["images"], output_names=["detections"], dynamo=False, external_data=False,
                  dynamic_axes={"images": {2: "height", 3: "width"}, "detections": {2: "anchors"}})
onnx.checker.check_model(onnx.load(path))
session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
export.eval()
checks = []
for height, width in [(640, 640), (640, 160), (480, 640), (640, 480), (640, 320), (320, 640)]:
    data = np.random.default_rng(42).random((1, 3, height, width), dtype=np.float32)
    with torch.inference_mode():
        expected = export(torch.from_numpy(data)).numpy()
    actual = session.run(None, {"images": data})[0]
    assert np.allclose(expected, actual, atol=.003, rtol=.001), (height, width)
    checks.append({"input": list(data.shape), "output": list(actual.shape),
                   "max_error": float(np.max(np.abs(expected - actual)))})
metadata = json.loads((output / "metadata.json").read_text())
metadata.update(growth_input=[1, 3, "height", "width"], growth_output=[1, 7, "anchors"],
                growth_letterbox="auto-stride", growth_stride=32, growth_max_size=640,
                growth_revision="rect32-v1", growth_dynamic_parity=checks, growth_model_bytes=path.stat().st_size)
(output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("PASS six input shapes: ONNX output agrees with the trained PyTorch graph.")
