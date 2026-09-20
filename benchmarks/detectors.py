"""Detector adapters for Phase 2c: a BGR frame in, `Detection` list out.

Benchmarks only — the runtime detector arrives in Phase 3, after ADR-009.
`load()` is separate from `detect()` so both can be timed on their own.

Third-party caches are pinned under `tools/` so inference writes nothing to
the home directory (docs/SETUP.md).
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from vectrax.evaluation.detection import Detection

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models" / "detectors"
CACHES = {
    "YOLO_CONFIG_DIR": ROOT / "tools" / "ultralytics",
    "HF_HOME": ROOT / "tools" / "hf",
    "RF_HOME": ROOT / "tools" / "roboflow",
}
CPU = "cpu"
# Emit almost everything: precision/recall thresholds are applied when scoring,
# and AP50 needs the low-score tail.
SCORE_MIN = 0.05
YOLO_IMGSZ = 640

for var, path in CACHES.items():
    # Ultralytics silently falls back to /tmp when the directory does not exist.
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(var, str(path))

__all__ = ["DETECTORS", "DFineDetector", "Detector", "RfDetrDetector", "YoloDetector", "to_detections"]


class Detector(Protocol):
    def load(self) -> None:
        """Build the model. Timed separately from inference."""

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        """`frame` is BGR uint8, as the pipeline delivers it."""


def to_detections(xyxy: Sequence[Sequence[float]], scores: Sequence[float], labels: Sequence[str],
                  frame: int, score_min: float) -> list[Detection]:
    dets = []
    for (x1, y1, x2, y2), score, label in zip(xyxy, scores, labels, strict=True):
        if score < score_min:
            continue

        dets.append(Detection(frame, (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                              float(score), label))

    return dets


class YoloDetector:
    """Ultralytics YOLO. AGPL-3.0: reference numbers only, never runtime (R5)."""

    def __init__(self, weights: Path, device: str = CPU, score_min: float = SCORE_MIN):
        self._weights = weights
        self._device = device
        self._score_min = score_min
        self._model = None

    def load(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(self._weights))
        self._model.to(self._device)

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        result = self._model.predict(frame, imgsz=YOLO_IMGSZ, conf=self._score_min,
                                     device=self._device, verbose=False)[0]
        boxes = result.boxes
        labels = [result.names[int(c)] for c in boxes.cls.tolist()]
        return to_detections(boxes.xyxy.tolist(), boxes.conf.tolist(), labels, frame_id, self._score_min)


class RfDetrDetector:
    """RF-DETR Nano (Apache-2.0) through the rfdetr package."""

    def __init__(self, weights: Path, device: str = CPU, score_min: float = SCORE_MIN):
        self._weights = weights
        self._device = device
        self._score_min = score_min
        self._model = None

    def load(self) -> None:
        from rfdetr import RFDETRNano

        self._model = RFDETRNano(pretrain_weights=str(self._weights), device=self._device)

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        dets = self._model.predict(rgb, threshold=self._score_min, include_source_image=False)
        return to_detections(dets.xyxy.tolist(), dets.confidence.tolist(),
                             list(dets.data["class_name"]), frame_id, self._score_min)


class DFineDetector:
    """D-FINE Nano (Apache-2.0) through transformers."""

    def __init__(self, weights: Path, device: str = CPU, score_min: float = SCORE_MIN):
        self._weights = weights
        self._device = device
        self._score_min = score_min
        self._model = None
        self._processor = None

    def load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, DFineForObjectDetection

        self._processor = AutoImageProcessor.from_pretrained(self._weights)
        self._model = DFineForObjectDetection.from_pretrained(self._weights).to(self._device).eval()
        self._no_grad = torch.no_grad

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        inputs = self._processor(images=rgb, return_tensors="pt").to(self._device)
        with self._no_grad():
            outputs = self._model(**inputs)

        height, width = frame.shape[:2]
        result = self._processor.post_process_object_detection(
            outputs, target_sizes=[(height, width)], threshold=self._score_min)[0]
        id2label = self._model.config.id2label
        labels = [id2label[int(i)] for i in result["labels"].tolist()]
        return to_detections(result["boxes"].tolist(), result["scores"].tolist(), labels,
                             frame_id, self._score_min)


DETECTORS = {
    "yolo26n": lambda **kw: YoloDetector(MODELS / "yolo26n.pt", **kw),
    "yolo11n": lambda **kw: YoloDetector(MODELS / "yolo11n.pt", **kw),
    "rfdetr_n": lambda **kw: RfDetrDetector(MODELS / "rf-detr-nano.pth", **kw),
    "dfine_n": lambda **kw: DFineDetector(MODELS / "dfine-nano-coco", **kw),
}
