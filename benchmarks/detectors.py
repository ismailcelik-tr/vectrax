"""Detector adapters for Phase 2c: a BGR frame in, `Detection` list out.

Benchmarks only: the runtime detector lives in `vectrax.detection` and these
adapters share its preprocessing and decode. `load()` is separate from
`detect()` so both can be timed on their own.

Third-party caches are pinned under `tools/` so inference writes nothing to
the home directory (docs/SETUP.md).
"""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from vectrax.detection.coreml import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE
from vectrax.detection.decode import preprocess, to_observations
from vectrax.evaluation.detection import Detection

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models" / "detectors"
CACHES = {
    "YOLO_CONFIG_DIR": ROOT / "tools" / "ultralytics",
    "HF_HOME": ROOT / "tools" / "hf",
    "RF_HOME": ROOT / "tools" / "roboflow",
}
EXPORTED = MODELS / "exported"
LABELS_FILE = "labels.json"
COREML_PRECISION = "fp16"  # what a deployment would ship; ANE requires it
CPU = "cpu"
# Emit almost everything: precision/recall thresholds are applied when scoring,
# and AP50 needs the low-score tail.
SCORE_MIN = 0.05
YOLO_IMGSZ = 640
BOX_DIMS = 4

for var, path in CACHES.items():
    # Ultralytics silently falls back to /tmp when the directory does not exist.
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(var, str(path))

__all__ = ["BACKENDS", "DETECTORS", "Detector", "build", "to_detections"]


@dataclass(frozen=True, slots=True)
class Spec:
    """What an exported graph needs around it; the graph itself stops at the model outputs."""

    size: int  # square input, pixels
    mean: tuple[float, float, float] | None  # None: rescale to [0, 1] only (D-FINE)
    std: tuple[float, float, float] | None


SPECS = {
    "rfdetr_n": Spec(INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD),
    "dfine_n": Spec(640, None, None),
}


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


def _as_detections(observations, size):
    """Observations carry normalized boxes; scoring wants pixels, unrounded."""
    width, height = size
    dets = []
    for o in observations:
        w, h = o.box.w * width, o.box.h * height
        dets.append(Detection(o.frame_id, (o.box.cx * width - w / 2, o.box.cy * height - h / 2, w, h),
                              o.score, o.label))

    return dets


def _split_outputs(outputs):
    """Exported graphs name their outputs differently; boxes are the (*, 4) tensor."""
    boxes = [o for o in outputs if o.shape[-1] == BOX_DIMS]
    logits = [o for o in outputs if o.shape[-1] != BOX_DIMS]
    if len(boxes) != 1 or len(logits) != 1:
        raise ValueError(f"cannot tell boxes from logits in shapes {[o.shape for o in outputs]}")

    return boxes[0][0], logits[0][0]


def _artifact(name, suffix):
    matches = sorted((EXPORTED / name).glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"no *{suffix} in {EXPORTED / name}; run benchmarks/export_detectors.py")

    return matches[0]


def _labels(name):
    path = EXPORTED / name / LABELS_FILE
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


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


class ExportedDetector:
    """Shared preprocessing for the exported graphs."""

    def _batch(self, frame: np.ndarray) -> np.ndarray:
        return preprocess(frame, self._spec.size, self._spec.mean, self._spec.std)


class OnnxDetector(ExportedDetector):
    """An exported ONNX graph on one ONNX Runtime provider."""

    def __init__(self, name: str, providers: list[str], score_min: float = SCORE_MIN):
        self._name = name
        self._spec = SPECS[name]
        self._providers = providers
        self._score_min = score_min
        self._session = None

    def load(self) -> None:
        import onnxruntime as ort

        self._labels = _labels(self._name)
        self._session = ort.InferenceSession(str(_artifact(self._name, ".onnx")), providers=self._providers)
        active = self._session.get_providers()[0]
        if active != self._providers[0]:
            raise RuntimeError(f"ORT runs on {active}, not the requested {self._providers[0]}")

        self._input = self._session.get_inputs()[0].name

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        outputs = self._session.run(None, {self._input: self._batch(frame)})
        boxes, logits = _split_outputs(outputs)
        size = (frame.shape[1], frame.shape[0])
        observations = to_observations(boxes, logits, self._labels, size, frame_id,
                                       capture_ns=0, score_min=self._score_min)
        return _as_detections(observations, size)


class CoreMlDetector(ExportedDetector):
    """A Core ML package pinned to one set of compute units."""

    def __init__(self, name: str, compute_units: str, precision: str = COREML_PRECISION,
                 score_min: float = SCORE_MIN):
        self._name = name
        self._spec = SPECS[name]
        self._compute_units = compute_units
        self._precision = precision
        self._score_min = score_min
        self._model = None

    def load(self) -> None:
        import coremltools as ct

        self._labels = _labels(self._name)
        self._model = ct.models.MLModel(str(_artifact(self._name, f"_{self._precision}.mlpackage")),
                                        compute_units=ct.ComputeUnit[self._compute_units])
        self._input = self._model.get_spec().description.input[0].name

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        outputs = self._model.predict({self._input: self._batch(frame)})
        boxes, logits = _split_outputs(list(outputs.values()))
        size = (frame.shape[1], frame.shape[0])
        observations = to_observations(boxes, logits, self._labels, size, frame_id,
                                       capture_ns=0, score_min=self._score_min)
        return _as_detections(observations, size)


DETECTORS = {
    "yolo26n": lambda **kw: YoloDetector(MODELS / "yolo26n.pt", **kw),
    "yolo11n": lambda **kw: YoloDetector(MODELS / "yolo11n.pt", **kw),
    "rfdetr_n": lambda **kw: RfDetrDetector(MODELS / "rf-detr-nano.pth", **kw),
    "dfine_n": lambda **kw: DFineDetector(MODELS / "dfine-nano-coco", **kw),
}
TORCH_DEVICES = {"pytorch-cpu": CPU, "pytorch-mps": "mps"}
ORT_PROVIDERS = {"onnx-cpu": ["CPUExecutionProvider"],
                 "onnx-coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"]}
COREML_UNITS = {"coreml-cpu": "CPU_ONLY", "coreml-gpu": "CPU_AND_GPU", "coreml-ane": "CPU_AND_NE"}
BACKENDS = [*TORCH_DEVICES, *ORT_PROVIDERS, *COREML_UNITS]


def build(name: str, backend: str, precision: str = COREML_PRECISION) -> Detector:
    if backend in TORCH_DEVICES:
        return DETECTORS[name](device=TORCH_DEVICES[backend])

    if backend in ORT_PROVIDERS:
        return OnnxDetector(name, ORT_PROVIDERS[backend])

    return CoreMlDetector(name, COREML_UNITS[backend], precision)
