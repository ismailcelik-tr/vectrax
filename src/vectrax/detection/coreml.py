"""RF-DETR Nano as a Core ML package (ADR-009).

The bundle compiles on first use — up to ~6 s cold, ~0.15 s once macOS has
cached it (docs/PERFORMANCE.md) — so `load()` pays it with a dummy frame
instead of letting the first real frame wait.
"""

import json
from enum import Enum
from pathlib import Path

import numpy as np

from vectrax.detection.decode import preprocess, to_observations
from vectrax.frames import FramePacket
from vectrax.tracking.observation import Observation

__all__ = ["ComputeUnits", "CoreMlDetector"]

LABELS_FILE = "labels.json"
INPUT_SIZE = 384  # RF-DETR Nano resolution
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SCORE_MIN = 0.5
BOX_DIMS = 4


class ComputeUnits(Enum):
    GPU = "CPU_AND_GPU"  # ADR-009 default: decodes identically to PyTorch
    ANE = "CPU_AND_NE"  # 11 W cheaper, slightly different numbers
    CPU = "CPU_ONLY"


class CoreMlDetector:
    def __init__(self, package: Path, units: ComputeUnits = ComputeUnits.GPU,
                 score_min: float = SCORE_MIN):
        self._package = Path(package)
        self._units = units
        self._score_min = score_min
        self._model = None

    def load(self) -> None:
        import coremltools as ct

        self._labels = self._load_labels()
        self._model = ct.models.MLModel(str(self._package),
                                        compute_units=ct.ComputeUnit[self._units.value])
        self._input = self._model.get_spec().description.input[0].name
        self._run(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))

    def detect(self, frame: FramePacket) -> list[Observation]:
        boxes, logits = self._run(frame.image)
        return to_observations(boxes, logits, self._labels, (frame.width, frame.height),
                               frame.frame_id, frame.capture_ns, self._score_min)

    def _run(self, image: np.ndarray):
        batch = preprocess(image, INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD)
        outputs = list(self._model.predict({self._input: batch}).values())
        return _split(outputs)

    def _load_labels(self):
        path = self._package.parent / LABELS_FILE
        return {int(k): v for k, v in json.loads(path.read_text()).items()}


def _split(outputs):
    """The converted graph names its outputs after MIL ops; boxes are the (*, 4) tensor."""
    boxes = [o for o in outputs if o.shape[-1] == BOX_DIMS]
    logits = [o for o in outputs if o.shape[-1] != BOX_DIMS]
    if len(boxes) != 1 or len(logits) != 1:
        raise ValueError(f"cannot tell boxes from logits in shapes {[o.shape for o in outputs]}")

    return boxes[0][0], logits[0][0]
