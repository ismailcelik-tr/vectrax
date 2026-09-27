"""DINOv2-S as a Core ML package on the Neural Engine (benchmarks/export_reid.py).

ANE because it is the fastest backend for this model and not the GPU the
detector runs on (docs/PERFORMANCE.md, DINOv2-S Re-ID embedding).
"""

from pathlib import Path

import cv2
import numpy as np

from vectrax.frames import FramePacket
from vectrax.tracking.geometry import Box

__all__ = ["CoreMlEmbedder"]

SIDE = 224
BATCHES = (8, 4, 2, 1)  # sizes the package was exported for, largest first
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT = "pixels"
OUTPUT = "embedding"
MIN_CROP_PX = 2


class CoreMlEmbedder:
    def __init__(self, package: Path):
        self._package = Path(package)
        self._model = None

    def load(self) -> None:
        """Pays the first-call compile here, not on the first real frame."""
        import coremltools as ct

        self._model = ct.models.MLModel(str(self._package), compute_units=ct.ComputeUnit.CPU_AND_NE)
        self._run(np.zeros((1, 3, SIDE, SIDE), dtype=np.float32))

    def embed(self, frame: FramePacket, boxes: list[Box]) -> list[np.ndarray | None]:
        """One L2-normalized embedding per box; None where too little of it is on the image."""
        crops = [_crop(frame.image, b) for b in boxes]
        kept = [c for c in crops if c is not None]
        found = iter(self._chunked(np.stack(kept)) if kept else [])
        return [None if c is None else next(found) for c in crops]

    def _chunked(self, batch):
        out, i = [], 0
        while i < len(batch):
            size = next(b for b in BATCHES if b <= len(batch) - i)
            out.extend(self._run(batch[i:i + size]))
            i += size

        return out

    def _run(self, batch):
        return self._model.predict({INPUT: batch})[OUTPUT]


def _crop(img, box: Box):
    h, w = img.shape[:2]
    x, y, bw, bh = box.to_xywh_px(w, h)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
    if x1 - x0 < MIN_CROP_PX or y1 - y0 < MIN_CROP_PX:
        return None

    rgb = cv2.resize(img[y0:y1, x0:x1], (SIDE, SIDE), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    return ((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
