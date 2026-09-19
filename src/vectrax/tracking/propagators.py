"""Per-frame, per-target, class-agnostic propagators (OpenCV trackers)."""

from enum import Enum
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from vectrax.frames import FramePacket
from vectrax.tracking.geometry import Box
from vectrax.tracking.observation import Observation, Origin

__all__ = [
    "MODELS_DIR",
    "CsrtPropagator",
    "KcfPropagator",
    "NanoPropagator",
    "Propagator",
    "ScoreSource",
    "VitPropagator",
]

MODELS_DIR = Path(__file__).resolve().parents[3] / "models" / "trackers"
VIT_MODEL = "vittrack_2023sep.onnx"
NANO_BACKBONE = "nanotrack_backbone_sim.onnx"
NANO_HEAD = "nanotrack_head_sim.onnx"

_TEMPLATE_PX = 32
# Blur so a few pixels of box misalignment do not read as appearance change.
_SCORE_BLUR_SIGMA = 1.5


class Propagator(Protocol):
    def init(self, frame: FramePacket, box: Box) -> None: ...

    def update(self, frame: FramePacket) -> Observation | None: ...


class ScoreSource(Enum):
    NCC = "ncc"  # our appearance check against the first frame (ADR-004)
    NATIVE = "native"  # the tracker's own confidence, where it has one


class _OpenCvPropagator:
    def __init__(self, scale: float, score: ScoreSource):
        if not 0 < scale <= 1:
            raise ValueError("scale must be in (0, 1]")

        self._scale = scale
        self._score = score
        self._tracker = None
        self._template = None

    def _create(self):
        raise NotImplementedError

    def init(self, frame: FramePacket, box: Box) -> None:
        img = self._resize(frame.image)
        h, w = img.shape[:2]
        self._tracker = self._create()
        self._tracker.init(img, box.to_xywh_px(w, h))
        self._template = _patch(img, box.to_xywh_px(w, h))

    def update(self, frame: FramePacket) -> Observation | None:
        img = self._resize(frame.image)
        h, w = img.shape[:2]
        ok, rect = self._tracker.update(img)
        if not ok:
            return None

        patch = _patch(img, rect)
        if patch is None:
            return None

        if self._score is ScoreSource.NATIVE:
            score = float(self._tracker.getTrackingScore())
        else:
            score = float(cv2.matchTemplate(patch, self._template, cv2.TM_CCOEFF_NORMED)[0, 0])

        return Observation(
            frame_id=frame.frame_id,
            capture_ns=frame.capture_ns,
            box=Box.from_xywh_px(*rect, w, h),
            score=min(max(score, 0.0), 1.0),
            origin=Origin.PROPAGATOR,
        )

    def _resize(self, image):
        if self._scale == 1.0:
            return image

        return cv2.resize(image, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)


class CsrtPropagator(_OpenCvPropagator):
    """OpenCV CSRT; no native confidence (ADR-004)."""

    def __init__(self, scale: float = 1.0):
        super().__init__(scale, ScoreSource.NCC)

    def _create(self):
        return cv2.TrackerCSRT.create()


class KcfPropagator(_OpenCvPropagator):
    """OpenCV KCF: fast correlation filter, no scale estimation."""

    def __init__(self, scale: float = 1.0):
        super().__init__(scale, ScoreSource.NCC)

    def _create(self):
        return cv2.TrackerKCF.create()


class VitPropagator(_OpenCvPropagator):
    """OpenCV ViTTrack (OpenCV Zoo, Apache-2.0)."""

    def __init__(self, scale: float = 1.0, score: ScoreSource = ScoreSource.NATIVE, models_dir: Path = MODELS_DIR):
        super().__init__(scale, score)
        self._params = cv2.TrackerVit_Params()
        self._params.net = str(models_dir / VIT_MODEL)

    def _create(self):
        return cv2.TrackerVit.create(self._params)


class NanoPropagator(_OpenCvPropagator):
    """OpenCV NanoTrack v2 (HonglinChu/SiamTrackers, Apache-2.0)."""

    def __init__(self, scale: float = 1.0, score: ScoreSource = ScoreSource.NATIVE, models_dir: Path = MODELS_DIR):
        super().__init__(scale, score)
        self._params = cv2.TrackerNano_Params()
        self._params.backbone = str(models_dir / NANO_BACKBONE)
        self._params.neckhead = str(models_dir / NANO_HEAD)

    def _create(self):
        return cv2.TrackerNano.create(self._params)


def _patch(img, rect):
    x, y, w, h = (int(v) for v in rect)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (_TEMPLATE_PX, _TEMPLATE_PX), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (0, 0), _SCORE_BLUR_SIGMA).astype(np.float32)
