"""Per-frame, per-target, class-agnostic propagators (OpenCV trackers)."""

from enum import Enum
from pathlib import Path
from typing import Protocol

import cv2

from vectrax.frames import FramePacket
from vectrax.tracking.appearance import patch
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
        self._template = patch(img, box.to_xywh_px(w, h))

    def update(self, frame: FramePacket) -> Observation | None:
        img = self._resize(frame.image)
        h, w = img.shape[:2]
        ok, rect = self._tracker.update(img)
        if not ok:
            return None

        current = patch(img, rect)
        if current is None:
            return None

        if self._score is ScoreSource.NATIVE:
            score = float(self._tracker.getTrackingScore())
        else:
            score = float(cv2.matchTemplate(current, self._template, cv2.TM_CCOEFF_NORMED)[0, 0])

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
