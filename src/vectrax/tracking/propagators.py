"""Per-frame, per-target, class-agnostic propagators."""

from typing import Protocol

import cv2
import numpy as np

from vectrax.frames import FramePacket
from vectrax.tracking.geometry import Box
from vectrax.tracking.observation import Observation, Origin

__all__ = ["CsrtPropagator", "Propagator"]

_TEMPLATE_PX = 32
# Blur so a few pixels of box misalignment do not read as appearance change.
_SCORE_BLUR_SIGMA = 1.5


class Propagator(Protocol):
    def init(self, frame: FramePacket, box: Box) -> None: ...

    def update(self, frame: FramePacket) -> Observation | None: ...


class CsrtPropagator:
    """OpenCV CSRT. CSRT reports no confidence, so score = NCC between the
    initial appearance and the current box (PROVISIONAL, ADR-004)."""

    def __init__(self, scale: float = 1.0):
        if not 0 < scale <= 1:
            raise ValueError("scale must be in (0, 1]")

        self._scale = scale
        self._tracker = None
        self._template = None

    def init(self, frame: FramePacket, box: Box) -> None:
        img = self._resize(frame.image)
        h, w = img.shape[:2]
        self._tracker = cv2.TrackerCSRT.create()
        self._tracker.init(img, box.to_xywh_px(w, h))
        self._template = self._patch(img, box.to_xywh_px(w, h))

    def update(self, frame: FramePacket) -> Observation | None:
        img = self._resize(frame.image)
        h, w = img.shape[:2]
        ok, rect = self._tracker.update(img)
        if not ok:
            return None

        patch = self._patch(img, rect)
        if patch is None:
            return None

        ncc = float(cv2.matchTemplate(patch, self._template, cv2.TM_CCOEFF_NORMED)[0, 0])
        return Observation(
            frame_id=frame.frame_id,
            capture_ns=frame.capture_ns,
            box=Box.from_xywh_px(*rect, w, h),
            score=max(0.0, ncc),
            origin=Origin.PROPAGATOR,
        )

    def _resize(self, image):
        if self._scale == 1.0:
            return image

        return cv2.resize(image, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _patch(img, rect):
        x, y, w, h = (int(v) for v in rect)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None

        gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (_TEMPLATE_PX, _TEMPLATE_PX), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (0, 0), _SCORE_BLUR_SIGMA).astype(np.float32)
