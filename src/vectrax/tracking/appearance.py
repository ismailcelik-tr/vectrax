"""Target appearance: a gray template, NCC scoring and a windowed search."""

import cv2
import numpy as np

from vectrax.frames import FramePacket
from vectrax.tracking.geometry import Box
from vectrax.tracking.observation import Observation, Origin

__all__ = ["Appearance", "patch"]

TEMPLATE_PX = 32
# Blur so a few pixels of box misalignment do not read as appearance change.
_BLUR_SIGMA = 1.5
# Best and runner-up; a close runner-up is a look-alike.
_PEAKS = 2


def patch(img, rect) -> np.ndarray | None:
    """Gray, blurred TEMPLATE_PX square of the rect; None if too little of it is on the image."""
    x, y, w, h = (int(v) for v in rect)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (TEMPLATE_PX, TEMPLATE_PX), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (0, 0), _BLUR_SIGMA).astype(np.float32)


class Appearance:
    """How the target looked when selected."""

    def __init__(self, frame: FramePacket, box: Box):
        self._template = patch(frame.image, box.to_xywh_px(frame.width, frame.height))

    def score(self, frame: FramePacket, box: Box) -> float | None:
        if self._template is None:
            return None

        p = patch(frame.image, box.to_xywh_px(frame.width, frame.height))
        if p is None:
            return None

        return _clamp(cv2.matchTemplate(p, self._template, cv2.TM_CCOEFF_NORMED)[0, 0])

    def search(self, frame: FramePacket, around: Box, reach: float) -> list[Observation]:
        """Best matches of around's size within `reach` box sizes of it, best first."""
        if self._template is None:
            return []

        w, h = frame.width, frame.height
        bw, bh = around.w * w, around.h * h
        x0, x1 = max(round(around.cx * w - (0.5 + reach) * bw), 0), min(round(around.cx * w + (0.5 + reach) * bw), w)
        y0, y1 = max(round(around.cy * h - (0.5 + reach) * bh), 0), min(round(around.cy * h + (0.5 + reach) * bh), h)
        # Scale the window so a target-sized box becomes the template.
        size = (round((x1 - x0) * TEMPLATE_PX / bw), round((y1 - y0) * TEMPLATE_PX / bh))
        if min(size) < TEMPLATE_PX:
            return []

        gray = cv2.cvtColor(frame.image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), _BLUR_SIGMA).astype(np.float32)
        response = cv2.matchTemplate(small, self._template, cv2.TM_CCOEFF_NORMED)
        sx, sy = size[0] / (x1 - x0), size[1] / (y1 - y0)

        found = []
        for _ in range(_PEAKS):
            _, best, _, (u, v) = cv2.minMaxLoc(response)
            if best <= 0:
                break

            cx, cy = x0 + (u + TEMPLATE_PX / 2) / sx, y0 + (v + TEMPLATE_PX / 2) / sy
            found.append(Observation(frame.frame_id, frame.capture_ns, Box(cx / w, cy / h, around.w, around.h),
                                     _clamp(best), Origin.SEARCH))
            # Blank one template around the peak so the runner-up is another object.
            response[max(v - TEMPLATE_PX, 0):v + TEMPLATE_PX, max(u - TEMPLATE_PX, 0):u + TEMPLATE_PX] = -1

        return found


def _clamp(v) -> float:
    return min(max(float(v), 0.0), 1.0)
