import cv2
import numpy as np
import pytest

from vectrax.frames import FramePacket, PixelFormat
from vectrax.tracking.geometry import Box
from vectrax.tracking.observation import Origin
from vectrax.tracking.propagators import CsrtPropagator

W, H, SIDE = 320, 240, 40
FRAME_NS = 33_333_333
STEP_PX = 3


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    background = rng.integers(60, 120, (H, W, 3), dtype=np.uint8)
    noise = rng.integers(0, 256, (SIDE, SIDE, 3), dtype=np.uint8)
    # Real objects have smooth texture; raw per-pixel noise does not.
    patch = cv2.GaussianBlur(noise, (0, 0), 3)
    patch = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX)
    return background, patch


def _frame(i, background, patch, visible=True):
    img = background.copy()
    x = 40 + STEP_PX * i
    if visible:
        img[100:100 + SIDE, x:x + SIDE] = patch

    return FramePacket(i, "synthetic", i * FRAME_NS, i * FRAME_NS, img, PixelFormat.BGR), x


def _iou(a: Box, b: Box):
    ax0, ay0, ax1, ay1 = a.cx - a.w / 2, a.cy - a.h / 2, a.cx + a.w / 2, a.cy + a.h / 2
    bx0, by0, bx1, by1 = b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / (a.w * a.h + b.w * b.h - inter)


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_follows_moving_patch_with_high_score(scale):
    background, patch = _scene()
    first, x0 = _frame(0, background, patch)
    prop = CsrtPropagator(scale=scale)
    prop.init(first, Box.from_xywh_px(x0, 100, SIDE, SIDE, W, H))

    for i in range(1, 30):
        frame, x = _frame(i, background, patch)
        obs = prop.update(frame)

        assert obs is not None
        assert obs.origin is Origin.PROPAGATOR
        assert obs.frame_id == i
        assert _iou(obs.box, Box.from_xywh_px(x, 100, SIDE, SIDE, W, H)) > 0.6
        assert obs.score > 0.6


def test_score_drops_when_target_disappears():
    background, patch = _scene()
    first, x0 = _frame(0, background, patch)
    prop = CsrtPropagator()
    prop.init(first, Box.from_xywh_px(x0, 100, SIDE, SIDE, W, H))
    for i in range(1, 10):
        prop.update(_frame(i, background, patch)[0])

    gone = prop.update(_frame(10, background, patch, visible=False)[0])

    assert gone is None or gone.score < 0.3


def test_invalid_scale_rejected():
    with pytest.raises(ValueError):
        CsrtPropagator(scale=0)
