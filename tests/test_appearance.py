import cv2
import numpy as np
import pytest

from vectrax.frames import FramePacket, PixelFormat
from vectrax.tracking.appearance import Appearance
from vectrax.tracking.geometry import Box
from vectrax.tracking.observation import Origin

W, H, SIDE = 320, 240, 40
HOME = (40, 100)


def _textures(n, seed=0):
    rng = np.random.default_rng(seed)
    background = rng.integers(60, 120, (H, W, 3), dtype=np.uint8)
    patches = []
    for _ in range(n):
        noise = cv2.GaussianBlur(rng.integers(0, 256, (SIDE, SIDE, 3), dtype=np.uint8), (0, 0), 3)
        patches.append(cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX))

    return background, patches


def _frame(background, placed, frame_id=0):
    img = background.copy()
    for patch, (x, y) in placed:
        img[y:y + SIDE, x:x + SIDE] = patch

    return FramePacket(frame_id, "synthetic", frame_id, frame_id, img, PixelFormat.BGR)


def _box(x, y):
    return Box.from_xywh_px(x, y, SIDE, SIDE, W, H)


def _near(box, x, y, px=3):
    bx, by, _, _ = box.to_xywh_px(W, H)
    return abs(bx - x) <= px and abs(by - y) <= px


@pytest.fixture
def scene():
    background, (target, other) = _textures(2)
    look = Appearance(_frame(background, [(target, HOME)]), _box(*HOME))
    return background, target, other, look


def test_score_is_high_on_the_target_and_low_elsewhere(scene):
    background, target, other, look = scene
    later = _frame(background, [(target, (200, 60)), (other, HOME)])

    assert look.score(later, _box(200, 60)) > 0.9
    assert look.score(later, _box(*HOME)) < 0.5


def test_search_finds_the_target_inside_the_window(scene):
    background, target, _, look = scene
    later = _frame(background, [(target, (150, 120))], frame_id=7)

    found = look.search(later, _box(*HOME), reach=4.0)

    assert _near(found[0].box, 150, 120)
    assert found[0].score > 0.9
    assert (found[0].frame_id, found[0].origin) == (7, Origin.SEARCH)


def test_search_does_not_look_outside_the_window(scene):
    background, target, _, look = scene
    later = _frame(background, [(target, (260, 180))])

    found = look.search(later, _box(*HOME), reach=1.0)

    assert not any(_near(c.box, 260, 180) for c in found)


def test_search_reports_a_twin_as_the_runner_up(scene):
    background, target, _, look = scene
    later = _frame(background, [(target, (120, 100)), (target, (220, 100))])

    found = look.search(later, _box(170, 100), reach=4.0)

    assert {(round(c.box.cx * W), round(c.box.cy * H)) for c in found[:2]} == {(140, 120), (240, 120)}
    assert min(c.score for c in found[:2]) > 0.9


def test_a_box_off_the_image_has_no_score(scene):
    background, target, _, look = scene

    assert look.score(_frame(background, [(target, HOME)]), Box(1.4, 0.5, 0.1, 0.1)) is None
