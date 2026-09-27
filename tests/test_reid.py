import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vectrax.evaluation.gt import load_mot
from vectrax.frames import FramePacket, PixelFormat
from vectrax.reid.coreml import CoreMlEmbedder
from vectrax.tracking.geometry import Box
from vectrax.tracking.observation import Observation, Origin

MODEL = Path("models/reid/exported/dinov2s_fp16.mlpackage")
CLIP = Path("data/fixtures/single_target.mp4")
LATER = 60
BACKGROUND = (40, 40, 113, 143)  # top-left corner: wall, no cup

needs_model = pytest.mark.skipif(not MODEL.exists() or not CLIP.exists(),
                                 reason="exported model or fixture missing (both git-ignored)")


def test_an_embedding_does_not_break_observation_equality():
    a = Observation(0, 0, Box(0.5, 0.5, 0.1, 0.1), 0.9, Origin.DETECTOR, "cup", np.ones(3))
    b = Observation(0, 0, Box(0.5, 0.5, 0.1, 0.1), 0.9, Origin.DETECTOR, "cup", np.zeros(3))

    assert a == b


@pytest.fixture(scope="module")
def cup():
    cap = cv2.VideoCapture(str(CLIP))
    frames = []
    for i in range(LATER + 1):
        ok, image = cap.read()
        assert ok
        frames.append(FramePacket(i, "test", i, i, image, PixelFormat.BGR))

    cap.release()
    gt = load_mot(CLIP.with_suffix("").with_suffix(".gt.zip"), json.loads(CLIP.with_suffix(".json").read_text())["frames"])
    h, w = frames[0].image.shape[:2]
    track = next(iter(gt.tracks.values()))
    embedder = CoreMlEmbedder(MODEL)
    embedder.load()
    return embedder, frames, {i: Box.from_xywh_px(*track[i].box, w, h) for i in (0, LATER)}, (w, h)


@needs_model
def test_the_same_cup_is_closer_than_the_wall(cup):
    embedder, frames, boxes, (w, h) = cup
    first, wall = embedder.embed(frames[0], [boxes[0], Box.from_xywh_px(*BACKGROUND, w, h)])
    (later,) = embedder.embed(frames[LATER], [boxes[LATER]])

    assert first @ later > first @ wall
    assert first @ first == pytest.approx(1.0, abs=1e-3)


@needs_model
def test_batches_match_single_calls(cup):
    embedder, frames, boxes, (w, h) = cup
    three = [boxes[0], Box.from_xywh_px(*BACKGROUND, w, h), Box(0.5, 0.5, 0.2, 0.2)]

    together = embedder.embed(frames[0], three)
    alone = [embedder.embed(frames[0], [b])[0] for b in three]

    assert [float(a @ b) for a, b in zip(together, alone, strict=True)] == pytest.approx([1.0] * 3, abs=1e-3)


@needs_model
def test_a_box_off_the_image_has_no_embedding(cup):
    embedder, frames, _, _ = cup

    assert embedder.embed(frames[0], [Box(1.4, 0.5, 0.1, 0.1)]) == [None]
