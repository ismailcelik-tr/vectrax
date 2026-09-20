from pathlib import Path

import cv2
import numpy as np
import pytest

from vectrax.detection.coreml import CoreMlDetector
from vectrax.detection.decode import preprocess, to_observations
from vectrax.frames import FramePacket, PixelFormat
from vectrax.tracking.observation import Origin

LABELS = {0: "person", 1: "cell phone"}
CENTER_BOX = np.array([[0.5, 0.5, 0.5, 0.5]])  # normalized cxcywh
SIZE = (1280, 720)  # width, height
CAPTURE_NS = 1_234_000_000


def _logits(*rows):
    return np.array(rows, dtype=np.float32)


def _observe(boxes, logits, score_min=0.5):
    return to_observations(boxes, logits, LABELS, SIZE, frame_id=7, capture_ns=CAPTURE_NS,
                           score_min=score_min)


def test_observation_carries_frame_capture_and_origin():
    obs = _observe(CENTER_BOX, _logits([-10.0, 2.0]))

    assert len(obs) == 1
    assert (obs[0].frame_id, obs[0].capture_ns, obs[0].origin) == (7, CAPTURE_NS, Origin.DETECTOR)
    assert obs[0].score == pytest.approx(0.8808, abs=1e-4)


def test_box_is_normalized_and_label_is_a_hint():
    obs = _observe(CENTER_BOX, _logits([-10.0, 2.0]))

    box = obs[0].box
    assert (box.cx, box.cy, box.w, box.h) == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert obs[0].label == "cell phone"


def test_low_scores_and_unmapped_classes_are_dropped():
    assert _observe(CENTER_BOX, _logits([-1.0, -1.0])) == []
    assert _observe(CENTER_BOX, _logits([-10.0, -10.0, 5.0])) == []


def test_preprocess_gives_nchw_float_in_model_size():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    batch = preprocess(frame, size=384, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    assert batch.shape == (1, 3, 384, 384)
    assert batch.dtype == np.float32
    assert batch.min() == pytest.approx(-1.0)


MODEL = Path("models/detectors/exported/rfdetr_n/rfdetr-nano_fp16.mlpackage")
CLIP = Path("data/fixtures/single_target.mp4")
CUP_GT = (795, 297, 113, 143)  # frame 0, docs/FIXTURES.md


@pytest.mark.skipif(not MODEL.exists() or not CLIP.exists(),
                    reason="exported model or fixture missing (both git-ignored)")
def test_detector_finds_the_cup_in_a_fixture_frame():
    cap = cv2.VideoCapture(str(CLIP))
    ok, image = cap.read()
    cap.release()
    assert ok

    detector = CoreMlDetector(MODEL)
    detector.load()
    frame = FramePacket(frame_id=0, source_id="test", capture_ns=CAPTURE_NS, arrival_ns=CAPTURE_NS,
                        image=image, pixel_format=PixelFormat.BGR)

    cups = [o for o in detector.detect(frame) if o.label == "cup"]

    assert cups, "no cup detected"
    best = max(cups, key=lambda o: o.score)
    x, y, w, h = best.box.to_xywh_px(image.shape[1], image.shape[0])
    assert best.origin is Origin.DETECTOR
    assert CUP_GT[0] <= x + w / 2 <= CUP_GT[0] + CUP_GT[2]
    assert CUP_GT[1] <= y + h / 2 <= CUP_GT[1] + CUP_GT[3]
