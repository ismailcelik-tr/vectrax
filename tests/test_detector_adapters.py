import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
import detectors as d

from vectrax.evaluation.detection import Detection

XYXY = [(10.0, 20.0, 40.0, 60.0)]


def test_xyxy_becomes_xywh():
    dets = d.to_detections(XYXY, [0.9], ["cup"], frame=3, score_min=0.5)

    assert dets == [Detection(3, (10.0, 20.0, 30.0, 40.0), 0.9, "cup")]


def test_scores_below_minimum_are_dropped():
    dets = d.to_detections(XYXY * 2, [0.9, 0.4], ["cup", "person"], frame=0, score_min=0.5)

    assert [x.label for x in dets] == ["cup"]


LABELS = {0: "person", 1: "cup"}
CENTER_BOX = [[0.5, 0.5, 0.5, 0.5]]  # normalized cxcywh
SIZE = (100, 200)  # width, height


def _logits(*rows):
    return np.array(rows, dtype=np.float32)


def test_decode_scales_box_to_pixels():
    dets = d.decode(np.array(CENTER_BOX), _logits([-10.0, 2.0]), LABELS, SIZE, frame=7, score_min=0.5)

    assert len(dets) == 1
    assert dets[0].frame == 7
    assert dets[0].label == "cup"
    assert dets[0].box == pytest.approx((25.0, 50.0, 50.0, 100.0))
    assert dets[0].score == pytest.approx(0.8808, abs=1e-4)


def test_decode_keeps_every_class_above_threshold_on_one_query():
    dets = d.decode(np.array(CENTER_BOX), _logits([1.0, 2.0]), LABELS, SIZE, frame=0, score_min=0.5)

    assert sorted(x.label for x in dets) == ["cup", "person"]


def test_decode_drops_unmapped_class_index():
    dets = d.decode(np.array(CENTER_BOX), _logits([-10.0, -10.0, 5.0]), LABELS, SIZE, frame=0, score_min=0.5)

    assert dets == []


def test_decode_clamps_box_to_frame():
    dets = d.decode(np.array([[0.0, 0.0, 0.5, 0.5]]), _logits([-10.0, 2.0]), LABELS, SIZE, frame=0, score_min=0.5)

    assert dets[0].box == pytest.approx((0.0, 0.0, 25.0, 50.0))
