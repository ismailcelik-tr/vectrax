import sys
from pathlib import Path

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

