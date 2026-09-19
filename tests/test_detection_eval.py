import pytest

from vectrax.evaluation.detection import Detection, evaluate_detections
from vectrax.evaluation.gt import GroundTruth, GtBox, Visibility

BOX = (100.0, 100.0, 50.0, 50.0)
OTHER = (300.0, 100.0, 50.0, 50.0)
FAR = (500.0, 500.0, 50.0, 50.0)


def _gt(frames, tracks):
    """tracks: {id: {frame: (box, visibility, label)}}"""
    gt = GroundTruth(frames=frames)
    for tid, boxes in tracks.items():
        gt.tracks[tid] = {f: GtBox(b, v, lab) for f, (b, v, lab) in boxes.items()}
    return gt


def _cups(n):
    return _gt(n, {1: {f: (BOX, Visibility.VISIBLE, "cup") for f in range(n)}})


def test_perfect_detections_score_one():
    dets = [Detection(f, BOX, 0.9, "cup") for f in range(3)]

    r = evaluate_detections(_cups(3), dets, label="cup", score_min=0.5)

    assert r.ap50 == pytest.approx(1.0)
    assert r.precision == pytest.approx(1.0)
    assert r.recall == pytest.approx(1.0)
    assert (r.gt_count, r.tp, r.fp) == (3, 3, 0)


def test_missed_frame_lowers_recall_not_precision():
    dets = [Detection(0, BOX, 0.9, "cup"), Detection(1, BOX, 0.9, "cup")]

    r = evaluate_detections(_cups(4), dets, label="cup", score_min=0.5)

    assert r.recall == pytest.approx(0.5)
    assert r.precision == pytest.approx(1.0)
    assert r.ap50 == pytest.approx(0.5)


def test_duplicate_detection_is_false_positive():
    dets = [Detection(0, BOX, 0.9, "cup"), Detection(0, BOX, 0.8, "cup")]

    r = evaluate_detections(_cups(1), dets, label="cup", score_min=0.5)

    assert (r.tp, r.fp) == (1, 1)


def test_score_min_filters_precision_and_recall_but_not_ap():
    # Low-score true positive ranks below a high-score false positive.
    dets = [Detection(0, FAR, 0.9, "cup"), Detection(0, BOX, 0.3, "cup")]

    r = evaluate_detections(_cups(1), dets, label="cup", score_min=0.5)

    assert (r.tp, r.fp) == (0, 1)
    assert r.recall == 0.0
    # PR points: (P 0, R 0) then (P 0.5, R 1) → all-point AP 0.5.
    assert r.ap50 == pytest.approx(0.5)


def test_other_class_is_ignored_for_class_metrics():
    dets = [Detection(0, BOX, 0.9, "cup"), Detection(0, FAR, 0.9, "person")]

    r = evaluate_detections(_cups(1), dets, label="cup", score_min=0.5)

    assert (r.tp, r.fp) == (1, 0)


def test_class_agnostic_matches_any_label():
    gt = _gt(1, {1: {0: (BOX, Visibility.VISIBLE, "object")}})
    dets = [Detection(0, BOX, 0.9, "bowl")]

    r = evaluate_detections(gt, dets, label=None, score_min=0.5)

    assert r.recall == pytest.approx(1.0)


def test_partial_gt_neither_counts_nor_penalizes():
    gt = _gt(2, {1: {0: (BOX, Visibility.VISIBLE, "cup"), 1: (BOX, Visibility.PARTIAL, "cup")}})
    dets = [Detection(0, BOX, 0.9, "cup"), Detection(1, BOX, 0.9, "cup")]

    r = evaluate_detections(gt, dets, label="cup", score_min=0.5)

    assert (r.gt_count, r.tp, r.fp) == (1, 1, 0)


def test_missed_partial_gt_does_not_lower_recall():
    gt = _gt(2, {1: {0: (BOX, Visibility.VISIBLE, "cup"), 1: (BOX, Visibility.PARTIAL, "cup")}})

    r = evaluate_detections(gt, [Detection(0, BOX, 0.9, "cup")], label="cup", score_min=0.5)

    assert r.recall == pytest.approx(1.0)


def test_two_targets_one_detection_each():
    gt = _gt(1, {1: {0: (BOX, Visibility.VISIBLE, "cup")}, 2: {0: (OTHER, Visibility.VISIBLE, "cup")}})
    dets = [Detection(0, BOX, 0.9, "cup"), Detection(0, OTHER, 0.8, "cup")]

    r = evaluate_detections(gt, dets, label="cup", score_min=0.5)

    assert (r.gt_count, r.tp, r.fp) == (2, 2, 0)


def test_no_gt_gives_zero_not_error():
    r = evaluate_detections(_gt(1, {}), [Detection(0, BOX, 0.9, "cup")], label="cup", score_min=0.5)

    assert (r.recall, r.ap50, r.fp) == (0.0, 0.0, 1)
