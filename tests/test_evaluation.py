import zipfile

import pytest

from vectrax.evaluation.gt import Visibility, load_mot
from vectrax.evaluation.metrics import PredBox, evaluate_track, iou
from vectrax.tracking.state import TrackState

S = TrackState
BOX = (100.0, 100.0, 50.0, 50.0)
FAR = (500.0, 500.0, 50.0, 50.0)


def _zip(tmp_path, lines):
    path = tmp_path / "gt.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("gt/gt.txt", "\n".join(lines) + "\n")
        z.writestr("gt/labels.txt", "cup\nperson\nobject\n")
    return path


def test_load_mot_converts_frames_and_visibility(tmp_path):
    path = _zip(tmp_path, [
        "1,1,100,100,50,50,1,1,1.0",
        "2,1,102,100,50,50,1,1,0.0",
        "4,1,110,100,50,50,1,1,1.0",
    ])

    gt = load_mot(path, frames=4)

    track = gt.tracks[1]
    assert track[0].visibility is Visibility.VISIBLE
    assert track[1].visibility is Visibility.PARTIAL
    assert 2 not in track
    assert track[3].box == (110.0, 100.0, 50.0, 50.0)
    assert track[0].label == "cup"
    assert gt.frames == 4


def test_iou():
    assert iou(BOX, BOX) == pytest.approx(1.0)
    assert iou(BOX, FAR) == 0.0
    assert iou(BOX, (125.0, 100.0, 50.0, 50.0)) == pytest.approx(1 / 3)


def _gt(present):
    """present: per frame, None = absent, else Visibility."""
    return {f: v for f, v in enumerate(present) if v is not None}


def _visible(n):
    return [Visibility.VISIBLE] * n


def test_perfect_tracking_scores_one():
    gt_vis = _gt(_visible(5))
    gt_box = dict.fromkeys(gt_vis, BOX)
    preds = [PredBox(S.TRACKING, BOX)] * 5

    r = evaluate_track(gt_vis, gt_box, preds, other_gt={})

    assert r.success_rate == 1.0
    assert r.false_visible == 0
    assert r.hijack_frames == 0


def test_absence_hidden_is_correct_claiming_visible_is_not():
    present = [Visibility.VISIBLE, None, None, Visibility.VISIBLE]
    gt_vis = _gt(present)
    gt_box = dict.fromkeys(gt_vis, BOX)
    honest = [PredBox(S.TRACKING, BOX), PredBox(S.OCCLUDED, BOX), PredBox(S.OCCLUDED, BOX), PredBox(S.TRACKING, BOX)]
    fooled = [PredBox(S.TRACKING, BOX), PredBox(S.TRACKING, FAR), PredBox(S.TRACKING, FAR), PredBox(S.TRACKING, BOX)]

    assert evaluate_track(gt_vis, gt_box, honest, other_gt={}).false_visible == 0
    assert evaluate_track(gt_vis, gt_box, fooled, other_gt={}).false_visible == 2


def test_recovery_frames_after_reappearance():
    present = [Visibility.VISIBLE, None, Visibility.VISIBLE, Visibility.VISIBLE, Visibility.VISIBLE]
    gt_vis = _gt(present)
    gt_box = dict.fromkeys(gt_vis, BOX)
    preds = [PredBox(S.TRACKING, BOX), PredBox(S.OCCLUDED, FAR), PredBox(S.OCCLUDED, FAR),
             PredBox(S.TRACKING, FAR), PredBox(S.TRACKING, BOX)]

    r = evaluate_track(gt_vis, gt_box, preds, other_gt={})

    assert r.recoveries == [2]


def test_never_recovered_is_none():
    present = [Visibility.VISIBLE, None, Visibility.VISIBLE, Visibility.VISIBLE]
    gt_vis = _gt(present)
    gt_box = dict.fromkeys(gt_vis, BOX)
    preds = [PredBox(S.TRACKING, BOX), PredBox(S.OCCLUDED, BOX), PredBox(S.LOST, BOX), PredBox(S.LOST, BOX)]

    assert evaluate_track(gt_vis, gt_box, preds, other_gt={}).recoveries == [None]


def test_state_counts_by_gt_visibility():
    present = [Visibility.VISIBLE, Visibility.PARTIAL, None]
    gt_vis = _gt(present)
    gt_box = dict.fromkeys(gt_vis, BOX)
    preds = [PredBox(S.TRACKING, BOX), PredBox(S.DEGRADED, BOX), PredBox(S.OCCLUDED, BOX)]

    r = evaluate_track(gt_vis, gt_box, preds, other_gt={})

    assert r.states[("visible", "tracking")] == 1
    assert r.states[("partial", "degraded")] == 1
    assert r.states[("absent", "occluded")] == 1


def test_hijack_counts_frames_on_other_target():
    gt_vis = _gt(_visible(3))
    gt_box = dict.fromkeys(gt_vis, BOX)
    other = {f: FAR for f in range(3)}
    preds = [PredBox(S.TRACKING, BOX), PredBox(S.TRACKING, FAR), PredBox(S.TRACKING, FAR)]

    r = evaluate_track(gt_vis, gt_box, preds, other_gt={2: other})

    assert r.hijack_frames == 2
    assert r.success_rate == pytest.approx(1 / 3)


def test_frames_before_selection_are_skipped():
    gt_vis = _gt(_visible(3))
    gt_box = dict.fromkeys(gt_vis, BOX)
    preds = [None, PredBox(S.TRACKING, BOX), PredBox(S.TRACKING, BOX)]

    r = evaluate_track(gt_vis, gt_box, preds, other_gt={})

    assert r.frames_scored == 2
    assert r.success_rate == 1.0


def test_run_fixture_on_synthetic_clip(tmp_path):
    import json

    import cv2
    import numpy as np

    from vectrax.evaluation.runner import run_fixture
    from vectrax.tracking.propagators import CsrtPropagator

    w, h, side, n = 320, 240, 40, 30
    rng = np.random.default_rng(5)
    background = rng.integers(60, 120, (h, w, 3), dtype=np.uint8)
    patch = cv2.normalize(cv2.GaussianBlur(rng.integers(0, 256, (side, side, 3), dtype=np.uint8), (0, 0), 3),
                          None, 0, 255, cv2.NORM_MINMAX)
    writer = cv2.VideoWriter(str(tmp_path / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    lines = []
    for i in range(n):
        img = background.copy()
        x = 40 + 3 * i
        img[100:100 + side, x:x + side] = patch
        writer.write(img)
        lines.append(f"{i + 1},1,{x},100,{side},{side},1,1,1.0")

    writer.release()
    (tmp_path / "clip.json").write_text(json.dumps({"frames": n, "stamps": [
        {"capture_ns": i * 33_333_333, "arrival_ns": i * 33_333_333} for i in range(n)]}))
    _zip(tmp_path, lines).rename(tmp_path / "clip.gt.zip")

    result = run_fixture(tmp_path / "clip.mp4", CsrtPropagator)

    track = result.tracks[1]
    assert track.frames_scored == n
    assert track.success_rate > 0.9
    assert result.track_ms["n"] == n
