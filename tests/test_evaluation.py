import zipfile

import pytest

from vectrax.evaluation.gt import Visibility, load_mot
from vectrax.evaluation.metrics import (
    PredBox,
    Relock,
    evaluate_track,
    identity_scores,
    iou,
)
from vectrax.tracking.state import TrackState

S = TrackState
BOX = (100.0, 100.0, 50.0, 50.0)
FAR = (500.0, 500.0, 50.0, 50.0)
NOWHERE = (300.0, 300.0, 50.0, 50.0)


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


CLIP_FRAMES = 30


def _synthetic_clip(tmp_path, rows):
    """rows: y of each target's lane; target k moves right along its lane."""
    import json

    import cv2
    import numpy as np

    w, h, side = 320, 240, 40
    rng = np.random.default_rng(5)
    background = rng.integers(60, 120, (h, w, 3), dtype=np.uint8)
    patches = [cv2.normalize(cv2.GaussianBlur(rng.integers(0, 256, (side, side, 3), dtype=np.uint8), (0, 0), 3),
                             None, 0, 255, cv2.NORM_MINMAX) for _ in rows]
    writer = cv2.VideoWriter(str(tmp_path / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    lines = []
    for i in range(CLIP_FRAMES):
        img = background.copy()
        x = 40 + 3 * i
        for k, (y, patch) in enumerate(zip(rows, patches, strict=True), start=1):
            img[y:y + side, x:x + side] = patch
            lines.append(f"{i + 1},{k},{x},{y},{side},{side},1,1,1.0")

        writer.write(img)

    writer.release()
    (tmp_path / "clip.json").write_text(json.dumps({"frames": CLIP_FRAMES, "stamps": [
        {"capture_ns": i * 33_333_333, "arrival_ns": i * 33_333_333} for i in range(CLIP_FRAMES)]}))
    _zip(tmp_path, lines).rename(tmp_path / "clip.gt.zip")
    return tmp_path / "clip.mp4"


def test_run_fixture_on_synthetic_clip(tmp_path):
    from vectrax.evaluation.runner import run_fixture
    from vectrax.tracking.propagators import CsrtPropagator

    n = CLIP_FRAMES
    result = run_fixture(_synthetic_clip(tmp_path, [100]), CsrtPropagator)

    track = result.tracks[1]
    assert track.frames_scored == n
    assert track.success_rate > 0.9
    assert result.track_ms["n"] == n
    assert result.identity["idsw"] == 0


def test_on_target_counts_small_boxes_inside_the_object():
    gt_vis = _gt(_visible(2))
    gt_box = dict.fromkeys(gt_vis, BOX)
    small_inside = (115.0, 115.0, 20.0, 20.0)
    preds = [PredBox(S.TRACKING, small_inside), PredBox(S.TRACKING, FAR)]

    r = evaluate_track(gt_vis, gt_box, preds, other_gt={})

    assert r.success_rate == 0.0
    assert r.on_target_rate == 0.5


@pytest.mark.parametrize(("back_at", "expected"), [(BOX, Relock.OWN), (FAR, Relock.OTHER), (NOWHERE, Relock.BACKGROUND)])
def test_relock_names_the_target_the_box_returned_to(back_at, expected):
    gt_vis = _gt(_visible(3))
    gt_box = dict.fromkeys(gt_vis, BOX)
    other = {f: FAR for f in range(3)}
    preds = [PredBox(S.TRACKING, BOX), PredBox(S.OCCLUDED, BOX), PredBox(S.TRACKING, back_at)]

    r = evaluate_track(gt_vis, gt_box, preds, other_gt={2: other})

    assert r.relocks == {2: expected}


def test_relock_while_own_target_is_absent_is_wrong():
    gt_vis = _gt([Visibility.VISIBLE, None, None])
    gt_box = dict.fromkeys(gt_vis, BOX)
    preds = [PredBox(S.TRACKING, BOX), PredBox(S.OCCLUDED, BOX), PredBox(S.TRACKING, BOX)]

    r = evaluate_track(gt_vis, gt_box, preds, other_gt={})

    assert r.relocks == {2: Relock.BACKGROUND}
    assert r.wrong_relocks == 1


def test_selection_is_not_a_relock():
    gt_vis = _gt(_visible(2))
    gt_box = dict.fromkeys(gt_vis, BOX)
    preds = [None, PredBox(S.INITIALIZING, BOX)]

    assert evaluate_track(gt_vis, gt_box, preds, other_gt={}).relocks == {}


def test_identity_counts_a_swap_between_two_targets():
    gt = {1: dict.fromkeys(range(6), BOX), 2: dict.fromkeys(range(6), FAR)}
    first = [PredBox(S.TRACKING, BOX)] * 3 + [PredBox(S.TRACKING, FAR)] * 3
    second = [PredBox(S.TRACKING, FAR)] * 3 + [PredBox(S.TRACKING, BOX)] * 3

    swapped = identity_scores(gt, {1: first, 2: second})
    kept = identity_scores(gt, {1: [PredBox(S.TRACKING, BOX)] * 6, 2: [PredBox(S.TRACKING, FAR)] * 6})

    assert swapped["idsw"] == 2
    assert swapped["idf1"] == pytest.approx(0.5)
    assert kept["idsw"] == 0
    assert kept["idf1"] == pytest.approx(1.0)
    assert kept["hota"] == pytest.approx(1.0)


def test_hidden_tracks_are_not_detections():
    gt = {1: dict.fromkeys(range(4), BOX)}
    track = [PredBox(S.TRACKING, BOX)] * 2 + [PredBox(S.OCCLUDED, BOX)] * 2

    assert identity_scores(gt, {1: track})["idf1"] == pytest.approx(2 / 3)


def test_run_fixture_can_select_one_target(tmp_path):
    # With one target selected, the others are look-alikes nobody tracks.
    from vectrax.evaluation.runner import run_fixture
    from vectrax.tracking.propagators import CsrtPropagator

    result = run_fixture(_synthetic_clip(tmp_path, [40, 160]), CsrtPropagator, only=2)

    assert set(result.tracks) == {2}
    assert result.tracks[2].success_rate > 0.9
