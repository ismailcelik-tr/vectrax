import cv2
import numpy as np
import pytest

from vectrax.events import EventBus, EventType
from vectrax.frames import FramePacket, PixelFormat
from vectrax.tracking.config import NS_PER_S, TrackingConfig
from vectrax.tracking.geometry import Box
from vectrax.tracking.manager import TrackManager
from vectrax.tracking.observation import Observation, Origin
from vectrax.tracking.state import Command, TrackState

FRAME_NS = NS_PER_S // 30
CFG = TrackingConfig(confirm_frames=3, occlusion_timeout_ns=NS_PER_S // 2, reacquire_timeout_ns=NS_PER_S // 2)
BOX = Box(0.5, 0.5, 0.1, 0.1)
S = TrackState


class FakePropagator:
    """Returns scripted scores at the init box; None in the script = no observation."""

    def __init__(self, script):
        self.script = script
        self.inits = []
        self.updates = 0
        self.box = None

    def init(self, frame, box):
        self.inits.append((frame.frame_id, box))
        self.box = box

    def update(self, frame):
        self.updates += 1
        score = self.script(frame.frame_id)
        if score is None:
            return None

        return Observation(frame.frame_id, frame.capture_ns, self.box, score, Origin.PROPAGATOR)


def _frame(i):
    return FramePacket(i, "t", i * FRAME_NS, i * FRAME_NS, np.zeros((4, 4, 3), np.uint8), PixelFormat.BGR)


def _manager(script=lambda i: 0.9):
    bus = EventBus()
    events = []
    bus.subscribe(events.append)
    props = []

    def factory():
        props.append(FakePropagator(script))
        return props[-1]

    return TrackManager(CFG, factory, bus), events, props


def _run(mgr, start, n):
    snaps = None
    for i in range(start, start + n):
        snaps = mgr.step(_frame(i))

    return {s.track_id: s for s in snaps}


def test_select_then_confirm_to_tracking():
    mgr, events, props = _manager()
    tid = mgr.select(BOX)

    snaps = _run(mgr, 0, 3)

    assert tid == 1
    assert snaps[tid].state is S.TRACKING
    assert props[0].inits[0] == (0, BOX)
    assert [e.type for e in events][:1] == [EventType.TARGET_SELECTED]
    changes = [(e.payload["from"], e.payload["to"]) for e in events if e.type is EventType.STATE_CHANGED]
    assert changes == [(S.INITIALIZING, S.TRACKING)]


def test_weak_then_missing_then_timeout():
    script = {**{i: 0.9 for i in range(3)}, 3: 0.4}
    mgr, _, _ = _manager(lambda i: script.get(i))
    tid = mgr.select(BOX)

    assert _run(mgr, 0, 3)[tid].state is S.TRACKING
    assert _run(mgr, 3, 1)[tid].state is S.DEGRADED
    assert _run(mgr, 4, 1)[tid].state is S.OCCLUDED

    frames_to_timeout = CFG.occlusion_timeout_ns // FRAME_NS + 1
    assert _run(mgr, 5, frames_to_timeout)[tid].state is S.REACQUIRING

    frames_to_lost = CFG.reacquire_timeout_ns // FRAME_NS + 2
    assert _run(mgr, 5 + frames_to_timeout, frames_to_lost)[tid].state is S.LOST


def test_occluded_track_keeps_predicting():
    mgr, _, _ = _manager(lambda i: 0.9 if i < 5 else None)
    tid = mgr.select(BOX)
    before = _run(mgr, 0, 5)[tid]

    during = _run(mgr, 5, 3)[tid]

    assert during.state is S.OCCLUDED
    assert during.box is not None
    assert during.capture_ns > before.capture_ns


def test_ids_are_never_reused():
    mgr, events, _ = _manager()
    first = mgr.select(BOX)
    mgr.remove(first)
    _run(mgr, 0, 1)

    second = mgr.select(BOX)

    assert second != first
    assert EventType.TRACK_REMOVED in [e.type for e in events]


def test_paused_track_is_not_updated_and_resume_reinitializes():
    mgr, _, props = _manager()
    tid = mgr.select(BOX)
    _run(mgr, 0, 3)
    updates_before = props[0].updates
    mgr.command(tid, Command.PAUSE)

    assert _run(mgr, 3, 5)[tid].state is S.PAUSED
    assert props[0].updates == updates_before

    mgr.command(tid, Command.RESUME)
    assert _run(mgr, 8, 1)[tid].state is S.INITIALIZING
    assert len(props[-1].inits) == 1 and props[-1].inits[0][0] == 8


def test_reselect_keeps_id_and_uses_new_box():
    mgr, _, props = _manager()
    tid = mgr.select(BOX)
    _run(mgr, 0, 3)
    new_box = Box(0.2, 0.2, 0.05, 0.05)

    mgr.command(tid, Command.RESELECT, box=new_box)
    snaps = _run(mgr, 3, 1)

    assert snaps[tid].state is S.INITIALIZING
    assert props[-1].inits[-1] == (3, new_box)


def test_invalid_command_is_reported_not_raised():
    mgr, events, _ = _manager()
    tid = mgr.select(BOX)
    mgr.command(tid, Command.STOP)
    mgr.command(tid, Command.RESUME)
    mgr.command(99, Command.PAUSE)

    snaps = _run(mgr, 0, 1)

    assert snaps[tid].state is S.STOPPED
    rejected = [e for e in events if e.type is EventType.COMMAND_REJECTED]
    assert [e.track_id for e in rejected] == [tid, 99]


def test_tracks_are_independent():
    mgr, _, _ = _manager()
    a = mgr.select(BOX)
    b = mgr.select(Box(0.2, 0.2, 0.1, 0.1))
    mgr.command(a, Command.PAUSE)

    snaps = _run(mgr, 0, 3)

    assert snaps[a].state is S.PAUSED
    assert snaps[b].state is S.TRACKING


def test_parallel_propagation_matches_sequential():
    from concurrent.futures import ThreadPoolExecutor

    import cv2

    from vectrax.tracking.propagators import CsrtPropagator

    rng = np.random.default_rng(3)
    background = rng.integers(60, 120, (240, 320, 3), dtype=np.uint8)
    patch = cv2.GaussianBlur(rng.integers(0, 256, (30, 30, 3), dtype=np.uint8), (0, 0), 3)

    def frame(i):
        img = background.copy()
        img[50:80, 20 + 3 * i:50 + 3 * i] = patch
        img[150:180, 250 - 3 * i:280 - 3 * i] = patch
        return FramePacket(i, "t", i * FRAME_NS, i * FRAME_NS, img, PixelFormat.BGR)

    def run(executor):
        mgr = TrackManager(CFG, CsrtPropagator, EventBus(), executor=executor)
        mgr.select(Box.from_xywh_px(20, 50, 30, 30, 320, 240))
        mgr.select(Box.from_xywh_px(250, 150, 30, 30, 320, 240))
        return [mgr.step(frame(i)) for i in range(20)]

    with ThreadPoolExecutor(2) as pool:
        assert run(pool) == run(None)


def test_jittering_score_still_confirms_within_the_window():
    """NanoTrack's box oscillates on a 3-frame rhythm (data/sessions/demo_detect,
    f303-333), so the score crosses good_quality up and down. Confirmation counts
    good frames in a window; consecutive ones would time out into LOST."""
    one_good_in_three = lambda i: 0.9 if i % 3 == 0 else 0.45
    manager, _, _ = _manager(one_good_in_three)
    track_id = manager.select(BOX)

    states = [next(s.state for s in manager.step(_frame(i)) if s.track_id == track_id)
              for i in range(15)]

    assert S.LOST not in states
    assert S.TRACKING in states



LABEL = "cup"


def _det(frame_id, box, label=LABEL):
    return Observation(frame_id, frame_id * FRAME_NS, box, 0.8, Origin.DETECTOR, label)


def _state(mgr, i, track_id, detections=()):
    return next(s.state for s in mgr.step(_frame(i), detections) if s.track_id == track_id)


HOLD_FRAMES = CFG.detection_hold_ns // FRAME_NS + 1
CONFIRM = CFG.confirm_frames
READY = CONFIRM + HOLD_FRAMES  # confirmed, and the evidence from selection has lapsed


def _confirm(mgr, seen):
    """Frames 0..READY-1. The detector reports `seen` while the track initializes, then nothing."""
    for i in range(READY):
        mgr.step(_frame(i), [_det(i - 1, box, label) for box, label in seen] if 0 < i < CONFIRM else [])


@pytest.mark.parametrize("score, weak", [(0.1, S.OCCLUDED), (0.45, S.DEGRADED)])
def test_a_detection_lifts_a_weak_track_to_tracking(score, weak):
    mgr, _, _ = _manager(lambda i: 0.9 if i < READY else score)
    tid = mgr.select(BOX)
    _confirm(mgr, [(BOX, LABEL)])
    assert _state(mgr, READY, tid) is weak

    assert _state(mgr, READY + 1, tid, [_det(READY, BOX)]) is S.TRACKING


def test_the_lift_lapses_when_detections_stop():
    mgr, _, _ = _manager(lambda i: 0.9 if i < READY else 0.1)
    tid = mgr.select(BOX)
    _confirm(mgr, [(BOX, LABEL)])
    _state(mgr, READY, tid, [_det(READY - 1, BOX)])

    states = [_state(mgr, i, tid) for i in range(READY + 1, READY + 1 + HOLD_FRAMES)]

    assert states[0] is S.TRACKING
    assert states[-1] is S.OCCLUDED


def test_no_detection_never_demotes():
    # A COCO detector cannot see arbitrary targets (R1).
    mgr, _, _ = _manager()
    tid = mgr.select(BOX)
    _confirm(mgr, [(BOX, LABEL)])
    elsewhere = Box(0.1, 0.1, 0.05, 0.05)

    states = [_state(mgr, i, tid, [_det(i - 1, elsewhere)]) for i in range(READY, READY + HOLD_FRAMES)]

    assert set(states) == {S.TRACKING}


class _Moving(FakePropagator):
    """Reports `path(frame_id)` as its box."""

    def __init__(self, script, path):
        super().__init__(script)
        self.path = path

    def update(self, frame):
        self.box = self.path(frame.frame_id)
        return super().update(frame)


def _moving_manager(script, path):
    return TrackManager(CFG, lambda: _Moving(script, path), EventBus())


def test_a_late_detection_is_matched_where_the_track_was_when_it_was_seen():
    speed, lag = 0.05, 3  # the track moves a box width in two frames

    def path(i):
        return Box(0.1 + speed * i, 0.5, 0.1, 0.1)

    mgr = _moving_manager(lambda i: 0.9 if i < READY else 0.45, path)
    tid = mgr.select(path(0))
    for i in range(READY):
        mgr.step(_frame(i), [_det(i - 1, path(i - 1))] if 0 < i < CONFIRM else [])

    _run(mgr, READY, lag)
    now = READY + lag

    assert _state(mgr, now, tid, [_det(now - lag, path(now - lag))]) is S.TRACKING


def test_a_matched_detection_pulls_the_box_onto_the_target():
    # The propagator loses a moving target and stays put; the detector keeps seeing it.
    speed, frames = 0.01, 30

    def target(i):
        return Box(0.3 + speed * i, 0.5, 0.1, 0.1)

    mgr = _moving_manager(lambda i: 0.9 if i < READY else 0.0, lambda i: target(min(i, READY - 1)))
    tid = mgr.select(target(0))
    _confirm(mgr, [(target(0), LABEL)])

    for i in range(READY, READY + frames):
        snap = next(s for s in mgr.step(_frame(i), [_det(i - 1, target(i - 1))] if i % 2 else [])
                    if s.track_id == tid)

    assert snap.state is S.TRACKING
    assert abs(snap.box.cx - target(READY + frames - 1).cx) < target(0).w / 2


def test_a_detection_does_not_confirm_an_initializing_track():
    mgr, _, _ = _manager(lambda i: 0.45)
    tid = mgr.select(BOX)
    _run(mgr, 0, 1)

    states = [_state(mgr, i, tid, [_det(i - 1, BOX)]) for i in range(1, 5)]

    assert set(states) == {S.INITIALIZING}


def test_a_detection_does_not_revive_a_lost_track():
    mgr, _, _ = _manager(lambda i: 0.9 if i < READY else None)
    tid = mgr.select(BOX)
    _confirm(mgr, [(BOX, LABEL)])
    frames_to_lost = READY + (CFG.occlusion_timeout_ns + CFG.reacquire_timeout_ns) // FRAME_NS + 3
    _run(mgr, READY, frames_to_lost - READY)
    assert _state(mgr, frames_to_lost, tid) is S.LOST

    assert _state(mgr, frames_to_lost + 1, tid, [_det(frames_to_lost, BOX)]) is S.LOST


def test_only_the_class_seen_at_selection_lifts():
    # demo_20260920: with the phone gone, a chair behind it lifted the phone's track.
    mgr, _, _ = _manager(lambda i: 0.9 if i < READY else 0.1)
    tid = mgr.select(BOX)
    _confirm(mgr, [(BOX, "cell phone")])

    assert _state(mgr, READY, tid, [_det(READY - 1, BOX, "chair")]) is S.OCCLUDED
    assert _state(mgr, READY + 1, tid, [_det(READY, BOX, "cell phone")]) is S.TRACKING


def test_a_target_the_detector_missed_at_selection_is_never_lifted():
    mgr, _, _ = _manager(lambda i: 0.9 if i < READY else 0.1)
    tid = mgr.select(BOX)
    _confirm(mgr, [])

    assert _state(mgr, READY, tid, [_det(READY - 1, BOX)]) is S.OCCLUDED


@pytest.mark.parametrize("right_label, lifted", [("bottle", 0), (LABEL, 1)])
def test_the_label_weighs_in_when_a_detection_sits_between_two_tracks(right_label, lifted):
    # The detection overlaps the right track more; a shared label pulls it to the left one.
    boxes = [Box(0.40, 0.5, 0.2, 0.2), Box(0.52, 0.5, 0.2, 0.2)]
    mgr, _, _ = _manager(lambda i: 0.9 if i < READY else 0.45)
    ids = [mgr.select(b) for b in boxes]
    _confirm(mgr, [(boxes[0], LABEL), (boxes[1], right_label)])

    between = _det(READY - 1, Box(0.465, 0.5, 0.2, 0.2))
    snaps = {s.track_id: s.state for s in mgr.step(_frame(READY), [between])}

    assert [snaps[i] for i in ids] == [S.TRACKING if k == lifted else S.DEGRADED for k in range(2)]


SAME = CFG.same_object_ns // FRAME_NS + 1
VISIBLE = {S.TRACKING, S.DEGRADED}


def _two_on(paths, scores):
    """Track i reports paths[i](frame) with score scores[i](frame)."""
    made = iter(zip(scores, paths, strict=True))
    mgr = TrackManager(CFG, lambda: _Moving(*next(made)), EventBus())
    return mgr, [mgr.select(path(0)) for path in paths]


def test_two_tracks_on_one_object_the_weaker_yields():
    # demo_detect, phase3_verify: both pupil tracks ended up on one pupil.
    mgr, (a, b) = _two_on([lambda i: BOX] * 2, [lambda i: 0.9, lambda i: 0.8])

    snaps = _run(mgr, 0, CONFIRM + SAME)
    later = [_state(mgr, i, b) for i in range(CONFIRM + SAME, CONFIRM + SAME + 10)]

    assert snaps[a].state is S.TRACKING
    assert snaps[b].state is S.OCCLUDED
    assert not VISIBLE & set(later)


def test_a_brief_overlap_keeps_both():
    mgr, ids = _two_on([lambda i: BOX] * 2, [lambda i: 0.9, lambda i: 0.8])

    snaps = _run(mgr, 0, CONFIRM + SAME - 2)

    assert {snaps[i].state for i in ids} == {S.TRACKING}


def test_on_a_tie_the_newer_track_yields():
    mgr, (a, b) = _two_on([lambda i: BOX] * 2, [lambda i: 0.9] * 2)

    snaps = _run(mgr, 0, CONFIRM + SAME)

    assert (snaps[a].state, snaps[b].state) == (S.TRACKING, S.OCCLUDED)




# Reacquisition (Phase 4 step 3). Frames carry a textured target so appearance can be scored.

IMG_W, IMG_H, SIDE_PX = 320, 240, 40
REACQ_CFG = TrackingConfig(confirm_frames=3, occlusion_timeout_ns=NS_PER_S // 2, reacquire_timeout_ns=2 * NS_PER_S)
REACQUIRING_BY = (REACQ_CFG.occlusion_timeout_ns + NS_PER_S) // FRAME_NS  # searched for 0.5 s


def _textures(n, seed=0):
    rng = np.random.default_rng(seed)
    background = rng.integers(60, 120, (IMG_H, IMG_W, 3), dtype=np.uint8)
    patches = []
    for _ in range(n):
        noise = cv2.GaussianBlur(rng.integers(0, 256, (SIDE_PX, SIDE_PX, 3), dtype=np.uint8), (0, 0), 3)
        patches.append(cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX))

    return background, patches


def _px(x, y):
    return Box.from_xywh_px(x, y, SIDE_PX, SIDE_PX, IMG_W, IMG_H)


class _Scene:
    """Draws `placed(i)` = [(patch, (x, y))] on frame i."""

    def __init__(self, background, placed):
        self.background, self.placed = background, placed

    def frame(self, i):
        img = self.background.copy()
        for patch, (x, y) in self.placed(i):
            img[y:y + SIDE_PX, x:x + SIDE_PX] = patch

        return FramePacket(i, "t", i * FRAME_NS, i * FRAME_NS, img, PixelFormat.BGR)


class _Relocking(_Moving):
    """_Moving until re-initialized; then a good score at its new box, or along `path` if it `follows`."""

    def __init__(self, script, path, follows=False):
        super().__init__(script, path)
        self.follows = follows

    def update(self, frame):
        if len(self.inits) == 1:
            return super().update(frame)

        self.updates += 1
        if self.follows:
            self.box = self.path(frame.frame_id)

        return Observation(frame.frame_id, frame.capture_ns, self.box, 0.9, Origin.PROPAGATOR)


def _scene_manager(tracks):
    """tracks: [(script, path[, follows])] per selection; each path(0) is where it is selected."""
    made = iter(tracks)
    mgr = TrackManager(REACQ_CFG, lambda: _Relocking(*next(made)), EventBus())
    return mgr, [mgr.select(path(0)) for _, path, *_ in tracks]


def _play(mgr, scene, start, stop, track_id):
    return [next(s for s in mgr.step(scene.frame(i)) if s.track_id == track_id) for i in range(start, stop)]


def _at(box, x, y, px=4):
    bx, by, _, _ = box.to_xywh_px(IMG_W, IMG_H)
    return abs(bx - x) <= px and abs(by - y) <= px


HOME_PX, AWAY_PX = (140, 100), (220, 140)
HIDE, BACK = 10, 25


def _hidden_then(target, back_at):
    return lambda i: [] if HIDE <= i < BACK else [(target, HOME_PX if i < HIDE else back_at)]


def _lost_propagator():
    return lambda i: 0.9 if i < HIDE else None, lambda i: _px(*HOME_PX)


def test_reacquiring_finds_the_target_where_it_came_back():
    background, (target,) = _textures(1)
    scene = _Scene(background, _hidden_then(target, AWAY_PX))
    mgr, (tid,) = _scene_manager([_lost_propagator()])

    snaps = _play(mgr, scene, 0, BACK + REACQUIRING_BY, tid)

    assert S.REACQUIRING in {s.state for s in snaps}
    assert snaps[-1].state is S.TRACKING
    assert _at(snaps[-1].box, *AWAY_PX)


def test_two_look_alikes_keep_the_track_reacquiring():
    background, (target,) = _textures(1)

    def placed(i):
        return [(target, HOME_PX)] if i < HIDE else [] if i < BACK else [(target, (60, 100)), (target, (220, 100))]

    mgr, (tid,) = _scene_manager([_lost_propagator()])

    snaps = _play(mgr, _Scene(background, placed), 0, BACK + REACQUIRING_BY, tid)

    assert snaps[-1].state is S.REACQUIRING


def test_the_look_alike_another_track_holds_is_not_taken():
    background, (target,) = _textures(1)
    held_px, free_px = (220, 100), (60, 100)

    def placed(i):
        own = [(target, HOME_PX)] if i < HIDE else [] if i < BACK else [(target, free_px)]
        return [(target, held_px), *own]

    holder = (lambda i: 0.9, lambda i: _px(*held_px))
    mgr, (_, tid) = _scene_manager([holder, _lost_propagator()])

    snaps = _play(mgr, _Scene(background, placed), 0, BACK + REACQUIRING_BY, tid)

    assert snaps[-1].state is S.TRACKING
    assert _at(snaps[-1].box, *free_px)


def test_a_propagator_box_on_background_is_not_reacquired():
    # phase3_verify2 f372: the propagator sat on a stale box far from the gone phone.
    background, (target,) = _textures(1)
    scene = _Scene(background, lambda i: [(target, HOME_PX)] if i < HIDE else [])
    stale = (lambda i: 0.9 if i < HIDE else 0.5, lambda i: _px(*HOME_PX) if i < HIDE else _px(*AWAY_PX))
    mgr, (tid,) = _scene_manager([stale])

    snaps = _play(mgr, scene, 0, BACK + REACQUIRING_BY, tid)

    assert snaps[-1].state is S.REACQUIRING


def test_the_propagator_box_is_reacquired_when_it_is_on_the_target():
    # non_coco, fast_motion: the Kalman stopped or overshot, the propagator kept the target.
    background, (target,) = _textures(1)

    def where(i):
        return HOME_PX if i < HIDE else (min(HOME_PX[0] + 4 * (i - HIDE), 270), HOME_PX[1])

    scene = _Scene(background, lambda i: [(target, where(i))])
    dips = (lambda i: 0.9 if i < HIDE else 0.1, lambda i: _px(*where(i)), True)
    mgr, (tid,) = _scene_manager([dips])

    snaps = _play(mgr, scene, 0, BACK + REACQUIRING_BY, tid)

    assert snaps[-1].state is S.TRACKING
    assert _at(snaps[-1].box, *where(BACK + REACQUIRING_BY - 1), px=8)


def test_an_occluded_track_does_not_take_a_far_box():
    # Coast long enough for the motion gate to pass anything, still inside the timeout.
    hidden, back = 5, 18
    far = Box(0.8, 0.5, 0.1, 0.1)
    mgr = TrackManager(CFG, lambda: _Moving(lambda i: None if hidden <= i < back else 0.9,
                                            lambda i: far if i >= back else BOX), EventBus())
    tid = mgr.select(BOX)
    _run(mgr, 0, back)

    assert _state(mgr, back, tid) is S.OCCLUDED


def test_weak_observations_do_not_hold_a_track_occluded():
    below_exit = CFG.min_quality + CFG.quality_hysteresis / 2
    mgr, _, _ = _manager(lambda i: 0.9 if i < 3 else None if i == 3 else below_exit)
    tid = mgr.select(BOX)

    snaps = _run(mgr, 0, 3 + CFG.occlusion_timeout_ns // FRAME_NS + 2)

    assert snaps[tid].state is S.REACQUIRING


def test_a_yielded_track_reacquires_its_own_object():
    # phase3_verify: one pupil track slid onto the other pupil and yielded.
    background, (left, right) = _textures(2)
    left_px, right_px = (60, 100), (180, 100)
    slide_from, slide_frames = 5, 40

    def slides(i):
        k = min(max(i - slide_from, 0) / slide_frames, 1.0)
        return _px(round(right_px[0] + (left_px[0] - right_px[0]) * k), right_px[1])

    scene = _Scene(background, lambda i: [(left, left_px), (right, right_px)])
    mgr, (_, b) = _scene_manager([(lambda i: 0.9, lambda i: _px(*left_px)), (lambda i: 0.9, slides)])

    snaps = _play(mgr, scene, 0, 150, b)

    assert S.OCCLUDED in {s.state for s in snaps}
    assert snaps[-1].state is S.TRACKING
    assert _at(snaps[-1].box, *right_px)
