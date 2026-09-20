import numpy as np

from vectrax.events import EventBus, EventType
from vectrax.frames import FramePacket, PixelFormat
from vectrax.tracking.config import NS_PER_S, TrackingConfig
from vectrax.tracking.geometry import Box
from vectrax.tracking.manager import TrackManager
from vectrax.tracking.observation import Observation, Origin
from vectrax.tracking.state import Command, TrackState

FRAME_NS = NS_PER_S // 30
CFG = TrackingConfig(confirm_frames=3, occlusion_timeout_ns=NS_PER_S // 2)
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
    assert _run(mgr, 5, frames_to_timeout)[tid].state is S.LOST


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
