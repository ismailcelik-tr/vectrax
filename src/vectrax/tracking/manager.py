"""TrackManager: sole owner of track IDs and lifecycle.

Operator calls (select, command, remove) are queued and applied at the start
of the next step, so every change is tied to a frame and replays identically.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

from vectrax.events import Event, EventBus, EventType
from vectrax.frames import FramePacket
from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.geometry import Box
from vectrax.tracking.kalman import CvKalman
from vectrax.tracking.propagators import Propagator
from vectrax.tracking.quality import TrackQuality
from vectrax.tracking.state import (
    Command,
    Evidence,
    InvalidCommand,
    TrackState,
    apply_command,
    next_state,
)

__all__ = ["TrackManager", "TrackSnapshot"]

OPERATOR_SCORE = 1.0
TRAIL_LEN = 90

_UPDATED = frozenset({TrackState.INITIALIZING, TrackState.TRACKING, TrackState.DEGRADED, TrackState.OCCLUDED})


@dataclass(frozen=True, slots=True)
class TrackSnapshot:
    track_id: int
    state: TrackState
    box: Box  # Kalman estimate at capture_ns
    observed: Box | None
    quality: TrackQuality
    velocity: tuple[float, float]
    frame_id: int
    capture_ns: int
    trail: tuple[tuple[float, float], ...]


class _Op(Enum):
    SELECT = auto()
    COMMAND = auto()
    REMOVE = auto()


class _Track:
    def __init__(self, track_id: int):
        self.id = track_id
        self.state = TrackState.INITIALIZING
        self.kf: CvKalman | None = None
        self.prop: Propagator | None = None
        self.quality = TrackQuality(None, None)
        self.observed: Box | None = None
        self.state_since = 0
        self.last_visible = 0
        self.streak = 0
        self.frame_id = -1
        self.trail: deque[tuple[float, float]] = deque(maxlen=TRAIL_LEN)


class TrackManager:
    def __init__(self, cfg: TrackingConfig, propagator_factory: Callable[[], Propagator], bus: EventBus):
        self._cfg = cfg
        self._factory = propagator_factory
        self._bus = bus
        self._tracks: dict[int, _Track] = {}
        self._pending: list[tuple] = []
        self._next_id = 1

    def select(self, box: Box) -> int:
        track_id = self._next_id
        self._next_id += 1
        self._pending.append((_Op.SELECT, track_id, box))
        return track_id

    def command(self, track_id: int, cmd: Command, box: Box | None = None) -> None:
        self._pending.append((_Op.COMMAND, track_id, cmd, box))

    def remove(self, track_id: int) -> None:
        self._pending.append((_Op.REMOVE, track_id))

    def step(self, frame: FramePacket) -> list[TrackSnapshot]:
        pending, self._pending = self._pending, []
        for op in pending:
            self._apply(op, frame)

        for t in self._tracks.values():
            if t.state not in _UPDATED or t.frame_id == frame.frame_id:
                continue

            self._update(t, frame)

        return [self._snapshot(t) for t in self._tracks.values()]

    def _apply(self, op, frame):
        kind, track_id, *args = op
        if kind is _Op.SELECT:
            t = _Track(track_id)
            self._tracks[track_id] = t
            self._init(t, frame, args[0])
            self._publish(EventType.TARGET_SELECTED, frame, track_id, box=args[0])
            return

        t = self._tracks.get(track_id)
        if t is None:
            self._publish(EventType.COMMAND_REJECTED, frame, track_id, reason="unknown track")
            return

        if kind is _Op.REMOVE:
            del self._tracks[track_id]
            self._publish(EventType.TRACK_REMOVED, frame, track_id)
            return

        cmd, box = args
        try:
            new = apply_command(t.state, cmd)
        except InvalidCommand as e:
            self._publish(EventType.COMMAND_REJECTED, frame, track_id, command=cmd, reason=str(e))
            return

        if cmd is Command.RESELECT and box is None:
            self._publish(EventType.COMMAND_REJECTED, frame, track_id, command=cmd, reason="reselect needs a box")
            return

        if new is TrackState.INITIALIZING:
            self._init(t, frame, box or t.kf.box)
            return

        self._transition(t, new, frame)

    def _init(self, t, frame, box):
        t.prop = self._factory()
        t.prop.init(frame, box)
        t.kf = CvKalman(box, frame.capture_ns, self._cfg)
        t.quality = TrackQuality(OPERATOR_SCORE, None)
        t.observed = box
        t.streak = 1
        t.last_visible = frame.capture_ns
        t.frame_id = frame.frame_id
        t.trail.append((box.cx, box.cy))
        # Re-entry (reselect during INITIALIZING) must restart the init timeout.
        t.state_since = frame.capture_ns
        self._transition(t, TrackState.INITIALIZING, frame)

    def _update(self, t, frame):
        ns = frame.capture_ns
        t.kf.predict(ns)
        obs = t.prop.update(frame)
        residual = t.kf.residual(obs.box) if obs else None
        t.quality = TrackQuality(obs.score if obs else None, residual)
        q = t.quality.combined(self._cfg)

        if q is not None and q >= self._cfg.min_quality:
            t.kf.update(obs.box)
            t.last_visible = ns
        else:
            t.kf.coast()

        t.streak = t.streak + 1 if q is not None and q >= self._cfg.good_quality else 0
        t.observed = obs.box if obs else None
        t.frame_id = frame.frame_id
        t.trail.append((t.kf.box.cx, t.kf.box.cy))

        ev = Evidence(q, ns - t.last_visible, ns - t.state_since, t.streak)
        self._transition(t, next_state(t.state, ev, self._cfg), frame)

    def _transition(self, t, new, frame):
        old = t.state
        if new is old:
            return

        t.state = new
        t.state_since = frame.capture_ns
        self._publish(EventType.STATE_CHANGED, frame, t.id, **{"from": old, "to": new})

    def _publish(self, kind, frame, track_id, **payload):
        self._bus.publish(Event(kind, frame.frame_id, frame.capture_ns, track_id, payload))

    def _snapshot(self, t):
        return TrackSnapshot(
            track_id=t.id,
            state=t.state,
            box=t.kf.box,
            observed=t.observed,
            quality=t.quality,
            velocity=t.kf.velocity,
            frame_id=t.frame_id,
            capture_ns=t.kf.t_ns,
            trail=tuple(t.trail),
        )
