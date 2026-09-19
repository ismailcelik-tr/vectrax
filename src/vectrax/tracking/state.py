"""Track state machine.

DEGRADED  = visible, quality below good.
OCCLUDED  = not visible, within occlusion timeout.
LOST      = not visible past the timeout. Leaves only by operator command.

    INITIALIZING ──streak──► TRACKING ◄──► DEGRADED
         │                     │  ▲          │
      timeout              bad │  │ good     │ bad
         ▼                     ▼  │          ▼
        LOST ◄──timeout──── OCCLUDED ◄───────┘

PAUSE / RESUME / RESELECT / STOP are operator commands (see apply_command).
REACQUIRING arrives in Phase 4.
"""

from dataclasses import dataclass
from enum import Enum

from vectrax.tracking.config import TrackingConfig

__all__ = ["ALLOWED", "Command", "Evidence", "InvalidCommand", "TrackState", "apply_command", "next_state"]


class TrackState(Enum):
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    DEGRADED = "degraded"
    OCCLUDED = "occluded"
    LOST = "lost"
    PAUSED = "paused"
    STOPPED = "stopped"


class Command(Enum):
    PAUSE = "pause"
    RESUME = "resume"
    RESELECT = "reselect"
    STOP = "stop"


class InvalidCommand(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Evidence:
    quality: float | None  # combined TrackQuality; None = no observation
    ns_since_visible: int
    ns_in_state: int
    good_streak: int


_S = TrackState
_ACTIVE = frozenset({_S.INITIALIZING, _S.TRACKING, _S.DEGRADED, _S.OCCLUDED})
_OPERATOR_ONLY = frozenset({_S.LOST, _S.PAUSED, _S.STOPPED})

ALLOWED: dict[TrackState, frozenset[TrackState]] = {
    _S.INITIALIZING: frozenset({_S.INITIALIZING, _S.TRACKING, _S.LOST, _S.PAUSED, _S.STOPPED}),
    _S.TRACKING: frozenset({_S.DEGRADED, _S.OCCLUDED, _S.INITIALIZING, _S.PAUSED, _S.STOPPED}),
    _S.DEGRADED: frozenset({_S.TRACKING, _S.OCCLUDED, _S.INITIALIZING, _S.PAUSED, _S.STOPPED}),
    _S.OCCLUDED: frozenset({_S.TRACKING, _S.DEGRADED, _S.LOST, _S.INITIALIZING, _S.PAUSED, _S.STOPPED}),
    _S.LOST: frozenset({_S.INITIALIZING, _S.STOPPED}),
    _S.PAUSED: frozenset({_S.INITIALIZING, _S.STOPPED}),
    _S.STOPPED: frozenset(),
}


def next_state(state: TrackState, ev: Evidence, cfg: TrackingConfig) -> TrackState:
    if state in _OPERATOR_ONLY:
        return state

    visible = ev.quality is not None and ev.quality >= cfg.min_quality
    if state is _S.INITIALIZING:
        if ev.good_streak >= cfg.confirm_frames:
            return _S.TRACKING

        if ev.ns_in_state > cfg.init_timeout_ns:
            return _S.LOST

        return _S.INITIALIZING

    if not visible:
        if state is _S.OCCLUDED and ev.ns_since_visible > cfg.occlusion_timeout_ns:
            return _S.LOST

        return _S.OCCLUDED

    if ev.quality >= cfg.good_quality:
        return _S.TRACKING

    return _S.DEGRADED


def apply_command(state: TrackState, cmd: Command) -> TrackState:
    if state is _S.STOPPED:
        raise InvalidCommand(f"{cmd.value} on stopped track")

    if cmd is Command.STOP:
        return _S.STOPPED

    if cmd is Command.RESELECT:
        return _S.INITIALIZING

    if cmd is Command.PAUSE and state in _ACTIVE:
        return _S.PAUSED

    if cmd is Command.RESUME and state is _S.PAUSED:
        return _S.INITIALIZING

    raise InvalidCommand(f"{cmd.value} not valid in {state.value}")
