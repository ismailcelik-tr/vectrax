import itertools

import pytest

from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.state import (
    ALLOWED,
    Command,
    Evidence,
    InvalidCommand,
    TrackState,
    apply_command,
    next_state,
)

CFG = TrackingConfig(min_quality=0.3, good_quality=0.6, confirm_frames=3,
                     init_timeout_ns=1_000, occlusion_timeout_ns=500)
GOOD, WEAK, BAD = 0.9, 0.4, 0.1


def ev(quality, since_visible=0, in_state=0, streak=0):
    return Evidence(quality=quality, ns_since_visible=since_visible, ns_in_state=in_state, good_streak=streak)


S = TrackState


# Automatic transitions: one test per edge.

def test_initializing_to_tracking_after_confirm_streak():
    assert next_state(S.INITIALIZING, ev(GOOD, streak=3), CFG) is S.TRACKING


def test_initializing_waits_for_streak():
    assert next_state(S.INITIALIZING, ev(GOOD, streak=2), CFG) is S.INITIALIZING


def test_initializing_to_lost_on_timeout():
    assert next_state(S.INITIALIZING, ev(WEAK, in_state=1_001, streak=0), CFG) is S.LOST


def test_tracking_stays_on_good():
    assert next_state(S.TRACKING, ev(GOOD), CFG) is S.TRACKING


def test_tracking_to_degraded_on_weak():
    assert next_state(S.TRACKING, ev(WEAK), CFG) is S.DEGRADED


def test_tracking_to_occluded_on_bad():
    assert next_state(S.TRACKING, ev(BAD), CFG) is S.OCCLUDED


def test_tracking_to_occluded_on_no_observation():
    assert next_state(S.TRACKING, ev(None), CFG) is S.OCCLUDED


def test_degraded_to_tracking_on_good():
    assert next_state(S.DEGRADED, ev(GOOD), CFG) is S.TRACKING


def test_degraded_to_occluded_on_bad():
    assert next_state(S.DEGRADED, ev(BAD), CFG) is S.OCCLUDED


def test_occluded_stays_within_timeout():
    assert next_state(S.OCCLUDED, ev(None, since_visible=500), CFG) is S.OCCLUDED


def test_occluded_to_lost_after_timeout():
    assert next_state(S.OCCLUDED, ev(None, since_visible=501), CFG) is S.LOST


def test_occluded_to_tracking_on_good():
    assert next_state(S.OCCLUDED, ev(GOOD), CFG) is S.TRACKING


def test_occluded_to_degraded_on_weak():
    assert next_state(S.OCCLUDED, ev(WEAK), CFG) is S.DEGRADED


@pytest.mark.parametrize("state", [S.LOST, S.PAUSED, S.STOPPED])
def test_operator_only_states_ignore_evidence(state):
    assert next_state(state, ev(GOOD, streak=10), CFG) is state


# Operator commands.

@pytest.mark.parametrize("state", [S.INITIALIZING, S.TRACKING, S.DEGRADED, S.OCCLUDED])
def test_pause_from_active(state):
    assert apply_command(state, Command.PAUSE) is S.PAUSED


def test_resume_reinitializes():
    assert apply_command(S.PAUSED, Command.RESUME) is S.INITIALIZING


@pytest.mark.parametrize("state", [s for s in S if s is not S.STOPPED])
def test_stop_from_any(state):
    assert apply_command(state, Command.STOP) is S.STOPPED


@pytest.mark.parametrize("state", [s for s in S if s is not S.STOPPED])
def test_reselect_reinitializes(state):
    assert apply_command(state, Command.RESELECT) is S.INITIALIZING


@pytest.mark.parametrize(("state", "cmd"), [
    (S.TRACKING, Command.RESUME),
    (S.LOST, Command.PAUSE),
    (S.PAUSED, Command.PAUSE),
    (S.STOPPED, Command.RESUME),
    (S.STOPPED, Command.RESELECT),
    (S.STOPPED, Command.STOP),
])
def test_invalid_commands_rejected(state, cmd):
    with pytest.raises(InvalidCommand):
        apply_command(state, cmd)


# Table consistency.

def _evidence_grid():
    for q, since, in_state, streak in itertools.product(
        [None, BAD, WEAK, GOOD], [0, 501], [0, 1_001], [0, 3]
    ):
        yield ev(q, since, in_state, streak)


def test_every_transition_is_in_table():
    seen = set()
    for state, e in itertools.product(S, _evidence_grid()):
        nxt = next_state(state, e, CFG)
        assert nxt is state or nxt in ALLOWED[state], (state, e, nxt)
        seen.add((state, nxt))

    for state, cmd in itertools.product(S, Command):
        try:
            nxt = apply_command(state, cmd)
        except InvalidCommand:
            continue

        assert nxt in ALLOWED[state], (state, cmd, nxt)
        seen.add((state, nxt))

    declared = {(a, b) for a, targets in ALLOWED.items() for b in targets}
    assert declared <= seen, f"declared but unreachable: {declared - seen}"


def test_stopped_is_terminal():
    assert ALLOWED[S.STOPPED] == frozenset()
