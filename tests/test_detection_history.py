import pytest

from vectrax.detection.history import TrackHistory, carry_forward
from vectrax.tracking.geometry import Box

BOX = Box(0.5, 0.5, 0.2, 0.2)


def test_history_returns_the_boxes_of_a_recorded_frame():
    history = TrackHistory(window=4)

    history.record(10, {1: BOX})

    assert history.at(10) == {1: BOX}


def test_frames_outside_the_window_are_forgotten():
    history = TrackHistory(window=2)

    for frame_id in range(5):
        history.record(frame_id, {1: BOX})

    assert history.at(4) == {1: BOX}
    assert history.at(2) == {}


def test_carry_forward_shifts_by_how_far_the_track_moved():
    detection = Box(0.40, 0.50, 0.10, 0.10)
    seen_at = Box(0.42, 0.50, 0.20, 0.20)  # track box on the frame the detector saw
    now = Box(0.52, 0.55, 0.24, 0.24)  # same track two frames later, inflated

    carried = carry_forward(detection, seen_at, now)

    assert (carried.cx, carried.cy) == pytest.approx((0.50, 0.55))
    # the detector's size is the correction, so it survives the carry
    assert (carried.w, carried.h) == pytest.approx((0.10, 0.10))
