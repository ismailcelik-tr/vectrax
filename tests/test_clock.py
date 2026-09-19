import pytest

from vectrax.clock import MonotonicClock, VirtualClock


def test_virtual_clock_starts_at_given_time():
    clock = VirtualClock(start_ns=1_000)

    assert clock.now_ns() == 1_000


def test_virtual_clock_advance_and_set():
    clock = VirtualClock()

    clock.advance(500)
    assert clock.now_ns() == 500

    clock.set(2_000)
    assert clock.now_ns() == 2_000


def test_virtual_clock_never_goes_back():
    clock = VirtualClock(start_ns=1_000)

    with pytest.raises(ValueError):
        clock.set(999)

    with pytest.raises(ValueError):
        clock.advance(-1)


def test_monotonic_clock_does_not_decrease():
    clock = MonotonicClock()
    first = clock.now_ns()

    assert clock.now_ns() >= first
