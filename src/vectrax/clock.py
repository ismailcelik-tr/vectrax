"""Time source. All VectraX timestamps are monotonic nanoseconds."""

import time
from typing import Protocol

__all__ = ["Clock", "MonotonicClock", "VirtualClock"]


class Clock(Protocol):
    def now_ns(self) -> int: ...


class MonotonicClock:
    def now_ns(self) -> int:
        return time.monotonic_ns()


class VirtualClock:
    """Driven by replay or tests; never moves backwards."""

    def __init__(self, start_ns: int = 0):
        self._now = start_ns

    def now_ns(self) -> int:
        return self._now

    def set(self, ns: int) -> None:
        if ns < self._now:
            raise ValueError(f"clock cannot go back: {ns} < {self._now}")

        self._now = ns

    def advance(self, delta_ns: int) -> None:
        self.set(self._now + delta_ns)
