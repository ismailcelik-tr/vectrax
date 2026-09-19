"""Bounded frame hand-off between threads. Stale frames are dropped, never queued."""

import threading
from collections import deque

__all__ = ["LatestFrameBuffer"]


class LatestFrameBuffer:
    def __init__(self, capacity: int = 1):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")

        self._items = deque()
        self._capacity = capacity
        self._cond = threading.Condition()
        self._dropped = 0
        self._closed = False

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def closed(self) -> bool:
        return self._closed

    def put(self, item) -> None:
        with self._cond:
            if len(self._items) == self._capacity:
                self._items.popleft()
                self._dropped += 1

            self._items.append(item)
            self._cond.notify()

    def get(self, timeout_s: float):
        """Oldest held item, or None on timeout or close."""
        with self._cond:
            self._cond.wait_for(lambda: self._items or self._closed, timeout=timeout_s)
            if not self._items:
                return None

            return self._items.popleft()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
