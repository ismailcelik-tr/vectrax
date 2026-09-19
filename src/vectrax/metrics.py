"""Per-frame stage timing and latency distributions."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = ["FrameTiming", "LatencyStats", "Metrics", "RunMode"]

NS_PER_MS = 1_000_000


class RunMode(Enum):
    REALTIME = "realtime"  # live source, frames may drop
    DETERMINISTIC = "deterministic"  # every frame, replay-identical tracking


@dataclass(frozen=True, slots=True)
class FrameTiming:
    capture_ns: int
    arrival_ns: int
    tick_ns: int
    tracked_ns: int
    render_ns: int | None = None


class LatencyStats:
    def __init__(self):
        self._values: list[int] = []

    def add(self, ns: int) -> None:
        self._values.append(ns)

    def summary(self) -> dict | None:
        if not self._values:
            return None

        a = np.asarray(self._values, dtype=np.float64) / NS_PER_MS
        return {
            "n": int(a.size),
            "p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3),
            "p99": round(float(np.percentile(a, 99)), 3),
            "max": round(float(a.max()), 3),
        }


class Metrics:
    """In DETERMINISTIC mode capture_ns comes from a recording, so spans
    against it are meaningless; only processing time is reported."""

    def __init__(self, mode: RunMode, warmup_frames: int = 0):
        self.mode = mode
        self._warmup = warmup_frames
        self._frames = 0
        self._spans = {"track_ms": LatencyStats()}
        if mode is RunMode.REALTIME:
            for name in ("capture_to_arrival_ms", "arrival_to_tick_ms", "capture_to_tracked_ms", "capture_to_render_ms"):
                self._spans[name] = LatencyStats()

    def observe(self, t: FrameTiming) -> None:
        if self._warmup > 0:
            self._warmup -= 1
            return

        self._frames += 1
        self._spans["track_ms"].add(t.tracked_ns - t.tick_ns)
        if self.mode is not RunMode.REALTIME:
            return

        self._spans["capture_to_arrival_ms"].add(t.arrival_ns - t.capture_ns)
        self._spans["arrival_to_tick_ms"].add(t.tick_ns - t.arrival_ns)
        self._spans["capture_to_tracked_ms"].add(t.tracked_ns - t.capture_ns)
        if t.render_ns is not None:
            self._spans["capture_to_render_ms"].add(t.render_ns - t.capture_ns)

    def summary(self) -> dict:
        return {"mode": self.mode.value, "frames": self._frames,
                **{name: stats.summary() for name, stats in self._spans.items()}}
