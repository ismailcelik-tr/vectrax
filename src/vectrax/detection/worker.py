"""Detection off the tick thread: the real-time path never waits for a model."""

import threading
from dataclasses import dataclass

from vectrax.buffer import LatestFrameBuffer
from vectrax.clock import Clock, MonotonicClock
from vectrax.frames import FramePacket
from vectrax.metrics import RunMode
from vectrax.tracking.observation import Observation

__all__ = ["DetectionResult", "InferenceWorker"]

STRIDE = 2  # every other frame at 30 fps: 15 Hz, which the eye does not catch
GET_TIMEOUT_S = 0.1


@dataclass(frozen=True, slots=True)
class DetectionResult:
    frame_id: int  # the frame the detector saw, not the current one
    capture_ns: int
    observations: list[Observation]
    inference_ns: int


class InferenceWorker:
    """REALTIME runs the detector on its own thread behind a capacity-1 buffer:
    a slow model drops frames, it never queues them or delays the tick.
    DETERMINISTIC runs it inline on `submit`, so a replay reproduces exactly.

    A failed detection is counted and skipped in REALTIME; DETERMINISTIC lets it
    raise, because a replay that silently skips inference is not a replay.
    """

    def __init__(self, detector, mode: RunMode, stride: int = STRIDE, clock: Clock | None = None):
        if stride < 1:
            raise ValueError("stride must be >= 1")

        self._detector = detector
        self._mode = mode
        self._stride = stride
        self._clock = clock or MonotonicClock()
        self._pending = LatestFrameBuffer(capacity=1)
        self._done = []
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._thread = None
        self._processed = 0
        self._failures = 0

    @property
    def dropped(self) -> int:
        return self._pending.dropped

    @property
    def processed(self) -> int:
        with self._lock:
            return self._processed

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    def start(self) -> None:
        if self._mode is RunMode.DETERMINISTIC or self._thread is not None:
            return

        self._thread = threading.Thread(target=self._run, name="inference", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        self._pending.close()
        if self._thread is not None:
            self._thread.join(timeout=GET_TIMEOUT_S * 2)
            self._thread = None

    def submit(self, frame: FramePacket) -> None:
        """Never blocks. Frames off the stride, and frames arriving while the
        detector is busy, are dropped."""
        if frame.frame_id % self._stride:
            return

        if self._mode is RunMode.DETERMINISTIC:
            self._record(self._detect(frame))
            return

        self._pending.put(frame)

    def results(self) -> list[DetectionResult]:
        """Everything finished since the last call, oldest first."""
        with self._lock:
            done, self._done = self._done, []

        return done

    def _run(self) -> None:
        while not self._stopping.is_set():
            frame = self._pending.get(GET_TIMEOUT_S)
            if frame is None:
                continue

            try:
                self._record(self._detect(frame))
            except Exception:  # noqa: BLE001 — one bad frame must not end detection
                with self._lock:
                    self._failures += 1

    def _detect(self, frame: FramePacket) -> DetectionResult:
        started = self._clock.now_ns()
        observations = self._detector.detect(frame)
        return DetectionResult(frame.frame_id, frame.capture_ns, observations,
                               self._clock.now_ns() - started)

    def _record(self, result: DetectionResult) -> None:
        with self._lock:
            self._done.append(result)
            self._processed += 1
