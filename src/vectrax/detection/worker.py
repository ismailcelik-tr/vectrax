"""Inference off the tick thread: the real-time path never waits for a model."""

import threading
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from vectrax.buffer import LatestFrameBuffer
from vectrax.clock import Clock, MonotonicClock
from vectrax.frames import FramePacket
from vectrax.metrics import RunMode
from vectrax.tracking.geometry import Box
from vectrax.tracking.observation import Observation

__all__ = ["EmbedRequest", "InferenceResult", "InferenceWorker"]

STRIDE = 2  # every other frame at 30 fps: 15 Hz, which the eye does not catch
GET_TIMEOUT_S = 0.1


@dataclass(frozen=True, slots=True)
class EmbedRequest:
    key: Hashable  # the caller's; comes back with the embedding
    box: Box


@dataclass(frozen=True, slots=True)
class InferenceResult:
    frame_id: int  # the frame the models saw, not the current one
    capture_ns: int
    observations: list[Observation]
    inference_ns: int
    embeddings: dict[Hashable, np.ndarray | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Job:
    frame: FramePacket
    detect: bool
    requests: Sequence[EmbedRequest]
    labels: frozenset[str]


class InferenceWorker:
    """REALTIME runs the detector on its own thread behind a capacity-1 buffer:
    a slow model drops frames, it never queues them or delays the tick.
    DETERMINISTIC runs it inline on `submit`, so a replay reproduces exactly.

    The detector must provide `load()` and `detect(frame)`; the optional
    embedder `load()` and `embed(frame, boxes)`. `start()` loads them eagerly
    and lets a bad model fail there, where the caller sees it: per-frame
    failures are counted and skipped in REALTIME, so a model that never works
    would otherwise stay silent. DETERMINISTIC lets a failure raise, because a
    replay that silently skips inference is not a replay.
    """

    def __init__(self, detector, mode: RunMode, stride: int = STRIDE, clock: Clock | None = None, embedder=None):
        if stride < 1:
            raise ValueError("stride must be >= 1")

        self._detector = detector
        self._embedder = embedder
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
        if self._thread is not None:
            return

        self._detector.load()
        if self._embedder is not None:
            self._embedder.load()

        if self._mode is RunMode.DETERMINISTIC:
            return

        self._thread = threading.Thread(target=self._run, name="inference", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        self._pending.close()
        if self._thread is not None:
            self._thread.join(timeout=GET_TIMEOUT_S * 2)
            self._thread = None

    def submit(self, frame: FramePacket, requests: Sequence[EmbedRequest] = (),
               labels: frozenset[str] = frozenset()) -> None:
        """Never blocks. Detects on the stride; embeds `requests` and detections
        of `labels` when there is an embedder. A frame with nothing to do, and a
        job arriving while the worker is busy, is dropped."""
        detect = frame.frame_id % self._stride == 0
        requests = requests if self._embedder is not None else ()
        if not detect and not requests:
            return

        job = _Job(frame, detect, tuple(requests), labels)
        if self._mode is RunMode.DETERMINISTIC:
            self._record(self._infer(job))
            return

        self._pending.put(job)

    def results(self) -> list[InferenceResult]:
        """Everything finished since the last call, oldest first."""
        with self._lock:
            done, self._done = self._done, []

        return done

    def _run(self) -> None:
        while not self._stopping.is_set():
            job = self._pending.get(GET_TIMEOUT_S)
            if job is None:
                continue

            try:
                self._record(self._infer(job))
            except Exception:  # noqa: BLE001 — one bad frame must not end detection
                with self._lock:
                    self._failures += 1

    def _infer(self, job: _Job) -> InferenceResult:
        started = self._clock.now_ns()
        frame = job.frame
        observations = self._detector.detect(frame) if job.detect else []
        embeddings = {}
        if self._embedder is not None:
            # One call: detections to check against tracks, and boxes the tracks asked for.
            watched = [i for i, o in enumerate(observations) if o.label in job.labels]
            boxes = [observations[i].box for i in watched] + [r.box for r in job.requests]
            vectors = self._embedder.embed(frame, boxes) if boxes else []
            for i, v in zip(watched, vectors, strict=False):
                observations[i] = replace(observations[i], embedding=v)

            embeddings = {r.key: v for r, v in zip(job.requests, vectors[len(watched):], strict=True)}

        return InferenceResult(frame.frame_id, frame.capture_ns, observations, self._clock.now_ns() - started,
                               embeddings)

    def _record(self, result: InferenceResult) -> None:
        with self._lock:
            self._done.append(result)
            self._processed += 1
