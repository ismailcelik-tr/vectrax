import threading
import time

import numpy as np
import pytest

from vectrax.clock import VirtualClock
from vectrax.detection.worker import InferenceWorker
from vectrax.frames import FramePacket, PixelFormat
from vectrax.metrics import RunMode
from vectrax.tracking.geometry import Box
from vectrax.tracking.observation import Observation, Origin

FRAME_NS = 33_333_333
JOIN_S = 2.0


def _wait_for(predicate):
    deadline = time.monotonic() + JOIN_S
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.005)

    return predicate()


def _frame(frame_id):
    return FramePacket(frame_id=frame_id, source_id="test", capture_ns=frame_id * FRAME_NS,
                       arrival_ns=frame_id * FRAME_NS, image=np.zeros((4, 4, 3), dtype=np.uint8),
                       pixel_format=PixelFormat.BGR)


class FakeDetector:
    """Records what it saw; `gate` holds it inside detect() until released."""

    def __init__(self, gate: threading.Event | None = None):
        self.seen = []
        self.entered = threading.Event()
        self.loaded = False
        self._gate = gate

    def load(self):
        self.loaded = True

    def detect(self, frame):
        if not self.loaded:
            raise RuntimeError("detector not loaded")

        self.entered.set()
        if self._gate is not None:
            self._gate.wait(JOIN_S)

        self.seen.append(frame.frame_id)
        return [Observation(frame.frame_id, frame.capture_ns, Box(0.5, 0.5, 0.1, 0.1), 0.9,
                            Origin.DETECTOR, "cup")]


def _worker(detector, **kwargs):
    worker = InferenceWorker(detector, RunMode.REALTIME, clock=VirtualClock(), **kwargs)
    worker.start()
    return worker


def test_submit_drops_frames_instead_of_queueing_them():
    gate = threading.Event()
    detector = FakeDetector(gate)
    worker = _worker(detector, stride=1)
    try:
        worker.submit(_frame(0))
        detector.entered.wait(JOIN_S)
        for i in range(1, 51):
            worker.submit(_frame(i))
    finally:
        gate.set()
        assert _wait_for(lambda: len(detector.seen) == 2)
        worker.stop()

    # frame 0 in flight, frame 50 held; the 49 in between never waited in a queue
    assert detector.seen == [0, 50]
    assert worker.dropped == 49


def test_worker_loads_the_detector_before_the_first_frame():
    detector = FakeDetector()
    worker = InferenceWorker(detector, RunMode.DETERMINISTIC, clock=VirtualClock(), stride=1)

    worker.start()
    worker.submit(_frame(0))

    assert detector.loaded
    assert worker.processed == 1


def test_stride_skips_frames_before_the_detector():
    detector = FakeDetector()
    worker = InferenceWorker(detector, RunMode.DETERMINISTIC, clock=VirtualClock(), stride=3)
    worker.start()

    for i in range(7):
        worker.submit(_frame(i))

    assert detector.seen == [0, 3, 6]


def test_results_carry_the_frame_they_saw_and_drain_once():
    detector = FakeDetector()
    worker = InferenceWorker(detector, RunMode.DETERMINISTIC, clock=VirtualClock(), stride=1)
    worker.start()

    worker.submit(_frame(4))
    results = worker.results()

    assert [r.frame_id for r in results] == [4]
    assert results[0].observations[0].origin is Origin.DETECTOR
    assert results[0].capture_ns == 4 * FRAME_NS
    assert worker.results() == []


class Boom:
    def __init__(self):
        self.calls = 0

    def load(self):
        pass

    def detect(self, frame):
        self.calls += 1
        raise RuntimeError("model blew up")


def test_a_failing_detection_never_stops_the_worker():
    detector = Boom()
    worker = _worker(detector, stride=1)
    try:
        worker.submit(_frame(0))
        worker.submit(_frame(1))
        assert _wait_for(lambda: worker.failures >= 1)
    finally:
        worker.stop()

    assert worker.failures >= 1
    assert worker.results() == []


def test_deterministic_mode_raises_instead_of_hiding_the_failure():
    worker = InferenceWorker(Boom(), RunMode.DETERMINISTIC, clock=VirtualClock(), stride=1)
    worker.start()

    with pytest.raises(RuntimeError):
        worker.submit(_frame(0))
