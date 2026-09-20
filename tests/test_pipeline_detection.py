import cv2
import numpy as np

from vectrax.clock import VirtualClock
from vectrax.detection.worker import InferenceWorker
from vectrax.metrics import RunMode
from vectrax.pipeline import build_file_pipeline
from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.geometry import Box
from vectrax.tracking.observation import Observation, Origin

W, H, SIDE, FPS, N = 320, 240, 40, 30, 12
START_X, Y, STEP_PX = 40, 100, 3
DET_BOX = Box(0.3, 0.4, 0.1, 0.1)


class StubDetector:
    def __init__(self):
        self.seen = []

    def detect(self, frame):
        self.seen.append(frame.frame_id)
        return [Observation(frame.frame_id, frame.capture_ns, DET_BOX, 0.8, Origin.DETECTOR, "cup")]


def _clip(tmp_path):
    rng = np.random.default_rng(1)
    background = rng.integers(60, 120, (H, W, 3), dtype=np.uint8)
    patch = cv2.normalize(cv2.GaussianBlur(rng.integers(0, 256, (SIDE, SIDE, 3), dtype=np.uint8), (0, 0), 3),
                          None, 0, 255, cv2.NORM_MINMAX)
    path = tmp_path / "moving.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        img = background.copy()
        x = START_X + STEP_PX * i
        img[Y:Y + SIDE, x:x + SIDE] = patch
        writer.write(img)

    writer.release()
    return path


def _run(tmp_path, stride):
    detector = StubDetector()
    worker = InferenceWorker(detector, RunMode.DETERMINISTIC, stride=stride, clock=VirtualClock())
    pipe = build_file_pipeline(_clip(tmp_path), TrackingConfig(), clock=VirtualClock(), worker=worker)
    pipe.select(Box.from_xywh_px(START_X, Y, SIDE, SIDE, W, H))
    ticks = []
    while (tick := pipe.tick()) is not None:
        ticks.append(tick)

    return detector, ticks


def test_tick_carries_the_detections(tmp_path):
    _, ticks = _run(tmp_path, stride=1)

    assert all(len(t.detections) == 1 for t in ticks)
    assert ticks[3].detections[0].origin is Origin.DETECTOR
    assert ticks[3].detections[0].frame_id == 3


def test_stride_leaves_the_frames_in_between_without_detections(tmp_path):
    detector, ticks = _run(tmp_path, stride=3)

    assert detector.seen == [0, 3, 6, 9]
    assert [t.frame.frame_id for t in ticks if t.detections] == [0, 3, 6, 9]


def test_pipeline_without_a_worker_reports_no_detections(tmp_path):
    pipe = build_file_pipeline(_clip(tmp_path), TrackingConfig(), clock=VirtualClock())
    pipe.select(Box.from_xywh_px(START_X, Y, SIDE, SIDE, W, H))

    tick = pipe.tick()

    assert tick.detections == []
