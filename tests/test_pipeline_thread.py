import cv2
import numpy as np

from vectrax.buffer import LatestFrameBuffer
from vectrax.clock import VirtualClock
from vectrax.pipeline import PipelineThread, build_file_pipeline
from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.geometry import Box

W, H, N, FPS = 160, 120, 25, 30


def _clip(tmp_path):
    path = tmp_path / "c.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(4)
    for _ in range(N):
        writer.write(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))

    writer.release()
    return path


def test_runs_every_frame_and_hands_latest_tick(tmp_path):
    pipe = build_file_pipeline(_clip(tmp_path), TrackingConfig(), clock=VirtualClock())
    pipe.select(Box(0.5, 0.5, 0.2, 0.2))
    ticks = LatestFrameBuffer()

    runner = PipelineThread(pipe, ticks, stop_at_end=True)
    runner.start()
    runner.join(timeout=10)

    assert not runner.is_alive()
    assert runner.error is None
    assert pipe.metrics.summary()["frames"] == N
    assert ticks.get(timeout_s=0).frame.frame_id == N - 1


def test_max_frames_stops_runner(tmp_path):
    pipe = build_file_pipeline(_clip(tmp_path), TrackingConfig(), clock=VirtualClock())
    runner = PipelineThread(pipe, LatestFrameBuffer(), stop_at_end=True, max_frames=5)
    runner.start()
    runner.join(timeout=10)

    assert pipe.metrics.summary()["frames"] == 5


def test_stop_ends_runner(tmp_path):
    pipe = build_file_pipeline(_clip(tmp_path), TrackingConfig(), clock=VirtualClock())
    runner = PipelineThread(pipe, LatestFrameBuffer(), stop_at_end=False)
    runner.start()
    runner.stop()
    runner.join(timeout=10)

    assert not runner.is_alive()
