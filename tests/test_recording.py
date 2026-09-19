import json

import cv2
import numpy as np

from vectrax.__main__ import main
from vectrax.clock import VirtualClock
from vectrax.frames import FramePacket, PixelFormat
from vectrax.pipeline import build_file_pipeline
from vectrax.recording import SessionRecorder
from vectrax.sources.file import FileSource
from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.geometry import Box
from vectrax.tracking.state import Command

W, H, SIDE, FPS, N = 320, 240, 40, 30, 30
START_X, Y, STEP_PX = 40, 100, 3
PAUSE_AT = 10


def _clip(tmp_path):
    rng = np.random.default_rng(2)
    background = rng.integers(60, 120, (H, W, 3), dtype=np.uint8)
    patch = cv2.normalize(cv2.GaussianBlur(rng.integers(0, 256, (SIDE, SIDE, 3), dtype=np.uint8), (0, 0), 3),
                          None, 0, 255, cv2.NORM_MINMAX)
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        img = background.copy()
        img[Y:Y + SIDE, START_X + STEP_PX * i:START_X + STEP_PX * i + SIDE] = patch
        writer.write(img)

    writer.release()
    return path


def test_recorder_round_trips_through_file_source(tmp_path):
    rec = SessionRecorder(tmp_path / "s", fps=FPS)
    for i in range(5):
        img = np.full((H, W, 3), i * 40, np.uint8)
        rec.write(FramePacket(i * 3, "cam", 1_000 + i, 2_000 + i, img, PixelFormat.BGR))

    rec.close()

    with FileSource(tmp_path / "s" / "video.mp4") as src:
        packets = []
        while (p := src.read()) is not None:
            packets.append(p)

    assert [p.frame_id for p in packets] == [0, 1, 2, 3, 4]
    assert [p.capture_ns for p in packets] == [1_000 + i for i in range(5)]
    assert rec.dropped == 0


def _states(path, operator):
    """Run a pipeline, calling operator(pipe, frame_index) before each frame."""
    pipe = build_file_pipeline(path, TrackingConfig(), clock=VirtualClock())
    states = []
    while (frame := pipe.read()) is not None:
        operator(pipe, frame.frame_id)
        tick = pipe.process(frame)
        states.append([(s.track_id, s.state.value) for s in tick.tracks])

    pipe.close()
    return pipe, states


def test_session_replay_matches_original_run(tmp_path):
    clip = _clip(tmp_path)
    session = tmp_path / "session"
    tid = {}

    def operator(pipe, i):
        if i == 0:
            pipe.start_recording(SessionRecorder(session, fps=FPS))
            tid["a"] = pipe.select(Box.from_xywh_px(START_X, Y, SIDE, SIDE, W, H))
        if i == PAUSE_AT:
            pipe.command(tid["a"], Command.PAUSE)

    _, original = _states(clip, operator)

    ops = [json.loads(line) for line in (session / "operator.jsonl").read_text().splitlines()]
    assert [(o["frame"], o["op"]) for o in ops] == [(0, "select"), (PAUSE_AT, "command")]

    out = tmp_path / "replay.jsonl"
    main(["--source", f"file:{session / 'video.mp4'}", "--headless", "--out", str(out)])
    replayed = [[(t["id"], t["state"]) for t in json.loads(line)["tracks"]] for line in out.read_text().splitlines()]

    assert replayed == original


def test_render_out_writes_annotated_video(tmp_path):
    clip = _clip(tmp_path)
    render = tmp_path / "review.mp4"

    main(["--source", f"file:{clip}", "--headless", "--init-boxes", f"{START_X},{Y},{SIDE},{SIDE}",
          "--out", str(tmp_path / "r.jsonl"), "--render-out", str(render)])

    cap = cv2.VideoCapture(str(render))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N
