import json

import cv2
import numpy as np

from vectrax.__main__ import main
from vectrax.clock import VirtualClock
from vectrax.metrics import RunMode
from vectrax.pipeline import build_file_pipeline
from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.geometry import Box
from vectrax.tracking.state import Command, TrackState

W, H, SIDE, FPS, N = 320, 240, 40, 30, 40
START_X, Y, STEP_PX = 40, 100, 3


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


def _run(path, commands=None):
    pipe = build_file_pipeline(path, TrackingConfig(), clock=VirtualClock())
    tid = pipe.select(Box.from_xywh_px(START_X, Y, SIDE, SIDE, W, H))
    states = []
    while (tick := pipe.tick()) is not None:
        for at, cmd in (commands or {}).items():
            if tick.frame.frame_id == at:
                pipe.command(tid, cmd)

        states.append(next(s.state for s in tick.tracks if s.track_id == tid))

    return pipe, states


def test_tracks_synthetic_clip_to_tracking(tmp_path):
    pipe, states = _run(_clip(tmp_path))

    assert len(states) == N
    assert states[-1] is TrackState.TRACKING
    assert pipe.metrics.mode is RunMode.DETERMINISTIC
    assert pipe.metrics.summary()["frames"] == N


def test_operator_command_goes_through_pipeline(tmp_path):
    _, states = _run(_clip(tmp_path), commands={10: Command.PAUSE})

    assert states[-1] is TrackState.PAUSED


def test_headless_cli_is_reproducible(tmp_path):
    clip = _clip(tmp_path)
    box = f"{START_X},{Y},{SIDE},{SIDE}"
    outs = []
    for run in ("a", "b"):
        out = tmp_path / f"{run}.jsonl"
        main(["--source", f"file:{clip}", "--headless", "--init-boxes", box, "--out", str(out)])
        outs.append(out.read_text())

    assert outs[0] == outs[1]
    lines = outs[0].splitlines()
    assert len(lines) == N
    last = json.loads(lines[-1])
    assert last["frame_id"] == N - 1
    assert last["tracks"][0]["state"] == "tracking"
