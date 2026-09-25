"""VectraX CLI.

  uv run vectrax --source camera:MacBook [--record [NAME]]
  uv run vectrax --source file:data/fixtures/single_target.mp4
  uv run vectrax --source file:clip.mp4 --headless --init-boxes 320,180,120,90 --out run.jsonl
  uv run vectrax --source file:data/sessions/NAME/video.mp4 --headless --render-out review.mp4
  uv run vectrax --source camera:MacBook --detect models/detectors/exported/rfdetr_n/rfdetr-nano_fp16.mlpackage

Headless operator input, first match wins: --init-boxes, the session's
operator.jsonl, the clip's <clip>.init.json.
"""

import argparse
import contextlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

from vectrax.detection.coreml import CoreMlDetector
from vectrax.detection.worker import STRIDE, InferenceWorker
from vectrax.metrics import RunMode
from vectrax.pipeline import Pipeline, build_camera_pipeline, build_file_pipeline
from vectrax.recording import OPERATOR_LOG, SessionRecorder
from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.geometry import Box
from vectrax.tracking.state import Command
from vectrax.ui.opencv_view import draw, load_init, run_ui

__all__ = ["main"]

FILE_PREFIX = "file:"
CAMERA_PREFIX = "camera:"
QUALITY_DECIMALS = 4
INIT_SUFFIX = ".init.json"
SESSIONS_DIR = Path("data/sessions")
RECORD_FPS = 30
REVIEW_FPS = 30


def _parse_boxes(text):
    return [tuple(float(v) for v in part.split(",")) for part in text.split(";")]


def _worker(args, mode) -> InferenceWorker | None:
    if not args.detect:
        return None

    return InferenceWorker(CoreMlDetector(Path(args.detect)), mode, stride=args.detect_stride)


def _build(args) -> Pipeline:
    cfg = TrackingConfig()
    if args.source.startswith(FILE_PREFIX):
        return build_file_pipeline(args.source[len(FILE_PREFIX):], cfg, scale=args.scale,
                                   worker=_worker(args, RunMode.DETERMINISTIC))

    if args.source.startswith(CAMERA_PREFIX):
        return build_camera_pipeline(args.source[len(CAMERA_PREFIX):], cfg, scale=args.scale,
                                     worker=_worker(args, RunMode.REALTIME))

    raise SystemExit(f"--source must start with {FILE_PREFIX} or {CAMERA_PREFIX}")


def _clip_path(args):
    if not args.source.startswith(FILE_PREFIX):
        return None

    return Path(args.source[len(FILE_PREFIX):])


def _init_path(args):
    clip = _clip_path(args)
    return clip.with_suffix(INIT_SUFFIX) if clip else None


def _init_boxes(args):
    if args.init_boxes:
        return _parse_boxes(args.init_boxes)

    path = _init_path(args)
    return load_init(path) if path else []


def _schedule(args, frame_size) -> dict[int, list[dict]]:
    """Operator ops keyed by the frame index they apply to."""
    ops = defaultdict(list)
    log = _clip_path(args).parent / OPERATOR_LOG
    if not args.init_boxes and log.exists():
        for line in log.read_text().splitlines():
            op = json.loads(line)
            ops[op["frame"]].append(op)

        return ops

    w, h = frame_size
    for x, y, bw, bh in _init_boxes(args):
        b = Box.from_xywh_px(x, y, bw, bh, w, h)
        ops[0].append({"op": "select", "box": [b.cx, b.cy, b.w, b.h]})

    return ops


def _apply(pipe, op):
    box = Box(*op["box"]) if op.get("box") else None
    if op["op"] == "select":
        track_id = pipe.select(box)
        expected = op.get("track_id", track_id)
        if track_id != expected:
            print(f"replay: select got id {track_id}, recording had {expected}", file=sys.stderr)
    elif op["op"] == "command":
        pipe.command(op["track_id"], Command(op["command"]), box)
    elif op["op"] == "remove":
        pipe.remove(op["track_id"])


def _record(tick, img_w, img_h):
    tracks = []
    for s in tick.tracks:
        q = s.quality.propagator_score
        tracks.append({
            "id": s.track_id,
            "state": s.state.value,
            "box": list(s.box.to_xywh_px(img_w, img_h)),
            "score": None if q is None else round(q, QUALITY_DECIMALS),
        })

    return {"frame_id": tick.frame.frame_id, "tracks": tracks}


def _headless(args):
    if _clip_path(args) is None:
        raise SystemExit("--headless needs a file source")

    pipe = _build(args)
    w, h = pipe.frame_size
    ops = _schedule(args, (w, h))
    review = None
    if args.render_out:
        review = cv2.VideoWriter(args.render_out, cv2.VideoWriter_fourcc(*"mp4v"), REVIEW_FPS, (w, h))

    sink = open(args.out, "w") if args.out else contextlib.nullcontext(sys.stdout)  # noqa: SIM115
    try:
        with sink as out:
            while (frame := pipe.read()) is not None:
                for op in ops.get(frame.frame_id, []):
                    _apply(pipe, op)

                tick = pipe.process(frame)
                out.write(json.dumps(_record(tick, w, h)) + "\n")
                if review is not None:
                    review.write(draw(frame.image, tick.tracks, {"frame": frame.frame_id},
                                      detections=tick.detections))
    finally:
        pipe.close()
        if review is not None:
            review.release()

    return pipe.metrics.summary()


def _interactive(args):
    pipe = _build(args)
    if args.record is not None:
        name = args.record or time.strftime("%Y%m%d-%H%M%S")
        recorder = SessionRecorder(SESSIONS_DIR / name, fps=RECORD_FPS)
        pipe.start_recording(recorder)
        print(f"Recording to {recorder.directory}", file=sys.stderr)

    return run_ui(pipe, _init_boxes(args), _init_path(args))


def main(argv=None):
    p = argparse.ArgumentParser(prog="vectrax")
    p.add_argument("--source", required=True, help="file:PATH or camera:NAME")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--init-boxes", help="x,y,w,h[;x,y,w,h] in pixels")
    p.add_argument("--out", help="headless: per-frame JSONL path")
    p.add_argument("--render-out", help="headless: annotated review video")
    p.add_argument("--record", nargs="?", const="", help=f"record session to {SESSIONS_DIR}/NAME")
    p.add_argument("--metrics-out", help="write latency summary JSON here")
    p.add_argument("--scale", type=float, default=1.0, help="propagator downscale (0,1]")
    p.add_argument("--detect", help="Core ML detector package; detections lift track state")
    p.add_argument("--detect-stride", type=int, default=STRIDE, help="detect every Nth frame")
    args = p.parse_args(argv)

    summary = _headless(args) if args.headless else _interactive(args)
    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
