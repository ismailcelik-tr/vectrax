"""VectraX CLI.

  uv run vectrax --source camera:MacBook
  uv run vectrax --source file:data/fixtures/single_target.mp4
  uv run vectrax --source file:clip.mp4 --headless --init-boxes 320,180,120,90 --out run.jsonl
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

from vectrax.pipeline import Pipeline, build_camera_pipeline, build_file_pipeline
from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.geometry import Box

__all__ = ["main"]

FILE_PREFIX = "file:"
CAMERA_PREFIX = "camera:"
QUALITY_DECIMALS = 4


def _parse_boxes(text):
    if not text:
        return []

    return [tuple(float(v) for v in part.split(",")) for part in text.split(";")]


def _build(args, renders) -> Pipeline:
    cfg = TrackingConfig()
    if args.source.startswith(FILE_PREFIX):
        return build_file_pipeline(args.source[len(FILE_PREFIX):], cfg, renders=renders, scale=args.scale)

    if args.source.startswith(CAMERA_PREFIX):
        return build_camera_pipeline(args.source[len(CAMERA_PREFIX):], cfg, renders=renders, scale=args.scale)

    raise SystemExit(f"--source must start with {FILE_PREFIX} or {CAMERA_PREFIX}")


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
    if not args.source.startswith(FILE_PREFIX):
        raise SystemExit("--headless needs a file source")

    pipe = _build(args, renders=False)
    w, h = pipe.frame_size
    for x, y, bw, bh in _parse_boxes(args.init_boxes):
        pipe.select(Box.from_xywh_px(x, y, bw, bh, w, h))

    sink = open(args.out, "w") if args.out else contextlib.nullcontext(sys.stdout)  # noqa: SIM115
    try:
        with sink as out:
            while (tick := pipe.tick()) is not None:
                out.write(json.dumps(_record(tick, w, h)) + "\n")
    finally:
        pipe.close()

    return pipe.metrics.summary()


def main(argv=None):
    p = argparse.ArgumentParser(prog="vectrax")
    p.add_argument("--source", required=True, help="file:PATH or camera:NAME")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--init-boxes", help="x,y,w,h[;x,y,w,h] in pixels")
    p.add_argument("--out", help="headless: per-frame JSONL path")
    p.add_argument("--metrics-out", help="write latency summary JSON here")
    p.add_argument("--scale", type=float, default=1.0, help="propagator downscale (0,1]")
    args = p.parse_args(argv)

    if args.headless:
        summary = _headless(args)
    else:
        from vectrax.ui.opencv_view import run_ui

        summary = run_ui(_build(args, renders=True), _parse_boxes(args.init_boxes))

    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
