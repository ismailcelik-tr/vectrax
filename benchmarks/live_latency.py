"""Live latency, sensor PTS → render, for a fixed number of targets.

Fixed square boxes at fixed places, so runs compare across commits.
Keep the window untouched while it runs (resizing stalls OpenCV).

  uv run benchmarks/live_latency.py --targets 1,3
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from vectrax.metrics import Metrics, RunMode
from vectrax.pipeline import build_camera_pipeline
from vectrax.tracking.config import TrackingConfig
from vectrax.ui.opencv_view import run_ui

FRAMES = 900
WARMUP_FRAMES = 30
BOX_PX = 160
FRAME_W, FRAME_H = 1280, 720
OUT_DIR = Path(__file__).resolve().parent / "results" / "latency"
R3_P95_MS = 100


def _boxes(n):
    """n boxes spread evenly along the horizontal center line."""
    y = FRAME_H // 2 - BOX_PX // 2
    return [(round(FRAME_W * (i + 1) / (n + 1)) - BOX_PX // 2, y, BOX_PX, BOX_PX) for i in range(n)]


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()


def _env():
    return {
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "git_sha": _run(["git", "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
        "power": _run(["pmset", "-g", "batt"]).splitlines()[0],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="MacBook")
    p.add_argument("--targets", default="1,3")
    p.add_argument("--frames", type=int, default=FRAMES)
    args = p.parse_args()

    env = _env()
    if "AC Power" not in env["power"]:
        raise SystemExit(f"Plug in the adapter first ({env['power']})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows = []
    for n in (int(v) for v in args.targets.split(",")):
        pipe = build_camera_pipeline(args.device, TrackingConfig())
        pipe.metrics = Metrics(RunMode.REALTIME, warmup_frames=WARMUP_FRAMES)
        summary = run_ui(pipe, _boxes(n), max_frames=args.frames + WARMUP_FRAMES)
        report = {"targets": n, "box_px": BOX_PX, "frames": args.frames, "warmup": WARMUP_FRAMES,
                  "dropped": pipe.dropped, "env": env, **summary}
        (OUT_DIR / f"{stamp}_{n}targets.json").write_text(json.dumps(report, indent=2))
        rows.append(report)

    print(f"\n{'targets':>7} {'track p50/p95':>14} {'render p50/p95/max':>22} {'R3':>4} {'dropped':>8}")
    for r in rows:
        t, c = r["track_ms"], r["capture_to_render_ms"]
        ok = "ok" if c["p95"] <= R3_P95_MS else "MISS"
        print(f"{r['targets']:>7} {t['p50']:>6.1f}/{t['p95']:<6.1f} {c['p50']:>7.1f}/{c['p95']:.1f}/{c['max']:.1f} "
              f"{ok:>4} {r['dropped']:>8}")

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
