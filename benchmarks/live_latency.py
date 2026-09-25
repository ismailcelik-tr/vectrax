"""Live latency, sensor PTS → render, for a fixed number of targets.

Fixed square boxes at fixed places, so runs compare across commits.
Keep the window untouched while it runs (resizing stalls OpenCV).

  uv run benchmarks/live_latency.py --targets 1,3
  uv run benchmarks/live_latency.py --targets 1,3 --detect models/detectors/exported/rfdetr_n/rfdetr-nano_fp16.mlpackage
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from vectrax.clock import MonotonicClock
from vectrax.detection.coreml import CoreMlDetector
from vectrax.detection.worker import STRIDE, InferenceWorker
from vectrax.metrics import LatencyStats, Metrics, RunMode
from vectrax.pipeline import build_camera_pipeline
from vectrax.tracking.config import TrackingConfig
from vectrax.ui.opencv_view import run_ui

FRAMES = 900
WARMUP_FRAMES = 30
BOX_PX = 160
FRAME_W, FRAME_H = 1280, 720
OUT_DIR = Path(__file__).resolve().parent / "results" / "latency"
R3A_P95_MS = 100  # sensor → guidance (≈ tracked until Phase 5)
R3B_P95_MS = 120  # sensor → render


def _boxes(n):
    """n boxes spread evenly along the horizontal center line."""
    y = FRAME_H // 2 - BOX_PX // 2
    return [(round(FRAME_W * (i + 1) / (n + 1)) - BOX_PX // 2, y, BOX_PX, BOX_PX) for i in range(n)]


class _WorkerProbe:
    """Wraps InferenceWorker for Pipeline. Measures what detection costs the
    tick thread (`submit`) and how old a result is when the tick fuses it.
    Pipeline.process calls `results()` then `submit(frame)` with the frame it
    fused them on, so ages are taken in `submit`."""

    def __init__(self, worker: InferenceWorker, warmup_frames: int):
        self._worker = worker
        self._warmup = warmup_frames
        self._clock = MonotonicClock()
        self._fused = []
        self._submit = LatencyStats()
        self._age_ms = LatencyStats()
        self._inference = LatencyStats()
        self._age_frames = []

    def start(self):
        self._worker.start()

    def stop(self):
        self._worker.stop()

    def results(self):
        self._fused = self._worker.results()
        return self._fused

    def submit(self, frame):
        started = self._clock.now_ns()
        self._worker.submit(frame)
        elapsed = self._clock.now_ns() - started
        if self._warmup > 0:
            self._warmup -= 1
            return

        self._submit.add(elapsed)
        for r in self._fused:
            self._age_frames.append(frame.frame_id - r.frame_id)
            self._age_ms.add(frame.capture_ns - r.capture_ns)
            self._inference.add(r.inference_ns)

    def summary(self):
        ages = np.asarray(self._age_frames)
        frames = None
        if ages.size:
            frames = {"n": int(ages.size), "p50": float(np.percentile(ages, 50)),
                      "p95": float(np.percentile(ages, 95)), "max": int(ages.max())}

        return {"submit_ms": self._submit.summary(), "age_frames": frames, "age_ms": self._age_ms.summary(),
                "inference_ms": self._inference.summary(), "processed": self._worker.processed,
                "dropped": self._worker.dropped, "failures": self._worker.failures}


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()


def _env():
    return {
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "git_sha": _run(["git", "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain", "--untracked-files=no"])),
        "power": _run(["pmset", "-g", "batt"]).splitlines()[0],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="MacBook")
    p.add_argument("--targets", default="1,3")
    p.add_argument("--frames", type=int, default=FRAMES)
    p.add_argument("--detect", help="Core ML detector package; detection runs alongside tracking")
    p.add_argument("--detect-stride", type=int, default=STRIDE)
    args = p.parse_args()

    env = _env()
    if "AC Power" not in env["power"]:
        raise SystemExit(f"Plug in the adapter first ({env['power']})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = "_detect" if args.detect else ""
    rows = []
    for n in (int(v) for v in args.targets.split(",")):
        probe = None
        if args.detect:
            worker = InferenceWorker(CoreMlDetector(Path(args.detect)), RunMode.REALTIME, stride=args.detect_stride)
            probe = _WorkerProbe(worker, WARMUP_FRAMES)

        # Loads and warms the detector before the first frame is timed.
        pipe = build_camera_pipeline(args.device, TrackingConfig(), worker=probe)
        pipe.metrics = Metrics(RunMode.REALTIME, warmup_frames=WARMUP_FRAMES)
        summary = run_ui(pipe, _boxes(n), max_frames=args.frames + WARMUP_FRAMES)
        report = {"targets": n, "box_px": BOX_PX, "frames": args.frames, "warmup": WARMUP_FRAMES,
                  "dropped": pipe.dropped, "env": env, **summary}
        if probe is not None:
            report["detection"] = {"detector": args.detect, "stride": args.detect_stride, **probe.summary()}

        (OUT_DIR / f"{stamp}_{n}targets{suffix}.json").write_text(json.dumps(report, indent=2))
        rows.append(report)

    print(f"\n{'targets':>7} {'tracked p50/p95':>16} {'R3a':>4} {'render p50/p95':>15} {'R3b':>4} {'dropped':>8}"
          f" {'det age p50/p95/max':>20}")
    for r in rows:
        t, c = r["capture_to_tracked_ms"], r["capture_to_render_ms"]
        a = "ok" if t["p95"] <= R3A_P95_MS else "MISS"
        b = "ok" if c["p95"] <= R3B_P95_MS else "MISS"
        age = (r.get("detection") or {}).get("age_frames")
        age_col = f"{age['p50']:g}/{age['p95']:g}/{age['max']}" if age else "-"
        print(f"{r['targets']:>7} {t['p50']:>7.1f}/{t['p95']:<7.1f} {a:>4} {c['p50']:>6.1f}/{c['p95']:<7.1f} {b:>4} "
              f"{r['dropped']:>8} {age_col:>20}")

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
