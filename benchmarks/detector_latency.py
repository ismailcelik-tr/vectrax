"""Latency, memory and power of one detector on one backend.

  uv run benchmarks/detector_latency.py --detector rfdetr_n --backend pytorch-cpu
  uv run benchmarks/detector_latency.py --detector rfdetr_n --backend pytorch-mps --sustained-s 300

Load and the first inference are timed apart from the steady-state loop: the
first call pays lazy model build and graph compile. Frames are decoded before
the loop, so only inference is timed. macmon samples CPU, GPU, ANE and package
power while the loop runs; per-minute percentiles expose thermal drift.
"""

import argparse
import json
import platform
import resource
import statistics
import subprocess
import threading
import time
from pathlib import Path

import cv2
from detectors import BACKENDS, DETECTORS, build

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "data" / "fixtures"
OUT_DIR = Path(__file__).resolve().parent / "results" / "detector_latency"
FIXTURE = "single_target"
FRAMES = 120  # decoded once, then cycled
WARMUP = 10
RUNS = 300
MINUTE_S = 60.0
POWER_FIELDS = ["cpu_usage_pct", "ecpu_active_ratio", "pcpu_active_ratio", "gpu_active_ratio",
                "ane_power", "gpu_power", "sys_power"]
BYTES_PER_MB = 1024 * 1024


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def _pmset(*args):
    return subprocess.run(["pmset", *args], capture_output=True, text=True, check=False).stdout


def _power_state():
    low_power = [ln.split()[-1] for ln in _pmset("-g").splitlines() if "lowpowermode" in ln]
    return {"pmset_ps": _pmset("-g", "ps").splitlines()[0],
            "lowpowermode": low_power[0] if low_power else None}


class PowerMonitor:
    """macmon samples collected in the background for the duration of a `with` block."""

    def __init__(self, interval_ms: int = 1000):
        self._interval_ms = interval_ms
        self._samples = []
        self._proc = None

    def __enter__(self):
        self._proc = subprocess.Popen(
            ["macmon", "pipe", "-s", "0", "-i", str(self._interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._proc.terminate()
        self._thread.join(timeout=2)

    def summary(self) -> dict:
        if not self._samples:
            return {}

        out = {"samples": len(self._samples)}
        for field in POWER_FIELDS:
            values = [s[field] for s in self._samples if field in s]
            out[f"{field}_mean"] = round(statistics.fmean(values), 3)
            out[f"{field}_max"] = round(max(values), 3)

        temps = [s["temp"]["cpu_temp_avg"] for s in self._samples]
        out["cpu_temp_c_first"], out["cpu_temp_c_last"] = round(temps[0], 1), round(temps[-1], 1)
        return out

    def _collect(self):
        for line in self._proc.stdout:
            try:
                self._samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue


def _frames(fixture, count):
    cap = cv2.VideoCapture(str(FIXTURES / f"{fixture}.mp4"))
    images = []
    while len(images) < count and (ok := cap.read())[0]:
        images.append(ok[1])

    cap.release()
    return images


def _percentiles(latencies_ms):
    ordered = sorted(latencies_ms)
    return {"n": len(ordered),
            "p50": round(statistics.median(ordered), 2),
            "p95": round(ordered[min(len(ordered) - 1, round(0.95 * len(ordered)))], 2),
            "max": round(ordered[-1], 2)}


def _loop(detector, images, runs, duration_s):
    """Returns (elapsed_s, latency_ms) pairs measured from the loop start."""
    started = time.perf_counter()
    marks = []
    i = 0
    while True:
        t0 = time.perf_counter()
        detector.detect(images[i % len(images)], i)
        t1 = time.perf_counter()
        marks.append((t0 - started, 1000 * (t1 - t0)))
        i += 1

        if duration_s is not None and t1 - started >= duration_s:
            return marks

        if duration_s is None and i >= runs:
            return marks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", choices=sorted(DETECTORS), required=True)
    p.add_argument("--backend", choices=BACKENDS, required=True)
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp16",
                   help="Core ML bundle precision")
    p.add_argument("--fixture", default=FIXTURE)
    p.add_argument("--frames", type=int, default=FRAMES)
    p.add_argument("--runs", type=int, default=RUNS)
    p.add_argument("--sustained-s", type=float, default=None,
                   help="run for this many seconds instead of --runs")
    args = p.parse_args()

    images = _frames(args.fixture, args.frames)
    detector = build(args.detector, args.backend, args.precision)

    t0 = time.perf_counter()
    detector.load()
    load_ms = 1000 * (time.perf_counter() - t0)

    t0 = time.perf_counter()
    detections = detector.detect(images[0], 0)
    first_ms = 1000 * (time.perf_counter() - t0)

    for i in range(WARMUP):
        detector.detect(images[i % len(images)], i)

    with PowerMonitor() as monitor:
        marks = _loop(detector, images, args.runs, args.sustained_s)

    latencies = [ms for _, ms in marks]
    minutes = []
    for start in range(int(marks[-1][0] // MINUTE_S) + 1):
        window = [ms for at, ms in marks if start * MINUTE_S <= at < (start + 1) * MINUTE_S]
        if window:
            minutes.append(_percentiles(window))

    report = {
        "detector": args.detector, "backend": args.backend, "precision": args.precision,
        "fixture": args.fixture,
        "load_ms": round(load_ms, 1), "first_inference_ms": round(first_ms, 1),
        "first_inference_detections": len(detections),
        "inference_ms": _percentiles(latencies),
        "elapsed_s": round(marks[-1][0], 1),
        "rss_peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / BYTES_PER_MB, 1),
        "per_minute": minutes if args.sustained_s else None,
        "power": monitor.summary(), **_power_state(),
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "macos": platform.mac_ver()[0],
    }

    print(f"{args.detector} {args.backend}: load {report['load_ms']:.0f} ms, first "
          f"{report['first_inference_ms']:.0f} ms, p50 {report['inference_ms']['p50']:.1f} ms, "
          f"p95 {report['inference_ms']['p95']:.1f} ms, RSS {report['rss_peak_mb']:.0f} MB")
    if report["power"]:
        pw = report["power"]
        print(f"  cpu {pw['cpu_usage_pct_mean']:.2f} gpu {pw['gpu_active_ratio_mean']:.2f} "
              f"ane {pw['ane_power_mean']:.2f} W sys {pw['sys_power_mean']:.1f} W "
              f"temp {pw['cpu_temp_c_first']:.0f}→{pw['cpu_temp_c_last']:.0f} °C")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.detector}_{args.backend}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
