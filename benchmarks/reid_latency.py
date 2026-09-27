"""Latency of one DINOv2-S embedding call per backend and batch size.

  uv run benchmarks/reid_latency.py

Times the model call only: a batch of 224x224 crops already preprocessed. The first call and
WARMUP more are excluded. Core ML runs the export of benchmarks/export_reid.py.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "models" / "reid" / "dinov2-small"
EXPORTED = ROOT / "models" / "reid" / "exported" / "dinov2s_fp16.mlpackage"
OUT_DIR = Path(__file__).resolve().parent / "results" / "reid_latency"
SIDE = 224
BATCHES = (1, 2, 4, 8)
WARMUP = 10
RUNS = 200
COREML_UNITS = {"coreml-all": "ALL", "coreml-gpu": "CPU_AND_GPU", "coreml-ane": "CPU_AND_NE"}
BACKENDS = ["torch-cpu", "torch-mps", *COREML_UNITS]


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def _pmset(*args):
    return subprocess.run(["pmset", *args], capture_output=True, text=True, check=False).stdout


def _power_state():
    low_power = [ln.split()[-1] for ln in _pmset("-g").splitlines() if "lowpowermode" in ln]
    return {"pmset_ps": _pmset("-g", "ps").splitlines()[0], "lowpowermode": low_power[0] if low_power else None}


def _runner(backend):
    if backend in COREML_UNITS:
        model = ct.models.MLModel(str(EXPORTED), compute_units=ct.ComputeUnit[COREML_UNITS[backend]])
        return lambda x: model.predict({"pixels": x})["embedding"]

    device = backend.removeprefix("torch-")
    model = AutoModel.from_pretrained(SOURCE).eval().to(device)

    def run(x):
        with torch.no_grad():
            out = torch.nn.functional.normalize(model(pixel_values=torch.from_numpy(x).to(device)).pooler_output, dim=1)
            return out.cpu().numpy()  # waits for MPS

    return run


def _percentiles(ms):
    ordered = sorted(ms)

    def at(q):
        return ordered[min(int(q * len(ordered)), len(ordered) - 1)]

    return {"p50": at(0.5), "p95": at(0.95), "p99": at(0.99), "max": ordered[-1], "mean": statistics.fmean(ms)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backends", default=",".join(BACKENDS))
    args = p.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    for backend in args.backends.split(","):
        t0 = time.perf_counter()
        run = _runner(backend)
        first_x = rng.standard_normal((1, 3, SIDE, SIDE), dtype=np.float32)
        run(first_x)
        load_first_s = time.perf_counter() - t0
        for batch in BATCHES:
            x = rng.standard_normal((batch, 3, SIDE, SIDE), dtype=np.float32)
            for _ in range(WARMUP):
                run(x)

            ms = []
            for _ in range(RUNS):
                t = time.perf_counter()
                run(x)
                ms.append((time.perf_counter() - t) * 1000)

            rows.append({"backend": backend, "batch": batch, "load_first_s": load_first_s, **_percentiles(ms)})
            print(f"{backend:11s} batch {batch}: p50 {rows[-1]['p50']:6.2f} ms  p95 {rows[-1]['p95']:6.2f} ms  "
                  f"(load + first {load_first_s:.1f} s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"git_sha": _git("rev-parse", "--short", "HEAD"),
              "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
              "macos": platform.mac_ver()[0], "runs": RUNS, "warmup": WARMUP, **_power_state(), "rows": rows}
    out = OUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_dinov2s.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
