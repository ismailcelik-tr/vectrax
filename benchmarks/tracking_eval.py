"""Score a propagator on all annotated fixtures (docs/FIXTURES.md).

  uv run benchmarks/tracking_eval.py --propagator csrt
  uv run benchmarks/tracking_eval.py --propagator nano_ncc --detect models/detectors/exported/rfdetr_n/rfdetr-nano_fp16.mlpackage
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from vectrax.detection.coreml import CoreMlDetector
from vectrax.detection.worker import InferenceWorker
from vectrax.evaluation.runner import run_fixture
from vectrax.metrics import RunMode
from vectrax.tracking.propagators import (
    CsrtPropagator,
    KcfPropagator,
    NanoPropagator,
    ScoreSource,
    VitPropagator,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "data" / "fixtures"
OUT_DIR = Path(__file__).resolve().parent / "results" / "tracking"
NAMES = ["single_target", "crossing_targets", "near_targets", "occlusion", "exit_reentry",
         "fast_motion", "non_coco", "low_light"]
PROPAGATORS = {
    "csrt": CsrtPropagator,
    "kcf": KcfPropagator,
    "vit": VitPropagator,
    "vit_ncc": lambda: VitPropagator(score=ScoreSource.NCC),
    "nano": NanoPropagator,
    "nano_ncc": lambda: NanoPropagator(score=ScoreSource.NCC),
}
HIDDEN = ("occluded", "lost")


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def _summary(name, result):
    tracks = result.tracks.values()
    present = sum(t.present_frames for t in tracks)
    partial = sum(n for t in tracks for (g, _), n in t.states.items() if g == "partial")
    partial_deg = sum(n for t in tracks for (g, p), n in t.states.items() if g == "partial" and p == "degraded")
    absent = sum(n for t in tracks for (g, _), n in t.states.items() if g == "absent")
    absent_hidden = sum(n for t in tracks for (g, p), n in t.states.items() if g == "absent" and p in HIDDEN)
    return {
        "fixture": name,
        "success": sum(t.successes for t in tracks) / present if present else 0.0,
        "on_target": sum(t.on_target for t in tracks) / present if present else 0.0,
        "false_visible": sum(t.false_visible for t in tracks),
        "hijack_frames": sum(t.hijack_frames for t in tracks),
        "recoveries": [r for t in tracks for r in t.recoveries],
        "partial_as_degraded": partial_deg / partial if partial else None,
        "absent_as_hidden": absent_hidden / absent if absent else None,
        "track_ms_p50": result.track_ms["p50"] if result.track_ms else None,
        "states": {f"{g}->{p}": n for t in tracks for (g, p), n in sorted(t.states.items())},
    }


def _pct(v):
    return "   -" if v is None else f"{100 * v:3.0f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--propagator", choices=sorted(PROPAGATORS), default="csrt")
    p.add_argument("--fixtures", default=",".join(NAMES))
    p.add_argument("--detect", help="Core ML detector package; fused into track state")
    args = p.parse_args()

    rows = []
    for name in args.fixtures.split(","):
        worker = InferenceWorker(CoreMlDetector(Path(args.detect)), RunMode.DETERMINISTIC) if args.detect else None
        rows.append(_summary(name, run_fixture(FIXTURES / f"{name}.mp4", PROPAGATORS[args.propagator], worker=worker)))

    print(f"\n{'fixture':17s} {'success':>7s} {'onTarget':>8s} {'falseVis':>8s} {'hijack':>6s} {'partial→deg':>11s} "
          f"{'absent→hid':>10s} {'ms p50':>6s}  recoveries (frames)")
    for r in rows:
        print(f"{r['fixture']:17s} {_pct(r['success']):>7s} {_pct(r['on_target']):>8s} {r['false_visible']:8d} {r['hijack_frames']:6d} "
              f"{_pct(r['partial_as_degraded']):>11s} {_pct(r['absent_as_hidden']):>10s} {r['track_ms_p50']:6.1f}  "
              f"{r['recoveries']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"propagator": args.propagator, "detector": args.detect, "git_sha": _git("rev-parse", "--short", "HEAD"),
              "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")), "macos": platform.mac_ver()[0], "fixtures": rows}
    suffix = "_detect" if args.detect else ""
    out = OUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.propagator}{suffix}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
