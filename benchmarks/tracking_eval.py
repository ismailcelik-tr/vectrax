"""Score a propagator on all annotated fixtures (docs/FIXTURES.md).

  uv run benchmarks/tracking_eval.py --propagator csrt
  uv run benchmarks/tracking_eval.py --propagator nano_ncc --detect models/detectors/exported/rfdetr_n/rfdetr-nano_fp16.mlpackage

--one-at-a-time runs multi-target fixtures once per target with only that one selected, as an
operator tracking one of two look-alikes would; rows read <fixture>#<target>.
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from vectrax.detection.coreml import CoreMlDetector
from vectrax.detection.worker import InferenceWorker
from vectrax.evaluation.gt import load_mot
from vectrax.evaluation.metrics import Relock
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
        "relocks": sum(len(t.relocks) for t in tracks),
        "wrong_relocks": sum(t.wrong_relocks for t in tracks),
        "wrong_relock_frames": sorted([f, k.value] for t in tracks for f, k in t.relocks.items() if k is not Relock.OWN),
        **result.identity,
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
    p.add_argument("--one-at-a-time", action="store_true", help="select one target per run")
    args = p.parse_args()

    rows = []
    for name in args.fixtures.split(","):
        video = FIXTURES / f"{name}.mp4"
        targets = sorted(load_mot(video.with_suffix("").with_suffix(".gt.zip"), 0).tracks)
        for only in targets if args.one_at_a_time and len(targets) > 1 else [None]:
            worker = InferenceWorker(CoreMlDetector(Path(args.detect)), RunMode.DETERMINISTIC) if args.detect else None
            result = run_fixture(video, PROPAGATORS[args.propagator], worker=worker, only=only)
            rows.append(_summary(name if only is None else f"{name}#{only}", result))

    print(f"\n{'fixture':19s} {'success':>7s} {'onTarget':>8s} {'falseVis':>8s} {'hijack':>6s} {'relock':>6s} {'wrong':>5s} "
          f"{'IDSW':>4s} {'IDF1':>5s} {'HOTA':>5s} {'partial→deg':>11s} {'absent→hid':>10s} {'ms p50':>6s}  recoveries (frames)")
    for r in rows:
        print(f"{r['fixture']:19s} {_pct(r['success']):>7s} {_pct(r['on_target']):>8s} {r['false_visible']:8d} {r['hijack_frames']:6d} "
              f"{r['relocks']:6d} {r['wrong_relocks']:5d} {r['idsw']:4d} {_pct(r['idf1']):>5s} {_pct(r['hota']):>5s} "
              f"{_pct(r['partial_as_degraded']):>11s} {_pct(r['absent_as_hidden']):>10s} {r['track_ms_p50']:6.1f}  "
              f"{r['recoveries']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"propagator": args.propagator, "detector": args.detect, "one_at_a_time": args.one_at_a_time,
              "git_sha": _git("rev-parse", "--short", "HEAD"),
              "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")), "macos": platform.mac_ver()[0], "fixtures": rows}
    suffix = ("_detect" if args.detect else "") + ("_one" if args.one_at_a_time else "")
    out = OUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.propagator}{suffix}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
