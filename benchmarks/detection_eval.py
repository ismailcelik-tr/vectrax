"""Score a detector on all annotated fixtures (docs/FIXTURES.md).

  uv run benchmarks/detection_eval.py --detector dfine_n

Every frame of every fixture, PyTorch CPU, no resizing beyond the model's own
preprocessing. Speed lives in PERFORMANCE.md, not here.

Fixture GT labels only the target objects, so scoring is restricted to the one
class each fixture annotates; non_coco's "object" has no COCO equivalent and is
scored class-agnostically (recall is the number that means anything there).
"""

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from detectors import DETECTORS

from vectrax.evaluation.detection import evaluate_detections
from vectrax.evaluation.gt import load_mot
from vectrax.sources.file import FileSource

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "data" / "fixtures"
OUT_DIR = Path(__file__).resolve().parent / "results" / "detection"
NAMES = ["single_target", "crossing_targets", "near_targets", "occlusion", "exit_reentry",
         "fast_motion", "non_coco", "low_light"]
NON_COCO_LABEL = "object"  # CVAT label for the target with no COCO class
SCORE_MIN = 0.5  # precision/recall threshold; AP50 uses every score


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def _gt_label(gt):
    """The single class a fixture annotates, or None for class-agnostic scoring."""
    labels = {b.label for boxes in gt.tracks.values() for b in boxes.values()}
    if len(labels) != 1:
        raise ValueError(f"expected one GT label, got {sorted(labels)}")

    label = labels.pop()
    return None if label == NON_COCO_LABEL else label


def _run(detector, name, score_min):
    video = FIXTURES / f"{name}.mp4"
    frames = json.loads(video.with_suffix(".json").read_text())["frames"]
    gt = load_mot(video.with_suffix("").with_suffix(".gt.zip"), frames)

    dets = []
    with FileSource(video) as src:
        while (frame := src.read()) is not None:
            dets.extend(detector.detect(frame.image, frame.frame_id))

    label = _gt_label(gt)
    result = evaluate_detections(gt, dets, label, score_min)
    return {"fixture": name, "label": label or "any", "frames": frames,
            "detections": len(dets), **asdict(result)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", choices=sorted(DETECTORS), required=True)
    p.add_argument("--fixtures", default=",".join(NAMES))
    p.add_argument("--score-min", type=float, default=SCORE_MIN)
    args = p.parse_args()

    detector = DETECTORS[args.detector]()
    detector.load()

    rows = []
    for name in args.fixtures.split(","):
        rows.append(_run(detector, name, args.score_min))
        print(f"{rows[-1]['fixture']:17s} done")

    print(f"\n{'fixture':17s} {'class':>8s} {'GT':>5s} {'TP':>5s} {'FP':>6s} {'prec':>5s} {'rec':>5s} {'AP50':>5s}")
    for r in rows:
        print(f"{r['fixture']:17s} {r['label']:>8s} {r['gt_count']:5d} {r['tp']:5d} {r['fp']:6d} "
              f"{100 * r['precision']:4.0f}% {100 * r['recall']:4.0f}% {100 * r['ap50']:4.0f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"detector": args.detector, "backend": "pytorch-cpu", "score_min": args.score_min,
              "git_sha": _git("rev-parse", "--short", "HEAD"),
              "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
              "macos": platform.mac_ver()[0], "fixtures": rows}
    out = OUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.detector}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
