"""Which appearance score tells a reacquisition candidate on the target from one elsewhere.

  uv run benchmarks/reid_eval.py --detect models/detectors/exported/rfdetr_n/rfdetr-nano_fp16.mlpackage

Fixtures run through the deterministic pipeline with reacquisition watching but never acting, so
every REACQUIRING frame's candidates are kept. GT labels each candidate: own (center on its
target), other (on another target), bg (target visible elsewhere), absent (target gone). Each
scorer compares a candidate with the look at selection, and with a gallery: the selection plus
earlier TRACKING frames at good quality. Reports ROC AUC (own vs the rest) and own recall at the
strictest threshold no other candidate reaches. Candidates come from the NCC search and the
propagator, so a scorer can only re-rank them; proposal recall says how often the target was
among them at all. --one-at-a-time runs multi-target fixtures once per target with only that one
selected, so the others are untracked look-alikes rather than objects a track holds.
"""

import argparse
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel

from vectrax.detection.coreml import CoreMlDetector
from vectrax.detection.worker import InferenceWorker
from vectrax.evaluation.gt import load_mot
from vectrax.metrics import RunMode
from vectrax.pipeline import build_file_pipeline
from vectrax.tracking.appearance import patch
from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.geometry import Box
from vectrax.tracking.propagators import NanoPropagator, ScoreSource
from vectrax.tracking.state import TrackState

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "data" / "fixtures"
DINO_DIR = ROOT / "models" / "reid" / "dinov2-small"
OUT_DIR = Path(__file__).resolve().parent / "results" / "reid"
NAMES = ["single_target", "crossing_targets", "near_targets", "occlusion", "exit_reentry",
         "fast_motion", "non_coco", "low_light"]
LABELS = ("own", "other", "bg", "absent")
GALLERY_EVERY = 5  # good TRACKING frames per gallery sample
CROP_PX = 112  # stored crop side; upscaled for DINOv2
DINO_PX = 224
DINO_BATCH = 64
# ImageNet statistics, as in the model's preprocessor_config.json.
DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DINO_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
HIST_BINS = [16, 16]  # hue, saturation; value left out for lighting
HIST_RANGES = [0, 180, 0, 256]


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def _crop(img, box: Box):
    h, w = img.shape[:2]
    x, y, bw, bh = box.to_xywh_px(w, h)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    return cv2.resize(img[y0:y1, x0:x1], (CROP_PX, CROP_PX), interpolation=cv2.INTER_AREA)


def _label(box: Box, gid, gt, f, w, h):
    cx, cy = box.cx * w, box.cy * h

    def on(g):
        return g[0] <= cx <= g[0] + g[2] and g[1] <= cy <= g[1] + g[3]

    own = gt.tracks[gid].get(f)
    if own is not None and on(own.box):
        return "own"

    if any(f in boxes and on(boxes[f].box) for o, boxes in gt.tracks.items() if o != gid):
        return "other"

    return "absent" if own is None else "bg"


def _collect(name, cfg, detect, only=None):
    video = FIXTURES / f"{name}.mp4"
    gt = load_mot(video.with_suffix("").with_suffix(".gt.zip"), json.loads(video.with_suffix(".json").read_text())["frames"])
    starts = {gid: min(boxes) for gid, boxes in gt.tracks.items() if only in (None, gid)}
    worker = InferenceWorker(CoreMlDetector(detect), RunMode.DETERMINISTIC) if detect else None
    pipe = build_file_pipeline(video, cfg, propagator_factory=lambda: NanoPropagator(score=ScoreSource.NCC),
                               worker=worker)
    w, h = pipe.frame_size
    track_of, good = {}, defaultdict(int)
    refs = {gid: [] for gid in starts}  # crops, selection first
    cands, frames = [], []
    try:
        while (frame := pipe.read()) is not None:
            f = frame.frame_id
            for gid, first in starts.items():
                if f == first:
                    box = Box.from_xywh_px(*gt.tracks[gid][first].box, w, h)
                    track_of[gid] = pipe.select(box)
                    refs[gid].append(_crop(frame.image, box))  # GT box at selection: inside the image

            snaps = {s.track_id: s for s in pipe.process(frame).tracks}
            for gid, tid in track_of.items():
                s = snaps.get(tid)
                if s is None:
                    continue

                q = s.quality.combined(cfg)
                if s.state is TrackState.TRACKING and q is not None and q >= cfg.good_quality:
                    good[gid] += 1
                    if good[gid] % GALLERY_EVERY == 0 and (c := _crop(frame.image, s.box)) is not None:
                        refs[gid].append(c)

                if s.state is not TrackState.REACQUIRING:
                    continue

                labels = []
                for c in s.candidates:
                    crop = _crop(frame.image, c.box)
                    if crop is None:
                        continue

                    labels.append(_label(c.box, gid, gt, f, w, h))
                    cands.append({"fixture": name, "gid": gid, "frame": f, "origin": c.origin.value,
                                  "box": [round(v, 4) for v in (c.box.cx, c.box.cy, c.box.w, c.box.h)],
                                  "label": labels[-1], "crop": crop, "refs": len(refs[gid])})

                frames.append({"fixture": name, "gid": gid, "frame": f, "visible": f in gt.tracks[gid],
                               "proposed": "own" in labels})
    finally:
        pipe.close()

    return refs, cands, frames


def _ncc(crop):
    return patch(crop, (0, 0, CROP_PX, CROP_PX))


def _ncc_sim(a, b):
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])


def _hist(crop):
    hist = cv2.calcHist([cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)], [0, 1], None, HIST_BINS, HIST_RANGES)
    return cv2.normalize(hist, None, 1, 0, cv2.NORM_L1)


def _hist_sim(a, b):
    return 1.0 - float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def _dino_all(crops):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained(DINO_DIR).eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), DINO_BATCH):
            rgb = [cv2.resize(c, (DINO_PX, DINO_PX), interpolation=cv2.INTER_CUBIC)[:, :, ::-1] for c in crops[i:i + DINO_BATCH]]
            x = (np.stack(rgb).astype(np.float32) / 255.0 - DINO_MEAN) / DINO_STD
            emb = model(pixel_values=torch.from_numpy(x).permute(0, 3, 1, 2).to(device)).pooler_output
            out.extend(torch.nn.functional.normalize(emb, dim=1).cpu().numpy())

    return out


def _dino_sim(a, b):
    return float(a @ b)


def _auc(pos, neg):
    """P(a random own candidate outscores a random other one); ties count half."""
    if not pos or not neg:
        return None

    neg = np.sort(np.asarray(neg))
    pos = np.asarray(pos)
    below = np.searchsorted(neg, pos, side="left")
    ties = np.searchsorted(neg, pos, side="right") - below
    return float((below + ties / 2).sum() / (len(pos) * len(neg)))


def _recall_clean(pos, neg):
    """Own recall at the strictest threshold no other candidate reaches."""
    if not pos:
        return None

    top = max(neg) if neg else -math.inf
    return float(np.mean(np.asarray(pos) > top))


def _spells(frames):
    """Runs of consecutive REACQUIRING frames per track."""
    runs = []
    for f in frames:
        last = runs[-1][-1] if runs else None
        if last and (last["fixture"], last["gid"]) == (f["fixture"], f["gid"]) and f["frame"] == last["frame"] + 1:
            runs[-1].append(f)
        else:
            runs.append([f])

    return runs


def _first_pick(spell, by_frame, cands, vals, cfg):
    """The manager's rule: best object candidate with good_quality and reacquire_margin over the next."""
    for f in spell:
        objects = []
        for i in sorted(by_frame.get((f["fixture"], f["gid"], f["frame"]), []), key=lambda i: -vals[i]):
            if not any(_same(cands[i]["box"], cands[j]["box"]) for j in objects):
                objects.append(i)

        if not objects:
            continue

        runner_up = vals[objects[1]] if len(objects) > 1 else 0.0
        if vals[objects[0]] >= cfg.good_quality and vals[objects[0]] - runner_up >= cfg.reacquire_margin:
            return f["frame"], cands[objects[0]]["label"]

    return None


def _same(a, b):
    """Each center lies inside the other box (TrackManager's rule)."""
    return abs(a[0] - b[0]) <= min(a[2], b[2]) / 2 and abs(a[1] - b[1]) <= min(a[3], b[3]) / 2


def _fmt(v):
    return "   -" if v is None else f"{v:.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", default=",".join(NAMES))
    p.add_argument("--detect", help="Core ML detector package, as in the live pipeline")
    p.add_argument("--one-at-a-time", action="store_true", help="select one target per run")
    args = p.parse_args()

    # Watch candidates, never act on them: no candidate leads anything by an infinite margin.
    cfg = replace(TrackingConfig(), reacquire_margin=math.inf)
    refs, cands, frames = {}, [], []
    detect = Path(args.detect) if args.detect else None
    for name in args.fixtures.split(","):
        runs = [None]
        if args.one_at_a_time:
            video = FIXTURES / f"{name}.mp4"
            runs = sorted(load_mot(video.with_suffix("").with_suffix(".gt.zip"), 0).tracks)

        for only in runs:
            label = name if only is None or len(runs) == 1 else f"{name}#{only}"
            r, c, f = _collect(name, cfg, detect, only)
            refs.update({(label, gid): crops for gid, crops in r.items()})
            cands += [{**x, "fixture": label} for x in c]
            frames += [{**x, "fixture": label} for x in f]

    order = [(k, i) for k, crops in refs.items() for i in range(len(crops))]
    ref_crops = [refs[k][i] for k, i in order]
    all_crops = [c["crop"] for c in cands] + ref_crops
    dino = _dino_all(all_crops)
    features = {
        "ncc": [_ncc(c) for c in all_crops],
        "hist": [_hist(c) for c in all_crops],
        "dino": dino,
    }
    sims = {"ncc": _ncc_sim, "hist": _hist_sim, "dino": _dino_sim}
    ref_at = {k_i: len(cands) + n for n, k_i in enumerate(order)}

    scores = {}
    for scorer, feat in features.items():
        for mode in ("selection", "gallery"):
            key = f"{scorer}/{mode}"
            scores[key] = []
            for n, c in enumerate(cands):
                k = (c["fixture"], c["gid"])
                count = 1 if mode == "selection" else c["refs"]
                scores[key].append(max(sims[scorer](feat[n], feat[ref_at[(k, i)]]) for i in range(count)))

    report = {"git_sha": _git("rev-parse", "--short", "HEAD"),
              "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
              "macos": platform.mac_ver()[0], "detector": args.detect, "fixtures": args.fixtures.split(","),
              "one_at_a_time": args.one_at_a_time,
              "counts": {lb: sum(c["label"] == lb for c in cands) for lb in LABELS},
              "proposal_recall": None, "scorers": {},
              "candidates": [{k: c[k] for k in ("fixture", "gid", "frame", "origin", "box", "label")} for c in cands]}
    seen = [f for f in frames if f["visible"]]
    report["proposal_recall"] = sum(f["proposed"] for f in seen) / len(seen) if seen else None

    print(f"candidates {report['counts']}, REACQUIRING frames with the target visible: {len(seen)}, "
          f"own among candidates: {report['proposal_recall']}")
    print(f"\n{'scorer':16s} {'AUC':>5s} {'vs bg':>6s} {'vs absent':>9s} {'vs other':>8s} {'clean recall':>12s}")
    for key, vals in scores.items():
        by = defaultdict(list)
        for c, v in zip(cands, vals, strict=True):
            by[c["label"]].append(v)

        rest = by["other"] + by["bg"] + by["absent"]
        row = {"auc": _auc(by["own"], rest), "auc_bg": _auc(by["own"], by["bg"]),
               "auc_absent": _auc(by["own"], by["absent"]), "auc_other": _auc(by["own"], by["other"]),
               "clean_recall": _recall_clean(by["own"], rest), "scores": [round(v, 4) for v in vals]}
        report["scorers"][key] = row
        print(f"{key:16s} {_fmt(row['auc']):>5s} {_fmt(row['auc_bg']):>6s} {_fmt(row['auc_absent']):>9s} "
              f"{_fmt(row['auc_other']):>8s} {_fmt(row['clean_recall']):>12s}")

    rule = TrackingConfig()
    by_frame = defaultdict(list)
    for i, c in enumerate(cands):
        by_frame[(c["fixture"], c["gid"], c["frame"])].append(i)

    spells = _spells(frames)
    report["spells"] = [{"fixture": s[0]["fixture"], "gid": s[0]["gid"], "frames": [s[0]["frame"], s[-1]["frame"]],
                         "proposed": sum(f["proposed"] for f in s)} for s in spells]
    print(f"\nFirst pick per REACQUIRING spell (good_quality {rule.good_quality}, margin {rule.reacquire_margin})")
    for key, vals in scores.items():
        picks = [_first_pick(s, by_frame, cands, vals, rule) for s in spells]
        report["scorers"][key]["picks"] = picks
        own = sum(p is not None and p[1] == "own" for p in picks)
        wrong = sum(p is not None and p[1] != "own" for p in picks)
        print(f"{key:16s} own {own}  wrong {wrong}  none {len(picks) - own - wrong}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_reid.json"
    out.write_text(json.dumps(report))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
