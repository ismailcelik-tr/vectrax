"""Tracking metrics against ground truth."""

import contextlib
import io
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from vectrax.evaluation.gt import Visibility
from vectrax.tracking.state import TrackState

__all__ = ["PredBox", "Relock", "TrackResult", "evaluate_track", "identity_scores", "iou"]

SUCCESS_IOU = 0.5
ABSENT = "absent"

# States in which the tracker claims the target is on screen at its box.
_CLAIMS_VISIBLE = frozenset({TrackState.INITIALIZING, TrackState.TRACKING, TrackState.DEGRADED})


class Relock(Enum):
    """Where a track claims to be visible again after a hidden spell."""

    OWN = "own"
    OTHER = "other"  # another annotated target
    BACKGROUND = "background"  # no annotated target, or its own target is absent


@dataclass(frozen=True, slots=True)
class PredBox:
    state: TrackState
    box: tuple[float, float, float, float]  # x, y, w, h in pixels


@dataclass
class TrackResult:
    frames_scored: int = 0
    present_frames: int = 0
    successes: int = 0
    on_target: int = 0  # box center inside the GT box, any size
    false_visible: int = 0  # target absent, tracker claims it is on screen
    hijack_frames: int = 0  # tracker box sits on another annotated target
    recoveries: list[int | None] = field(default_factory=list)  # frames to re-lock after each reappearance
    relocks: dict[int, Relock] = field(default_factory=dict)  # frame → target the track came back on
    states: Counter = field(default_factory=Counter)  # (gt visibility, predicted state) → frames

    @property
    def success_rate(self) -> float:
        return self.successes / self.present_frames if self.present_frames else 0.0

    @property
    def on_target_rate(self) -> float:
        """Separates 'right place, wrong size' from 'wrong place'."""
        return self.on_target / self.present_frames if self.present_frames else 0.0

    @property
    def wrong_relocks(self) -> int:
        return sum(k is not Relock.OWN for k in self.relocks.values())


def iou(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    iw = min(ax1 + aw, bx1 + bw) - max(ax1, bx1)
    ih = min(ay1 + ah, by1 + bh) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0

    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


def evaluate_track(gt_vis: dict[int, Visibility], gt_box: dict[int, tuple], preds: list[PredBox | None],
                   other_gt: dict[int, dict[int, tuple]]) -> TrackResult:
    """preds[f] is None before the operator selected the target."""
    r = TrackResult()
    hits = set()
    prev = None
    for f, pred in enumerate(preds):
        if pred is None:
            continue

        r.frames_scored += 1
        vis = gt_vis.get(f)
        r.states[(vis.value if vis else ABSENT, pred.state.value)] += 1
        shown = pred.state in _CLAIMS_VISIBLE
        if shown and prev is not None and prev.state not in _CLAIMS_VISIBLE:
            others = [boxes[f] for boxes in other_gt.values() if f in boxes]
            r.relocks[f] = _relock(pred.box, gt_box.get(f), others)

        prev = pred
        if vis is None:
            r.false_visible += shown
            continue

        r.present_frames += 1
        if not shown:
            continue

        r.on_target += _center_inside(pred.box, gt_box[f])
        if iou(pred.box, gt_box[f]) >= SUCCESS_IOU:
            r.successes += 1
            hits.add(f)
        elif any(f in boxes and iou(pred.box, boxes[f]) >= SUCCESS_IOU for boxes in other_gt.values()):
            r.hijack_frames += 1

    first = next((f for f, p in enumerate(preds) if p is not None), len(preds))
    r.recoveries = [_recovery(r_frame, gt_vis, hits, len(preds)) for r_frame in _reappearances(gt_vis, first, len(preds))]
    return r


def _center_inside(pred, gt):
    cx, cy = pred[0] + pred[2] / 2, pred[1] + pred[3] / 2
    return gt[0] <= cx <= gt[0] + gt[2] and gt[1] <= cy <= gt[1] + gt[3]


def _relock(box, own, others) -> Relock:
    """The target whose box holds the center; best IoU wins when several do."""
    targets = [(own, Relock.OWN)] if own is not None else []
    targets += [(o, Relock.OTHER) for o in others]
    hits = [(iou(box, b), kind) for b, kind in targets if _center_inside(box, b)]
    return max(hits, key=lambda h: h[0])[1] if hits else Relock.BACKGROUND


def _reappearances(gt_vis, first, n):
    return [f for f in range(first + 1, n) if f in gt_vis and f - 1 not in gt_vis]


def _recovery(start, gt_vis, hits, n):
    f = start
    while f < n and f in gt_vis:
        if f in hits:
            return f - start

        f += 1

    return None


def identity_scores(gt: dict[int, dict[int, tuple]], tracks: dict[int, list[PredBox | None]]) -> dict[str, float]:
    """TrackEval IDSW, IDF1 and HOTA over all targets of one clip.

    gt: target id → frame → box. tracks: track id → per-frame PredBox, None
    before selection. Only states that claim the target is visible count.
    """
    metrics = _trackeval_metrics()
    quiet = {"PRINT_CONFIG": False}
    data = _sequence(gt, tracks)
    with _numpy_aliases():
        clear = metrics.CLEAR(quiet).eval_sequence(data)
        ident = metrics.Identity(quiet).eval_sequence(data)
        hota = metrics.HOTA().eval_sequence(data)

    return {"idsw": int(clear["IDSW"]), "idf1": float(ident["IDF1"]), "hota": float(np.mean(hota["HOTA"]))}


def _trackeval_metrics():
    # Importing trackeval prints a warning when the optional BURST dataset's pycocotools is missing.
    with contextlib.redirect_stdout(io.StringIO()):
        from trackeval import metrics

    return metrics


@contextlib.contextmanager
def _numpy_aliases():
    """TrackEval @12c8791 still uses np.int and np.float, removed in NumPy 1.24."""
    np.int, np.float = int, float
    try:
        yield
    finally:
        del np.int, np.float


def _sequence(gt, tracks) -> dict:
    """TrackEval's per-clip input: ids renumbered from 0, IoU per frame."""
    n = max([len(p) for p in tracks.values()] + [max(b) + 1 for b in gt.values() if b])
    gt_index = {g: i for i, g in enumerate(gt)}
    track_index = {t: i for i, t in enumerate(tracks)}
    gt_ids, track_ids, sims = [], [], []
    for f in range(n):
        gts = [(gt_index[g], boxes[f]) for g, boxes in gt.items() if f in boxes]
        shown = [(track_index[t], p[f].box) for t, p in tracks.items()
                 if f < len(p) and p[f] is not None and p[f].state in _CLAIMS_VISIBLE]
        gt_ids.append(np.array([i for i, _ in gts], dtype=int))
        track_ids.append(np.array([i for i, _ in shown], dtype=int))
        sims.append(np.array([[iou(g, b) for _, b in shown] for _, g in gts]).reshape(len(gts), len(shown)))

    return {
        "num_timesteps": n,
        "num_gt_ids": len(gt),
        "num_tracker_ids": len(tracks),
        "num_gt_dets": sum(map(len, gt_ids)),
        "num_tracker_dets": sum(map(len, track_ids)),
        "gt_ids": gt_ids,
        "tracker_ids": track_ids,
        "similarity_scores": sims,
    }
