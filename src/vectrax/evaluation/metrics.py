"""Single-target tracking metrics against ground truth."""

from collections import Counter
from dataclasses import dataclass, field

from vectrax.evaluation.gt import Visibility
from vectrax.tracking.state import TrackState

__all__ = ["PredBox", "TrackResult", "evaluate_track", "iou"]

SUCCESS_IOU = 0.5
ABSENT = "absent"

# States in which the tracker claims the target is on screen at its box.
_CLAIMS_VISIBLE = frozenset({TrackState.INITIALIZING, TrackState.TRACKING, TrackState.DEGRADED})


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
    states: Counter = field(default_factory=Counter)  # (gt visibility, predicted state) → frames

    @property
    def success_rate(self) -> float:
        return self.successes / self.present_frames if self.present_frames else 0.0

    @property
    def on_target_rate(self) -> float:
        """Separates 'right place, wrong size' from 'wrong place'."""
        return self.on_target / self.present_frames if self.present_frames else 0.0


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
    for f, pred in enumerate(preds):
        if pred is None:
            continue

        r.frames_scored += 1
        vis = gt_vis.get(f)
        r.states[(vis.value if vis else ABSENT, pred.state.value)] += 1
        shown = pred.state in _CLAIMS_VISIBLE
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


def _reappearances(gt_vis, first, n):
    return [f for f in range(first + 1, n) if f in gt_vis and f - 1 not in gt_vis]


def _recovery(start, gt_vis, hits, n):
    f = start
    while f < n and f in gt_vis:
        if f in hits:
            return f - start

        f += 1

    return None
