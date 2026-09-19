"""Detection precision, recall and AP50 against fixture ground truth.

Partial GT (CVAT "occluded") is an ignore region: a detection on it is
neither a hit nor a false positive, and missing it costs no recall.
"""

from dataclasses import dataclass

from vectrax.evaluation.gt import GroundTruth, Visibility
from vectrax.evaluation.metrics import SUCCESS_IOU, iou

__all__ = ["DetResult", "Detection", "evaluate_detections"]


@dataclass(frozen=True, slots=True)
class Detection:
    frame: int
    box: tuple[float, float, float, float]  # x, y, w, h in pixels
    score: float
    label: str  # COCO class name


@dataclass(frozen=True, slots=True)
class DetResult:
    gt_count: int  # visible GT boxes
    tp: int  # at score ≥ score_min
    fp: int
    precision: float
    recall: float
    ap50: float  # over all scores


def evaluate_detections(gt: GroundTruth, dets: list[Detection], label: str | None,
                        score_min: float) -> DetResult:
    """label None = class-agnostic: any detection may match any GT box."""
    visible, partial = {}, {}
    for boxes in gt.tracks.values():
        for f, b in boxes.items():
            if label is not None and b.label != label:
                continue

            target = visible if b.visibility is Visibility.VISIBLE else partial
            target.setdefault(f, []).append(b.box)

    ranked = sorted((d for d in dets if label is None or d.label == label), key=lambda d: -d.score)
    taken = set()
    hits = []  # (score, is_tp) in score order, ignored detections left out
    for d in ranked:
        match = _best(d.box, visible.get(d.frame, []), d.frame, taken)
        if match is not None:
            taken.add(match)
            hits.append((d.score, True))
            continue

        if any(iou(d.box, g) >= SUCCESS_IOU for g in partial.get(d.frame, [])):
            continue

        hits.append((d.score, False))

    gt_count = sum(len(v) for v in visible.values())
    tp = sum(1 for s, ok in hits if ok and s >= score_min)
    fp = sum(1 for s, ok in hits if not ok and s >= score_min)
    return DetResult(
        gt_count=gt_count,
        tp=tp,
        fp=fp,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / gt_count if gt_count else 0.0,
        ap50=_ap([ok for _, ok in hits], gt_count),
    )


def _best(box, candidates, frame, taken):
    best, best_iou = None, SUCCESS_IOU
    for i, g in enumerate(candidates):
        if (frame, i) in taken:
            continue

        v = iou(box, g)
        if v >= best_iou:
            best, best_iou = (frame, i), v

    return best


def _ap(hits, gt_count):
    """All-point interpolated area under the precision-recall curve."""
    if not gt_count:
        return 0.0

    precisions, recalls = [], []
    tp = 0
    for n, ok in enumerate(hits, start=1):
        tp += ok
        precisions.append(tp / n)
        recalls.append(tp / gt_count)

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap, prev = 0.0, 0.0
    for p, r in zip(precisions, recalls, strict=True):
        ap += (r - prev) * p
        prev = r

    return ap
