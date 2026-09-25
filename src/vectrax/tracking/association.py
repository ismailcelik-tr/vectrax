"""Associator: which detection belongs to which track (SPEC, two tracking roles).

Mutual best only: a pair matches when each is the other's strict favourite.
A detection that suits two tracks equally, or a track with two equal
detections, stays unmatched. Missing a match costs nothing, because the
absence of a detection never demotes a track; a wrong match moves a track
onto another object.
"""

from collections.abc import Mapping, Sequence

from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.geometry import Box
from vectrax.tracking.observation import Observation

__all__ = ["associate"]


def associate(tracks: Mapping[int, Box], detections: Sequence[Observation], cfg: TrackingConfig,
              labels: Mapping[int, str] | None = None) -> dict[int, Observation]:
    """tracks: boxes on the frame the detector saw. labels: each track's class hint, if any."""
    labels = labels or {}
    by_track: dict[int, dict[int, float]] = {t: {} for t in tracks}
    by_det: dict[int, dict[int, float]] = {d: {} for d in range(len(detections))}
    for t, box in tracks.items():
        for d, det in enumerate(detections):
            overlap = _iou(box, det.box)
            if overlap < cfg.assoc_min_iou:
                continue

            same_label = det.label is not None and det.label == labels.get(t)
            by_track[t][d] = by_det[d][t] = overlap * (cfg.assoc_label_weight if same_label else 1.0)

    matches = {}
    for t, candidates in by_track.items():
        d = _strict_best(candidates)
        if d is not None and _strict_best(by_det[d]) == t:
            matches[t] = detections[d]

    return matches


def _strict_best(scores: dict[int, float]) -> int | None:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return None

    if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
        return None

    return ranked[0][0]


def _iou(a: Box, b: Box) -> float:
    iw = min(a.cx + a.w / 2, b.cx + b.w / 2) - max(a.cx - a.w / 2, b.cx - b.w / 2)
    ih = min(a.cy + a.h / 2, b.cy + b.h / 2) - max(a.cy - a.h / 2, b.cy - b.h / 2)
    if iw <= 0 or ih <= 0:
        return 0.0

    inter = iw * ih
    return inter / (a.w * a.h + b.w * b.h - inter)
