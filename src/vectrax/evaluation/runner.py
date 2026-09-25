"""Run a fixture through the deterministic pipeline and score it.

Each GT track is selected at its first annotated frame with its GT box, as an
operator would; that pipeline track is then scored against that GT track.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vectrax.evaluation.gt import load_mot
from vectrax.evaluation.metrics import PredBox, TrackResult, evaluate_track
from vectrax.pipeline import build_file_pipeline
from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.geometry import Box
from vectrax.tracking.propagators import Propagator

__all__ = ["FixtureResult", "run_fixture"]

GT_SUFFIX = ".gt.zip"


@dataclass
class FixtureResult:
    tracks: dict[int, TrackResult]  # GT track id → result
    track_ms: dict


def run_fixture(video: Path | str, propagator_factory: Callable[[], Propagator],
                cfg: TrackingConfig | None = None, worker=None) -> FixtureResult:
    video = Path(video)
    frames = json.loads(video.with_suffix(".json").read_text())["frames"]
    gt = load_mot(video.with_suffix("").with_suffix(GT_SUFFIX), frames)
    starts = {gid: min(boxes) for gid, boxes in gt.tracks.items()}

    pipe = build_file_pipeline(video, cfg or TrackingConfig(), propagator_factory=propagator_factory, worker=worker)
    w, h = pipe.frame_size
    pred_of = {}
    preds = {gid: [] for gid in gt.tracks}
    try:
        while (frame := pipe.read()) is not None:
            for gid, first in starts.items():
                if frame.frame_id == first:
                    pred_of[gid] = pipe.select(Box.from_xywh_px(*gt.tracks[gid][first].box, w, h))

            snaps = {s.track_id: s for s in pipe.process(frame).tracks}
            for gid in gt.tracks:
                snap = snaps.get(pred_of.get(gid))
                preds[gid].append(None if snap is None else PredBox(snap.state, snap.box.to_xywh_px(w, h)))
    finally:
        pipe.close()

    results = {}
    for gid, boxes in gt.tracks.items():
        vis = {f: b.visibility for f, b in boxes.items()}
        own = {f: b.box for f, b in boxes.items()}
        others = {o: {f: b.box for f, b in ob.items()} for o, ob in gt.tracks.items() if o != gid}
        results[gid] = evaluate_track(vis, own, preds[gid], others)

    return FixtureResult(results, pipe.metrics.summary()["track_ms"])
