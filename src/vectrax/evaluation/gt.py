"""Ground truth from CVAT MOT 1.1 exports."""

import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

__all__ = ["GroundTruth", "GtBox", "Visibility", "load_mot"]

GT_FILE = "gt/gt.txt"
LABELS_FILE = "gt/labels.txt"
VISIBLE_MIN = 1.0


class Visibility(Enum):
    VISIBLE = "visible"
    PARTIAL = "partial"  # CVAT "occluded": partly hidden, box on the visible part


@dataclass(frozen=True, slots=True)
class GtBox:
    box: tuple[float, float, float, float]  # x, y, w, h in pixels
    visibility: Visibility
    label: str


@dataclass
class GroundTruth:
    frames: int
    # track id → frame → box; a missing frame means the target is absent.
    tracks: dict[int, dict[int, GtBox]] = field(default_factory=dict)


def load_mot(path: Path | str, frames: int) -> GroundTruth:
    with zipfile.ZipFile(path) as z:
        labels = z.read(LABELS_FILE).decode().split()
        rows = z.read(GT_FILE).decode().splitlines()

    gt = GroundTruth(frames=frames)
    for row in rows:
        mot_frame, track_id, x, y, w, h, _, cls, vis = row.split(",")
        visibility = Visibility.VISIBLE if float(vis) >= VISIBLE_MIN else Visibility.PARTIAL
        # MOT frames are 1-based.
        gt.tracks.setdefault(int(track_id), {})[int(mot_frame) - 1] = GtBox(
            (float(x), float(y), float(w), float(h)), visibility, labels[int(cls) - 1])

    return gt
