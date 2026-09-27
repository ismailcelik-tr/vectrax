"""What a propagator, detector or operator reports about one target in one frame."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from vectrax.tracking.geometry import Box

__all__ = ["Observation", "Origin"]


class Origin(Enum):
    OPERATOR = "operator"
    PROPAGATOR = "propagator"
    DETECTOR = "detector"
    SEARCH = "search"  # appearance search while reacquiring


@dataclass(frozen=True, slots=True)
class Observation:
    frame_id: int
    capture_ns: int
    box: Box
    score: float  # 0..1
    origin: Origin
    label: str | None = None  # detector class, a hint only (R1)
    embedding: np.ndarray | None = field(default=None, compare=False)  # Re-ID, L2-normalized
