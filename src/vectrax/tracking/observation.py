"""What a propagator, detector or operator reports about one target in one frame."""

from dataclasses import dataclass
from enum import Enum

from vectrax.tracking.geometry import Box

__all__ = ["Observation", "Origin"]


class Origin(Enum):
    OPERATOR = "operator"
    PROPAGATOR = "propagator"
    DETECTOR = "detector"


@dataclass(frozen=True, slots=True)
class Observation:
    frame_id: int
    capture_ns: int
    box: Box
    score: float  # 0..1
    origin: Origin
    label: str | None = None  # detector class, a hint only (R1)
