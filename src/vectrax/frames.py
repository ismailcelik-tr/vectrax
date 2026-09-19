"""Source-independent frame representation."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = ["FramePacket", "PixelFormat"]


class PixelFormat(Enum):
    BGR = "bgr"


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame_id: int
    source_id: str
    capture_ns: int  # sensor PTS on the monotonic clock (ADR-003)
    arrival_ns: int
    image: np.ndarray
    pixel_format: PixelFormat

    def __post_init__(self):
        if self.arrival_ns < self.capture_ns:
            raise ValueError("arrival_ns precedes capture_ns")

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def age_ns(self) -> int:
        return self.arrival_ns - self.capture_ns
