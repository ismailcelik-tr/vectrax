"""Time alignment for late detections.

Inference finishes a few frames after the frame it saw, so a result has to be
compared against where the tracks were *then*, and the correction moved onto
where they are *now*.
"""

from collections import OrderedDict

from vectrax.tracking.geometry import Box

__all__ = ["TrackHistory", "carry_forward"]

WINDOW = 30  # 1 s at 30 fps; older results are not worth fusing


class TrackHistory:
    """Track boxes per frame, for the last `window` frames."""

    def __init__(self, window: int = WINDOW):
        if window < 1:
            raise ValueError("window must be >= 1")

        self._window = window
        self._frames: OrderedDict[int, dict[int, Box]] = OrderedDict()

    def record(self, frame_id: int, boxes: dict[int, Box]) -> None:
        self._frames[frame_id] = dict(boxes)
        while len(self._frames) > self._window:
            self._frames.popitem(last=False)

    def at(self, frame_id: int) -> dict[int, Box]:
        """Empty when that frame has already fallen out of the window."""
        return self._frames.get(frame_id, {})


def carry_forward(detection: Box, seen_at: Box, now: Box) -> Box:
    """Move a detection from the frame it saw onto the current frame.

    The centre shifts by however far the track moved in between; the size stays
    the detector's, because the propagator's size is the part we distrust (it
    inflates under motion blur, docs/ROADMAP.md).
    """
    return Box(detection.cx + (now.cx - seen_at.cx), detection.cy + (now.cy - seen_at.cy),
               detection.w, detection.h)
