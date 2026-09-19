"""Mac camera (built-in or Continuity) as a CameraSource."""

import cv2

from vectrax.buffer import LatestFrameBuffer
from vectrax.frames import FramePacket, PixelFormat
from vectrax.sources.avf import AvfCapture, ensure_permission, find_device

__all__ = ["MacCamera"]


class MacCamera:
    def __init__(self, query: str, width: int = 1280, height: int = 720, fps: int = 30, buffer_capacity: int = 1):
        ensure_permission()
        device = find_device(query)
        self.source_id = f"camera:{device.uniqueID()}"
        self._size = (height, width)
        self._buffer = LatestFrameBuffer(buffer_capacity)
        self._next_id = 0
        self._capture = AvfCapture(device, width, height, fps, self._on_frame)

    @property
    def dropped(self) -> int:
        return self._buffer.dropped + self._capture.dropped

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._size[1], self._size[0]

    def open(self) -> None:
        self._capture.start()

    def close(self) -> None:
        self._capture.stop()
        self._buffer.close()

    def read(self, timeout_s: float) -> FramePacket | None:
        return self._buffer.get(timeout_s)

    def _on_frame(self, view, capture_ns, arrival_ns):
        # Frames before the format switch arrive at the preset size (ADR-001).
        if view.shape[:2] != self._size:
            return

        packet = FramePacket(
            frame_id=self._next_id,
            source_id=self.source_id,
            capture_ns=capture_ns,
            arrival_ns=arrival_ns,
            image=cv2.cvtColor(view, cv2.COLOR_BGRA2BGR),
            pixel_format=PixelFormat.BGR,
        )
        self._next_id += 1
        self._buffer.put(packet)
