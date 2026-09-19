"""Frame sources. Perception never knows which one produced a frame."""

from typing import Protocol

from vectrax.frames import FramePacket

__all__ = ["CameraSource"]


class CameraSource(Protocol):
    source_id: str

    def open(self) -> None: ...

    def close(self) -> None: ...

    def read(self, timeout_s: float) -> FramePacket | None:
        """Next frame; None on timeout or end of stream."""
        ...

    @property
    def dropped(self) -> int: ...
