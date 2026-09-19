"""Recorded clip replay. Timestamps come from the sidecar written at record time."""

import json
from pathlib import Path

import cv2

from vectrax.frames import FramePacket, PixelFormat

__all__ = ["FileSource"]

NS_PER_S = 1_000_000_000


class FileSource:
    def __init__(self, path: Path | str):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(self._path)

        self.source_id = f"file:{self._path.stem}"
        self._stamps = self._load_stamps()
        self._cap = None
        self._next_id = 0

    @property
    def dropped(self) -> int:
        return 0

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._size

    def open(self) -> None:
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise OSError(f"cannot decode {self._path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self, timeout_s: float = 0) -> FramePacket | None:
        ok, image = self._cap.read()
        if not ok:
            return None

        frame_id = self._next_id
        self._next_id += 1
        capture_ns, arrival_ns = self._stamp(frame_id)
        return FramePacket(
            frame_id=frame_id,
            source_id=self.source_id,
            capture_ns=capture_ns,
            arrival_ns=arrival_ns,
            image=image,
            pixel_format=PixelFormat.BGR,
        )

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    def _load_stamps(self):
        sidecar = self._path.with_suffix(".json")
        if not sidecar.exists():
            return None

        return json.loads(sidecar.read_text())["stamps"]

    def _stamp(self, frame_id):
        if self._stamps is None:
            ns = round(frame_id * NS_PER_S / self._fps)
            return ns, ns

        if frame_id >= len(self._stamps):
            raise ValueError(f"{self._path.name}: frame {frame_id} has no sidecar stamp")

        s = self._stamps[frame_id]
        return s["capture_ns"], s["arrival_ns"]
