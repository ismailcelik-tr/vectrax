"""Session recording: the frames the pipeline processed plus operator actions.

Layout (FileSource reads video.mp4 + video.json directly):
    <dir>/video.mp4       H.264, processed frames only
    <dir>/video.json      per-frame capture/arrival stamps
    <dir>/operator.jsonl  {"frame": index, "op": ...} per operator call
"""

import json
import queue
import subprocess
import threading
from pathlib import Path

from vectrax.frames import FramePacket

__all__ = ["SessionRecorder"]

QUEUE_FRAMES = 120
CRF = 12
VIDEO = "video.mp4"
OPERATOR_LOG = "operator.jsonl"


class SessionRecorder:
    """Encoding runs on its own thread so recording does not add tick latency."""

    def __init__(self, directory: Path | str, fps: int):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=False)
        self._fps = fps
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_FRAMES)
        self._stamps: list[dict] = []
        self._ops = open(self._dir / OPERATOR_LOG, "w")  # noqa: SIM115
        self._encoder = None
        self._writer = None
        self._size = None
        self.dropped = 0

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def next_index(self) -> int:
        return len(self._stamps)

    def write(self, frame: FramePacket) -> int | None:
        """Index of the frame in the recording; None if the encoder fell behind."""
        if self._encoder is None:
            self._start(frame.width, frame.height)

        try:
            self._queue.put_nowait(frame.image)
        except queue.Full:
            self.dropped += 1
            return None

        self._stamps.append({"capture_ns": frame.capture_ns, "arrival_ns": frame.arrival_ns})
        return len(self._stamps) - 1

    def log(self, entry: dict) -> None:
        self._ops.write(json.dumps(entry) + "\n")

    def close(self) -> None:
        self._ops.close()
        if self._encoder is not None:
            self._queue.put(None)
            self._writer.join()
            self._encoder.stdin.close()
            self._encoder.wait()

        w, h = self._size or (0, 0)
        meta = {"fps": self._fps, "width": w, "height": h, "frames": len(self._stamps),
                "recording_dropped": self.dropped, "stamps": self._stamps}
        (self._dir / VIDEO).with_suffix(".json").write_text(json.dumps(meta, indent=1))

    def _start(self, width, height):
        self._size = (width, height)
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(self._fps), "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(CRF), "-pix_fmt", "yuv420p",
            str(self._dir / VIDEO),
        ]
        self._encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._writer = threading.Thread(target=self._drain, daemon=True)
        self._writer.start()

    def _drain(self):
        while (image := self._queue.get()) is not None:
            self._encoder.stdin.write(image.tobytes())
