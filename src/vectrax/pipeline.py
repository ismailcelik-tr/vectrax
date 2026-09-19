"""Pipeline tick: source → TrackManager → snapshots. The UI talks only to this layer."""

from dataclasses import dataclass, replace
from pathlib import Path

from vectrax.clock import Clock, MonotonicClock
from vectrax.events import EventBus
from vectrax.frames import FramePacket
from vectrax.metrics import FrameTiming, Metrics, RunMode
from vectrax.sources import CameraSource
from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.geometry import Box
from vectrax.tracking.manager import TrackManager, TrackSnapshot
from vectrax.tracking.propagators import CsrtPropagator
from vectrax.tracking.state import Command

__all__ = ["Pipeline", "Tick", "build_camera_pipeline", "build_file_pipeline"]

READ_TIMEOUT_S = 0.1


@dataclass(frozen=True, slots=True)
class Tick:
    frame: FramePacket
    tracks: list[TrackSnapshot]
    timing: FrameTiming


class Pipeline:
    def __init__(self, source: CameraSource, manager: TrackManager, bus: EventBus,
                 clock: Clock, mode: RunMode, renders: bool = False):
        self._source = source
        self._manager = manager
        self._clock = clock
        self._renders = renders
        self.bus = bus
        self.mode = mode
        self.metrics = Metrics(mode)
        self._recorder = None
        self._ops: list[dict] = []
        source.open()

    @property
    def dropped(self) -> int:
        return self._source.dropped

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._source.frame_size

    def close(self) -> None:
        self._source.close()
        if self._recorder is not None:
            self._recorder.close()

    def start_recording(self, recorder) -> None:
        """Record every frame processed from now on, with operator calls."""
        self._recorder = recorder

    def tick(self, timeout_s: float = READ_TIMEOUT_S) -> Tick | None:
        """None: end of a recorded stream, or no live frame within timeout."""
        frame = self.read(timeout_s)
        if frame is None:
            return None

        return self.process(frame)

    def read(self, timeout_s: float = READ_TIMEOUT_S) -> FramePacket | None:
        return self._source.read(timeout_s)

    def process(self, frame: FramePacket) -> Tick:
        """Operator calls made before this apply to this frame."""
        if self._recorder is not None:
            index = self._recorder.write(frame)
            for op in self._ops:
                self._recorder.log({"frame": self._recorder.next_index if index is None else index, **op})

        self._ops = []
        tick_ns = self._clock.now_ns()
        tracks = self._manager.step(frame)
        timing = FrameTiming(frame.capture_ns, frame.arrival_ns, tick_ns, self._clock.now_ns())
        if not self._renders:
            self.metrics.observe(timing)

        return Tick(frame, tracks, timing)

    def rendered(self, tick: Tick) -> None:
        self.metrics.observe(replace(tick.timing, render_ns=self._clock.now_ns()))

    def select(self, box: Box) -> int:
        track_id = self._manager.select(box)
        self._note("select", track_id, box=_box_list(box))
        return track_id

    def command(self, track_id: int, cmd: Command, box: Box | None = None) -> None:
        self._manager.command(track_id, cmd, box)
        self._note("command", track_id, command=cmd.value, box=_box_list(box))

    def remove(self, track_id: int) -> None:
        self._manager.remove(track_id)
        self._note("remove", track_id)

    def _note(self, op, track_id, **fields):
        if self._recorder is not None:
            self._ops.append({"op": op, "track_id": track_id, **fields})


def _box_list(box):
    return None if box is None else [box.cx, box.cy, box.w, box.h]


def _assemble(source, cfg, clock, mode, renders, scale):
    bus = EventBus()
    manager = TrackManager(cfg, lambda: CsrtPropagator(scale), bus)
    return Pipeline(source, manager, bus, clock or MonotonicClock(), mode, renders)


def build_file_pipeline(path: Path | str, cfg: TrackingConfig, clock: Clock | None = None,
                        renders: bool = False, scale: float = 1.0) -> Pipeline:
    from vectrax.sources.file import FileSource

    return _assemble(FileSource(path), cfg, clock, RunMode.DETERMINISTIC, renders, scale)


def build_camera_pipeline(query: str, cfg: TrackingConfig, clock: Clock | None = None,
                          renders: bool = True, scale: float = 1.0) -> Pipeline:
    # Imported lazily: AVFoundation is macOS-only.
    from vectrax.sources.mac import MacCamera

    return _assemble(MacCamera(query), cfg, clock, RunMode.REALTIME, renders, scale)
