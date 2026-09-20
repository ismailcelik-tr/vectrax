"""Pipeline tick: source → TrackManager → snapshots. The UI talks only to this layer."""

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from vectrax.clock import Clock, MonotonicClock
from vectrax.detection.history import TrackHistory
from vectrax.events import EventBus
from vectrax.frames import FramePacket
from vectrax.metrics import FrameTiming, Metrics, RunMode
from vectrax.sources import CameraSource
from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.geometry import Box
from vectrax.tracking.manager import TrackManager, TrackSnapshot
from vectrax.tracking.observation import Observation
from vectrax.tracking.propagators import NanoPropagator, Propagator, ScoreSource
from vectrax.tracking.state import Command

__all__ = ["Pipeline", "PipelineThread", "Tick", "build_camera_pipeline", "build_file_pipeline"]

READ_TIMEOUT_S = 0.1
PROPAGATOR_WORKERS = 4


@dataclass(frozen=True, slots=True)
class Tick:
    frame: FramePacket
    tracks: list[TrackSnapshot]
    timing: FrameTiming
    # Detections that finished by this tick; they saw an earlier frame and do
    # not touch the tracks yet (fusion is Phase 3 step 5).
    detections: list[Observation] = field(default_factory=list)


class Pipeline:
    def __init__(self, source: CameraSource, manager: TrackManager, bus: EventBus, clock: Clock,
                 mode: RunMode, worker=None):
        self._source = source
        self._manager = manager
        self._clock = clock
        self._worker = worker
        self._history = TrackHistory()
        self._ops_lock = threading.Lock()
        self.bus = bus
        self.mode = mode
        self.metrics = Metrics(mode)
        self._recorder = None
        self._ops: list[dict] = []
        source.open()
        if worker is not None:
            worker.start()

    @property
    def dropped(self) -> int:
        return self._source.dropped

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._source.frame_size

    def close(self) -> None:
        if self._worker is not None:
            self._worker.stop()

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
        with self._ops_lock:
            ops, self._ops = self._ops, []

        if self._recorder is not None:
            index = self._recorder.write(frame)
            for op in ops:
                self._recorder.log({"frame": self._recorder.next_index if index is None else index, **op})

        tick_ns = self._clock.now_ns()
        tracks = self._manager.step(frame)
        timing = FrameTiming(frame.capture_ns, frame.arrival_ns, tick_ns, self._clock.now_ns())
        self.metrics.observe(timing)
        self._history.record(frame.frame_id, {t.track_id: t.box for t in tracks})
        return Tick(frame, tracks, timing, self._detections(frame))

    def _detections(self, frame: FramePacket) -> list[Observation]:
        """Hand this frame to the detector and collect whatever has finished."""
        if self._worker is None:
            return []

        self._worker.submit(frame)
        return [o for result in self._worker.results() for o in result.observations]

    def rendered(self, tick: Tick) -> None:
        self.metrics.observe_render(tick.timing, self._clock.now_ns())

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
        if self._recorder is None:
            return

        with self._ops_lock:
            self._ops.append({"op": op, "track_id": track_id, **fields})


class PipelineThread(threading.Thread):
    """Runs read → process off the UI thread, so a blocking UI (macOS
    waitKey ~16 ms) never delays tracking. The newest Tick goes to `ticks`."""

    def __init__(self, pipe: Pipeline, ticks, stop_at_end: bool, max_frames: int | None = None):
        super().__init__(name="pipeline", daemon=True)
        self._pipe = pipe
        self._ticks = ticks
        self._stop_at_end = stop_at_end
        self._max_frames = max_frames
        self._stopping = threading.Event()
        self.error: Exception | None = None

    def stop(self) -> None:
        self._stopping.set()

    def run(self) -> None:
        processed = 0
        try:
            while not self._stopping.is_set():
                frame = self._pipe.read()
                if frame is None:
                    if self._stop_at_end:
                        return

                    continue

                self._ticks.put(self._pipe.process(frame))
                processed += 1
                if self._max_frames is not None and processed >= self._max_frames:
                    return
        except Exception as e:  # noqa: BLE001 — re-raised on the UI thread
            self.error = e
        finally:
            self._ticks.close()


def _box_list(box):
    return None if box is None else [box.cx, box.cy, box.w, box.h]


def _assemble(source, cfg, clock, mode, scale, propagator_factory=None, worker=None):
    bus = EventBus()
    # Worker threads live for the process; OpenCV releases the GIL in CSRT.
    executor = ThreadPoolExecutor(PROPAGATOR_WORKERS, thread_name_prefix="propagator")
    # ADR-008: NanoTrack, scored by NCC so a vanished target is not reported visible.
    factory = propagator_factory or (lambda: NanoPropagator(scale, score=ScoreSource.NCC))
    manager = TrackManager(cfg, factory, bus, executor)
    return Pipeline(source, manager, bus, clock or MonotonicClock(), mode, worker)


def build_file_pipeline(path: Path | str, cfg: TrackingConfig, clock: Clock | None = None,
                        scale: float = 1.0, propagator_factory: Callable[[], Propagator] | None = None,
                        worker=None) -> Pipeline:
    from vectrax.sources.file import FileSource

    return _assemble(FileSource(path), cfg, clock, RunMode.DETERMINISTIC, scale, propagator_factory,
                     worker)


def build_camera_pipeline(query: str, cfg: TrackingConfig, clock: Clock | None = None,
                          scale: float = 1.0, worker=None) -> Pipeline:
    # Imported lazily: AVFoundation is macOS-only.
    from vectrax.sources.mac import MacCamera

    return _assemble(MacCamera(query), cfg, clock, RunMode.REALTIME, scale, worker=worker)
