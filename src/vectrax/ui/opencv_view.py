"""Local operator UI (OpenCV window). Talks only to Pipeline.

macOS requires the window on the main thread. Live tracking runs on
PipelineThread so the ~16 ms waitKey never delays it (ADR-006).
"""

import json
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from vectrax.buffer import LatestFrameBuffer
from vectrax.clock import Clock, MonotonicClock
from vectrax.metrics import RunMode
from vectrax.pipeline import Pipeline, PipelineThread
from vectrax.tracking.geometry import Box
from vectrax.tracking.manager import TrackSnapshot
from vectrax.tracking.state import Command, TrackState

__all__ = ["TITLE_BAR_PT", "Action", "Controller", "draw", "load_init", "run_ui", "window_size"]

WINDOW = "VectraX"
MIN_DRAG_PX = 8
LIVE_WAIT_MS = 1
RENDER_POLL_S = 0.005
FROZEN_WAIT_MS = 15
KEY_NONE = -1
KEY_ESC = 27
FPS_SMOOTHING = 0.9
NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000
HELP = "drag: select  click: focus  1-9: focus id  p: pause/resume  s: stop  x: remove  r: reselect  space: freeze  q: quit"
PAUSED_HINT = "paused: p resumes tracking at this box; to follow a moved object press r and drag"
TITLE_BAR_PT = 32
SCREEN_FILL = 0.97
FOCUS_PAD_PX = 5

_COLORS = {
    TrackState.INITIALIZING: (255, 255, 0),
    TrackState.TRACKING: (0, 220, 0),
    TrackState.DEGRADED: (0, 220, 255),
    TrackState.OCCLUDED: (0, 140, 255),
    TrackState.LOST: (0, 0, 255),
    TrackState.PAUSED: (160, 160, 160),
    TrackState.STOPPED: (80, 80, 80),
}
_PAUSABLE = frozenset({TrackState.INITIALIZING, TrackState.TRACKING, TrackState.DEGRADED, TrackState.OCCLUDED})
_DRAG_COLOR = (255, 0, 255)
_DETECTION_COLOR = (0, 200, 255)
_FOCUS_COLOR = (255, 255, 255)
_TEXT_COLOR = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Action(Enum):
    NONE = auto()
    QUIT = auto()
    TOGGLE_FREEZE = auto()


class Controller:
    """Mouse/keyboard → operator calls. Pixel coordinates of the frame."""

    def __init__(self, pipe, frame_size: tuple[int, int]):
        self._pipe = pipe
        self._w, self._h = frame_size
        self._tracks: dict[int, TrackSnapshot] = {}
        self._drag_start = None
        self._drag_now = None
        self._reselect = False
        self.focus: int | None = None
        self.selected: list[tuple[int, int, int, int]] = []

    @property
    def drag(self):
        if self._drag_start is None:
            return None

        return (*self._drag_start, *self._drag_now)

    def update_tracks(self, tracks: list[TrackSnapshot]) -> None:
        self._tracks = {t.track_id: t for t in tracks}
        if self.focus is not None and self.focus not in self._tracks:
            self.focus = None

    def on_mouse(self, event, x, y) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = self._drag_now = (x, y)
            return

        if self._drag_start is None:
            return

        self._drag_now = (x, y)
        if event != cv2.EVENT_LBUTTONUP:
            return

        (x0, y0), (x1, y1) = self._drag_start, self._drag_now
        self._drag_start = self._drag_now = None
        rect = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        if rect[2] < MIN_DRAG_PX or rect[3] < MIN_DRAG_PX:
            self._focus_at(x1, y1)
            return

        box = Box.from_xywh_px(*rect, self._w, self._h)
        if self._reselect and self.focus is not None:
            self._reselect = False
            self._pipe.command(self.focus, Command.RESELECT, box)
            return

        self.focus = self._pipe.select(box)
        self.selected.append(rect)

    def on_key(self, key: int) -> Action:
        ch = chr(key & 0xFF) if key != KEY_NONE else ""
        if ch == "q" or key == KEY_ESC:
            return Action.QUIT

        if ch == " ":
            return Action.TOGGLE_FREEZE

        if ch.isdigit() and int(ch) in self._tracks:
            self.focus = int(ch)
            return Action.NONE

        if ch == "c":
            self.focus = None
            return Action.NONE

        track = self._tracks.get(self.focus)
        if track is None:
            return Action.NONE

        if ch == "p":
            if track.state is TrackState.PAUSED:
                self._pipe.command(track.track_id, Command.RESUME)
            elif track.state in _PAUSABLE:
                self._pipe.command(track.track_id, Command.PAUSE)
        elif ch == "s":
            self._pipe.command(track.track_id, Command.STOP)
        elif ch == "x":
            self._pipe.remove(track.track_id)
            self.focus = None
        elif ch == "r":
            self._reselect = True

        return Action.NONE

    def _focus_at(self, x, y):
        hits = []
        for t in self._tracks.values():
            bx, by, bw, bh = t.box.to_xywh_px(self._w, self._h)
            if bx <= x <= bx + bw and by <= y <= by + bh:
                hits.append((bw * bh, t.track_id))

        self.focus = min(hits)[1] if hits else None


def draw(image, tracks: list[TrackSnapshot], hud: dict, focus: int | None = None, drag=None,
         pending=(), detections=()):
    out = image.copy()
    h, w = out.shape[:2]
    # Detections are drawn as evidence only; they do not drive any track yet.
    for d in detections:
        x, y, bw, bh = d.box.to_xywh_px(w, h)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), _DETECTION_COLOR, 1)
        cv2.putText(out, f"{d.label} {d.score:.2f}", (x, y + bh + 14), _FONT, 0.45,
                    _DETECTION_COLOR, 1)

    for t in tracks:
        color = _COLORS[t.state]
        x, y, bw, bh = t.box.to_xywh_px(w, h)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
        if t.track_id == focus:
            pad = FOCUS_PAD_PX
            cv2.rectangle(out, (x - pad, y - pad), (x + bw + pad, y + bh + pad), _FOCUS_COLOR, 2)

        trail = np.array([(round(cx * w), round(cy * h)) for cx, cy in t.trail], np.int32)
        cv2.polylines(out, [trail], False, color, 1)

        score = t.quality.propagator_score
        label = f"{t.track_id} {t.state.value}" + ("" if score is None else f" {score:.2f}")
        cv2.putText(out, label, (x, max(y - 6, 12)), _FONT, 0.5, color, 2)

    if drag is not None:
        x0, y0, x1, y1 = drag
        cv2.rectangle(out, (x0, y0), (x1, y1), _DRAG_COLOR, 1)

    # Selections queued on a frozen frame, not yet processed.
    for x, y, bw, bh in pending:
        cv2.rectangle(out, (x, y), (x + bw, y + bh), _DRAG_COLOR, 2)

    lines = [" ".join(f"{k}={v}" for k, v in hud.items()), HELP]
    focused = next((t for t in tracks if t.track_id == focus), None)
    if focused is not None and focused.state is TrackState.PAUSED:
        lines.append(PAUSED_HINT)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (10, 20 + 18 * i), _FONT, 0.45, _TEXT_COLOR, 1)

    return out


def load_init(path: Path) -> list[tuple[float, ...]]:
    if not path.exists():
        return []

    return [tuple(b) for b in json.loads(path.read_text())["boxes"]]


def _save_init(path, rects):
    path.write_text(json.dumps({"frame_id": 0, "boxes": [list(r) for r in rects]}))
    print(f"Saved init boxes to {path}")


def _hud(pipe, tracks, fps, last_tick, focus):
    counts = {}
    for t in tracks:
        counts[t.state.value] = counts.get(t.state.value, 0) + 1

    hud = {"mode": pipe.mode.value, "fps": f"{fps:.1f}", "dropped": pipe.dropped,
           "focus": "-" if focus is None else focus}
    if last_tick is not None and pipe.mode is RunMode.REALTIME:
        hud["age_ms"] = f"{(last_tick.timing.tracked_ns - last_tick.timing.capture_ns) / NS_PER_MS:.0f}"

    return hud | counts


def window_size(frame_size, screen_size):
    """Largest window with the frame's aspect that fits the screen (points)."""
    fw, fh = frame_size
    sw, sh = screen_size
    scale = min(sw * SCREEN_FILL / fw, (sh - TITLE_BAR_PT) * SCREEN_FILL / fh)
    return round(fw * scale), round(fh * scale)


def _screen_size():
    try:
        from AppKit import NSScreen  # macOS only
    except ImportError:
        return None

    frame = NSScreen.mainScreen().visibleFrame()
    return frame.size.width, frame.size.height


def run_ui(pipe: Pipeline, init_boxes, init_path: Path | None = None, clock: Clock | None = None,
           max_frames: int | None = None) -> dict:
    """max_frames: stop after that many processed frames (benchmarks)."""
    clock = clock or MonotonicClock()
    w, h = pipe.frame_size
    ctl = Controller(pipe, (w, h))
    for x, y, bw, bh in init_boxes:
        pipe.select(Box.from_xywh_px(x, y, bw, bh, w, h))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # Resizing by hand stalls the loop (OpenCV); open at full size instead.
    screen = _screen_size()
    if screen is not None:
        cv2.resizeWindow(WINDOW, *window_size((w, h), screen))
    cv2.setMouseCallback(WINDOW, lambda e, x, y, *_: ctl.on_mouse(e, x, y))

    try:
        if pipe.mode is RunMode.REALTIME:
            _run_live(pipe, ctl, clock, max_frames)
        else:
            _run_clip(pipe, ctl, clock, init_path, max_frames)
    finally:
        pipe.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    return pipe.metrics.summary()


class _FpsMeter:
    def __init__(self, clock):
        self._clock = clock
        self._last = clock.now_ns()
        self.fps = 0.0

    def tick(self):
        now = self._clock.now_ns()
        self.fps = FPS_SMOOTHING * self.fps + (1 - FPS_SMOOTHING) * NS_PER_S / max(now - self._last, 1)
        self._last = now


def _run_live(pipe, ctl, clock, max_frames):
    """Tracking on PipelineThread; this (main) thread only draws the newest tick."""
    ticks = LatestFrameBuffer()
    runner = PipelineThread(pipe, ticks, stop_at_end=False, max_frames=max_frames)
    runner.start()
    meter = _FpsMeter(clock)
    try:
        while True:
            tick = ticks.get(RENDER_POLL_S)
            if tick is None and ticks.closed:
                break

            if tick is not None:
                ctl.update_tracks(tick.tracks)
                meter.tick()
                hud = _hud(pipe, tick.tracks, meter.fps, tick, ctl.focus)
                cv2.imshow(WINDOW, draw(tick.frame.image, tick.tracks, hud, ctl.focus, ctl.drag,
                                        detections=tick.detections))

            key = cv2.waitKey(LIVE_WAIT_MS)
            if tick is not None:
                pipe.rendered(tick)

            if ctl.on_key(key) is Action.QUIT:
                break
    finally:
        runner.stop()
        runner.join()

    if runner.error is not None:
        raise runner.error


def _run_clip(pipe, ctl, clock, init_path, max_frames):
    """Single-threaded: starts frozen on frame 0 so targets can be drawn on it."""
    frozen = True
    first = pipe.read(0)
    shown, tracks, last_tick = first, [], None
    anchor = None
    meter = _FpsMeter(clock)
    processed = 0
    while True:
        if frozen:
            if shown is None:
                return

            pending = ctl.selected if last_tick is None else ()
            hud = _hud(pipe, tracks, meter.fps, last_tick, ctl.focus)
            cv2.imshow(WINDOW, draw(shown.image, tracks, hud, ctl.focus, ctl.drag, pending))
            key = cv2.waitKey(FROZEN_WAIT_MS)
        else:
            frame, first = (first, None) if first is not None else (pipe.read(), None)
            if frame is None:
                return

            anchor = anchor or (clock.now_ns(), frame.capture_ns)
            _pace(clock, anchor, frame.capture_ns)
            last_tick = pipe.process(frame)
            tracks = last_tick.tracks
            ctl.update_tracks(tracks)
            meter.tick()
            hud = _hud(pipe, tracks, meter.fps, last_tick, ctl.focus)
            cv2.imshow(WINDOW, draw(frame.image, tracks, hud, ctl.focus, ctl.drag,
                                    detections=last_tick.detections))
            key = cv2.waitKey(LIVE_WAIT_MS)
            pipe.rendered(last_tick)
            shown = frame
            processed += 1
            if max_frames is not None and processed >= max_frames:
                return

        action = ctl.on_key(key) if key != KEY_NONE else Action.NONE
        if action is Action.QUIT:
            return

        if action is Action.TOGGLE_FREEZE:
            if frozen and last_tick is None and init_path is not None and ctl.selected:
                _save_init(init_path, ctl.selected)

            frozen = not frozen
            anchor = None


def _pace(clock, anchor, capture_ns):
    """Play a clip at its recorded speed."""
    wall0, cap0 = anchor
    wait_ns = (capture_ns - cap0) - (clock.now_ns() - wall0)
    if wait_ns > 0:
        time.sleep(wait_ns / NS_PER_S)
