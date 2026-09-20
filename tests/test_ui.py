import cv2
import numpy as np

from vectrax.tracking.geometry import Box
from vectrax.tracking.manager import TrackSnapshot
from vectrax.tracking.quality import TrackQuality
from vectrax.tracking.state import Command, TrackState
from vectrax.ui.opencv_view import TITLE_BAR_PT, Action, Controller, draw, window_size

W, H = 640, 360
CENTER = Box(0.5, 0.5, 0.2, 0.2)


class FakePipe:
    def __init__(self):
        self.calls = []
        self.next_id = 1

    def select(self, box):
        self.calls.append(("select", box))
        self.next_id += 1
        return self.next_id - 1

    def command(self, track_id, cmd, box=None):
        self.calls.append(("command", track_id, cmd, box))

    def remove(self, track_id):
        self.calls.append(("remove", track_id))


def _snap(track_id, state, box=CENTER):
    return TrackSnapshot(track_id, state, box, box, TrackQuality(0.9, 1.0), (0.0, 0.0), 0, 0, ((0.5, 0.5),))


def _drag(ctl, x0, y0, x1, y1):
    ctl.on_mouse(cv2.EVENT_LBUTTONDOWN, x0, y0)
    ctl.on_mouse(cv2.EVENT_MOUSEMOVE, x1, y1)
    ctl.on_mouse(cv2.EVENT_LBUTTONUP, x1, y1)


def test_drag_selects_new_target():
    pipe = FakePipe()
    ctl = Controller(pipe, (W, H))

    _drag(ctl, 64, 36, 192, 108)

    kind, box = pipe.calls[0]
    assert kind == "select"
    assert (box.cx, box.cy, box.w, box.h) == (0.2, 0.2, 0.2, 0.2)
    assert ctl.focus == 1


def test_reverse_drag_is_normalized():
    pipe = FakePipe()
    ctl = Controller(pipe, (W, H))

    _drag(ctl, 192, 108, 64, 36)

    assert pipe.calls[-1][1] == Box(0.2, 0.2, 0.2, 0.2)


def test_click_inside_track_focuses_without_selecting():
    pipe = FakePipe()
    ctl = Controller(pipe, (W, H))
    ctl.update_tracks([_snap(7, TrackState.TRACKING)])

    _drag(ctl, W // 2, H // 2, W // 2 + 1, H // 2 + 1)

    assert ctl.focus == 7
    assert pipe.calls == []


def test_pause_toggles_with_state():
    pipe = FakePipe()
    ctl = Controller(pipe, (W, H))
    ctl.focus = 7

    ctl.update_tracks([_snap(7, TrackState.TRACKING)])
    ctl.on_key(ord("p"))
    ctl.update_tracks([_snap(7, TrackState.PAUSED)])
    ctl.on_key(ord("p"))

    assert [c[2] for c in pipe.calls] == [Command.PAUSE, Command.RESUME]


def test_stop_and_remove_focused():
    pipe = FakePipe()
    ctl = Controller(pipe, (W, H))
    ctl.focus = 3
    ctl.update_tracks([_snap(3, TrackState.TRACKING)])

    ctl.on_key(ord("s"))
    ctl.on_key(ord("x"))

    assert pipe.calls == [("command", 3, Command.STOP, None), ("remove", 3)]
    assert ctl.focus is None


def test_reselect_mode_uses_next_drag():
    pipe = FakePipe()
    ctl = Controller(pipe, (W, H))
    ctl.focus = 4
    ctl.update_tracks([_snap(4, TrackState.LOST)])

    ctl.on_key(ord("r"))
    _drag(ctl, 64, 36, 192, 108)

    assert pipe.calls == [("command", 4, Command.RESELECT, Box(0.2, 0.2, 0.2, 0.2))]


def test_keys_without_focus_do_nothing():
    pipe = FakePipe()
    ctl = Controller(pipe, (W, H))

    for key in "psxr":
        assert ctl.on_key(ord(key)) is Action.NONE

    assert pipe.calls == []


def test_quit_and_freeze_actions():
    ctl = Controller(FakePipe(), (W, H))

    assert ctl.on_key(ord("q")) is Action.QUIT
    assert ctl.on_key(ord(" ")) is Action.TOGGLE_FREEZE


def test_draw_does_not_modify_input():
    img = np.zeros((H, W, 3), np.uint8)
    tracks = [_snap(1, s) for s in TrackState]

    out = draw(img, tracks, {"fps": 30.0}, focus=1, drag=(10, 10, 50, 50))

    assert out.shape == img.shape
    assert not img.any()
    assert out.any()


def test_focused_track_is_drawn_differently():
    img = np.zeros((H, W, 3), np.uint8)
    tracks = [_snap(1, TrackState.TRACKING)]

    plain = draw(img, tracks, {}, focus=None)
    focused = draw(img, tracks, {}, focus=1)

    assert (plain != focused).any()


def test_window_fits_screen_keeping_aspect():
    w, h = window_size((1280, 720), (1512, 949))

    assert w <= 1512 and h + TITLE_BAR_PT <= 949
    assert abs(w / h - 1280 / 720) < 0.01
    assert w > 1280


def test_window_never_shrinks_below_frame_on_big_screen():
    assert window_size((1280, 720), (3000, 2000))[0] >= 1280


def test_too_small_a_drag_says_so_instead_of_doing_nothing():
    """A drag under MIN_DRAG_PX outside any track used to be a silent focus click,
    so an operator aiming at a pupil got no target and no explanation."""
    pipe = FakePipe()
    ctl = Controller(pipe, (W, H))

    _drag(ctl, 300, 200, 304, 203)

    assert pipe.calls == []
    assert "small" in ctl.notice.lower()


def test_a_click_inside_a_track_focuses_it_without_a_notice():
    pipe = FakePipe()
    ctl = Controller(pipe, (W, H))
    ctl.update_tracks([_snap(7, TrackState.TRACKING)])

    _drag(ctl, W // 2, H // 2, W // 2 + 2, H // 2 + 2)

    assert ctl.focus == 7
    assert ctl.notice is None


def test_a_real_drag_clears_an_earlier_notice():
    pipe = FakePipe()
    ctl = Controller(pipe, (W, H))

    _drag(ctl, 300, 200, 304, 203)
    _drag(ctl, 64, 36, 192, 108)

    assert ctl.notice is None
    assert pipe.calls[0][0] == "select"
