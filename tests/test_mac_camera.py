import os

import pytest

from vectrax.frames import PixelFormat

pytestmark = pytest.mark.skipif(os.environ.get("VECTRAX_CAMERA") != "1", reason="needs camera; set VECTRAX_CAMERA=1")

FRAMES = 30
READ_TIMEOUT_S = 2


def test_mac_camera_delivers_ordered_bgr_frames():
    from vectrax.sources.mac import MacCamera

    cam = MacCamera("MacBook", width=1280, height=720, fps=30)
    cam.open()
    try:
        packets = [cam.read(timeout_s=READ_TIMEOUT_S) for _ in range(FRAMES)]
    finally:
        cam.close()

    assert all(p is not None for p in packets)
    ids = [p.frame_id for p in packets]
    assert ids == sorted(ids)
    assert all(p.capture_ns < p.arrival_ns for p in packets)
    assert all((p.width, p.height) == (1280, 720) for p in packets)
    assert all(p.pixel_format is PixelFormat.BGR for p in packets)
    assert cam.source_id.startswith("camera:")
    assert cam.read(timeout_s=0.1) is None
