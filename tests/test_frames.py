import dataclasses

import numpy as np
import pytest

from vectrax.frames import FramePacket, PixelFormat


def _packet(h=720, w=1280):
    return FramePacket(
        frame_id=7,
        source_id="test",
        capture_ns=100,
        arrival_ns=150,
        image=np.zeros((h, w, 3), np.uint8),
        pixel_format=PixelFormat.BGR,
    )


def test_size_derives_from_image():
    p = _packet(h=360, w=640)

    assert (p.width, p.height) == (640, 360)


def test_age_is_arrival_minus_capture():
    assert _packet().age_ns == 50


def test_packet_is_immutable():
    p = _packet()

    with pytest.raises(dataclasses.FrozenInstanceError):
        p.frame_id = 8


def test_arrival_before_capture_rejected():
    with pytest.raises(ValueError):
        FramePacket(
            frame_id=0,
            source_id="test",
            capture_ns=200,
            arrival_ns=100,
            image=np.zeros((2, 2, 3), np.uint8),
            pixel_format=PixelFormat.BGR,
        )
