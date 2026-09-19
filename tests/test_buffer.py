import threading

import pytest

from vectrax.buffer import LatestFrameBuffer


def test_get_returns_items_in_order():
    buf = LatestFrameBuffer(capacity=3)
    buf.put(1)
    buf.put(2)

    assert buf.get(timeout_s=0) == 1
    assert buf.get(timeout_s=0) == 2


def test_full_buffer_drops_oldest():
    buf = LatestFrameBuffer(capacity=2)
    for i in range(5):
        buf.put(i)

    assert buf.dropped == 3
    assert buf.get(timeout_s=0) == 3
    assert buf.get(timeout_s=0) == 4


def test_empty_get_times_out_with_none():
    assert LatestFrameBuffer().get(timeout_s=0.01) is None


def test_get_wakes_on_put():
    buf = LatestFrameBuffer()
    threading.Timer(0.02, buf.put, args=("x",)).start()

    assert buf.get(timeout_s=1) == "x"


def test_close_wakes_waiting_reader():
    buf = LatestFrameBuffer()
    threading.Timer(0.02, buf.close).start()

    assert buf.get(timeout_s=1) is None
    assert buf.closed


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        LatestFrameBuffer(capacity=0)
