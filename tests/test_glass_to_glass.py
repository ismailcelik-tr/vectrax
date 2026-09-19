import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import glass_to_glass as g

FRAME_NS = 33_333_333
MS = 1_000_000
DARK, BRIGHT = 20.0, 80.0


def _synth(latency_ms, age_ms, toggles=10):
    """Frames every 33 ms; brightness follows the screen `latency_ms` late."""
    half = int(g.HALF_PERIOD_S * g.NS_PER_S)
    t0 = 1_000 * MS
    marks = [(t0 + i * half, i % 2 == 0) for i in range(toggles)]
    frames = []
    t = t0 - half
    while t < t0 + toggles * half:
        seen = t - latency_ms * MS
        state = [w for tm, w in marks if tm <= seen]
        lum = BRIGHT if state and state[-1] else DARK
        frames.append((t - age_ms * MS, t, lum))
        t += FRAME_NS

    return frames, marks


def test_latency_within_one_frame():
    frames, marks = _synth(latency_ms=100, age_ms=50)
    r = g._analyze(frames, marks)

    assert r["missed_toggles"] == 0
    assert 100 <= r["toggle_to_arrival_ms"]["mean"] <= 100 + FRAME_NS / MS
    assert r["toggle_to_pts_ms"]["mean"] == pytest.approx(r["toggle_to_arrival_ms"]["mean"] - 50, abs=0.1)


def test_flat_signal_rejected():
    frames, marks = _synth(latency_ms=100, age_ms=50)
    flat = [(c, a, DARK) for c, a, _ in frames]

    with pytest.raises(SystemExit):
        g._analyze(flat, marks)
