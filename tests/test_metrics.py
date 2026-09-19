import pytest

from vectrax.metrics import FrameTiming, LatencyStats, Metrics, RunMode

MS = 1_000_000


def test_latency_stats_percentiles():
    stats = LatencyStats()
    for v in range(1, 101):
        stats.add(v * MS)

    s = stats.summary()

    assert s["n"] == 100
    assert s["p50"] == pytest.approx(50.5)
    assert s["p95"] == pytest.approx(95.05)
    assert s["max"] == pytest.approx(100)


def test_empty_stats_summary_is_none():
    assert LatencyStats().summary() is None


def _timing():
    return FrameTiming(capture_ns=0, arrival_ns=50 * MS, tick_ns=60 * MS, tracked_ns=70 * MS)


def test_realtime_reports_capture_based_spans():
    m = Metrics(RunMode.REALTIME)
    m.observe(_timing())
    m.observe_render(_timing(), render_ns=80 * MS)

    s = m.summary()

    assert s["capture_to_arrival_ms"]["p50"] == pytest.approx(50)
    assert s["arrival_to_tick_ms"]["p50"] == pytest.approx(10)
    assert s["track_ms"]["p50"] == pytest.approx(10)
    assert s["capture_to_tracked_ms"]["p50"] == pytest.approx(70)
    assert s["capture_to_render_ms"]["p50"] == pytest.approx(80)
    assert s["frames"] == 1
    assert s["rendered"] == 1


def test_deterministic_reports_processing_only():
    m = Metrics(RunMode.DETERMINISTIC)
    m.observe(_timing())
    m.observe_render(_timing(), render_ns=80 * MS)

    s = m.summary()

    assert s["track_ms"]["p50"] == pytest.approx(10)
    assert "capture_to_tracked_ms" not in s
    assert "capture_to_render_ms" not in s


def test_unrendered_frames_still_count():
    m = Metrics(RunMode.REALTIME)
    m.observe(_timing())

    s = m.summary()

    assert s["frames"] == 1
    assert s["rendered"] == 0
    assert s["capture_to_render_ms"] is None


def test_warmup_frames_are_excluded():
    m = Metrics(RunMode.REALTIME, warmup_frames=2)
    for _ in range(5):
        m.observe(_timing())

    s = m.summary()

    assert s["frames"] == 3
    assert s["track_ms"]["n"] == 3
