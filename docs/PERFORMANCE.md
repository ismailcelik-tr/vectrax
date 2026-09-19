# Performance

Measured only. Each table cites its command and raw output.

## Camera capture, 1280x720 @ 30 (2026-09-19, AC power)

Command: `uv run scripts/camera_probe.py --device <name> [--label <l>]`
(600 frames after 30 warm-up frames, AVFoundation BGRA).
Raw: `benchmarks/results/probe/`, git `1b3f0b0`.

Frame age = host time at delivery − sensor PTS.

| Source | eff. fps | gaps | interval p95 / max ms | frame age p50 / p95 / max ms | copy to numpy p50 ms | CPU % |
|---|---|---|---|---|---|---|
| MacBook, AVF | 30.00 | 0 | 37.2 / 45.2 | 56.8 / 60.7 / 69.8 | 0.32 | 6.9 |
| MacBook, OpenCV | 30.00 | 0 | 36.9 / 41.4 | n/a | n/a | 8.7 |
| iPhone wired, AVF | 29.99 | 0 | 34.7 / 40.5 | 44.7 / 47.8 / 52.5 | 0.33 | 4.4 |
| iPhone wired, OpenCV | 29.98 | 0 | 34.8 / 38.5 | n/a | n/a | 7.2 |
| iPhone wireless, AVF | 27.61 | 2 | 36.0 / 1502 | 52.7 / 66.4 / 102.2 | 0.34 | 4.7 |
| iPhone wireless, OpenCV | 30.01 | 0 | 36.3 / 42.0 | n/a | n/a | 7.8 |

Caveats: single run per row. iPhone PTS semantics unverified (ADR-002).
Sensor-side latency before PTS is not visible; glass-to-glass pending.
