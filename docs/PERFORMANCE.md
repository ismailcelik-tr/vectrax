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

## Live pipeline latency, CSRT, 1280x720 @ 30 (2026-09-19, AC power)

Command: `uv run benchmarks/live_latency.py --targets 1,3` (MacBook camera,
160 px boxes at fixed places, 900 frames after 30 warm-up, OpenCV UI
rendering). Raw: `benchmarks/results/latency/20260919-171137_*`, git `1abf5b5`.

| Stage (ms) | 1 target p50 / p95 / max | 3 targets p50 / p95 / max |
|---|---|---|
| sensor PTS → arrival | 66.1 / 67.5 / 70.8 | 61.4 / 64.3 / 68.0 |
| arrival → tick (queue) | 0.2 / 0.2 / 1.0 | 16.5 / 32.6 / 35.3 |
| tracking (all targets) | 9.1 / 9.9 / 11.4 | 23.6 / 26.5 / 28.9 |
| **sensor PTS → render** | **89.3 / 91.0 / 92.8** | **116.1 / 131.1 / 136.6** |
| frames dropped | 1 | 118 (≈13 %) |

R3 (p95 ≤ 100 ms): met with 1 target, missed with 3.

Reading: ~66 ms elapse before Python sees a frame, leaving ~34 ms. CSRT
costs ~8 ms per target; draw + imshow + waitKey ~14 ms. With 3 targets
one loop (~38 ms) exceeds the 33 ms frame period, so frames wait and drop.
The earlier GIL-contention hypothesis is not supported: arrival latency
did not rise with 3 targets. Single run per row.
