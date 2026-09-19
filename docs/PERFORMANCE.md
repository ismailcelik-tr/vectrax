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

## Camera format vs frame age (2026-09-19 ~17:20, AC power, MacBook)

Command: `uv run scripts/camera_probe.py --device MacBook --method avf
--width W --height H --pixel-format F --label fmt`, two rounds, order varied.
Raw: `benchmarks/results/probe/macbook_fmt_*`, git `010a265`.

| Format | round 1 p50 / p95 | round 2 p50 / p95 |
|---|---|---|
| 1280x720 BGRA | 77.4 / 83.6 | 67.2 / 71.0 |
| 1280x720 420v | 68.3 / 70.3 | 77.2 / 82.9 |
| 1920x1080 420v | 68.4 / 71.0 | 68.1 / 70.0 |
| 1920x1080 BGRA | 84.1 / 88.5 | 85.2 / 88.8 |

Reading: run-to-run spread (~10 ms, two clusters ~68 and ~77 ms) exceeds
any format effect, except 1080p BGRA (~+8 ms, conversion). Keep 720p BGRA.
The same 720p BGRA probe read 56.8 ms in the morning. An exposure-length
hypothesis (dimmer light → later delivery) was contradicted: in a dark room
at 19:30 frame age fell to p50 51.5 ms (`low_light` sidecar). Cause
unknown. Compare latency only within one session and lighting.

## Live latency after parallel propagators + PipelineThread (2026-09-19 ~17:24)

Same command and setup as the 17:11 run; similar light. Raw:
`benchmarks/results/latency/20260919-172403_*`, git `fd49895`.

| Stage (ms, p50 / p95) | 1 target 17:11 → 17:24 | 3 targets 17:11 → 17:24 |
|---|---|---|
| sensor PTS → arrival | 66.1/67.5 → 65.0/66.7 | 61.4/64.3 → 64.7/68.0 |
| arrival → tick | 0.2/0.2 → 0.2/0.2 | 16.5/32.6 → 0.2/0.2 |
| tracking | 9.1/9.9 → 10.8/12.2 | 23.6/26.5 → 10.8/14.9 |
| **sensor → tracked (R3a)** | 75.4/77.1 → **76.1/78.2** | 102.5/117.8 → **76.3/79.5** |
| **sensor → render (R3b)** | 89.3/91.0 → **90.3/98.6** | 116.1/131.1 → **90.7/100.5** |
| frames dropped | 1 → 0 | 118 → 1 |

R3a (≤ 100) met for 1 and 3 targets; R3b (≤ 120) met for both (ADR-007).
Cost of the split: 1-target render p95 +7.6 ms, since a finished tick may
wait for the UI's current waitKey (0–16 ms). Known nit: `rendered` counts
one frame more than `frames` (warm-up boundary race between threads).

## Glass-to-glass, screen flash (2026-09-19 ~19:25, dark room, AC power)

Command: `uv run scripts/glass_to_glass.py --device <name> [--label wired]`,
60 black/white toggles, 0.5 s each. Raw: `benchmarks/results/glass/`.
Includes display latency (same for both cameras). The flash window did
not cover the full screen; contrast was sufficient (0 missed toggles).

| Camera | command → arrival mean / p50 / p95 | command → PTS mean / p50 / p95 |
|---|---|---|
| MacBook | 120.7 / 122.2 / 151.6 | 58.0 / 56.2 / 86.3 |
| iPhone wired (rear camera) | 102.3 / 101.8 / 116.1 | 51.7 / 51.7 / 65.3 |

Frame quantization (33 ms) spreads single samples; compare means.
