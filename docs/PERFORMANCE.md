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

## Detector backends, 1280x720 frames (2026-09-20, AC power, Low Power Mode off)

Command: `uv run benchmarks/detector_latency.py --detector <d> --backend <b>
[--precision fp16|fp32]`. Raw: `benchmarks/results/detector_latency/`,
git `f87b9e1`. 300 inferences after 10 warm-up, cycling 120 decoded
single_target frames; decoding is outside the timed loop. Load and the first
inference are timed separately, since the first call pays lazy build or graph
compile. RSS is the process peak, so it includes the framework (torch ~0.5 GB).
The `git_dirty: true` flag in every file comes from the benchmark scripts these
runs used, committed immediately afterwards; no code changed between runs.

| Detector | Backend | load ms | first ms | p50 ms | p95 ms | RSS MB |
|---|---|---|---|---|---|---|
| YOLO26n (ref) | pytorch-cpu | 606 | 533 | 15.8 | 20.4 | 740 |
| YOLO26n (ref) | pytorch-mps | 598 | 2286 | 8.3 | 12.5 | 887 |
| YOLO11n (ref) | pytorch-cpu | 484 | 451 | 15.9 | 20.2 | 730 |
| YOLO11n (ref) | pytorch-mps | 581 | 885 | 7.9 | 11.4 | 885 |
| RF-DETR-N | pytorch-cpu | 4061 | 119 | 42.4 | 51.4 | 1223 |
| RF-DETR-N | pytorch-mps | 3585 | 1571 | 27.6 | 33.6 | 1278 |
| RF-DETR-N | onnx-cpu | 89 | 48 | 46.6 | 59.7 | 630 |
| RF-DETR-N | onnx-coreml | 4035 | 95 | 67.0 | 76.3 | 1079 |
| RF-DETR-N | coreml-cpu fp32 | 1642 | 61 | 31.6 | 34.1 | 968 |
| RF-DETR-N | coreml-gpu fp32 | 1611 | 173 | 16.0 | 18.0 | 976 |
| RF-DETR-N | coreml-ane fp32 | 1727 | 41 | 31.5 | 33.8 | 971 |
| RF-DETR-N | coreml-cpu fp16 | 1570 | 39 | 17.3 | 18.5 | 865 |
| **RF-DETR-N** | **coreml-gpu fp16** | 1579 | 5694 | **8.3** | **9.7** | 873 |
| RF-DETR-N | coreml-ane fp16 | 4157 | 15 | 10.8 | 12.9 | 817 |
| D-FINE-N | pytorch-cpu | 1771 | 88 | 61.5 | 77.3 | 908 |
| D-FINE-N | pytorch-mps | 1852 | 2611 | 24.0 | 30.4 | 985 |
| D-FINE-N | onnx-cpu | 124 | 31 | 25.4 | 32.6 | 567 |
| D-FINE-N | onnx-coreml | — | — | — | — | — |

Reading:
- Core ML fp16 is the fastest path for RF-DETR-N: 8.3 ms p50 on the GPU, 10.8 ms
  on the ANE — 3–5× PyTorch CPU (42.4 ms) and 3× faster than MPS (27.6 ms).
  fp32 Core ML halves that advantage (16.0 ms GPU) and gives the ANE nothing
  (31.5 ms, same as CPU_ONLY): the ANE needs fp16.
- ONNX Runtime is the slowest option for RF-DETR (46.6 ms CPU) and its CoreML EP
  makes it worse (67.0 ms) — the graph is partitioned, not handed over whole. It
  does have the cheapest load (89 ms) and smallest RSS (630 MB).
- D-FINE-N's ONNX CPU path (25.4 ms) beats its PyTorch CPU path (61.5 ms) by
  2.4×; it has no Core ML numbers (conversion fails, docs/SETUP.md) and ORT's
  CoreML EP rejects its graph.
- First-inference cost is where compile lands, and it moves: the same
  coreml-gpu fp16 bundle showed 5694 ms cold and 142 ms once macOS had cached
  the compiled model. Treat it as "up to ~6 s once per bundle", not per process.
- Load is ~4 s for anything that goes through the rfdetr package (checkpoint
  read plus MD5 check) and ~0.1 s for a bare ONNX session.
- Power figures in these files rest on 2–20 macmon samples (runs last 2.5–20 s);
  only the sustained runs below are worth quoting.

### 5-minute sustained, RF-DETR-N Core ML fp16 (same conditions)

Command: as above with `--sustained-s 300`. Raw:
`20260920-171746_rfdetr_n_coreml-gpu.json`,
`20260920-172252_rfdetr_n_coreml-ane.json` (298 macmon samples each).

| Backend | inferences | p50 / p95 ms | per-minute p50 | sys W | ANE W | GPU act. | CPU % | temp °C |
|---|---|---|---|---|---|---|---|---|
| coreml-gpu fp16 | 34471 | 8.4 / 10.1 | 8.31, 8.41, 8.49, 8.36, 8.35 | 42.2 | 0.18 | 0.81 | 16 | 66 → 77 |
| coreml-ane fp16 | 27528 | 10.8 / 11.5 | 10.81, 10.82, 10.81, 10.81, 10.80 | 31.1 | 1.41 | 0.27 | 13 | 73 → 60 |

- Neither path drifts over five minutes: GPU p50 stays within 0.2 ms, ANE within
  0.02 ms. No thermal throttling at 30 fps-equivalent load.
- The ANE run draws 11 W less at the package and cools the machine down
  (73 → 60 °C) while the GPU run heats it (66 → 77 °C). ANE power only registers
  in macmon on the ANE run (1.41 W vs 0.18 W), which is also how we know the ANE
  is actually being used — the fp32 bundle reported 0.0 W and CPU-level latency.
- These two runs back to back ran hotter than the short runs (the GPU run
  started at 66 °C); the absolute package watts include whatever else the
  machine was doing, so compare the two rows with each other, not with vendor
  figures.
