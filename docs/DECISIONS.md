# Decisions

ADR log. Newest last. Status: ACCEPTED, PROVISIONAL, SUPERSEDED.

## ADR-001 Camera capture via AVFoundation (2026-09-19, ACCEPTED)

**Decision:** Live Mac sources capture through AVFoundation (PyObjC).
OpenCV is used for image processing, not capture.

**Alternatives:** OpenCV `VideoCapture` (AVFoundation backend).

**Evidence:** `scripts/camera_probe.py`, results in
`benchmarks/results/probe/`. Both methods deliver 30 fps with no gaps
on MacBook and wired iPhone. Only AVFoundation exposes the sensor PTS,
which showed ~45–57 ms of frame age before Python sees the frame.
OpenCV hides it. AVFoundation also selects devices by unique ID; OpenCV
uses an index that shifts when Continuity Camera connects. CPU: AVF
4.4–6.9 %, OpenCV 7.2–8.7 %.

**Tradeoff:** More code (PyObjC delegate, dispatch queue); macOS only.
Acceptable: other platforms get their own CameraSource.

**Gotcha:** `startRunning` re-applies the session preset and overrides
`activeFormat`; `InputPriority` preset is unsupported on macOS. Set the
format after `startRunning` and verify delivered frame size.

## ADR-002 MacBook camera primary, wired iPhone secondary (2026-09-19, PROVISIONAL)

**Decision:** Baseline and fixtures use the built-in camera. Wired iPhone
(Continuity Camera) is the second source, proving source independence.
Wireless Continuity Camera is not used for evaluation.

**Evidence:** Wireless run: one 1.5 s stream stall (sensor PTS gap
1549 ms), 27.6 effective fps, p95 frame age 66 ms. Wired: no gaps,
p50 frame age 44.7 ms vs 56.8 ms on MacBook.

**Open:** iPhone PTS is translated to the Mac host clock; whether it marks
exposure or transmit time is unknown, so the wired advantage is
unverified. Revisit after the glass-to-glass measurement.

## ADR-003 Latency budget starts at sensor PTS (2026-09-19, ACCEPTED)

**Decision:** R3 (p95 ≤ 100 ms) is measured from the sensor PTS to
guidance. `FramePacket.capture_ns` is the sensor PTS on the monotonic
host clock; arrival time is stored separately.

**Alternatives:** Count from arrival in Python (hides 45–57 ms the
operator experiences); separate budgets for both.

**Consequence:** About 40 ms (MacBook) remains for processing, rendering
excluded. Every stage budget derives from this.

## ADR-004 CSRT as baseline propagator (2026-09-19, PROVISIONAL)

**Decision:** Phase 1 uses OpenCV CSRT behind the `Propagator` interface.
Score = NCC between the initial appearance and the current box, both
downsized to 32 px and blurred (σ 1.5) so small misalignment does not
read as appearance change.

**Alternatives:** KCF, MIL (weaker), NanoTrack, ViTTrack, DaSiamRPN
(need ONNX weights). Compared in a benchmark after Phase 1.

**Why first:** no model files, class-agnostic (R1), Apache-2.0 (R5).

**Known limits:** CSRT reports no confidence; NCC against the initial
template drops when the object rotates or changes pose. Cost per target
at 720p unmeasured; a `scale` option runs it downsized.

## ADR-005 Phase 1 threading: capture queue + main-thread tick (2026-09-19, ACCEPTED)

**Decision:** AVFoundation delivers frames on its dispatch queue into a
`LatestFrameBuffer` (capacity 1, drop-oldest). The main thread reads the
newest frame, runs the tracking tick, draws and pumps the OpenCV window.

**Why:** macOS requires GUI calls on the main thread. With no inference
worker yet, one consumer thread is the simplest correct design; a slow
tick drops frames instead of building a queue.

**Consequence:** tracking and rendering costs add up on one thread.
First smoke run (1 target, not a benchmark): tick→render ~14 ms, more
than tracking itself. Revisit when the inference worker arrives
(Phase 3) or if render cost blocks R3.

## ADR-006 Tracking on PipelineThread, UI renders the newest tick (2026-09-19, ACCEPTED)

Supersedes the single-consumer part of ADR-005.

**Decision:** Live mode runs read → process on `PipelineThread`; the main
thread draws the newest `Tick` from a capacity-1 buffer. Clip mode stays
single-threaded (frozen start, paced playback).

**Evidence:** macOS `cv2.waitKey(1)` and `cv2.pollKey()` both block ~14–16 ms
(render probe: draw 0.3 ms, imshow 0.9 ms, waitKey 13.7 ms). With 3 targets
tracking + waitKey exceeded the 33 ms frame period: 118/930 frames dropped,
queue wait p95 32.6 ms (benchmarks/results/latency/20260919-171137_3targets).
Smoke run after the split: queue wait p95 0.2 ms, 1 frame dropped.

**Consequence:** every frame is tracked and measured; the UI may skip
frames. Operator calls cross threads, so `TrackManager` and `Pipeline`
guard their pending queues with locks. `Metrics` is lock-protected;
capture→render counts rendered frames only.

## ADR-007 R3 split into system and display latency (2026-09-19, ACCEPTED)

**Decision:** R3a p95 sensor PTS → guidance ≤ 100 ms; R3b p95 sensor PTS →
render ≤ 120 ms. Until guidance exists (Phase 5), R3a is read from
capture→tracked, since guidance runs right after tracking on the same thread.

**Why:** after ADR-006 the two diverge: tracking finishes ~79 ms after the
sensor, the OpenCV window shows it ~20 ms later (waitKey cycle). Future
PTZ control consumes guidance, not the display. The display path depends
on OpenCV HighGUI, which a later UI replaces.
