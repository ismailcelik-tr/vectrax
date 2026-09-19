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
