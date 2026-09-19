# Roadmap

Phases: docs/SPEC.md. This file tracks the current phase and open items.

## Phase 0 — in progress
- [x] Environment inspection (docs/ENVIRONMENT.md)
- [x] Camera probe, ADR-001..003 (docs/PERFORMANCE.md)
- [x] Fixtures: single_target, crossing_targets, near_targets, occlusion, exit_reentry, fast_motion
- [x] Fixture non_coco: transparent food container (R1 test)
- [ ] Evening (dark room): glass-to-glass MacBook + iPhone wired → close ADR-002
- [ ] Evening: low_light fixture
- [ ] CVAT up, annotate fixtures, export MOT 1.1

## Phase 1 — walking skeleton (approved 2026-09-19)
Steps, one commit each, test first:
- [x] clock, frames
- [x] LatestFrameBuffer, sources: FileSource (sidecar timestamps), MacCamera
- [x] Kalman (constant velocity), TrackState transitions, TrackQuality
- [x] CsrtPropagator (ADR-004 PROVISIONAL), TrackManager, events
- [x] metrics, pipeline (REALTIME / DETERMINISTIC), headless `--init-boxes`
- [x] OpenCV UI (ADR-005 threading) — owner verification pending
- [x] Pulled from Phase 6: `--record` session (video, stamps, operator.jsonl);
      headless replay reproduces it; `--render-out` review video
- [ ] Latency 1 and 3 targets → PERFORMANCE.md

Acceptance: tests green, ruff clean; owner verifies UI; deterministic
single_target run reproducible; latency measured, not claimed.
REACQUIRING arrives in Phase 4.

### Observed on fixtures (CSRT baseline, single_target, 1 target)
Headless, init box 795,297,113,143. Not yet in PERFORMANCE.md (no GT).
- track_ms p50/p95: 8.1/9.2 ms at scale 1.0; 6.0/6.8 ms at 0.5.
- Scale 0.5: quality hovers near good_quality → TRACKING/DEGRADED flapping.
  Needs hysteresis.
- Box does not grow as the cup approaches the camera (CSRT scale drift).
- Frames 318–334: box drifts onto the wall; score catches it, but state
  reads OCCLUDED while the cup is visible. Propagator alone cannot tell
  occlusion from drift; detector/reacquisition must (Phase 3–4).

## Next after Phase 1
- Propagator benchmark: CSRT vs NanoTrack vs ViTTrack on annotated fixtures.
- State hysteresis for TRACKING/DEGRADED.
