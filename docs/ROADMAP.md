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
- [ ] CsrtPropagator (ADR-004 PROVISIONAL), TrackManager, events
- [ ] metrics, pipeline (REALTIME / DETERMINISTIC), headless `--init-boxes`
- [ ] OpenCV UI (ADR-005 threading) — owner verifies on camera
- [ ] Latency 1 and 3 targets → PERFORMANCE.md

Acceptance: tests green, ruff clean; owner verifies UI; deterministic
single_target run reproducible; latency measured, not claimed.
REACQUIRING arrives in Phase 4.
