# Roadmap

Phases: docs/SPEC.md. This file tracks the current phase and open items.

## Phase 0 — in progress
- [x] Environment inspection (docs/ENVIRONMENT.md)
- [x] Camera probe, ADR-001..003 (docs/PERFORMANCE.md)
- [x] Fixtures: single_target, crossing_targets, near_targets, occlusion, exit_reentry, fast_motion
- [x] Fixture non_coco: transparent food container (R1 test)
- [x] Evening (dark room): glass-to-glass MacBook + iPhone wired → ADR-002 closed
- [x] Evening: low_light fixture (SAM 2 pre-label, owner reviewed)
- [ ] CVAT up, annotate fixtures, export MOT 1.1

## Phase 1 — walking skeleton (approved 2026-09-19, closed 2026-09-19)
Steps, one commit each, test first:
- [x] clock, frames
- [x] LatestFrameBuffer, sources: FileSource (sidecar timestamps), MacCamera
- [x] Kalman (constant velocity), TrackState transitions, TrackQuality
- [x] CsrtPropagator (ADR-004 PROVISIONAL), TrackManager, events
- [x] metrics, pipeline (REALTIME / DETERMINISTIC), headless `--init-boxes`
- [x] OpenCV UI (ADR-005 threading) — owner verified; issues logged below
- [x] Pulled from Phase 6: `--record` session (video, stamps, operator.jsonl);
      headless replay reproduces it; `--render-out` review video
- [x] Latency 1 and 3 targets → PERFORMANCE.md (R3: 1 target ok, 3 miss)

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

### Live sessions 2026-09-19 (data/sessions/, replayed headless)
| Session | Targets | capture→render p50/p95 ms | Finding |
|---|---|---|---|
| occlusion_live | 1 cup | 78.3 / 88.7 | Book hides cup (f449) → OCCLUDED, correct. Cup returns (~f510), box stuck by the hand → LOST f538. No reacquisition. |
| two_cups_live | 2 cups | 91.2 / 103.9 | 1st crossing OK. 2nd (f414–418): printed cup passes in front, id 2 jumps onto it with score 0.79 → both ids on one cup. Score cannot tell two white cups apart; manager allows two tracks on one object. |
| pupils_live | 2 eyes + cup | 100.4 / 131.5 | Eyes tracked well until the cup crosses the face (f858, f948) → OCCLUDED → LOST. Boxes cover the whole eye, not the pupil. |

Latency figures include frames before selection; not a controlled benchmark.

Root causes, in order of evidence:
1. Bug: Kalman prediction during OCCLUDED runs away (x −2109) and size goes
   negative (w −72). Needs velocity damping and size/position clamps.
2. No reacquisition after occlusion (occlusion_live, pupils_live). CSRT keeps
   learning the occluder.
3. Identity hijack (two_cups_live): grayscale NCC score too weak; no rule
   against two tracks on one object.
4. TRACKING/DEGRADED flapping around good_quality: needs hysteresis.

Fixed 1 and 4 (coast damping τ 0.3 s, clamps, hysteresis 0.1). Replay of the
same sessions: state changes 90→59 (two_cups), 82→45 (pupils); worst box
escape 2109→122 px; no negative sizes. Outcomes unchanged: 2 and 3 remain.
5. R3 (p95 ≤ 100 ms) met with 1 target only; ~8–10 ms CSRT per target plus
   ~14 ms render.

Acceptance: tests green, ruff clean; owner verified UI on camera;
deterministic replay reproduces runs (test + session replay); latency
measured. Open issues carried forward: reacquisition, identity hijack,
3-target latency.

## After Phase 1 (decided 2026-09-19: latency → annotation → Phase 2)
- [x] Latency: parallel propagators, PipelineThread (ADR-006), R3 split
      (ADR-007). R3a and R3b met for 1 and 3 targets. Format experiment:
      no gain; frame age depends on lighting.
- [x] Evening: glass-to-glass, low_light fixture
- [x] CVAT up; fixtures annotated (MOT 1.1): occlusion, single_target,
      crossing_targets by owner; near_targets, exit_reentry, fast_motion,
      non_coco pre-labeled with SAM 2 (owner reviewed near_targets)
- [x] Owner reviewed all 7; exported to data/fixtures/*.gt.zip (CVAT stopped)
- [ ] Phase 2 (approved 2026-09-19), see below

## Phase 2 — evidence-based model selection
2a [done] Evaluation harness (no downloads): MOT GT loader; tracking metrics
   (success IoU≥0.5, state correctness vs visible/partial/absent, recovery
   after absence, identity hijack); detection P/R/AP50 for cup/person and
   class-agnostic recall for non_coco; GT frame-0 boxes as operator input.
   CSRT baseline on all 8 fixtures.
2b Propagators: CSRT, KCF, ViTTrack (OpenCV Zoo, Apache-2.0), NanoTrack
   (only if licence verifies). Accuracy + ms/target → ADR-008.
2c Detectors: YOLO26n, YOLO11n (AGPL, reference only); RF-DETR-N, D-FINE-N
   (Apache-2.0, runtime candidates). Backends: PyTorch CPU/MPS, ONNX Runtime
   CPU/CoreML EP, Core ML. Load/compile, p50/p95, CPU/GPU/ANE, RSS, 5-min
   sustained for top 2; fixture accuracy. COCO val only if results look
   wrong. → ADR-009.
Out of scope: open-vocabulary detectors (AGPL or too heavy); R1
reacquisition goes through appearance (Phase 4). SAM 2 is not a runtime
candidate (~1.1 s/frame on MPS), so SAM 2 pre-labels do not bias results;
SAM-family candidates would be scored on owner-labelled fixtures only.
