# Evaluation

Quality on annotated fixtures (docs/FIXTURES.md). Speed lives in
PERFORMANCE.md.

## Metrics (`src/vectrax/evaluation/`)
Each GT track is selected at its first annotated frame with its GT box and
scored against the pipeline track it created. Frames before selection are
not scored.

| Metric | Meaning |
|---|---|
| success | GT present, tracker claims visible (INITIALIZING/TRACKING/DEGRADED), IoU ≥ 0.5 |
| onTarget | same, but only the box center must lie inside the GT box (right place, any size) |
| falseVis | GT absent, tracker claims visible |
| hijack | tracker box on another GT target (IoU ≥ 0.5) instead of its own |
| partial→deg | share of GT-partial frames the tracker reports as DEGRADED |
| absent→hid | share of GT-absent frames reported OCCLUDED or LOST |
| recoveries | per reappearance, frames until success again; None = never |

Command: `uv run benchmarks/tracking_eval.py --propagator <name>`.
Raw: `benchmarks/results/tracking/`.

## CSRT baseline (2026-09-19, git 7091752, raw `20260919-201105_csrt.json`)

| Fixture | success | onTarget | falseVis | hijack | partial→deg | absent→hid | recoveries |
|---|---|---|---|---|---|---|---|
| single_target | 80% | 92% | 0 | 0 | – | – | – |
| crossing_targets | 39% | 52% | 0 | 0 | 5% | 100% | [None] |
| near_targets | 39% | 68% | 0 | 0 | 0% | – | – |
| occlusion | 8% | 9% | 306 | 0 | 22% | 0% | [None ×3] |
| exit_reentry | 32% | 32% | 0 | 0 | – | 100% | [None ×2] |
| fast_motion | 13% | 17% | 0 | 0 | – | – | – |
| non_coco | 51% | 96% | 0 | 0 | – | – | – |
| low_light | 65% | 83% | 0 | 0 | – | – | – |

Reading (verified visually on occlusion and near_targets):
- Scale: onTarget ≫ success (non_coco 96 vs 51) — CSRT stays on the
  object but its box does not follow size changes.
- occlusion: the box locks onto the book edge when the cup hides (~f50) and
  reports TRACKING for all 306 absent frames. The NCC score (blurred 32 px
  grayscale) cannot tell a white cup from the book's light edge.
- No recovery after any absence (5 reappearances): no reacquisition.
- fast_motion: lost within the first seconds.
- partial→deg near 0: the quality score does not see partial occlusion.
- hijack 0: tracks are lost, not swapped, on these clips.
