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

## Propagator comparison (2b, 2026-09-19, git 5d72a1c)

Raw: `benchmarks/results/tracking/20260919-2013*` … `-201510_*`. The
`git_dirty: true` flags in four files come from untracked result files of
the previous run, not code changes (check fixed afterwards).
ms = tracking per frame for all targets, p50, when tracks are alive.

Success (IoU ≥ 0.5) per fixture:

| Fixture | CSRT | KCF | ViT | ViT+NCC | Nano | Nano+NCC |
|---|---|---|---|---|---|---|
| single_target | 80 | 44 | 77 | 69 | **94** | 93 |
| crossing_targets | 39 | 25 | 16 | 15 | **40** | 29 |
| near_targets | 39 | 38 | 47 | 46 | **58** | 50 |
| occlusion | 8 | 9 | 8 | 9 | **13** | 9 |
| exit_reentry | 32 | 36 | 57 | 29 | **76** | 36 |
| fast_motion | 13 | 11 | 10 | 10 | **21** | 20 |
| non_coco | 51 | 5 | 40 | 41 | **52** | 47 |
| low_light | 65 | 15 | 81 | 73 | **99** | 96 |
| **mean** | 41 | 23 | 42 | 37 | **57** | 48 |

| | CSRT | KCF | ViT | ViT+NCC | Nano | Nano+NCC |
|---|---|---|---|---|---|---|
| ms per target (approx.) | ~10 | ~2.5 | 2.5–7 | 2.5–7 | ~3 | ~3 |
| hijack frames (all fixtures) | 0 | 0 | 0 | 0 | **286** | 0 |
| false-visible frames | 306 | 0 | 344 | 0 | **539** | 0 |
| recoveries (of 5) | 0 | 0 | 2 | 0 | 4 | 0 |

Reading:
- NanoTrack tracks best and costs ~3× less than CSRT.
- Native scores (ViT, Nano) rarely drop when the target is gone. That keeps
  the tracker searching (Nano re-finds 4/5) but also makes it follow
  something else while reporting it visible, and swap identity on
  crossing_targets (279 frames). NCC removes both at the cost of recovery.
- No variant reports partial occlusion as DEGRADED (0–28 %): open issue in
  the quality model, not the propagator.
