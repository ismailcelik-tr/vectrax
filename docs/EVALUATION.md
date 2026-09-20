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

## Detector accuracy (2c, 2026-09-20, git 6c4db5d, PyTorch CPU)

Command: `uv run benchmarks/detection_eval.py --detector <name>`. Raw:
`benchmarks/results/detection/20260920-1559*` … `-161007_*`. Every frame of
all 8 fixtures; precision and recall at score ≥ 0.5, AP50 over all scores.

Scoring rule: fixture GT labels the target objects only, so each fixture is
scored on the single class it annotates (6 × cup, exit_reentry person).
non_coco's food container has no COCO class and is scored class-agnostically;
that makes its precision uninterpretable — every unlabelled person, chair or
other cup in frame counts as a false positive — so only its recall compares.

Recall (%) at score ≥ 0.5:

| Fixture | YOLO26n | YOLO11n | RF-DETR-N | D-FINE-N |
|---|---|---|---|---|
| single_target | 25 | 58 | **96** | 14 |
| crossing_targets | 16 | 20 | **89** | 12 |
| near_targets | 11 | 30 | **95** | 13 |
| occlusion | 15 | 8 | **71** | 17 |
| exit_reentry (person) | 96 | 96 | 96 | 96 |
| fast_motion | 0 | 2 | **40** | 2 |
| non_coco (class-agnostic) | 8 | 16 | 39 | **63** |
| low_light | 15 | 15 | **88** | 27 |
| **mean** | 23 | 31 | **77** | 31 |

AP50 (%):

| Fixture | YOLO26n | YOLO11n | RF-DETR-N | D-FINE-N |
|---|---|---|---|---|
| single_target | 80 | 74 | **100** | 44 |
| crossing_targets | 58 | 49 | **94** | 51 |
| near_targets | 64 | 65 | **99** | 69 |
| occlusion | 48 | 21 | **81** | 51 |
| exit_reentry (person) | **96** | **96** | 95 | **96** |
| fast_motion | 11 | 11 | **49** | 30 |
| non_coco (class-agnostic) | 15 | 18 | 17 | **19** |
| low_light | 59 | 37 | **99** | 73 |
| **mean** | 54 | 46 | **79** | 54 |

Precision on the labelled classes is 88–100 % for all four, except
fast_motion, where every model has under 10 true positives (50–83 %).

Reading:
- RF-DETR-N is the only candidate that finds the cup reliably: 71–96 % recall
  on seven fixtures against 8–30 % for the others, mean AP50 79 vs 46–54.
- person is easy for everyone (96 % recall): exit_reentry separates nothing.
- Low recall next to much higher AP50 (D-FINE single_target 14 % vs 44 AP50)
  means the box is found but scored under 0.5 — calibration, not blindness. A
  per-class threshold would recover part of it; RF-DETR needs no such tuning.
- fast_motion is hard for all (motion blur); best is RF-DETR at 40 % recall.
- non_coco: D-FINE reaches 63 % recall, RF-DETR 39 %. The transparent
  container is detected as some COCO class; R1 cannot rely on the label.
- The YOLOs are AGPL reference only (R5) and are also the weakest here.

## Backend parity, RF-DETR-N and D-FINE-N (2c, 2026-09-20)

The exported graphs are decoded by our own code (`benchmarks/detectors.py`
`decode`), so each backend was scored on two fixtures against the PyTorch
adapter. Command: `uv run benchmarks/detection_eval.py --detector <d>
--backend <b> [--precision fp16]`. Raw: `benchmarks/results/detection/`.

| Detector, backend | single_target rec / AP50 | occlusion rec / AP50 / prec |
|---|---|---|
| RF-DETR-N pytorch-cpu | 96 / 100 | 71 / 81 / 95 |
| RF-DETR-N onnx-cpu | 96 / 100 | 71 / 81 / 94 |
| RF-DETR-N coreml-gpu fp16 | 96 / 100 | 71 / 81 / 94 |
| RF-DETR-N coreml-ane fp16 | 96 / 100 | 69 / 77 / 91 |
| D-FINE-N pytorch-cpu | 14 / 44 | 17 / 51 / 88 |
| D-FINE-N onnx-cpu | 15 / 46 | 18 / 52 / 88 |

- ONNX and Core ML GPU match PyTorch on RF-DETR to within one box.
- The Core ML ANE path drifts: 10 fewer true positives and 10 more false
  positives on occlusion (fp16 plus ANE arithmetic). Small, but it is the one
  backend whose numbers are not interchangeable with the others.
- D-FINE-N ONNX scores a little higher than its PyTorch adapter because the
  resize filter differs: the HF processor uses antialiased PIL bilinear, our
  adapter `cv2.INTER_LINEAR`. RF-DETR has no such gap — its own `predict()`
  resizes with `antialias=False`, which is what cv2 does.
