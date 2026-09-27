<div align="center">

# VectraX

Real-time, operator-driven visual tracking.

[![License](https://img.shields.io/github/license/ismailcelik-tr/vectrax)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-macOS-000000?logo=apple&logoColor=white)](#platform-support)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Last commit](https://img.shields.io/github/last-commit/ismailcelik-tr/vectrax)](https://github.com/ismailcelik-tr/vectrax/commits/main)

[![English](https://img.shields.io/badge/English-555555?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2MCAzMCI%2BPGNsaXBQYXRoIGlkPSJzIj48cGF0aCBkPSJNMCwwdjMwaDYwVjB6Ii8%2BPC9jbGlwUGF0aD48Y2xpcFBhdGggaWQ9InQiPjxwYXRoIGQ9Ik0zMCwxNWgzMHYxNXp2MTVIMHpIMFYwelYwaDMweiIvPjwvY2xpcFBhdGg%2BPGcgY2xpcC1wYXRoPSJ1cmwoI3MpIj48cGF0aCBkPSJNMCwwdjMwaDYwVjB6IiBmaWxsPSIjMDEyMTY5Ii8%2BPHBhdGggZD0iTTAsMEw2MCwzME02MCwwTDAsMzAiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSI2Ii8%2BPHBhdGggZD0iTTAsMEw2MCwzME02MCwwTDAsMzAiIGNsaXAtcGF0aD0idXJsKCN0KSIgc3Ryb2tlPSIjQzgxMDJFIiBzdHJva2Utd2lkdGg9IjQiLz48cGF0aCBkPSJNMzAsMHYzME0wLDE1aDYwIiBzdHJva2U9IiNmZmYiIHN0cm9rZS13aWR0aD0iMTAiLz48cGF0aCBkPSJNMzAsMHYzME0wLDE1aDYwIiBzdHJva2U9IiNDODEwMkUiIHN0cm9rZS13aWR0aD0iNiIvPjwvZz48L3N2Zz4%3D&logoSize=auto)](README.md)
[![Türkçe](https://img.shields.io/badge/T%C3%BCrk%C3%A7e-555555?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMCAyMCI%2BPHJlY3Qgd2lkdGg9IjMwIiBoZWlnaHQ9IjIwIiBmaWxsPSIjRTMwQTE3Ii8%2BPGNpcmNsZSBjeD0iMTAuMCIgY3k9IjEwIiByPSI1LjAiIGZpbGw9IiNmZmYiLz48Y2lyY2xlIGN4PSIxMS4yNSIgY3k9IjEwIiByPSI0LjAiIGZpbGw9IiNFMzBBMTciLz48cG9seWdvbiBwb2ludHM9IjE1LjkxNywxMC4wMDAgMTcuNjQ0LDkuNDM5IDE3LjY0NCw3LjYyMiAxOC43MTIsOS4wOTIgMjAuNDM5LDguNTMxIDE5LjM3MiwxMC4wMDAgMjAuNDM5LDExLjQ2OSAxOC43MTIsMTAuOTA4IDE3LjY0NCwxMi4zNzggMTcuNjQ0LDEwLjU2MSIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg%3D%3D&logoSize=auto)](README.tr.md)

</div>

The operator draws a box around any object; VectraX follows it frame by
frame, reports how confident it is, and keeps working on objects no
detector knows. Intended for monitoring, robotics, inspection and research.
v1 never moves hardware.

## Status

Phase 3 of 7 is closed: a detector runs beside the trackers and can raise
a track's confidence. Next is Phase 4, occlusion and reacquisition: a
target that disappears and comes back is not re-found yet. Prediction and
on-screen guidance come in Phase 5. Progress: [docs/ROADMAP.md](docs/ROADMAP.md).

## How it works

```
Camera / file → LatestFrameBuffer (drop-oldest) → pipeline tick, every frame:
  1. propagate each track
  2. fuse finished detector results (late-result fusion)
  3. TrackManager: associate, update quality, run the state machine
  4. publish TrackSnapshot → UI, recorder, metrics

InferenceWorker (own thread, every Nth frame) → results → step 2.
The tick never waits on inference.
```

- **Propagator:** one class-agnostic single-object tracker per target:
  OpenCV NanoTrack, with our NCC appearance score driving quality (ADR-008).
- **Detector:** RF-DETR Nano on Core ML fp16 (ADR-009). It supplies
  evidence only: a matching detection can lift a track, a missing one
  never demotes it (ADR-010).
- **TrackManager:** the only component that creates, transitions or
  deletes tracks. One Kalman filter per track. States: INITIALIZING,
  TRACKING, DEGRADED, OCCLUDED, LOST, PAUSED, STOPPED.
- **Run modes:** realtime drops frames and runs inference asynchronously;
  deterministic processes every frame and reproduces a run exactly. All
  time comes from an injected monotonic clock.

Design decisions and their evidence: [docs/DECISIONS.md](docs/DECISIONS.md).

## Platform support

Frames enter through the `CameraSource` interface; tracking never knows
which source produced a frame. New cameras are new sources, not changes to
the tracker.

| Layer | Today | Later ([SPEC](docs/SPEC.md)) |
|---|---|---|
| Camera source | macOS cameras (AVFoundation) | USB (UVC), RTSP |
| File source | video clips | — |
| Detector backend | Core ML | CUDA / TensorRT, edge devices |

## Requirements

- Today: macOS on Apple Silicon. Development machine:
  [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
- Python 3.13 (coremltools has no 3.14 wheel) and [uv](https://docs.astral.sh/uv/).
- macOS camera: grant the terminal camera access; turn Center Stage off.

## Setup

```sh
git clone https://github.com/ismailcelik-tr/vectrax.git
cd vectrax
uv sync
```

`uv sync` installs every dependency group, including torch and the
AGPL-licensed `ultralytics`, which is a benchmark reference only. Groups:
[docs/SETUP.md](docs/SETUP.md).

Model weights are not in the repository (`models/` is git-ignored). Sources
and SHA-256 checksums: [docs/SETUP.md](docs/SETUP.md).

- **Tracker (required):** `models/trackers/nanotrack_backbone_sim.onnx`
  and `models/trackers/nanotrack_head_sim.onnx`.
- **Detector (optional, for `--detect`):** place
  `models/detectors/rf-detr-nano.pth`, then export it to Core ML:

  ```sh
  uv run benchmarks/export_detectors.py --detector rfdetr_n --format coreml
  ```

  Output: `models/detectors/exported/rfdetr_n/rfdetr-nano_fp16.mlpackage`.

## Usage

```sh
# Live camera; drag a box around each target
uv run vectrax --source camera:MacBook

# With the detector, recording the session to data/sessions/NAME
uv run vectrax --source camera:MacBook \
  --detect models/detectors/exported/rfdetr_n/rfdetr-nano_fp16.mlpackage \
  --record NAME

# Video file; opens frozen on frame 0 so targets can be drawn
uv run vectrax --source file:path/to/clip.mp4

# Headless, deterministic, per-frame JSONL
uv run vectrax --source file:clip.mp4 --headless \
  --init-boxes 320,180,120,90 --out run.jsonl

# Replay a recorded session and render a review video
uv run vectrax --source file:data/sessions/NAME/video.mp4 --headless \
  --render-out review.mp4
```

On macOS, `camera:NAME` matches part of the device name or its unique ID. Headless
runs take operator input from the first that exists: `--init-boxes`, the
session's `operator.jsonl`, the clip's `<clip>.init.json`.

Other flags: `--scale` (propagator downscale), `--detect-stride` (detect
every Nth frame), `--metrics-out` (latency summary JSON). See
`uv run vectrax --help`.

### Keys

| Key | Action |
|---|---|
| drag | select a new target |
| click, `1`–`9` | focus a track |
| `c` | clear focus |
| `p` | pause / resume the focused track |
| `s` | stop the focused track |
| `x` | remove the focused track |
| `r`, then drag | reselect the focused track |
| space | freeze / unfreeze (file source only) |
| `q`, Esc | quit |

## Development

```sh
uv run pytest
uv run ruff check .
VECTRAX_CAMERA=1 uv run pytest tests/test_mac_camera.py  # needs the camera
```

Rules that shape the code ([CLAUDE.md](CLAUDE.md)):

- Algorithmic code starts with a failing test.
- The real-time path has no database, no network, and never blocks on
  inference.
- Every performance or quality number is measured in this repo and cites
  its command and raw output.

## Benchmarks and evaluation

| Script | Measures |
|---|---|
| `benchmarks/tracking_eval.py` | tracking accuracy on annotated fixtures |
| `benchmarks/live_latency.py` | live camera latency, 1–3 targets |
| `benchmarks/detection_eval.py` | detector precision / recall on fixtures |
| `benchmarks/detector_latency.py` | detector latency, memory, power per backend |

Raw results: `benchmarks/results/`. Numbers:
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) (speed),
[docs/EVALUATION.md](docs/EVALUATION.md) (accuracy).

Fixtures are clips recorded on the development camera and annotated in
CVAT. They live in `data/fixtures/`, which is git-ignored, so they are not
distributed. Recording and annotation: [docs/FIXTURES.md](docs/FIXTURES.md).

## Project layout

```
src/vectrax/
  sources/      CameraSource interface; macOS camera, file
  tracking/     propagators, Kalman, quality, states, association, TrackManager
  detection/    Core ML detector, inference worker
  evaluation/   ground-truth loader, metrics
  ui/           OpenCV window
  pipeline.py   pipeline tick, run modes
  recording.py  session recording
benchmarks/     measurements
scripts/        camera probe, fixture recording, CVAT, SAM 2 pre-labels
tests/
docs/
```

## Documentation

| File | Contents |
|---|---|
| [SPEC.md](docs/SPEC.md) | requirements, architecture, phases |
| [ROADMAP.md](docs/ROADMAP.md) | current phase, open items |
| [DECISIONS.md](docs/DECISIONS.md) | ADR log |
| [PERFORMANCE.md](docs/PERFORMANCE.md) | measured latency and throughput |
| [EVALUATION.md](docs/EVALUATION.md) | measured accuracy |
| [FIXTURES.md](docs/FIXTURES.md) | ground-truth clips |
| [SETUP.md](docs/SETUP.md) | install ledger, model weights, known issues |
| [ENVIRONMENT.md](docs/ENVIRONMENT.md) | development machine |

## Privacy

Recordings, fixtures and weights stay on the machine: `data/`, `models/`
and `tools/` are git-ignored. Record only yourself and people who consent.

## Scope

v1 does not control a physical camera. Weapon or engagement logic is out
of scope ([docs/SPEC.md](docs/SPEC.md), Non-goals).

## Uninstall

`scripts/uninstall.sh` removes every install recorded in
[docs/SETUP.md](docs/SETUP.md), including `.venv/`. It does not delete the
project folder.

## License

[Apache License 2.0](LICENSE).

Model weights and third-party packages keep their own licenses
([docs/SETUP.md](docs/SETUP.md)). `ultralytics` (AGPL-3.0) is a benchmark
reference only, never a runtime dependency.
