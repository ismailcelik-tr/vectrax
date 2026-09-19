# Environment (inspected 2026-09-19)

| Item | Value |
|---|---|
| Machine | MacBook Pro, Apple M5, 16 GB |
| OS | macOS 27.0 (26A428) |
| Python | 3.13.14 (project), uv 0.11.22 |
| Xcode | installed, license accepted |
| Docker | Docker Desktop, running |

## Cameras
| Device | Unique ID | Formats |
|---|---|---|
| MacBook Pro camera | `6C707041-05AC-0011-0002-000000000001` | up to 1920x1080 @ 30 |
| iPhone "Fox" (Continuity) | `D79D41AE-D87C-4A73-9667-8DFF00000001` | up to 1920x1440 @ 60; 720p @ 60 |

Center Stage: disabled. Terminal has camera permission.

## Runtimes (import-verified)
| Package | Version | Notes |
|---|---|---|
| opencv-contrib-python | 5.0.0 | CSRT, KCF, MIL, Nano, Vit, DaSiamRPN present |
| pyobjc AVFoundation/CoreMedia/libdispatch | 12.2.2 | |
| torch | 2.14.0 | MPS available |
| onnxruntime | 1.30.0 | CoreMLExecutionProvider available |
| coremltools | 9.0 | tested only up to torch 2.7 |
| TrackEval | 1.0.dev1 (git 12c8791) | |

coremltools has no Python 3.14 wheel; this pinned the project to 3.13.

Measured camera timing: docs/PERFORMANCE.md.
