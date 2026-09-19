# Setup and install ledger

Every install made for VectraX outside the repo is recorded here, so
`scripts/uninstall.sh` can remove exactly that and nothing else.

## Inside the project (removed by deleting the folder)
| What | Where | Install |
|---|---|---|
| Python deps | `.venv/` | `uv sync` (all groups by default) |
| Brew snapshot, CVAT checkout | `tools/` (git-ignored) | — |
| Recordings, model weights | `data/`, `models/` (git-ignored) | — |

Dependency groups: default (runtime), `dev` (pytest, ruff), `ml` (torch,
onnxruntime, coremltools — benchmarks only), `eval` (TrackEval).

## Outside the project
| What | Install | Remove | Status |
|---|---|---|---|
| Homebrew formulae | `brew install ffmpeg macmon` | formulae in `tools/brew_added.txt` (diff vs `tools/brew_before.txt`) | installed: ffmpeg 9.0.2, macmon 0.8.2 + 9 deps |
| CVAT v2.76.0 (Docker) | `tools/cvat`, `docker compose up -d` | `docker compose down -v`, then images in `tools/docker_added.txt` (10 images, ~7 GB, all newly pulled) | pulled |
| uv wheel cache | automatic | `uv cache prune` (shared with other uv projects) | — |

Python 3.13 (Homebrew) was already installed; it is not removed.

## Known issues
- coremltools 9.0 is tested up to torch 2.7; installed torch is 2.14.
  PROVISIONAL — resolve in Phase 2 (pin torch or convert via ONNX).
- TrackEval's BURST module needs pycocotools; not used.
