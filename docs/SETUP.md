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
onnxruntime, coremltools — benchmarks only), `eval` (TrackEval),
`annotate` (SAM 2 from facebookresearch/sam2 @ 2b90b9f, Apache-2.0; the PyPI
`sam2` package is an unofficial fork and is not used).

Model weights in `models/` (git-ignored):
| File | Source | SHA-256 |
|---|---|---|
| `trackers/vittrack_2023sep.onnx` (0.7 MB) | opencv/opencv_zoo (Apache-2.0) | `2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d` |
| `trackers/nanotrack_backbone_sim.onnx` (1.1 MB) | HonglinChu/SiamTrackers nanotrackv2 (Apache-2.0) | `530bdd0cd00f19afab79a863e71ba71e3312395a5dc9151af675082bdaaa2fc4` |
| `trackers/nanotrack_head_sim.onnx` (0.7 MB) | same | `0d8c0637be849f092cc7236cae02e55c8b9455ebe37ba50601d6115db4247cd9` |
| `sam2/sam2.1_hiera_small.pt` (184 MB) | dl.fbaipublicfiles.com/segment_anything_2/092824/ | `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38` |

## Outside the project
| What | Install | Remove | Status |
|---|---|---|---|
| Homebrew formulae | `brew install ffmpeg macmon` | formulae in `tools/brew_added.txt` (diff vs `tools/brew_before.txt`) | installed: ffmpeg 9.0.2, macmon 0.8.2 + 9 deps |
| CVAT v2.76.0 (Docker) | `tools/cvat`, `docker compose up -d` | `docker compose down -v`, then images in `tools/docker_added.txt` (10 images, ~7 GB, all newly pulled) | running on demand |
| uv wheel cache | automatic | `uv cache prune` (shared with other uv projects) | — |

Python 3.13 (Homebrew) was already installed; it is not removed.

## CVAT notes
- `tools/cvat/docker-compose.override.yml` (git-ignored, regenerate if lost):
  every service `restart: "no"` (no auto-start with Docker); traefik ports
  bound to `127.0.0.1` only (R6). Open via `http://localhost:8080`; the
  router matches Host `localhost`, so `127.0.0.1` returns 404.
- Admin credentials: `tools/cvat_admin.txt` (mode 600, random password).
- `cvat/server` is amd64-only; runs under Rosetta. Task creation ~2 s per
  clip, export ~3 s. Uses ~5 GB RAM: stop with
  `docker compose -f tools/cvat/docker-compose.yml stop` when not annotating.
- Tasks and export: `scripts/cvat_tasks.py`.

## Known issues
- coremltools 9.0 is tested up to torch 2.7; installed torch is 2.14.
  PROVISIONAL — resolve in Phase 2 (pin torch or convert via ONNX).
- TrackEval's BURST module needs pycocotools; not used.
