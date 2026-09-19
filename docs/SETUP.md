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
