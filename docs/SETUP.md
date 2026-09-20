# Setup and install ledger

Every install made for VectraX outside the repo is recorded here, so
`scripts/uninstall.sh` can remove exactly that and nothing else.

## Inside the project (removed by deleting the folder)
| What | Where | Install |
|---|---|---|
| Python deps | `.venv/` | `uv sync` (all groups by default) |
| Brew snapshot, CVAT checkout | `tools/` (git-ignored) | — |
| Detector library caches | `tools/ultralytics`, `tools/hf`, `tools/roboflow` | set by `benchmarks/detectors.py` |
| Recordings, model weights | `data/`, `models/` (git-ignored) | — |
| Exported detector graphs | `models/detectors/exported/` | `uv run benchmarks/export_detectors.py` |

Dependency groups: default (runtime), `dev` (pytest, ruff), `ml` (torch,
onnx, onnxruntime, coremltools, rfdetr, transformers — benchmarks only),
`reference` (ultralytics, AGPL-3.0 — benchmark reference only, R5), `eval` (TrackEval),
`annotate` (SAM 2 from facebookresearch/sam2 @ 2b90b9f, Apache-2.0; the PyPI
`sam2` package is an unofficial fork and is not used).
`opencv-python` (pulled by ultralytics and supervision) is overridden out
in `pyproject.toml`; it would overwrite `cv2` from opencv-contrib-python.

Model weights in `models/` (git-ignored):
| File | Source | SHA-256 |
|---|---|---|
| `trackers/vittrack_2023sep.onnx` (0.7 MB) | opencv/opencv_zoo (Apache-2.0) | `2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d` |
| `trackers/nanotrack_backbone_sim.onnx` (1.1 MB) | HonglinChu/SiamTrackers nanotrackv2 (Apache-2.0) | `530bdd0cd00f19afab79a863e71ba71e3312395a5dc9151af675082bdaaa2fc4` |
| `trackers/nanotrack_head_sim.onnx` (0.7 MB) | same | `0d8c0637be849f092cc7236cae02e55c8b9455ebe37ba50601d6115db4247cd9` |
| `detectors/yolo26n.pt` (5.5 MB) | ultralytics/assets v8.4.0 (AGPL-3.0, reference only) | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| `detectors/yolo11n.pt` (5.6 MB) | same | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| `detectors/rf-detr-nano.pth` (366 MB) | storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth (Apache-2.0; MD5 matches rfdetr 1.10.1) | `d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d` |
| `detectors/dfine-nano-coco/` (15 MB) | huggingface.co/ustc-community/dfine-nano-coco @ 066438d (Apache-2.0) | `19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b` (model.safetensors) |
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

## Detector benchmark notes
- `benchmarks/detectors.py` pins `YOLO_CONFIG_DIR`, `HF_HOME` and `RF_HOME`
  under `tools/` and creates the directories; ultralytics falls back to
  `/tmp/Ultralytics` if its directory is missing. Verified: only
  `tools/ultralytics/Ultralytics/settings.json` is written, nothing in the
  home directory.
- RF-DETR Nano downloads no DINOv2 weights: patch size 16 and its positional
  encoding size disable `load_dinov2_weights`. It re-checks the local
  checkpoint's MD5 on every load. Verified with `HF_HUB_OFFLINE=1`.
- D-FINE and the YOLOs load from `models/detectors/` only; no downloads.
- Exports (`benchmarks/export_detectors.py`, artifacts git-ignored):
  `rfdetr_n --format onnx` → `rfdetr-nano.onnx` (108 MB, input 1x3x384x384,
  outputs `dets` 1x300x4 and `labels` 1x300x91); `--format coreml
  --precision fp16|fp32` → `rfdetr-nano_fp16/_fp32.mlpackage`;
  `dfine_n --format onnx` → `dfine_n.onnx` (input 1x3x640x640, outputs
  `logits` and `pred_boxes`). Each directory also gets a `labels.json` (class
  index → COCO name) so inference needs no model stack.
- `torch.onnx.export` needs `dynamo=False`: torch 2.14's default exporter
  imports `onnxscript`, which is not installed. The YOLOs are not exported
  (AGPL, reference only).

## Known issues
- coremltools 9.0 is tested up to torch 2.7; installed torch is 2.14. It
  converts RF-DETR (warning only) but fails on D-FINE, so D-FINE has no Core ML
  numbers — PROVISIONAL. TorchScript frontend: `TypeError: only 0-dimensional
  arrays can be converted to Python scalars` (ops.py `_cast`). torch.export
  frontend after `run_decompositions({})`: `NotImplementedError: Unsupported fx
  node _is_all_true`. A fix means pinning torch < 2.12 in a separate
  environment; not done, since D-FINE lost on accuracy anyway.
- ONNX Runtime's CoreML EP fails on the D-FINE graph:
  `CoreML static output shape ({1,1,1,300,300}) and inferred shape ({1,300})
  have different ranks`. ORT CPU works.
- TrackEval's BURST module needs pycocotools; not used.
