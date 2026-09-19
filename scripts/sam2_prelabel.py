"""Pre-label a fixture with SAM 2 (not a Phase 2 detector candidate, so it
does not bias detector evaluation). Output needs human review in CVAT.

  uv run scripts/sam2_prelabel.py near_targets --step 10 --box 383,263,536,400 --box 662,276,790,425

Runs only on every --step-th frame (the CVAT keyframes; CVAT interpolates).
Writes data/fixtures/<name>.sam2.json: per keyframe, per object, [x1,y1,x2,y2]
of the mask (None when SAM 2 sees no object) and the mask area in pixels.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "data" / "fixtures"
CHECKPOINT = ROOT / "models" / "sam2" / "sam2.1_hiera_small.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
JPEG_QUALITY = 95
MIN_MASK_PX = 50


def _extract(video, out_dir, step):
    """Every step-th frame as consecutive JPEGs; returns their source frame numbers."""
    cap = cv2.VideoCapture(str(video))
    kept = []
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            return kept

        if i % step == 0:
            cv2.imwrite(str(out_dir / f"{len(kept):05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            kept.append(i)

        i += 1


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    if xs.size < MIN_MASK_PX:
        return None, int(xs.size)

    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], int(xs.size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--box", action="append", required=True, help="x1,y1,x2,y2 on frame 0; one per object")
    p.add_argument("--step", type=int, default=10)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    predictor = build_sam2_video_predictor(CONFIG, str(CHECKPOINT), device=device)
    boxes = [[float(v) for v in b.split(",")] for b in args.box]
    result = {}
    with tempfile.TemporaryDirectory() as tmp:
        kept = _extract(FIXTURES / f"{args.name}.mp4", Path(tmp), args.step)
        t0 = time.monotonic()
        with torch.inference_mode():
            state = predictor.init_state(video_path=tmp, offload_video_to_cpu=True)
            for obj_id, box in enumerate(boxes, start=1):
                predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, box=np.array(box))

            for frame_idx, obj_ids, logits in predictor.propagate_in_video(state):
                masks = (logits > 0).cpu().numpy()[:, 0]
                result[frame_idx] = {int(o): _bbox(m) for o, m in zip(obj_ids, masks, strict=True)}

    out = FIXTURES / f"{args.name}.sam2.json"
    out.write_text(json.dumps({"keyframes": kept, "device": device, "checkpoint": CHECKPOINT.name,
                               "objects": {str(i): [result[k][i] for k in range(len(kept))]
                                           for i in range(1, len(boxes) + 1)}}))
    print(f"{args.name}: {len(kept)} keyframes on {device} in {time.monotonic() - t0:.0f}s → {out}")


if __name__ == "__main__":
    main()
