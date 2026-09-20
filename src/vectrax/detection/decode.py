"""Preprocessing and DETR decoding for the exported detector graph.

The graph stops at the model outputs, so the box decode lives here: per-class
sigmoid, top-k query/class pairs, cxcywh to image coordinates. Mirrors rfdetr
`PostProcess`; one query may report several classes.
"""

import cv2
import numpy as np

from vectrax.tracking.geometry import Box
from vectrax.tracking.observation import Observation, Origin

__all__ = ["preprocess", "to_observations"]

TOP_K = 300  # query/class pairs the head keeps (rfdetr PostProcess.num_select)
SIGMOID_CLIP = 88.0  # exp overflow guard in float32
UINT8_MAX = 255.0


def preprocess(frame: np.ndarray, size: int, mean: tuple[float, float, float] | None,
               std: tuple[float, float, float] | None) -> np.ndarray:
    """BGR frame → 1xCxHxW float32. `mean` None rescales to [0, 1] only."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    chw = resized.transpose(2, 0, 1).astype(np.float32) / UINT8_MAX
    if mean is not None:
        chw = (chw - np.array(mean, dtype=np.float32)[:, None, None]) / \
            np.array(std, dtype=np.float32)[:, None, None]

    return chw[None]


def to_observations(boxes: np.ndarray, logits: np.ndarray, labels: dict[int, str],
                    size: tuple[int, int], frame_id: int, capture_ns: int,
                    score_min: float, top_k: int = TOP_K) -> list[Observation]:
    """`boxes` are normalized cxcywh, `size` is (width, height) of the frame.

    Class indices missing from `labels` are dropped: the graph carries a
    background slot the model never means as an object.
    """
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -SIGMOID_CLIP, SIGMOID_CLIP)))
    flat = scores.ravel()
    classes = scores.shape[1]
    keep = min(top_k, flat.size)
    order = np.argpartition(-flat, keep - 1)[:keep]

    width, height = size
    observations = []
    for i in order:
        score = float(flat[i])
        label = labels.get(int(i) % classes)
        if score < score_min or label is None:
            continue

        cx, cy, w, h = (float(v) for v in boxes[int(i) // classes])
        x = min(max((cx - w / 2) * width, 0.0), width)
        y = min(max((cy - h / 2) * height, 0.0), height)
        right = min(max((cx + w / 2) * width, 0.0), width)
        bottom = min(max((cy + h / 2) * height, 0.0), height)
        observations.append(Observation(
            frame_id=frame_id, capture_ns=capture_ns,
            box=Box.from_xywh_px(x, y, right - x, bottom - y, width, height),
            score=score, origin=Origin.DETECTOR, label=label))

    return observations
