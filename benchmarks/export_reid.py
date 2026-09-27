"""Export DINOv2-S, the Re-ID embedder, to Core ML fp16.

  uv run benchmarks/export_reid.py

Input `pixels`: batch 1, 2, 4 or 8 of 224x224 RGB, ImageNet-normalized. Output `embedding`: the
L2-normalized CLS embedding (384). Lands in models/reid/exported/ (git-ignored).

coremltools 9 cannot convert the positional-embedding interpolation traced under torch 2.14 (the
`_cast` error in docs/SETUP.md). The input size is fixed, so the table interpolated for 224 is
computed once and baked in; PyTorch output is unchanged by it.
"""

import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "models" / "reid" / "dinov2-small"
OUT = ROOT / "models" / "reid" / "exported" / "dinov2s_fp16.mlpackage"
SIDE = 224
BATCHES = (1, 2, 4, 8)
PARITY_BATCH = 4
MIN_COSINE = 0.999


class _Embedder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixels):
        return torch.nn.functional.normalize(self.model(pixel_values=pixels).pooler_output, dim=1)


def _fixed_size(model):
    emb = model.embeddings
    with torch.no_grad():
        tokens = emb.patch_embeddings(torch.zeros(1, 3, SIDE, SIDE))
        table = emb.interpolate_pos_encoding(torch.cat([emb.cls_token, tokens], 1), SIDE, SIDE).clone()

    emb.interpolate_pos_encoding = lambda embeddings, height, width: table
    return model


def main():
    reference = _Embedder(AutoModel.from_pretrained(SOURCE).eval()).eval()
    fixed = _Embedder(_fixed_size(AutoModel.from_pretrained(SOURCE).eval())).eval()
    x = torch.rand(PARITY_BATCH, 3, SIDE, SIDE)
    with torch.no_grad():
        expected = reference(x).numpy()
        traced = torch.jit.trace(fixed, x[:1])

    t0 = time.perf_counter()
    shapes = ct.EnumeratedShapes(shapes=[(b, 3, SIDE, SIDE) for b in BATCHES], default=(1, 3, SIDE, SIDE))
    model = ct.convert(traced, inputs=[ct.TensorType(name="pixels", shape=shapes)],
                       outputs=[ct.TensorType(name="embedding")], compute_precision=ct.precision.FLOAT16,
                       minimum_deployment_target=ct.target.macOS15, convert_to="mlprogram")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT))

    got = model.predict({"pixels": x.numpy()})["embedding"]
    cosine = [float(a @ b / np.linalg.norm(a)) for a, b in zip(got, expected, strict=True)]
    print(f"exported {OUT} in {time.perf_counter() - t0:.1f} s; cosine vs PyTorch fp32 min {min(cosine):.4f}")
    if min(cosine) < MIN_COSINE:
        raise SystemExit(f"parity below {MIN_COSINE}")


if __name__ == "__main__":
    main()
