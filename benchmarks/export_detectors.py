"""Export the Apache-2.0 detector candidates to ONNX and Core ML.

  uv run benchmarks/export_detectors.py --detector rfdetr_n --format onnx

Artifacts land in `models/detectors/exported/` (git-ignored); docs/SETUP.md
records the command that produced them. The YOLOs are not exported: AGPL,
reference numbers only (R5), and PyTorch covers that role.

Boxes are decoded in Python (see benchmarks/detectors.py), so both graphs stop
at the model outputs: RF-DETR logits + boxes, D-FINE logits + pred_boxes.
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models" / "detectors"
EXPORTED = MODELS / "exported"
DFINE_SIZE = 640  # preprocessor_config.json
FORMATS = ["onnx", "coreml"]
PRECISIONS = {"fp32": "float32", "fp16": "float16"}  # fp16 is the ANE-oriented bundle
LABELS_FILE = "labels.json"


def _dfine_wrapper():
    import torch
    from transformers import DFineForObjectDetection

    model = DFineForObjectDetection.from_pretrained(MODELS / "dfine-nano-coco").eval()

    class Wrapper(torch.nn.Module):
        """torch.onnx and coremltools both need plain tensors out, not a model output object."""

        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            out = self.model(pixel_values=pixel_values)
            return out.logits, out.pred_boxes

    return Wrapper().eval(), torch.zeros(1, 3, DFINE_SIZE, DFINE_SIZE)


def _write_labels(out_dir, labels):
    """Class index in the graph's logits → COCO name, so inference needs no model stack."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / LABELS_FILE).write_text(json.dumps({str(k): v for k, v in labels.items()}, indent=2))


def rfdetr(fmt, out_dir, precision):
    from rfdetr import RFDETRNano
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRNano(pretrain_weights=str(MODELS / "rf-detr-nano.pth"), device="cpu")
    _write_labels(out_dir, COCO_CLASSES)
    return model.export(output_dir=str(out_dir), format=fmt, verbose=False,
                        coreml_precision=PRECISIONS[precision])


def dfine(fmt, out_dir, precision):
    import torch

    wrapper, dummy = _dfine_wrapper()
    _write_labels(out_dir, wrapper.model.config.id2label)
    if fmt == "onnx":
        path = out_dir / "dfine_n.onnx"
        torch.onnx.export(wrapper, (dummy,), str(path), input_names=["pixel_values"],
                          output_names=["logits", "pred_boxes"], opset_version=17, dynamo=False)
        return path

    import coremltools as ct

    traced = torch.jit.trace(wrapper, dummy, strict=False)
    model = ct.convert(traced, inputs=[ct.TensorType(name="pixel_values", shape=dummy.shape)],
                       outputs=[ct.TensorType(name="logits"), ct.TensorType(name="pred_boxes")],
                       convert_to="mlprogram")
    path = out_dir / "dfine_n.mlpackage"
    model.save(str(path))
    return path


EXPORTERS = {"rfdetr_n": rfdetr, "dfine_n": dfine}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", choices=sorted(EXPORTERS), required=True)
    p.add_argument("--format", choices=FORMATS, required=True)
    p.add_argument("--precision", choices=sorted(PRECISIONS), default="fp16",
                   help="Core ML compute precision; ANE needs fp16")
    args = p.parse_args()

    out_dir = EXPORTED / args.detector
    t0 = time.perf_counter()
    path = EXPORTERS[args.detector](args.format, out_dir, args.precision)
    print(f"{args.detector} → {args.format} in {time.perf_counter() - t0:.1f} s: {path}")


if __name__ == "__main__":
    main()
