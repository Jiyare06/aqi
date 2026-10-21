"""Export the smoke/fire checkpoint to ONNX (outputs: final conv features + logits), verify, save CAM samples.

    uv run python -m hazeml.export_onnx runs/smokefire/<run>/best.pt

Writes next to the checkpoint: smokefire.onnx, head.npz (classifier W, b for cam.py),
export_report.json (torch vs ONNX max abs diff) and cam/*.png (6 test overlays). All gitignored.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn

from hazeml import cam
from hazeml.data import eval_transform
from hazeml.model import build_model
from hazeml.smokefire_data import load_dfire
from hazeml.train_smokefire import LABELS


class FeaturesAndLogits(nn.Module):
    """EfficientNet-B0 -> (final conv feature map, logits)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        f = self.model.features(x)
        return f, self.model.classifier(torch.flatten(self.model.avgpool(f), 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--n-verify", type=int, default=64)
    args = ap.parse_args()
    out = args.checkpoint.parent
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], pretrained=False, num_outputs=len(LABELS))
    model.load_state_dict(ckpt["state_dict"])
    wrapper = FeaturesAndLogits(model).eval()

    onnx_path = out / "smokefire.onnx"
    torch.onnx.export(wrapper, torch.zeros(1, 3, 224, 224), onnx_path, input_names=["input"],
                      output_names=["features", "logits"], dynamic_axes={"input": {0: "n"}, "features": {0: "n"}, "logits": {0: "n"}},
                      opset_version=17, dynamo=False)
    head = model.classifier[-1]
    np.savez(out / "head.npz", weight=head.weight.detach().numpy(), bias=head.bias.detach().numpy(), labels=np.array(LABELS))

    # Verify on real D-Fire test images (eval preprocessing from the checkpoint).
    tf = eval_transform(ckpt["preprocess"])
    test = load_dfire().query("split == 'test'").sample(args.n_verify, random_state=0)
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p in test["path"]])
    with torch.no_grad():
        f_t, l_t = wrapper(x)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    f_o, l_o = sess.run(["features", "logits"], {"input": x.numpy()})
    w = np.load(out / "head.npz")
    # GAP identity: mean of the raw CAM + bias reproduces the logits (checks cam.py against the model).
    gap = np.stack([[cam.cam_raw(f, w["weight"], c).mean() + w["bias"][c] for c in range(len(LABELS))] for f in f_o])
    report = {
        "onnx": str(onnx_path), "onnx_mb": onnx_path.stat().st_size / 2**20, "n_verify": args.n_verify,
        "features_shape": list(f_o.shape), "logits_max_abs_diff": float(np.abs(l_o - l_t.numpy()).max()),
        "features_max_abs_diff": float(np.abs(f_o - f_t.numpy()).max()),
        "cam_gap_vs_onnx_logits_max_abs_diff": float(np.abs(gap - l_o).max()),
        "onnxruntime": ort.__version__,
    }

    # 6 CAM overlays: top-scoring true positives for each label + a few others, from the test split.
    probs = 1 / (1 + np.exp(-l_o))
    tdf = test.reset_index(drop=True)
    picks, used = [], set()
    for j, lab in enumerate(LABELS):
        pos = [int(i) for i in np.argsort(-probs[:, j]) if tdf.loc[int(i), lab] == 1 and int(i) not in used][:2]
        picks += [(i, j) for i in pos]
        used.update(pos)
    neg = [int(i) for i in np.argsort(-probs[:, 0]) if tdf.loc[int(i), "label"] == "clear"][:1]  # most smoke-like clear image
    both = [i for i in range(len(tdf)) if tdf.loc[i, "label"] == "smoke_fire" and i not in used][:1]
    picks += [(i, 0) for i in neg] + [(i, 1) for i in both]
    (out / "cam").mkdir(exist_ok=True)
    saved = []
    for k, (i, j) in enumerate(picks[:6]):
        rgb = np.asarray(Image.open(tdf.loc[i, "path"]).convert("RGB").resize((448, 448)))
        heat = cam.cam(f_o[i], w["weight"], j)
        name = f"{k}_{tdf.loc[i, 'label']}_{LABELS[j]}cam_p{probs[i, j]:.2f}_{Path(tdf.loc[i, 'path']).stem}.png"
        Image.fromarray(np.concatenate([rgb, cam.overlay(rgb, heat)], axis=1)).save(out / "cam" / name)
        saved.append(name)
    report["cam_samples"] = saved
    (out / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
