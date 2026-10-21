"""Why doesn't the smoke model transfer to Pyro-SDIS? Diagnostics only (no training).

    uv run python -m hazeml.pyro_diag runs/smokefire/<run>/best.pt

Writes runs/pyro_diag/: summary.json, pr_curve_224.csv, pr_curve_448.csv, pr_curves.png, by_box_size.csv.
- AP + PR curve at 224 px (reuses the run's probs.npz) and at 448 px (same weights, whole frame squashed to 448x448).
- Recall / FPR at thresholds 0.5, 0.3, 0.1 (+ the val-picked smoke threshold).
- Recall by plume size: largest box area per positive image (normalised), split into terciles.
- Whether Pyro's train/val split is camera-disjoint.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader

from hazeml.data import MultiLabelImages, eval_transform
from hazeml.evaluate import predict
from hazeml.model import build_model
from hazeml.paths import data_dir
from hazeml.smokefire_data import load_dfire, load_pyro
from hazeml.train_smokefire import LABELS

OUT = Path(__file__).resolve().parents[2] / "runs" / "pyro_diag"
THRESHOLDS = (0.5, 0.3, 0.1)


def max_box_area() -> pd.Series:
    """image_name -> largest normalised box area (w*h) from Pyro's YOLO annotations; 0 if none."""
    areas = {}
    for f in sorted((data_dir() / "pyro_sdis" / "data").glob("*.parquet")):
        for r in pq.read_table(f, columns=["image_name", "annotations"]).to_pylist():
            boxes = [line.split() for line in (r["annotations"] or "").splitlines() if line.strip()]
            areas[r["image_name"]] = max((float(b[3]) * float(b[4]) for b in boxes), default=0.0)
    return pd.Series(areas)


def at_threshold(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = p >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    return {"threshold": t, "recall": tp / int(y.sum()), "fpr": fp / int((y == 0).sum()),
            "precision": tp / (tp + fp) if tp + fp else None, "flagged": int(pred.sum())}


def pr_points(y, p) -> pd.DataFrame:
    prec, rec, thr = precision_recall_curve(y, p)
    return pd.DataFrame({"threshold": np.r_[thr, np.nan], "precision": prec, "recall": rec})


def plot(curves: dict[str, pd.DataFrame], base: float, path: Path) -> None:
    W, H, m = 520, 420, 50
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([m, 20, W - 20, H - m], outline="black")
    xy = lambda r, pr: (m + r * (W - 20 - m), H - m - pr * (H - m - 20))
    d.line([xy(0, base), xy(1, base)], fill=(150, 150, 150))  # chance = prevalence
    colours = {"224 px": (31, 119, 180), "448 px": (214, 39, 40)}
    for k, (name, c) in enumerate(curves.items()):
        pts = [xy(r, pr) for r, pr in zip(c["recall"], c["precision"])]
        d.line(pts, fill=colours[name], width=2)
        d.text((W - 150, 30 + 15 * k), name, fill=colours[name])
    d.text((W - 150, 30 + 15 * len(curves)), f"chance {base:.3f}", fill=(120, 120, 120))
    d.text((m, H - m + 10), "recall 0 -> 1", fill="black")
    d.text((5, 20), "precision", fill="black")
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run = args.checkpoint.parent
    report = json.loads((run / "report.json").read_text())
    val_thr = report["thresholds"]["smoke"]

    pyro = load_pyro()
    y = pyro["smoke"].to_numpy()
    probs = np.load(run / "probs.npz", allow_pickle=True)
    assert (probs["pyro_files"] == pyro["image_name"].to_numpy()).all(), "probs.npz order != meta.csv order"
    p224 = probs["pyro"][:, 0]

    # Same weights at 448 px (Pyro + D-Fire test as an in-domain sanity check).
    device = torch.device("cuda")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], pretrained=False, num_outputs=len(LABELS))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).to(memory_format=torch.channels_last)
    pre448 = dict(ckpt["preprocess"], resize_hw=[448, 448])

    def run_at(df, pre, draft):
        ds = MultiLabelImages(df["path"], np.zeros((len(df), 2)), eval_transform(pre), draft=draft)
        return predict(model, DataLoader(ds, batch_size=32, num_workers=args.workers, pin_memory=True), device, "sigmoid")

    p448 = run_at(pyro, pre448, 640)[:, 0]
    dtest = load_dfire().query("split == 'test'").reset_index(drop=True)
    pd448 = run_at(dtest, pre448, 640)

    base = float(y.mean())
    res = {"n": int(len(y)), "positives": int(y.sum()), "negatives": int((y == 0).sum()), "ap_chance_level": base}
    for name, p in (("224", p224), ("448", p448)):
        res[f"ap_{name}"] = float(average_precision_score(y, p))
        res[f"thresholds_{name}"] = [at_threshold(y, p, t) for t in (*THRESHOLDS, val_thr)]
        res[f"score_quantiles_{name}"] = {"positives": np.quantile(p[y == 1], [0.1, 0.5, 0.9]).round(4).tolist(),
                                          "negatives": np.quantile(p[y == 0], [0.1, 0.5, 0.9]).round(4).tolist()}
        pr_points(y, p).to_csv(OUT / f"pr_curve_{name}.csv", index=False)
    res["dfire_test_448_ap"] = {lab: float(average_precision_score(dtest[lab], pd448[:, j])) for j, lab in enumerate(LABELS)}
    res["dfire_test_224_ap"] = {lab: report["dfire_test"]["per_label"][lab]["ap"] for lab in LABELS}
    plot({"224 px": pr_points(y, p224), "448 px": pr_points(y, p448)}, base, OUT / "pr_curves.png")

    # Recall by plume size (positives only), terciles of the largest box's normalised area.
    area = max_box_area().reindex(pyro["image_name"]).to_numpy()
    pos = y == 1
    cuts = np.quantile(area[pos], [1 / 3, 2 / 3])
    size = np.where(area < cuts[0], "small", np.where(area < cuts[1], "medium", "large"))
    rows = []
    for name in ("small", "medium", "large"):
        m = pos & (size == name)
        a = area[m]
        rows.append({"size": name, "n": int(m.sum()), "area_min": float(a.min()), "area_max": float(a.max()),
                     "side_px_at_224_median": float(np.median(np.sqrt(a)) * 224),
                     "side_px_at_448_median": float(np.median(np.sqrt(a)) * 448),
                     **{f"recall_224_t{t}": float((p224[m] >= t).mean()) for t in (*THRESHOLDS, val_thr)},
                     **{f"recall_448_t{t}": float((p448[m] >= t).mean()) for t in (*THRESHOLDS, val_thr)}})
    pd.DataFrame(rows).to_csv(OUT / "by_box_size.csv", index=False)
    res["by_box_size"] = rows
    res["box_area_tercile_cuts"] = cuts.tolist()

    # Camera-disjointness of Pyro's own train/val split.
    cams = pyro.groupby("hf_split")["camera"].apply(set)
    shared = cams["train"] & cams["val"]
    res["pyro_split"] = {"cameras_train": len(cams["train"]), "cameras_val": len(cams["val"]),
                         "cameras_in_both": len(shared),
                         "val_images_from_shared_cameras": int(pyro[(pyro["hf_split"] == "val") & pyro["camera"].isin(shared)].shape[0]),
                         "val_images": int((pyro["hf_split"] == "val").sum()),
                         "camera_disjoint": not shared}
    res["val_picked_smoke_threshold"] = val_thr
    (OUT / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
