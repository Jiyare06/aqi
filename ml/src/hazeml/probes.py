"""Cheap probes: does any image signal transfer across dates / cities?

    uv run python -m hazeml.probes features   # cache frozen B0 embeddings + handcrafted features
    uv run python -m hazeml.probes run        # all probes -> runs/probes/*.json + summary.md

No CNN training: frozen ImageNet EfficientNet-B0 embeddings (1280-d) and four
handcrafted haze features, each fed to a logistic regression.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2

from hazeml.data import IMAGENET_MEAN, IMAGENET_STD
from hazeml.india_nepal import load_india_nepal
from hazeml.metrics import confusion, precision_recall_f1
from hazeml.traqid import load_traqid

OUT = Path(__file__).resolve().parents[2] / "runs" / "probes"
HAND = ("dark_channel", "contrast", "saturation", "brightness")
C_EMB, C_HAND = 0.1, 1.0  # fixed, not tuned


# --- features -----------------------------------------------------------------


def handcrafted(x: torch.Tensor) -> torch.Tensor:
    """x: 3xHxW in [0, 1] -> [dark channel mean, RMS contrast, mean saturation, mean brightness]."""
    mx, mn = x.max(dim=0).values, x.min(dim=0).values
    dark = -F.max_pool2d(-mn[None, None], 15, stride=1, padding=7)[0, 0]  # 15x15 min filter (He et al.)
    gray = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    sat = torch.where(mx > 0, (mx - mn) / mx.clamp_min(1e-6), torch.zeros_like(mx))
    return torch.stack([dark.mean(), gray.std(), sat.mean(), gray.mean()])


class FeatureImages(Dataset):
    """-> (normalised 224 crop for the CNN, handcrafted features on the whole frame)."""

    def __init__(self, paths: list[str], resize: int):
        self.paths = paths
        self.to_float = v2.Compose([v2.ToImage(), v2.Resize(224, antialias=True), v2.ToDtype(torch.float32, scale=True)])
        self.cnn = v2.Compose([v2.ToImage(), v2.Resize(resize, antialias=True), v2.CenterCrop(224),
                               v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            image = im.convert("RGB")
        return self.cnn(image), handcrafted(self.to_float(image))


@torch.no_grad()
def extract(paths: list[str], resize: int, workers: int = 6) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = nn.Identity()
    model = model.eval().to(device).to(memory_format=torch.channels_last)
    loader = DataLoader(FeatureImages(paths, resize), batch_size=128, num_workers=workers, pin_memory=True)
    emb, hand = [], []
    for x, h in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb.append(model(x.to(device).to(memory_format=torch.channels_last)).float().cpu().numpy())
        hand.append(h.numpy())
    return np.concatenate(emb), np.concatenate(hand)


def cache_features() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # TRAQID 640x360 -> same eval crop as the CNN (256 -> 224); India/Nepal is already 224x224.
    for name, df, resize in (("traqid", load_traqid(), 256), ("india_nepal", load_india_nepal(), 224)):
        t = time.perf_counter()
        emb, hand = extract(df["path"].tolist(), resize)
        np.savez(OUT / f"features_{name}.npz", emb=emb, hand=hand)
        df.to_csv(OUT / f"meta_{name}.csv", index=False)
        print(f"{name}: {len(df)} frames, emb {emb.shape}, hand {hand.shape}, {time.perf_counter() - t:.0f}s", flush=True)


# --- evaluation -------------------------------------------------------------------


def _mode(a: np.ndarray) -> int:
    return int(np.bincount(a).argmax())


def scores(y: np.ndarray, pred: np.ndarray, base: np.ndarray, k: int) -> dict:
    """macro-F1, accuracy, baseline accuracy (per-fold majority), recall of the top class."""
    cm = confusion(y, pred, k)
    _, recall, f1 = precision_recall_f1(cm)
    present = cm.sum(axis=1) > 0
    return {
        "n": int(len(y)),
        "macro_f1": float(f1[present].mean()),
        "accuracy": float((y == pred).mean()),
        "baseline_accuracy": float((y == base).mean()),
        "top_recall": float(recall[k - 1]) if cm[k - 1].sum() else None,
        "confusion": cm.tolist(),
    }


def grouped(y, pred, base, groups, k) -> dict:
    """Majority vote per group (date / moment) vs the group's majority true label."""
    g = pd.DataFrame({"y": y, "p": pred, "b": base, "g": groups}).groupby("g")
    yy = g["y"].agg(lambda s: _mode(s.to_numpy())).to_numpy()
    pp = g["p"].agg(lambda s: _mode(s.to_numpy())).to_numpy()
    bb = g["b"].first().to_numpy()
    return scores(yy, pp, bb, k)


def run_cv(X, y, folds, groups, k: int, C: float) -> dict:
    """Out-of-fold predictions; `folds` = fold id per row (one held-out group of rows per fold)."""
    pred, base = np.empty_like(y), np.empty_like(y)
    per_fold = {}
    for f in pd.unique(folds):
        te = folds == f
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
        clf.fit(X[~te], y[~te])
        pred[te] = clf.predict(X[te])
        base[te] = _mode(y[~te])
        per_fold[str(f)] = {"n": int(te.sum()), "accuracy": float((pred[te] == y[te]).mean()),
                            "baseline_accuracy": float((base[te] == y[te]).mean()),
                            "true_mix": np.bincount(y[te], minlength=k).tolist(),
                            "pred_mix": np.bincount(pred[te], minlength=k).tolist()}
    return {"frame": scores(y, pred, base, k), "group": grouped(y, pred, base, groups, k), "per_fold": per_fold}


def both_tasks(X, band, folds, groups, C) -> dict:
    return {"3band": run_cv(X, band, folds, groups, 3, C),
            "2class_vp_severe": run_cv(X, (band == 2).astype(int), folds, groups, 2, C)}


# --- probes ---------------------------------------------------------------------------


def probe1_dates(tq: pd.DataFrame) -> dict:
    def eta2(values: pd.Series, groups: pd.Series) -> float:
        total = ((values - values.mean()) ** 2).sum()
        between = values.groupby(groups).agg(lambda s: len(s) * (s.mean() - values.mean()) ** 2).sum()
        return float(between / total)

    rows = []
    for date, g in tq.groupby("date"):
        mix = np.bincount(g["band"], minlength=3) / len(g)
        rows.append({"date": date, "n": len(g), "season": g["season"].iloc[0], "hours": f"{g['hour'].min()}-{g['hour'].max()}",
                     "band0": mix[0], "band1": mix[1], "band2": mix[2], "dominant_share": mix.max(),
                     "pm25_median": g["pm25"].median()})
    return {"per_date": rows,
            "between_date_share_band": eta2(tq["band"].astype(float), tq["date"]),
            "between_date_share_aqi": eta2(tq["aqi"], tq["date"])}


def run_probes() -> None:
    tq = pd.read_csv(OUT / "meta_traqid.csv")
    inp = pd.read_csv(OUT / "meta_india_nepal.csv")
    ftq, fin = np.load(OUT / "features_traqid.npz"), np.load(OUT / "features_india_nepal.npz")
    band_tq, band_in = tq["band"].to_numpy(), inp["band"].to_numpy()
    dates = tq["date"].to_numpy()
    results = {}

    def save(name, res):
        results[name] = res
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(f"done {name}", flush=True)

    save("p1_per_date", probe1_dates(tq))
    save("p3_traqid_lodo_emb", both_tasks(ftq["emb"], band_tq, dates, dates, C_EMB))
    save("p4_traqid_lodo_hand", both_tasks(ftq["hand"], band_tq, dates, dates, C_HAND))
    hand_hum = np.column_stack([ftq["hand"], tq["humidity"].to_numpy()])
    save("p4_traqid_lodo_hand_humidity", both_tasks(hand_hum, band_tq, dates, dates, C_HAND))
    rnd = np.empty(len(tq), dtype=int)
    for i, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ftq["emb"], band_tq)):
        rnd[te] = i
    save("p5_traqid_random5fold_emb", both_tasks(ftq["emb"], band_tq, rnd, dates, C_EMB))
    p6 = both_tasks(fin["emb"], band_in, inp["city"].to_numpy(), inp["moment"].to_numpy(), C_EMB)
    p6["cities"] = {c: {"frames": int(len(g)), "moments": int(g["moment"].nunique()), "country": g["country"].iloc[0],
                        "band_mix": np.bincount(g["band"], minlength=3).tolist()} for c, g in inp.groupby("city")}
    save("p6_india_nepal_loco_emb", p6)
    (OUT / "summary.md").write_text(summary_md(results))
    print((OUT / "summary.md").read_text())


def summary_md(r: dict) -> str:
    def pct(v):
        return "n/a" if v is None else f"{v:.1%}"

    lines = ["# Probes summary (generated from runs/probes/*.json)", "",
             "Logistic regression (balanced class weights; C=0.1 on embeddings, 1.0 on handcrafted; standardised).",
             "Baseline = majority class of each training fold. Group = majority vote per held-out date (TRAQID) or moment (India/Nepal).",
             "", "| probe | task | level | n | macro-F1 | accuracy | baseline | top-class recall |", "|---|---|---|---|---|---|---|---|"]
    for name, res in r.items():
        if name == "p1_per_date":
            continue
        for task in ("3band", "2class_vp_severe"):
            for level in ("frame", "group"):
                s = res[task][level]
                lines.append(f"| {name} | {task} | {level} | {s['n']} | {s['macro_f1']:.3f} | {s['accuracy']:.1%} | "
                             f"{s['baseline_accuracy']:.1%} | {pct(s['top_recall'])} |")
    p1 = r["p1_per_date"]
    lines += ["", "## Probe 1: TRAQID day/front per date", "",
              f"Share of variance between dates: band {p1['between_date_share_band']:.1%}, AQI {p1['between_date_share_aqi']:.1%}", "",
              "| date | n | season | hours | band 0/1/2 | dominant | PM2.5 median |", "|---|---|---|---|---|---|---|"]
    for d in p1["per_date"]:
        lines.append(f"| {d['date']} | {d['n']} | {d['season']} | {d['hours']} | {d['band0']:.0%}/{d['band1']:.0%}/{d['band2']:.0%} | "
                     f"{d['dominant_share']:.0%} | {d['pm25_median']:.0f} |")
    p6 = r["p6_india_nepal_loco_emb"]
    lines += ["", "## Probe 6: India/Nepal cities", "", "| city | country | frames | moments | band 0/1/2 frames | 3-band acc (held out) | baseline |",
              "|---|---|---|---|---|---|---|"]
    for c, info in p6["cities"].items():
        f = p6["3band"]["per_fold"][c]
        lines.append(f"| {c} | {info['country']} | {info['frames']} | {info['moments']} | {'/'.join(map(str, info['band_mix']))} | "
                     f"{f['accuracy']:.1%} | {f['baseline_accuracy']:.1%} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=("features", "run"))
    args = p.parse_args()
    cache_features() if args.step == "features" else run_probes()


if __name__ == "__main__":
    main()
