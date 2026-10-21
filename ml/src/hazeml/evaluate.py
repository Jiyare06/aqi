"""Evaluate a checkpoint on a TRAQID split; overall, by time of day, by date.

    uv run python -m hazeml.evaluate runs/<run>/best.pt [--split test]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hazeml.aqi import BANDS
from hazeml.data import TraqidImages, eval_transform
from hazeml.metrics import summarize
from hazeml.model import build_model
from hazeml.traqid import assign_split, load_traqid

# Hour buckets for the time-of-day confound (CLAUDE.md §4).
TIME_OF_DAY = (("morning (<11)", 0, 11), ("midday (11-15)", 11, 15), ("evening (>=15)", 15, 24))


@torch.no_grad()
def predict(model, loader, device, activation: str = "softmax") -> np.ndarray:
    """Probabilities (softmax over classes, or per-label sigmoid), one row per sample, in loader order."""
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        logits = logits.float()
        probs = torch.sigmoid(logits) if activation == "sigmoid" else torch.softmax(logits, dim=1)
        out.append(probs.cpu().numpy())
    return np.concatenate(out)


def _group(df: pd.DataFrame, pred: np.ndarray, majority: int) -> dict:
    y = df["band"].to_numpy()
    rep = summarize(y, pred)
    rep["majority_baseline"] = float((y == majority).mean()) if len(y) else 0.0
    rep["band_mix"] = [float((y == b).mean()) if len(y) else 0.0 for b in range(len(BANDS))]
    if not (y == 2).any():
        rep["band2_recall"] = None  # undefined, not 0
    return rep


def report(df: pd.DataFrame, probs: np.ndarray, majority: int) -> dict:
    """`majority` = most common band in train (the baseline always predicts it)."""
    pred = probs.argmax(axis=1)
    rep = {"overall": _group(df, pred, majority), "by_time_of_day": {}, "by_date": {}}
    rep["overall"]["mean_confidence"] = float(probs.max(axis=1).mean())
    for name, lo, hi in TIME_OF_DAY:
        m = ((df["hour"] >= lo) & (df["hour"] < hi)).to_numpy()
        if m.any():
            rep["by_time_of_day"][name] = _group(df[m], pred[m], majority) | {
                "dates": sorted(df.loc[m, "date"].unique())
            }
    for date in sorted(df["date"].unique()):
        m = (df["date"] == date).to_numpy()
        rep["by_date"][date] = _group(df[m], pred[m], majority)
    return rep


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def to_markdown(rep: dict, title: str = "Test report") -> str:
    o = rep["overall"]
    lines = [
        f"# {title}",
        "",
        f"n = {o['n']} | **macro-F1 {o['macro_f1']:.3f}** | accuracy {o['accuracy']:.1%} "
        f"(majority baseline {o['majority_baseline']:.1%}) | within-one {o['within_one']:.1%} | "
        f"**band-2 recall {_pct(o['band2_recall'])}** | mean confidence {o['mean_confidence']:.2f}",
        "",
        "| band | precision | recall | F1 | share |",
        "|---|---|---|---|---|",
    ]
    for b, name in enumerate(BANDS):
        lines.append(f"| {b} {name} | {o['precision'][b]:.3f} | {o['recall'][b]:.3f} | {o['f1'][b]:.3f} | {o['band_mix'][b]:.1%} |")
    lines += ["", "Confusion (rows = true, cols = predicted):", "", "| true \\ pred | " + " | ".join(map(str, range(len(BANDS)))) + " |",
              "|---|" + "---|" * len(BANDS)]
    for b, row in enumerate(o["confusion"]):
        lines.append(f"| {b} | " + " | ".join(map(str, row)) + " |")

    def table(groups: dict, extra: bool) -> list[str]:
        head = "| group | n | band mix (0/1/2) | accuracy | majority baseline | macro-F1 | band-2 recall |"
        out = [head + (" dates |" if extra else ""), "|---" * (7 + extra) + "|"]
        for name, g in groups.items():
            mix = "/".join(f"{m:.0%}" for m in g["band_mix"])
            row = (f"| {name} | {g['n']} | {mix} | {g['accuracy']:.1%} | {g['majority_baseline']:.1%} | "
                   f"{g['macro_f1']:.3f} | {_pct(g['band2_recall'])} |")
            out.append(row + (f" {', '.join(g['dates'])} |" if extra else ""))
        return out

    lines += ["", "## By time of day", "",
              "Caveat: on test, morning frames come (almost) only from 2023-12-21, so this is confounded with the date.", ""]
    lines += table(rep["by_time_of_day"], extra=True)
    lines += ["", "## By date", ""] + table(rep["by_date"], extra=False)
    return "\n".join(lines) + "\n"


def load_checkpoint(path: Path, device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).to(memory_format=torch.channels_last), ckpt


def evaluate_checkpoint(path: Path, split: str = "test", workers: int = 6, df: pd.DataFrame | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(path, device)
    if df is None:
        df = assign_split(load_traqid())
        df = df[df["split"] == split].reset_index(drop=True)
    loader = DataLoader(TraqidImages(df, eval_transform(ckpt["preprocess"])), batch_size=64,
                        num_workers=workers, pin_memory=True)
    rep = report(df, predict(model, loader, device), ckpt["majority_band"])
    rep["checkpoint"] = str(path)
    rep["split"] = split
    out = Path(path).parent
    (out / f"{split}_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    (out / f"{split}_report.md").write_text(to_markdown(rep, f"{split.capitalize()} report: {out.name}"))
    return rep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.split, args.workers)
    print((args.checkpoint.parent / f"{args.split}_report.md").read_text())


if __name__ == "__main__":
    main()
