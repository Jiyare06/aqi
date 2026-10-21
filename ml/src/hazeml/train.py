"""Train the image-only 3-band CNN on TRAQID (day/front, frozen date split).

    uv run python -m hazeml.train --smoke          # 1-epoch end-to-end check
    uv run python -m hazeml.train --seed 0         # full run

Outputs go to ml/runs/<timestamp>_<arch>_s<seed>/ (gitignored).
"""

import argparse
import hashlib
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hazeml.aqi import BANDS
from hazeml.data import EVAL_PREPROCESS, TraqidImages, eval_transform, train_transform
from hazeml.evaluate import evaluate_checkpoint, predict, to_markdown
from hazeml.metrics import summarize
from hazeml.model import ARCHS, build_model, param_groups
from hazeml.traqid import SPLIT_FILE, assign_split, load_traqid

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
SMOKE_ROWS = {"train": 512, "val": 256, "test": 256}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arch", default="efficientnet_b0", choices=ARCHS)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4, help="epochs without val macro-F1 gain before stopping")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="head LR")
    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help=f"1 epoch on {SMOKE_ROWS} rows")
    p.add_argument("--out", type=Path, default=None, help="run dir (default: runs/<timestamp>_<arch>_s<seed>)")
    return p.parse_args(argv)


class Logger:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        with self.path.open("a") as f:
            f.write(msg + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(bands: np.ndarray) -> torch.Tensor:
    """N / (K * n_c) from train labels."""
    counts = np.bincount(bands, minlength=len(BANDS))
    return torch.tensor(len(bands) / (len(BANDS) * np.maximum(counts, 1)), dtype=torch.float32)


def fit_batch_size(arch: str, batch_size: int, device, log) -> int:
    """Halve the batch until a forward+backward at 224 px fits in VRAM."""
    while batch_size >= 4:
        try:
            model = build_model(arch, pretrained=False).to(device).to(memory_format=torch.channels_last)
            x = torch.randn(batch_size, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(x).float().mean()
            loss.backward()
            del model, x, loss
            torch.cuda.empty_cache()
            return batch_size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"OOM at batch {batch_size}; halving")
            batch_size //= 2
    raise RuntimeError("model does not fit even at batch 4")


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.smoke:
        args.epochs = 1
    run = args.out or RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{args.arch}_s{args.seed}{'_smoke' if args.smoke else ''}"
    run.mkdir(parents=True, exist_ok=False)
    log = Logger(run / "train.log")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    df = assign_split(load_traqid())
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    if args.smoke:
        parts = {s: p.sample(min(len(p), SMOKE_ROWS[s]), random_state=args.seed).reset_index(drop=True)
                 for s, p in parts.items()}
    y_train = parts["train"]["band"].to_numpy()
    weights = class_weights(y_train)
    majority = int(np.bincount(y_train).argmax())

    batch_size = fit_batch_size(args.arch, args.batch_size, device, log) if device.type == "cuda" else args.batch_size
    config = vars(args) | {
        "out": str(run), "batch_size": batch_size, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "split_file": SPLIT_FILE.name, "split_sha1": hashlib.sha1(SPLIT_FILE.read_bytes()).hexdigest(),
        "rows": {s: len(p) for s, p in parts.items()}, "class_weights": weights.tolist(), "majority_band": majority,
        "torch": torch.__version__, "eval_preprocess": EVAL_PREPROCESS,
    }
    (run / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    log(f"run {run.name} | rows {config['rows']} | batch {batch_size} | class weights {[round(w, 3) for w in weights.tolist()]}")

    gen = torch.Generator().manual_seed(args.seed)
    loader_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(TraqidImages(parts["train"], train_transform()), batch_size=batch_size, shuffle=True,
                              drop_last=True, generator=gen, **loader_kw)
    val_loader = DataLoader(TraqidImages(parts["val"], eval_transform()), batch_size=64, **loader_kw)

    model = build_model(args.arch).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(param_groups(model, args.lr, args.backbone_lr), weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = cosine_with_warmup(optimizer, steps_per_epoch, steps_per_epoch * args.epochs)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    def checkpoint(epoch: int, val: dict) -> dict:
        return {"state_dict": model.state_dict(), "arch": args.arch, "bands": list(BANDS), "preprocess": EVAL_PREPROCESS,
                "majority_band": majority, "epoch": epoch, "val": val, "config": config}

    best_f1, best_epoch = -1.0, 0
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        wait = compute = loss_sum = 0.0
        seen = 0
        t_ready = time.perf_counter()
        for x, y in train_loader:
            t_batch = time.perf_counter()
            wait += t_batch - t_ready
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = criterion(model(x).float(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(y)  # .item() syncs, so compute time is real
            seen += len(y)
            t_ready = time.perf_counter()
            compute += t_ready - t_batch
        train_secs = time.perf_counter() - t0

        val = summarize(parts["val"]["band"].to_numpy(), predict(model, val_loader, device).argmax(axis=1))
        row = {
            "epoch": epoch, "train_loss": loss_sum / max(seen, 1), "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"],
            "val_band2_recall": val["band2_recall"], "lr_head": optimizer.param_groups[1]["lr"],
            "train_secs": train_secs, "images_per_sec": seen / train_secs, "loader_wait_share": wait / max(wait + compute, 1e-9),
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
        }
        with (run / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"epoch {epoch:2d} | loss {row['train_loss']:.4f} | val acc {val['accuracy']:.3f} macro-F1 {val['macro_f1']:.3f} "
            f"band2-rec {val['band2_recall']:.3f} | {row['images_per_sec']:.0f} img/s, loader wait {row['loader_wait_share']:.0%}, "
            f"peak VRAM {row['peak_vram_mb']:.0f} MB")

        torch.save(checkpoint(epoch, val), run / "last.pt")
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch = val["macro_f1"], epoch
            torch.save(checkpoint(epoch, val), run / "best.pt")
        elif epoch - best_epoch >= args.patience:
            log(f"early stop: no val macro-F1 gain for {args.patience} epochs")
            break

    log(f"best epoch {best_epoch} (val macro-F1 {best_f1:.3f}); evaluating best.pt on test")
    rep = evaluate_checkpoint(run / "best.pt", "test", args.workers, df=parts["test"])
    log(to_markdown(rep, f"Test report: {run.name}"))


if __name__ == "__main__":
    main()
