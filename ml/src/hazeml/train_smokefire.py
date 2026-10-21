"""Smoke/fire multi-label classifier (EfficientNet-B0, 2 sigmoid outputs) on D-Fire's grouped split.

    uv run python -m hazeml.train_smokefire --smoke                      # tiny end-to-end check
    uv run python -m hazeml.train_smokefire                              # full run
    uv run python -m hazeml.train_smokefire --eval-only runs/smokefire/<run>/best.pt
    uv run python -m hazeml.train_smokefire --init runs/smokefire/<run>/best.pt --pyro-negatives   # hard-negative fine-tune

Early stopping on val mean AP. After training: per-label thresholds = max F1 on val; reports on
D-Fire test and on Pyro-SDIS (external, smoke only, grouped by camera) -> report.json / report.md.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader

from hazeml.data import SQUASH_PREPROCESS, MultiLabelImages, eval_transform, smokefire_train_transform
from hazeml.evaluate import predict
from hazeml.model import build_model, param_groups
from hazeml.pyro_split import SPLIT_FILE as PYRO_SPLIT
from hazeml.smokefire_data import load_dfire, load_pyro
from hazeml.train import Logger, cosine_with_warmup, fit_batch_size, seed_everything

LABELS = ("smoke", "fire")
COMBOS = ("clear", "smoke", "fire", "smoke_fire")
RUNS = Path(__file__).resolve().parents[2] / "runs" / "smokefire"
SMOKE_ROWS = {"train": 512, "val": 256, "test": 256, "pyro": 256}


# --- metrics -------------------------------------------------------------------------------------


def pick_thresholds(y: np.ndarray, p: np.ndarray) -> list[float]:
    """Per label: the threshold maximising F1 on (y, p)."""
    out = []
    for j in range(y.shape[1]):
        prec, rec, thr = precision_recall_curve(y[:, j], p[:, j])
        f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
        out.append(float(thr[int(np.argmax(f1))]))
    return out


def label_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "n": int(len(y)), "n_pos": int(y.sum()), "threshold": thr,
        "ap": float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else None,
        "precision": precision, "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if recall and precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else None, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def combo(y: np.ndarray) -> np.ndarray:
    """(smoke, fire) 0/1 -> index into COMBOS."""
    return (y[:, 0] + 2 * y[:, 1]).astype(int)


def dfire_report(y: np.ndarray, p: np.ndarray, thr: list[float]) -> dict:
    pred = (p >= np.array(thr)).astype(int)
    cm = np.zeros((4, 4), dtype=int)
    np.add.at(cm, (combo(y), combo(pred)), 1)
    per = {lab: label_metrics(y[:, j], p[:, j], thr[j]) for j, lab in enumerate(LABELS)}
    return {"per_label": per, "mean_ap": float(np.mean([per[lab]["ap"] for lab in LABELS])),
            "combo_confusion": {"labels": list(COMBOS), "rows_true_cols_pred": cm.tolist()},
            "exact_match": float((cm.trace()) / cm.sum())}


def pyro_report(df: pd.DataFrame, p_smoke: np.ndarray, thr: float) -> dict:
    y = df["smoke"].to_numpy()
    overall = label_metrics(y, p_smoke, thr)
    cams = []
    for cam, idx in df.groupby("camera").indices.items():
        m = label_metrics(y[idx], p_smoke[idx], thr)
        cams.append({"camera": cam, "n": m["n"], "n_pos": m["n_pos"], "recall": m["recall"], "fpr": m["fpr"], "ap": m["ap"]})
    cams.sort(key=lambda c: -c["n"])
    rec = [c["recall"] for c in cams if c["recall"] is not None]
    fpr = [c["fpr"] for c in cams if c["fpr"] is not None]
    aps = [c["ap"] for c in cams if c["ap"] is not None]
    return {"overall": overall, "cameras": len(cams),
            "by_camera_macro": {"recall_mean": float(np.mean(rec)), "cameras_with_pos": len(rec),
                                "fpr_mean": float(np.mean(fpr)) if fpr else None, "cameras_with_neg": len(fpr),
                                "ap_mean": float(np.mean(aps)) if aps else None, "cameras_with_both": len(aps)},
            "per_camera": cams}


# --- evaluation of a checkpoint ------------------------------------------------------------------


def _loader(df: pd.DataFrame, targets, pre: dict, workers: int) -> DataLoader:
    return DataLoader(MultiLabelImages(df["path"], targets, eval_transform(pre)), batch_size=64,
                      num_workers=workers, pin_memory=True)


def evaluate_run(ckpt_path: Path, workers: int = 3, smoke: bool = False, log=print) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], pretrained=False, num_outputs=len(LABELS))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).to(memory_format=torch.channels_last)
    pre = ckpt["preprocess"]

    d = load_dfire()
    parts = {s: d[d["split"] == s].reset_index(drop=True) for s in ("val", "test")}
    pyro = load_pyro()
    if smoke:
        parts = {s: x.sample(min(len(x), SMOKE_ROWS[s]), random_state=0).reset_index(drop=True) for s, x in parts.items()}
        pyro = pyro.sample(SMOKE_ROWS["pyro"], random_state=0).reset_index(drop=True)
    y = {s: x[list(LABELS)].to_numpy() for s, x in parts.items()}
    p = {s: predict(model, _loader(x, y[s], pre, workers), device, "sigmoid") for s, x in parts.items()}
    thr = pick_thresholds(y["val"], p["val"])
    log(f"val-picked thresholds (max F1): {dict(zip(LABELS, [round(t, 3) for t in thr]))}")
    t0 = time.perf_counter()
    p_pyro = predict(model, _loader(pyro, np.c_[pyro["smoke"], np.zeros(len(pyro))], pre, workers), device, "sigmoid")
    log(f"Pyro-SDIS: {len(pyro)} images in {time.perf_counter() - t0:.0f}s")

    out = ckpt_path.parent
    np.savez(out / "probs.npz", val=p["val"], test=p["test"], pyro=p_pyro,
             val_files=parts["val"]["filename"].to_numpy(), test_files=parts["test"]["filename"].to_numpy(),
             pyro_files=pyro["image_name"].to_numpy())
    rep = {"checkpoint": str(ckpt_path), "epoch": ckpt["epoch"], "thresholds": dict(zip(LABELS, thr)),
           "val": dfire_report(y["val"], p["val"], thr), "dfire_test": dfire_report(y["test"], p["test"], thr),
           "pyro_sdis": pyro_report(pyro, p_pyro[:, 0], thr[0])}
    (out / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    (out / "report.md").write_text(to_markdown(rep))
    log(to_markdown(rep))
    return rep


def _f(v, pct=True):
    return "n/a" if v is None else (f"{v:.1%}" if pct else f"{v:.3f}")


def to_markdown(rep: dict) -> str:
    t = rep["dfire_test"]
    lines = [f"# Smoke/fire report ({Path(rep['checkpoint']).parent.name}, best epoch {rep['epoch']})", "",
             f"Thresholds (max F1 on val): smoke {rep['thresholds']['smoke']:.3f}, fire {rep['thresholds']['fire']:.3f}", "",
             "## D-Fire test", "", "| label | n pos / n | AP | precision | recall | F1 |", "|---|---|---|---|---|---|"]
    for lab in LABELS:
        m = t["per_label"][lab]
        lines.append(f"| {lab} | {m['n_pos']} / {m['n']} | {_f(m['ap'], False)} | {_f(m['precision'])} | {_f(m['recall'])} | {_f(m['f1'], False)} |")
    lines += ["", f"Mean AP {t['mean_ap']:.3f}; exact 4-way match {t['exact_match']:.1%}", "",
              "Confusion (rows = true, cols = predicted):", "", "| true \\ pred | " + " | ".join(COMBOS) + " |", "|---|" + "---|" * 4]
    for name, row in zip(COMBOS, t["combo_confusion"]["rows_true_cols_pred"]):
        lines.append(f"| {name} | " + " | ".join(map(str, row)) + " |")
    py = rep["pyro_sdis"]
    o, mac = py["overall"], py["by_camera_macro"]
    lines += ["", "## Pyro-SDIS (external; smoke only; val-picked smoke threshold)", "",
              f"n = {o['n']} ({o['n_pos']} smoke, {o['n'] - o['n_pos']} no smoke), {py['cameras']} cameras", "",
              f"Overall: AP {_f(o['ap'], False)}, precision {_f(o['precision'])}, recall {_f(o['recall'])}, "
              f"false-positive rate on negatives {_f(o['fpr'])} ({o['fp']} / {o['fp'] + o['tn']})", "",
              f"By camera (macro): mean recall {_f(mac['recall_mean'])} over {mac['cameras_with_pos']} cameras with smoke; "
              f"mean FPR {_f(mac['fpr_mean'])} over {mac['cameras_with_neg']} cameras with negatives; "
              f"mean AP {_f(mac['ap_mean'], False)} over {mac['cameras_with_both']} cameras with both", "",
              "Largest 15 cameras:", "", "| camera | n | smoke | recall | FPR | AP |", "|---|---|---|---|---|---|"]
    for c in py["per_camera"][:15]:
        lines.append(f"| {c['camera']} | {c['n']} | {c['n_pos']} | {_f(c['recall'])} | {_f(c['fpr'])} | {_f(c['ap'], False)} |")
    return "\n".join(lines) + "\n"


# --- hard negatives (Pyro-SDIS, site-grouped split) ---------------------------------------------------


def pyro_by_split() -> pd.DataFrame:
    split = json.loads(PYRO_SPLIT.read_text())["cameras"]
    d = load_pyro()
    return d.assign(split=d["camera"].map(split), fire=0)


def as_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """Pyro rows -> D-Fire-like rows labelled smoke=0, fire=0 (only call on smoke == 0 rows)."""
    assert (df["smoke"] == 0).all()
    return pd.DataFrame({"path": df["path"], "smoke": 0, "fire": 0, "source": "pyro_neg"})


def compare_hardneg(models: dict[str, Path], out: Path, workers: int = 3, smoke: bool = False) -> dict:
    """Same procedure for every model: thresholds = max F1 on D-Fire val + val-site Pyro negatives;
    report D-Fire test, FPR on held-out-site Pyro negatives, recall on held-out-site Pyro positives (info)."""
    device = torch.device("cuda")
    d = load_dfire()
    dval, dtest = d[d["split"] == "val"].reset_index(drop=True), d[d["split"] == "test"].reset_index(drop=True)
    py = pyro_by_split()
    pval_neg = py[(py["split"] == "val") & (py["smoke"] == 0)].reset_index(drop=True)
    ptest = py[py["split"] == "test"].reset_index(drop=True)
    if smoke:
        dval, dtest, ptest = (x.sample(min(len(x), 128), random_state=0).reset_index(drop=True) for x in (dval, dtest, ptest))
        pval_neg = pval_neg.sample(min(len(pval_neg), 64), random_state=0).reset_index(drop=True)
    y_val = np.r_[dval[list(LABELS)].to_numpy(), np.zeros((len(pval_neg), 2), dtype=int)]
    y_test = dtest[list(LABELS)].to_numpy()
    res = {"sets": {"dfire_val": len(dval), "pyro_val_negatives": len(pval_neg), "dfire_test": len(dtest),
                    "pyro_test_negatives": int((ptest["smoke"] == 0).sum()), "pyro_test_positives": int(ptest["smoke"].sum())},
           "models": {}}
    for name, ckpt_path in models.items():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(ckpt["arch"], pretrained=False, num_outputs=len(LABELS))
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device).to(memory_format=torch.channels_last)
        pre = ckpt["preprocess"]
        run = lambda df: predict(model, _loader(df, np.zeros((len(df), 2)), pre, workers), device, "sigmoid")  # noqa: E731
        p_val = np.r_[run(dval), run(pval_neg)]
        p_test, p_py = run(dtest), run(ptest)
        thr = pick_thresholds(y_val, p_val)
        neg, pos = (ptest["smoke"] == 0).to_numpy(), (ptest["smoke"] == 1).to_numpy()
        flags = p_py >= np.array(thr)
        res["models"][name] = {
            "checkpoint": str(ckpt_path), "thresholds": dict(zip(LABELS, thr)),
            "dfire_test": dfire_report(y_test, p_test, thr),
            "pyro_test_negatives": {"n": int(neg.sum()), "smoke_fpr": float(flags[neg, 0].mean()),
                                    "fire_fpr": float(flags[neg, 1].mean()), "any_flag_fpr": float(flags[neg].any(1).mean()),
                                    "smoke_score_median": float(np.median(p_py[neg, 0]))},
            "pyro_test_positives_info": {"n": int(pos.sum()), "smoke_recall": float(flags[pos, 0].mean()),
                                         "smoke_ap_test_sites": float(average_precision_score(ptest["smoke"], p_py[:, 0]))},
        }
    out.mkdir(parents=True, exist_ok=True)
    (out / "hardneg_compare.json").write_text(json.dumps(res, indent=2) + "\n")
    lines = ["# Hard-negative fine-tune vs current model", "",
             f"Sets: {res['sets']}", "", "Thresholds for both models: max F1 on D-Fire val + val-site Pyro negatives.", "",
             "| model | thr smoke / fire | D-Fire test AP smoke / fire | smoke P / R | fire P / R | Pyro held-out neg FPR smoke / fire / any | Pyro held-out pos smoke recall (info) |",
             "|---|---|---|---|---|---|---|"]
    for name, m in res["models"].items():
        t, pl, ng, ps = m["thresholds"], m["dfire_test"]["per_label"], m["pyro_test_negatives"], m["pyro_test_positives_info"]
        lines.append(f"| {name} | {t['smoke']:.3f} / {t['fire']:.3f} | {pl['smoke']['ap']:.3f} / {pl['fire']['ap']:.3f} | "
                     f"{pl['smoke']['precision']:.1%} / {pl['smoke']['recall']:.1%} | {pl['fire']['precision']:.1%} / {pl['fire']['recall']:.1%} | "
                     f"{ng['smoke_fpr']:.1%} / {ng['fire_fpr']:.1%} / {ng['any_flag_fpr']:.1%} | {ps['smoke_recall']:.1%} |")
    (out / "hardneg_compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


# --- training --------------------------------------------------------------------------------------


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-only", type=Path)
    ap.add_argument("--init", type=Path, help="start from this checkpoint's weights (fine-tune)")
    ap.add_argument("--pyro-negatives", action="store_true",
                    help="add Pyro-SDIS negatives from train sites; early-stop on D-Fire val + val-site negatives")
    args = ap.parse_args(argv)

    if args.eval_only:
        evaluate_run(args.eval_only, args.workers, args.smoke)
        return
    if args.smoke:
        args.epochs = 1
    tag = ("_hardneg" if args.pyro_negatives else "") + ("_smoke" if args.smoke else "")
    run = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_{args.arch}_s{args.seed}{tag}"
    run.mkdir(parents=True)
    log = Logger(run / "train.log")
    seed_everything(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    d = load_dfire()
    parts = {s: d[d["split"] == s].reset_index(drop=True) for s in ("train", "val")}
    if args.pyro_negatives:
        py = pyro_by_split()
        neg = {s: as_negatives(py[(py["split"] == s) & (py["smoke"] == 0)]) for s in ("train", "val")}
        parts = {s: pd.concat([parts[s].assign(source="dfire"), neg[s]], ignore_index=True) for s in parts}
    if args.smoke:
        parts = {s: x.sample(SMOKE_ROWS[s], random_state=args.seed).reset_index(drop=True) for s, x in parts.items()}
    y_train = parts["train"][list(LABELS)].to_numpy()
    pos = y_train.sum(axis=0)
    pos_weight = torch.tensor((len(y_train) - pos) / pos, dtype=torch.float32)
    batch = fit_batch_size(args.arch, args.batch_size, device, log)
    config = vars(args) | {"eval_only": None, "batch_size": batch, "rows": {s: len(x) for s, x in parts.items()},
                           "pyro_negatives_rows": {s: int((x.get("source") == "pyro_neg").sum()) if "source" in x else 0
                                                   for s, x in parts.items()},
                           "pyro_split_file": "ml/splits/pyro_cameras.json" if args.pyro_negatives else None,
                           "pos_weight": dict(zip(LABELS, pos_weight.tolist())), "preprocess": SQUASH_PREPROCESS,
                           "split_file": "ml/splits/dfire.json", "torch": torch.__version__}
    (run / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    log(f"run {run.name} | rows {config['rows']} | batch {batch} | pos_weight {config['pos_weight']}")

    gen = torch.Generator().manual_seed(args.seed)
    kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(MultiLabelImages(parts["train"]["path"], y_train, smokefire_train_transform()),
                              batch_size=batch, shuffle=True, drop_last=True, generator=gen, **kw)
    y_val = parts["val"][list(LABELS)].to_numpy()
    val_loader = DataLoader(MultiLabelImages(parts["val"]["path"], y_val, eval_transform(SQUASH_PREPROCESS)),
                            batch_size=64, **kw)

    model = build_model(args.arch, num_outputs=len(LABELS))
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["state_dict"])
        log(f"initialised from {args.init}")
    model = model.to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(param_groups(model, args.lr, args.backbone_lr), weight_decay=args.weight_decay)
    scheduler = cosine_with_warmup(optimizer, len(train_loader), len(train_loader) * args.epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    def checkpoint(epoch, val):
        return {"state_dict": model.state_dict(), "arch": args.arch, "labels": list(LABELS), "preprocess": SQUASH_PREPROCESS,
                "epoch": epoch, "val": val, "config": config}

    best, best_epoch = -1.0, 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, seen, loss_sum = time.perf_counter(), 0, 0.0
        for x, yb in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(x).float(), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(yb)
            seen += len(yb)
        secs = time.perf_counter() - t0
        p_val = predict(model, val_loader, device, "sigmoid")
        aps = [float(average_precision_score(y_val[:, j], p_val[:, j])) for j in range(len(LABELS))]
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "val_ap_smoke": aps[0], "val_ap_fire": aps[1],
               "val_mean_ap": float(np.mean(aps)), "train_secs": secs, "images_per_sec": seen / secs,
               "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20}
        with (run / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"epoch {epoch:2d} | loss {row['train_loss']:.4f} | val AP smoke {aps[0]:.3f} fire {aps[1]:.3f} "
            f"mean {row['val_mean_ap']:.3f} | {row['images_per_sec']:.0f} img/s | peak VRAM {row['peak_vram_mb']:.0f} MB")
        torch.save(checkpoint(epoch, row), run / "last.pt")
        if row["val_mean_ap"] > best:
            best, best_epoch = row["val_mean_ap"], epoch
            torch.save(checkpoint(epoch, row), run / "best.pt")
        elif epoch - best_epoch >= args.patience:
            log(f"early stop: no val mean-AP gain for {args.patience} epochs")
            break
    log(f"best epoch {best_epoch} (val mean AP {best:.3f}); evaluating")
    del train_loader, val_loader
    evaluate_run(run / "best.pt", args.workers, args.smoke, log)
    if args.pyro_negatives:
        compare_hardneg({"current": args.init, "hardneg": run / "best.pt"} if args.init else {"hardneg": run / "best.pt"},
                        run, args.workers, args.smoke)
    log("DONE")


if __name__ == "__main__":
    main()
