"""Score our own phone clips with the smoke/fire model settings, mirroring the deployed processor.

    uv run python -m hazeml.score_clips

Clips: ~/datasets/own_clips/*.mp4|*.mov (read-only). Name = [stock_]<label>_<tags…>; label in
smoke / fire / nosmoke / reject; a stock_ prefix is reported separately.
Pipeline = infra/processor (imported unchanged): frames.sample (~1 fps, max 30) -> quality.gate
(< MIN_PASS good frames -> cause models skipped) -> causes.preprocess -> ONNX (features, logits) -> sigmoid;
"present" = >= 2 frames over the threshold (>= 1 if fewer than 3 frames), as causes.aggregate.
YOLO11n vehicle count = median of per-frame counts on the passing frames (vehicles.py, as deployed).
Correct: smoke -> smoke present; fire -> fire present; nosmoke -> neither present; reject -> gate skipped.
A non-reject clip skipped by the gate counts as incorrect.
Writes ml/runs/own_clips/: report.md, results.json, cam/<clip>__<model>.png (gitignored).
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "infra" / "processor"))
import cam as pcam  # noqa: E402  (processor modules, unchanged)
import causes  # noqa: E402
import frames  # noqa: E402
import quality  # noqa: E402
import vehicles  # noqa: E402

CLIPS = Path.home() / "datasets" / "own_clips"
RUNS = REPO / "ml" / "runs"
OUT = RUNS / "own_clips"
MODELS = {"current": RUNS / "smokefire" / "20261009-220554_efficientnet_b0_s0",
          "finetuned": RUNS / "smokefire" / "20261010-072055_efficientnet_b0_s0_hardneg"}
LABELS = causes.LABELS


def settings() -> list[dict]:
    ft = json.loads((MODELS["finetuned"] / "hardneg_compare.json").read_text())["models"]["hardneg"]["thresholds"]
    return [
        {"name": "current@0.359", "model": "current", "thr": dict(causes.THRESHOLDS)},  # as deployed
        {"name": "current@0.600", "model": "current", "thr": {"smoke": 0.600, "fire": causes.THRESHOLDS["fire"]}},
        {"name": f"finetuned@{ft['smoke']:.3f}", "model": "finetuned", "thr": {"smoke": ft["smoke"], "fire": ft["fire"]}},
    ]


def parse(name: str) -> dict:
    stem = Path(name).stem
    stock = stem.startswith("stock_")
    tokens = (stem[len("stock_"):] if stock else stem).split("_")
    return {"clip": stem, "stock": stock, "label": tokens[0], "tags": [t for t in tokens[1:] if not re.fullmatch(r"\d+", t)]}


def present(probs: np.ndarray, thr: dict) -> dict:
    need = 2 if len(probs) >= 3 else 1
    return {lab: {"count": int((probs[:, j] >= thr[lab]).sum()), "present": bool((probs[:, j] >= thr[lab]).sum() >= need)}
            for j, lab in enumerate(LABELS)}


def correct(label: str, skipped: bool, pres: dict | None) -> bool:
    if label == "reject":
        return skipped
    if skipped:
        return False
    if label == "smoke":
        return pres["smoke"]["present"]
    if label == "fire":
        return pres["fire"]["present"]
    return not (pres["smoke"]["present"] or pres["fire"]["present"])


def save_cam(rgb_frames, probs, feats, head, path: Path) -> dict:
    j = int(np.argmax(probs.max(axis=0)))  # the stronger label
    i = int(np.argmax(probs[:, j]))  # its best frame
    rgb = rgb_frames[i]
    h, w = rgb.shape[:2]
    if w > causes.OVERLAY_MAX_W:
        rgb = np.asarray(Image.fromarray(rgb).resize((causes.OVERLAY_MAX_W, round(h * causes.OVERLAY_MAX_W / w)),
                                                     Image.Resampling.BILINEAR))
    over = pcam.overlay(rgb, pcam.cam(feats[i], head["weight"], j))
    Image.fromarray(np.concatenate([rgb, over], axis=1)).save(path)
    return {"label": LABELS[j], "frame": i, "prob": round(float(probs[i, j]), 3), "png": path.name}


def main() -> None:
    (OUT / "cam").mkdir(parents=True, exist_ok=True)
    sess = {k: ort.InferenceSession(str(d / "smokefire.onnx"), providers=["CPUExecutionProvider"]) for k, d in MODELS.items()}
    heads = {k: dict(np.load(d / "head.npz")) for k, d in MODELS.items()}
    yolo = ort.InferenceSession(str(RUNS / "yolo" / "yolo11n.onnx"), providers=["CPUExecutionProvider"])
    sets = settings()

    rows = []
    for path in sorted(p for p in CLIPS.iterdir() if p.suffix.lower() in (".mp4", ".mov")):
        info = parse(path.name)
        bgr = frames.sample(path)
        kept, why, _ = quality.gate(bgr)
        skipped = len(kept) < quality.MIN_PASS
        row = info | {"frames_sampled": len(bgr), "frames_used": len(kept), "frames_rejected": len(bgr) - len(kept),
                      "reject_reasons": why, "skipped": skipped, "models": {}, "settings": {}, "vehicle_count": None}
        if not skipped:
            rgb = [np.ascontiguousarray(f[..., ::-1]) for f in kept]
            x = causes.preprocess(rgb)
            for m, s in sess.items():
                feats, logits = s.run(["features", "logits"], {"input": x})
                probs = causes.sigmoid(logits)
                row["models"][m] = {"max": {lab: round(float(probs[:, j].max()), 3) for j, lab in enumerate(LABELS)},
                                    "cam": save_cam(rgb, probs, feats, heads[m], OUT / "cam" / f"{info['clip']}__{m}.png"),
                                    "_probs": probs}
            run = lambda t: yolo.run(["output0"], {"images": t})[0]  # noqa: E731
            per_frame = [sum(vehicles.count_by_class(vehicles.detect(run, f)).values()) for f in rgb]
            row["vehicle_count"] = vehicles.clip_vehicle_count(per_frame)
            row["vehicles_per_frame"] = per_frame
        for st in sets:
            pres = None if skipped else present(row["models"][st["model"]]["_probs"], st["thr"])
            row["settings"][st["name"]] = {"present": pres, "correct": correct(info["label"], skipped, pres)}
        if not skipped:  # the deployed setting must agree with the processor's own aggregate()
            agg = causes.aggregate(row["models"]["current"]["_probs"])
            assert all(agg[lab]["present"] == row["settings"]["current@0.359"]["present"][lab]["present"] for lab in LABELS)
        for m in row["models"].values():
            m.pop("_probs")
        rows.append(row)
        print(f"{info['clip']}: used {len(kept)}/{len(bgr)} " + " ".join(
            f"{k}={'ok' if v['correct'] else 'X'}" for k, v in row["settings"].items()), flush=True)

    summary = summarize(rows, sets)
    (OUT / "results.json").write_text(json.dumps({"settings": sets, "summary": summary, "clips": rows}, indent=2) + "\n")
    (OUT / "report.md").write_text(to_markdown(rows, sets, summary))
    print(json.dumps(summary, indent=2))


def summarize(rows, sets) -> dict:
    def acc(subset, name):
        return {"correct": sum(r["settings"][name]["correct"] for r in subset), "n": len(subset)}

    out = {"clips": len(rows), "own": sum(not r["stock"] for r in rows), "stock": sum(r["stock"] for r in rows), "per_setting": {}}
    for st in sets:
        n = st["name"]
        by_label, by_tag = defaultdict(list), defaultdict(list)
        for r in rows:
            by_label[r["label"]].append(r)
            for t in r["tags"]:
                by_tag[t].append(r)
        own = [r for r in rows if not r["stock"]]
        out["per_setting"][n] = {
            "all": acc(rows, n), "own": acc(own, n), "stock": acc([r for r in rows if r["stock"]], n),
            "own_excluding_reject": acc([r for r in own if r["label"] != "reject"], n),
            "per_label": {k: acc(v, n) for k, v in sorted(by_label.items())},
            "per_tag": {k: acc(v, n) for k, v in sorted(by_tag.items())},
            "wrong": [r["clip"] for r in rows if not r["settings"][n]["correct"]],
        }
    best = max(out["per_setting"], key=lambda n: out["per_setting"][n]["own"]["correct"])
    top = out["per_setting"][best]["own"]["correct"]
    out["best_on_own"] = [n for n, v in out["per_setting"].items() if v["own"]["correct"] == top]
    return out


def to_markdown(rows, sets, summary) -> str:
    names = [s["name"] for s in sets]
    f = lambda c: f"{c['correct']}/{c['n']}" if c["n"] else "–"  # noqa: E731
    lines = ["# Own clips: smoke/fire settings", "",
             "Pipeline mirrors infra/processor (frames.sample ~1 fps → quality.gate → causes.preprocess → ONNX; "
             "present = ≥ 2 frames over threshold, ≥ 1 if < 3 frames). Settings: "
             + "; ".join(f"**{s['name']}** ({s['model']}, smoke {s['thr']['smoke']:.3f}, fire {s['thr']['fire']:.3f})" for s in sets), "",
             f"Clips: {summary['clips']} ({summary['own']} own, {summary['stock']} stock). Best on own clips: **{', '.join(summary['best_on_own'])}**.", "",
             "## Accuracy", "", "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    rowdefs = [("all clips", "all"), ("own", "own"), ("own, excl. reject", "own_excluding_reject"), ("stock", "stock")]
    for title, key in rowdefs:
        lines.append(f"| {title} | " + " | ".join(f(summary["per_setting"][n][key]) for n in names) + " |")
    for kind in ("per_label", "per_tag"):
        keys = sorted(summary["per_setting"][names[0]][kind])
        for k in keys:
            lines.append(f"| {kind.split('_')[1]}: {k} | " + " | ".join(f(summary["per_setting"][n][kind][k]) for n in names) + " |")
    lines += ["", "Wrong clips: " + "; ".join(f"**{n}**: {', '.join(summary['per_setting'][n]['wrong']) or 'none'}" for n in names), "",
              "## Per clip", "",
              "| clip | label | tags | frames used / rejected (reasons) | max smoke cur / ft | max fire cur / ft | "
              + " | ".join(f"{n} smoke-frames / fire-frames → ok?" for n in names) + " | vehicles (median) | CAM (current; fine-tuned) |",
              "|---|---|---|---|---|---|" + "---|" * len(names) + "---|---|"]
    for r in rows:
        mx = lambda lab: " / ".join(f"{r['models'][m]['max'][lab]:.2f}" for m in ("current", "finetuned")) if r["models"] else "–"  # noqa: E731
        cells = []
        for n in names:
            s = r["settings"][n]
            mark = "✓" if s["correct"] else "✗"
            cells.append(f"skipped {mark}" if r["skipped"] else f"{s['present']['smoke']['count']} / {s['present']['fire']['count']} {mark}")
        cams = "; ".join(f"{r['models'][m]['cam']['png']} ({r['models'][m]['cam']['label']} {r['models'][m]['cam']['prob']:.2f})"
                         for m in ("current", "finetuned")) if r["models"] else "–"
        lines.append(f"| {r['clip']}{' (stock)' if r['stock'] else ''} | {r['label']} | {', '.join(r['tags'])} | "
                     f"{r['frames_used']} / {r['frames_rejected']} ({', '.join(r['reject_reasons']) or '–'}) | {mx('smoke')} | {mx('fire')} | "
                     + " | ".join(cells) + f" | {r['vehicle_count'] if r['vehicle_count'] is not None else '–'} | {cams} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
