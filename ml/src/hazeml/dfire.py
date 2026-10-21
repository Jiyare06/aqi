"""D-Fire: image-level smoke/fire/clear labels + a perceptual-hash grouped train/val/test split.

    uv run python -m hazeml.dfire sweep          # group sizes for several hash thresholds
    uv run python -m hazeml.dfire split --max-dist 6

Data: $DATA_DIR/dfire_hf/data/*.parquet (HF mirror badsaarow/d-fire of D-Fire, CC0 1.0;
Venâncio et al. 2022). Columns: image{bytes}, label (YOLO lines "cls xc yc w h"), filename.
Class ids are not documented in the D-Fire README; they're inferred from its published
box counts (fire 14,692, smoke 11,865) and checked exactly.

Grouping: 64-bit dHash per image; images within Hamming distance <= max_dist are linked, and
consecutive AoF video frames (AoF<n>, AoF<n+1>) within distance <= SEQ_DIST are linked too, unless
that would make a group larger than MAX_GROUP_SHARE of all images (skipped links are counted).
Connected components become groups, so near-duplicates never straddle splits. Split: StratifiedGroupKFold(20) on the 4-way label -> 14/3/3 folds = ~70/15/15.
Frozen to ml/splits/dfire.json (filenames, split, group only — no images or boxes).
"""

import argparse
import io
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from hazeml.paths import data_dir

SPLIT_FILE = Path(__file__).resolve().parents[2] / "splits" / "dfire.json"
OUT = Path(__file__).resolve().parents[2] / "runs" / "dfire_split"
PUBLISHED = {"boxes": {"fire": 14692, "smoke": 11865},
             "images": {"fire_only": 1164, "smoke_only": 5867, "fire_and_smoke": 4658, "none": 9838}}
LABELS = ("clear", "smoke", "fire", "smoke_fire")
SEQ_DIST = 15
MAX_GROUP_SHARE = 0.10


def root() -> Path:
    return data_dir() / "dfire_hf" / "data"


def dhash(image_bytes: bytes) -> int:
    with Image.open(io.BytesIO(image_bytes)) as im:
        im.draft("L", (64, 64))  # fast JPEG decode at reduced scale
        px = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def read_all() -> list[dict]:
    rows = []
    for f in sorted(root().glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for i in range(pf.num_row_groups):
            for r in pf.read_row_group(i, columns=["image", "label", "filename"]).to_pylist():
                rows.append({"filename": r["filename"], "mirror_split": f.name.split("-")[0],
                             "label_text": r["label"] or "", "hash": dhash(r["image"]["bytes"])})
    return rows


def class_ids(rows: list[dict]) -> dict[str, int]:
    """Infer {fire: id, smoke: id} from total box counts; fail unless they match D-Fire exactly."""
    counts = Counter(int(line.split()[0]) for r in rows for line in r["label_text"].splitlines() if line.strip())
    by_count = {n: c for c, n in counts.items()}
    pub = PUBLISHED["boxes"]
    if set(counts.values()) != set(pub.values()) or len(counts) != 2:
        raise SystemExit(f"box counts {dict(counts)} don't match D-Fire's published {pub}")
    return {"fire": by_count[pub["fire"]], "smoke": by_count[pub["smoke"]]}


def add_labels(rows: list[dict], ids: dict[str, int]) -> Counter:
    for r in rows:
        cls = {int(line.split()[0]) for line in r["label_text"].splitlines() if line.strip()}
        r["smoke"], r["fire"] = ids["smoke"] in cls, ids["fire"] in cls
        r["label"] = LABELS[int(r["smoke"]) + 2 * int(r["fire"])]
    return Counter(r["label"] for r in rows)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def aof_pairs(filenames: list[str]) -> list[tuple[int, int]]:
    """Index pairs of consecutive AoF frames (AoF<n>.jpg, AoF<n+1>.jpg)."""
    num = {}
    for i, f in enumerate(filenames):
        m = re.fullmatch(r"AoF(\d+)\.jpg", f)
        if m:
            num[int(m.group(1))] = i
    return [(num[k], num[k + 1]) for k in sorted(num) if k + 1 in num]


def groups(hashes: np.ndarray, max_dist: int, extra_pairs: list[tuple[int, int]] = (),
           max_share: float | None = None, chunk: int = 1024) -> tuple[np.ndarray, dict]:
    """Union-find over 'Hamming(dHash) <= max_dist' links plus `extra_pairs`.

    Extra links that would create a group > max_share * n are skipped (counted in the stats)."""
    n = len(hashes)
    parent, size = np.arange(n), np.ones(n, dtype=int)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b, cap=None) -> str:
        ra, rb = find(a), find(b)
        if ra == rb:
            return "same"
        if cap is not None and size[ra] + size[rb] > cap:
            return "capped"
        ra, rb = min(ra, rb), max(ra, rb)
        parent[rb] = ra
        size[ra] += size[rb]
        return "merged"

    for s in range(0, n, chunk):
        d = np.bitwise_count(hashes[s:s + chunk, None] ^ hashes[None, :])
        for a, b in zip(*np.nonzero(d <= max_dist)):
            if a + s < b:
                union(a + s, b)
    cap = int(max_share * n) if max_share else None
    outcomes = [union(a, b, cap) for a, b in extra_pairs]
    stats = {"extra_links": len(extra_pairs), "extra_merged": outcomes.count("merged"),
             "extra_already_grouped": outcomes.count("same"), "extra_skipped_by_cap": outcomes.count("capped"),
             "cap_images": cap}
    return np.array([find(i) for i in range(n)]), stats


def sizes_summary(g: np.ndarray) -> dict:
    sizes = np.bincount(np.unique(g, return_inverse=True)[1])
    return {"groups": int(len(sizes)), "singletons": int((sizes == 1).sum()), "largest": int(sizes.max()),
            "largest_share": float(sizes.max() / len(g)), "images_in_groups_ge2": int(sizes[sizes > 1].sum())}


def leak_check(rows: list[dict], split: dict[str, str]) -> dict:
    """Consecutive AoF frame pairs in different splits, with their dHash distances."""
    pairs = aof_pairs([r["filename"] for r in rows])
    cross = [hamming(rows[a]["hash"], rows[b]["hash"]) for a, b in pairs
             if split[rows[a]["filename"]] != split[rows[b]["filename"]]]
    bins = [(0, 6), (6, 10), (10, 16), (16, 24), (24, 65)]
    return {"aof_consecutive_pairs": len(pairs), "pairs_in_different_splits": len(cross),
            "cross_split_distance_hist": {f"{lo}-{hi - 1}": sum(lo <= x < hi for x in cross) for lo, hi in bins},
            "cross_split_distance_median": float(np.median(cross)) if cross else None}


def cmd_sweep(rows) -> None:
    hashes = np.array([r["hash"] for r in rows], dtype=np.uint64)
    res = {t: sizes_summary(groups(hashes, t)[0]) for t in (0, 4, 6, 8, 10, 12)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep.json").write_text(json.dumps(res, indent=2) + "\n")
    for t, s in res.items():
        print(f"max_dist {t:2d}: {s}")


def cmd_split(rows, max_dist: int, seed: int = 0) -> None:
    ids = class_ids(rows)
    label_counts = add_labels(rows, ids)
    pub = PUBLISHED["images"]
    ours = {"fire_only": label_counts["fire"], "smoke_only": label_counts["smoke"],
            "fire_and_smoke": label_counts["smoke_fire"], "none": label_counts["clear"]}
    if ours != pub:
        raise SystemExit(f"image-level counts {ours} don't match D-Fire's published {pub}")
    if len({r["filename"] for r in rows}) != len(rows):
        raise SystemExit("duplicate filenames")

    hashes = np.array([r["hash"] for r in rows], dtype=np.uint64)
    seq = [(a, b) for a, b in aof_pairs([r["filename"] for r in rows])
           if hamming(rows[a]["hash"], rows[b]["hash"]) <= SEQ_DIST]
    g, link_stats = groups(hashes, max_dist, seq, MAX_GROUP_SHARE)
    y = np.array([LABELS.index(r["label"]) for r in rows])
    fold = np.empty(len(rows), dtype=int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(20, shuffle=True, random_state=seed).split(y, y, g)):
        fold[te] = k
    names = np.where(fold < 14, "train", np.where(fold < 17, "val", "test"))
    split = {r["filename"]: str(s) for r, s in zip(rows, names)}

    per_split = {s: dict(Counter(r["label"] for r, n in zip(rows, names) if n == s)) | {"total": int((names == s).sum())}
                 for s in ("train", "val", "test")}
    group_ids = {gid: i for i, gid in enumerate(sorted(set(g.tolist())))}
    payload = {
        "dataset": "D-Fire (Venâncio et al. 2022), CC0 1.0, via Hugging Face mirror badsaarow/d-fire",
        "created": date.today().isoformat(),
        "class_ids": ids, "labels": list(LABELS),
        "label_rule": "smoke/fire flags = any box of that class; 'smoke_fire' = both; 'clear' = no boxes",
        "grouping": {"hash": "dHash 64-bit (9x8 grayscale)", "max_hamming": max_dist,
                     "aof_sequence_links": {"max_hamming": SEQ_DIST, "max_group_share": MAX_GROUP_SHARE} | link_stats}
                    | sizes_summary(g),
        "split_method": f"StratifiedGroupKFold(n_splits=20, shuffle=True, random_state={seed}) on 4-way label; folds 0-13 train, 14-16 val, 17-19 test",
        "counts": per_split,
        "leakage_check": leak_check(rows, split),
        "files": {r["filename"]: [split[r["filename"]], group_ids[int(gi)]] for r, gi in zip(rows, g)},
    }
    SPLIT_FILE.write_text(json.dumps(payload, indent=1) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split_summary.json").write_text(json.dumps({k: v for k, v in payload.items() if k != "files"}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "files"}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=("sweep", "split"))
    p.add_argument("--max-dist", type=int, default=6)  # see runs/dfire_split/sweep.json: 8+ chains into a 27% group
    args = p.parse_args()
    rows = read_all()
    print(f"read {len(rows)} images from {root()}", flush=True)
    cmd_sweep(rows) if args.step == "sweep" else cmd_split(rows, args.max_dist)


if __name__ == "__main__":
    main()
