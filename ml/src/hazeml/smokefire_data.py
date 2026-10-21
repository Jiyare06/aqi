"""Smoke/fire data on disk: extract images from the HF parquet files once, then load metadata tables.

    uv run python -m hazeml.smokefire_data extract

D-Fire  -> $DATA_DIR/dfire_hf/images/<filename>   + meta.csv (filename, smoke, fire, label, split)
Pyro-SDIS -> $DATA_DIR/pyro_sdis/images/<image_name> + meta.csv (image_name, hf_split, partner, camera, date, smoke, n_boxes)
Image bytes are written unchanged. Labels: D-Fire per ml/splits/dfire.json class ids; Pyro-SDIS
"class_id: always 0 (smoke). Images without smoke have empty annotations" (dataset card).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from hazeml.dfire import SPLIT_FILE as DFIRE_SPLIT
from hazeml.paths import data_dir

PYRO_PUBLISHED = {"images": 12378, "smoke": 10744, "no_smoke": 1634}


def _rows(parquet_dir: Path, columns: list[str]):
    for f in sorted(parquet_dir.glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for i in range(pf.num_row_groups):
            for r in pf.read_row_group(i, columns=columns).to_pylist():
                yield f.name.split("-")[0], r


def _write(path: Path, data: bytes) -> None:
    if not (path.exists() and path.stat().st_size == len(data)):
        path.write_bytes(data)


def extract_dfire() -> pd.DataFrame:
    root = data_dir() / "dfire_hf"
    split = json.loads(DFIRE_SPLIT.read_text())
    ids = split["class_ids"]
    (root / "images").mkdir(exist_ok=True)
    meta = []
    for _, r in _rows(root / "data", ["image", "label", "filename"]):
        cls = {int(line.split()[0]) for line in (r["label"] or "").splitlines() if line.strip()}
        smoke, fire = ids["smoke"] in cls, ids["fire"] in cls
        _write(root / "images" / r["filename"], r["image"]["bytes"])
        meta.append({"filename": r["filename"], "smoke": int(smoke), "fire": int(fire),
                     "label": ("clear", "smoke", "fire", "smoke_fire")[int(smoke) + 2 * int(fire)],
                     "split": split["files"][r["filename"]][0]})
    df = pd.DataFrame(meta)
    assert len(df) == len(split["files"]) == df["filename"].nunique(), "D-Fire rows don't match the split file"
    df.to_csv(root / "meta.csv", index=False)
    return df


def extract_pyro() -> pd.DataFrame:
    root = data_dir() / "pyro_sdis"
    (root / "images").mkdir(exist_ok=True)
    meta = []
    for hf_split, r in _rows(root / "data", ["image", "annotations", "image_name", "partner", "camera", "date"]):
        boxes = [line for line in (r["annotations"] or "").splitlines() if line.strip()]
        if any(int(b.split()[0]) != 0 for b in boxes):
            raise SystemExit(f"unexpected class id in {r['image_name']}")
        _write(root / "images" / r["image_name"], r["image"]["bytes"])
        meta.append({"image_name": r["image_name"], "hf_split": hf_split, "partner": r["partner"], "camera": r["camera"],
                     "date": r["date"], "smoke": int(bool(boxes)), "n_boxes": len(boxes)})
    df = pd.DataFrame(meta)
    got = {"images": len(df), "smoke": int(df["smoke"].sum()), "no_smoke": int((df["smoke"] == 0).sum())}
    if got != PYRO_PUBLISHED or df["image_name"].nunique() != len(df):
        raise SystemExit(f"Pyro-SDIS counts {got} (unique names {df['image_name'].nunique()}) != card {PYRO_PUBLISHED}")
    df.to_csv(root / "meta.csv", index=False)
    return df


def load_dfire() -> pd.DataFrame:
    root = data_dir() / "dfire_hf"
    df = pd.read_csv(root / "meta.csv")
    return df.assign(path=[str(root / "images" / f) for f in df["filename"]])


def load_pyro() -> pd.DataFrame:
    root = data_dir() / "pyro_sdis"
    df = pd.read_csv(root / "meta.csv")
    return df.assign(path=[str(root / "images" / f) for f in df["image_name"]])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=("extract",))
    p.parse_args()
    d = extract_dfire()
    print("D-Fire", len(d), d.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict("index"), flush=True)
    py = extract_pyro()
    print("Pyro-SDIS", len(py), "smoke", int(py["smoke"].sum()), "cameras", py["camera"].nunique(),
          "by hf_split", py["hf_split"].value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
