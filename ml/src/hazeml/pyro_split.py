"""Site-grouped (camera-disjoint) train/val/test split of Pyro-SDIS -> ml/splits/pyro_cameras.json.

    uv run python -m hazeml.pyro_split

Camera names are <site>-<heading> (e.g. brison-110, brison-127 = two views from one mast), and some sites
have numbered stations (croix-augas-01-…, salaunes-2-…). Neighbouring views share scenery, so the split
groups by town-level site: strip a trailing "-<digits>" twice. Sites are shuffled (seed 0) and taken in
order until each side holds its share of Pyro's NEGATIVES (the look-alikes we measure FPR on):
test 20%, val 10% (for thresholds), train the rest. Every camera is in exactly one split.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from hazeml.smokefire_data import load_pyro

SPLIT_FILE = Path(__file__).resolve().parents[2] / "splits" / "pyro_cameras.json"
SHARES = {"test": 0.20, "val": 0.10}


def site_of(camera: pd.Series) -> pd.Series:
    return camera.str.replace(r"-\d+$", "", regex=True).str.replace(r"-\d+$", "", regex=True)


def make_split(df: pd.DataFrame, seed: int = 0) -> dict[str, str]:
    df = df.assign(site=site_of(df["camera"]))
    neg = df[df["smoke"] == 0].groupby("site").size().reindex(df["site"].unique(), fill_value=0)
    sites = np.random.default_rng(seed).permutation(sorted(neg.index))
    total, out, filled = neg.sum(), {}, {"test": 0, "val": 0}
    for s in sites:
        side = next((k for k in ("test", "val") if filled[k] < SHARES[k] * total), "train")
        out[s] = side
        if side != "train":
            filled[side] += int(neg[s])
    return out


def main() -> None:
    df = load_pyro()
    sites = make_split(df)
    df = df.assign(site=site_of(df["camera"]))
    df["split"] = df["site"].map(sites)
    counts = {s: {"sites": int(g["site"].nunique()), "cameras": int(g["camera"].nunique()), "images": int(len(g)),
                  "positives": int(g["smoke"].sum()), "negatives": int((g["smoke"] == 0).sum())}
              for s, g in df.groupby("split")}
    assert df.groupby("camera")["split"].nunique().max() == 1
    payload = {"dataset": "Pyro-SDIS (Apache-2.0), pyronear/pyro-sdis", "created": date.today().isoformat(),
               "method": "town-level site = camera minus trailing '-<digits>' (twice); sites shuffled with seed 0, "
                         "assigned until test holds 20% and val 10% of all negatives; rest train",
               "counts": counts, "sites": dict(sorted(sites.items())),
               "cameras": dict(sorted(df.groupby("camera")["split"].first().items()))}
    SPLIT_FILE.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({k: payload[k] for k in ("method", "counts")}, indent=2))
    print({s: sorted(k for k, v in sites.items() if v == s) for s in ("val", "test")})


if __name__ == "__main__":
    main()
