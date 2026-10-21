"""TRAQID metadata loader + calendar-date split (CLAUDE.md §4).

Reads only TRAQID.csv; image paths are built from the `Image` id, never by
listing the image folders. Labels are recomputed from raw PM with hazeml.aqi.

Frames are 5 s apart, so random splits leak; every calendar date goes wholly
to one split. TRAQID's daytime data covers only 17 dates and the Very Poor /
Severe rows sit on a few of them, so the split is a small exhaustive search
for date sets that hit the target row fractions while keeping each split's
band mix close to the overall mix. The result is frozen in
ml/splits/traqid_dates.json; regenerate with `python -m hazeml.traqid --make-split`.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hazeml.aqi import BANDS, label_pm
from hazeml.paths import data_dir

SPLIT_FILE = Path(__file__).resolve().parents[2] / "splits" / "traqid_dates.json"
SPLITS = ("train", "val", "test")


def load_traqid(root: Path | None = None, view: str = "front", daytime_only: bool = True) -> pd.DataFrame:
    """One row per frame, with recomputed CPCB-style labels and image paths."""
    root = root or data_dir() / "traqid"
    raw = pd.read_csv(root / "TRAQID.csv", parse_dates=["created_at"])
    if daytime_only:
        raw = raw[raw["Day_or_Night"] == "Day"]

    labels = [label_pm(a, b) for a, b in zip(raw["PM2.5"], raw["PM10"])]
    df = pd.DataFrame({
        "image_id": raw["Image"],
        "path": [str(root / view / f"{i}.png") for i in raw["Image"]],
        "created_at": raw["created_at"],
        "date": raw["created_at"].dt.strftime("%Y-%m-%d"),
        "hour": raw["created_at"].dt.hour,
        "season": raw["Season"],
        "day_or_night": raw["Day_or_Night"],
        "pm25": raw["PM2.5"],
        "pm10": raw["PM10"],
        "temperature": raw["Temperature"],
        "humidity": raw["Humidity"],
        "aqi": [lab[0] for lab in labels],
        "category": [lab[1] for lab in labels],
        "band": [lab[2] for lab in labels],
        # TRAQID's own values, kept only for sanity checks.
        "src_aqi": raw["aqi"],
        "src_aqi_cat": raw["aqi_cat"],
    })
    return df.reset_index(drop=True)


def _best_subset(counts: pd.DataFrame, target: pd.Series, frac: float, lo: float, hi: float, total: int):
    """Date subset whose row share is in [lo, hi] and whose band mix is closest to target."""
    dates, table, target = list(counts.index), counts.to_numpy(), target.to_numpy()
    best, best_score = None, float("inf")
    for k in range(1, len(dates)):
        for subset in itertools.combinations(range(len(dates)), k):
            rows = table[list(subset)].sum(axis=0)
            n = rows.sum()
            if not lo <= n / total <= hi:
                continue
            score = np.abs(rows / n - target).sum() + abs(n / total - frac)
            if score < best_score:
                best, best_score = subset, score
    if best is None:
        raise ValueError(f"no date subset with row share in [{lo}, {hi}]")
    return [dates[i] for i in best]


def make_date_split(df: pd.DataFrame, test_frac: float = 0.2, val_frac: float = 0.1, tol: float = 0.05) -> dict[str, str]:
    """Deterministic {date: split}. Test is chosen first, then val from what's left."""
    counts = pd.crosstab(df["date"], df["band"]).reindex(columns=range(len(BANDS)), fill_value=0)
    target = df["band"].value_counts(normalize=True).reindex(range(len(BANDS)), fill_value=0)
    total = len(df)

    test = _best_subset(counts, target, test_frac, test_frac - tol, test_frac + tol, total)
    val = _best_subset(counts.drop(index=test), target, val_frac, val_frac - tol, val_frac + tol, total)
    return {d: "test" if d in test else "val" if d in val else "train" for d in counts.index}


def save_split(split: dict[str, str], path: Path = SPLIT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(split.items())), indent=2) + "\n")


def assign_split(df: pd.DataFrame, path: Path = SPLIT_FILE) -> pd.DataFrame:
    """Add a `split` column from the frozen date split."""
    split = json.loads(path.read_text())
    missing = set(df["date"]) - set(split)
    if missing:
        raise ValueError(f"dates not in {path.name}: {sorted(missing)}; regenerate the split")
    return df.assign(split=df["date"].map(split))


def summary(df: pd.DataFrame) -> str:
    """Rows, dates, band mix and majority baseline per split."""
    majority = df.loc[df["split"] == "train", "band"].mode()[0]
    lines = [f"{len(df)} rows, {df['date'].nunique()} dates; majority band in train = {majority} ({BANDS[majority]})"]
    for name in SPLITS:
        part = df[df["split"] == name]
        mix = part["band"].value_counts(normalize=True).reindex(range(len(BANDS)), fill_value=0)
        lines.append(
            f"  {name:5s} {len(part):6d} rows ({len(part) / len(df):5.1%}) {part['date'].nunique():2d} dates | "
            + " ".join(f"b{b}={mix[b]:5.1%}" for b in mix.index)
            + f" | majority baseline {(part['band'] == majority).mean():5.1%}"
        )
    lines.append("  dates: " + ", ".join(f"{s}={sorted(df.loc[df['split'] == s, 'date'].unique())}" for s in ("val", "test")))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--make-split", action="store_true", help=f"recompute and overwrite {SPLIT_FILE.name}")
    args = parser.parse_args()

    df = load_traqid()
    if args.make_split:
        save_split(make_date_split(df))
        print(f"wrote {SPLIT_FILE}")
    print(summary(assign_split(df)))


if __name__ == "__main__":
    main()
