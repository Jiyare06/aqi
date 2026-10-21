"""India/Nepal air-pollution image dataset (Kaggle, CC BY-NC-SA 3.0 IGO).

Reads only the CSV; image paths are built from `Filename` (Combined_Dataset/All_img).
Labels are recomputed from raw PM with hazeml.aqi (the CSV's AQI uses US bands).
A "moment" is location + date + hour: frames within one are near-duplicates.
"""

from pathlib import Path

import pandas as pd

from hazeml.aqi import label_pm
from hazeml.paths import data_dir

COMBINED = Path("india-nepal/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset")


def load_india_nepal(root: Path | None = None) -> pd.DataFrame:
    root = root or data_dir() / COMBINED
    raw = pd.read_csv(root / "IND_and_Nep_AQI_Dataset.csv")
    labels = [label_pm(a, b) for a, b in zip(raw["PM2.5"], raw["PM10"])]
    date = pd.to_datetime(dict(year=raw["Year"], month=raw["Month"], day=raw["Day"])).dt.strftime("%Y-%m-%d")
    return pd.DataFrame({
        "path": [str(root / "All_img" / f) for f in raw["Filename"]],
        "city": raw["Location"],
        "country": ["Nepal" if "Nepal" in loc else "India" for loc in raw["Location"]],
        "date": date,
        "hour": raw["Hour"],
        "moment": raw["Location"] + "|" + date + "|" + raw["Hour"],
        "pm25": raw["PM2.5"],
        "pm10": raw["PM10"],
        "aqi": [lab[0] for lab in labels],
        "category": [lab[1] for lab in labels],
        "band": [lab[2] for lab in labels],
        "src_aqi": raw["AQI"],
        "src_aqi_class": raw["AQI_Class"],
    })
