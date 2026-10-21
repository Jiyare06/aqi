"""Check our CPCB-style AQI against the AQI column of cpcb_2015_2020/city_day.csv.

Only rows where PM2.5 or PM10 drives the AQI are compared. The oracle was
computed from hourly data (24-h rolling means, 8-h max for CO/O3); we only
have daily means, so an exact match isn't expected. Run with `-s` to see the
report.
"""

import pandas as pd
import pytest

from hazeml.aqi import CATEGORIES, _subindex, aqi_band, aqi_category, cpcb_aqi
from hazeml.paths import data_dir

# Official CPCB breakpoints for the other pollutants (µg/m³, CO in mg/m³).
# Test-only: used to decide whether PM is the driving pollutant.
OTHER_BREAKPOINTS = {
    "NO2": ((0, 0), (40, 50), (80, 100), (180, 200), (280, 300), (400, 400)),
    "SO2": ((0, 0), (40, 50), (80, 100), (380, 200), (800, 300), (1600, 400)),
    "O3": ((0, 0), (50, 50), (100, 100), (168, 200), (208, 300), (748, 400)),
    "CO": ((0, 0), (1, 50), (2, 100), (10, 200), (17, 300), (34, 400)),
    "NH3": ((0, 0), (200, 50), (400, 100), (800, 200), (1200, 300), (1800, 400)),
}


def _city_day() -> pd.DataFrame:
    try:
        path = data_dir() / "cpcb_2015_2020" / "city_day.csv"
    except RuntimeError:
        pytest.skip("DATA_DIR not set")
    if not path.exists():
        pytest.skip(f"{path} not found")
    return pd.read_csv(path)


def _other_max(row) -> float:
    subs = [_subindex(row[col], bps) for col, bps in OTHER_BREAKPOINTS.items()]
    subs = [s for s in subs if s is not None]
    return max(subs) if subs else 0.0


def test_matches_cpcb_city_day():
    df = _city_day()
    df = df[df["AQI"].notna() & (df["PM2.5"].notna() | df["PM10"].notna())].copy()
    n_with_aqi = len(df)
    # A few rows have small negative readings; treat them as missing.
    for col in ("PM2.5", "PM10", *OTHER_BREAKPOINTS):
        df.loc[df[col] < 0, col] = float("nan")

    df["ours"] = [cpcb_aqi(a, b) for a, b in zip(df["PM2.5"], df["PM10"])]
    df["other"] = df.apply(_other_max, axis=1)
    df = df[df["ours"].notna() & (df["ours"] >= df["other"])]

    ours_cat = df["ours"].map(aqi_category)
    true_cat = df["AQI"].map(aqi_category)
    idx = {c: i for i, c in enumerate(CATEGORIES)}
    cat_dist = (ours_cat.map(idx) - true_cat.map(idx)).abs()
    diff = (df["ours"] - df["AQI"]).abs()

    n = len(df)
    category_match = (cat_dist == 0).mean()
    within_one = (cat_dist <= 1).mean()
    band_match = (df["ours"].map(aqi_band) == df["AQI"].map(aqi_band)).mean()
    within_10 = (diff <= 10).mean()
    bucket_agrees = (true_cat == df["AQI_Bucket"]).mean()

    print(
        f"\n\ncity_day.csv: {n_with_aqi} rows with AQI + PM; "
        f"{n} where PM drives the AQI ({n / n_with_aqi:.1%})\n"
        f"  category match       {category_match:.1%}\n"
        f"  within one category  {within_one:.1%}\n"
        f"  3-band match         {band_match:.1%}\n"
        f"  |dAQI| <= 10         {within_10:.1%}\n"
        f"  median |dAQI|        {diff.median():.1f}\n"
        f"  (oracle AQI vs its own AQI_Bucket: {bucket_agrees:.1%})\n"
        "  rows = oracle category, cols = ours:"
    )
    table = pd.crosstab(
        pd.Categorical(true_cat, CATEGORIES), pd.Categorical(ours_cat, CATEGORIES),
        rownames=["oracle"], colnames=["ours"], dropna=False,
    )
    print(table.to_string())

    # Floors sit just under the observed rates (74% / 85% / 99% on first run).
    # Misses are almost all off by one category, and our AQI runs ~10 lower than
    # the oracle's; the oracle tracks a d/d-1 PM average better than same-day PM,
    # consistent with 24-h rolling means rather than a breakpoint bug.
    assert n > 1000
    assert within_one >= 0.98
    assert band_match >= 0.83
    assert category_match >= 0.72
