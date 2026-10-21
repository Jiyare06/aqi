from pathlib import Path

import pandas as pd
import pytest

from hazeml.paths import data_dir
from hazeml.traqid import SPLIT_FILE, SPLITS, assign_split, load_traqid, make_date_split


def _synthetic() -> pd.DataFrame:
    # 10 dates x 100 rows; band mix varies by date.
    rows = []
    for d in range(10):
        for i in range(100):
            rows.append({"date": f"2024-01-{d + 1:02d}", "band": (i + d * 7) % 3 if i < 60 else 0})
    return pd.DataFrame(rows)


def test_make_date_split_is_by_date_and_sized():
    df = _synthetic()
    split = make_date_split(df)
    assert set(split) == set(df["date"])
    assert set(split.values()) == set(SPLITS)
    share = df["date"].map(split).value_counts(normalize=True)
    assert share["test"] == pytest.approx(0.2, abs=0.05)
    assert share["val"] == pytest.approx(0.1, abs=0.05)


def test_make_date_split_is_deterministic():
    df = _synthetic()
    assert make_date_split(df) == make_date_split(df.sample(frac=1, random_state=0))


def test_assign_split_rejects_unknown_dates(tmp_path):
    path = tmp_path / "split.json"
    path.write_text('{"2024-01-01": "train"}')
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"]})
    with pytest.raises(ValueError, match="2024-01-02"):
        assign_split(df, path)


# --- real data (skipped without $DATA_DIR/traqid) ---------------------------


@pytest.fixture(scope="module")
def traqid() -> pd.DataFrame:
    try:
        root = data_dir() / "traqid"
    except RuntimeError:
        pytest.skip("DATA_DIR not set")
    if not (root / "TRAQID.csv").exists():
        pytest.skip(f"{root / 'TRAQID.csv'} not found")
    return assign_split(load_traqid(root))


def test_day_front_only(traqid):
    assert (traqid["day_or_night"] == "Day").all()
    assert traqid["path"].str.contains("/front/").all()
    assert traqid["image_id"].is_unique
    # Spot-check a couple of paths without listing the folder.
    assert Path(traqid["path"].iloc[0]).is_file()
    assert Path(traqid["path"].iloc[-1]).is_file()


def test_labels_agree_with_traqid(traqid):
    assert traqid["band"].notna().all()
    assert (traqid["category"] == traqid["src_aqi_cat"]).mean() > 0.99


def test_no_date_in_two_splits(traqid):
    assert (traqid.groupby("date")["split"].nunique() == 1).all()


def test_every_split_has_every_band(traqid):
    counts = pd.crosstab(traqid["split"], traqid["band"])
    assert (counts > 0).all().all()
    assert (counts[2] >= 100).all()


def test_frozen_split_matches_recomputed(traqid):
    # The committed JSON should be what make_date_split gives on today's data.
    assert make_date_split(traqid) == traqid.groupby("date")["split"].first().to_dict()
    assert SPLIT_FILE.exists()
