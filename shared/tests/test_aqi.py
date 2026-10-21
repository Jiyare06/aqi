import math

import pytest

from cpcb_aqi import (
    BANDS,
    CATEGORIES,
    aqi_band,
    aqi_category,
    category_to_band,
    cpcb_aqi,
    label_pm,
    pm10_subindex,
    pm25_subindex,
)


@pytest.mark.parametrize(
    "conc, expected",
    [(0, 0), (30, 50), (60, 100), (90, 200), (120, 300), (250, 400),
     (15, 25), (75, 150), (185, 350), (380, 500)],
)
def test_pm25_subindex(conc, expected):
    assert pm25_subindex(conc) == pytest.approx(expected)


@pytest.mark.parametrize(
    "conc, expected",
    [(0, 0), (50, 50), (100, 100), (250, 200), (350, 300), (430, 400),
     (175, 150), (390, 350), (510, 500)],
)
def test_pm10_subindex(conc, expected):
    assert pm10_subindex(conc) == pytest.approx(expected)


@pytest.mark.parametrize("fn", [pm25_subindex, pm10_subindex])
def test_subindex_is_continuous(fn):
    for bp in (30, 50, 60, 90, 100, 120, 250, 350, 430):
        assert fn(bp - 1e-6) == pytest.approx(fn(bp + 1e-6), abs=1e-3)


def test_no_cap_above_500():
    assert pm25_subindex(1000) > 500


@pytest.mark.parametrize("fn", [pm25_subindex, pm10_subindex])
@pytest.mark.parametrize("missing", [None, math.nan])
def test_missing_is_none(fn, missing):
    assert fn(missing) is None


@pytest.mark.parametrize("fn", [pm25_subindex, pm10_subindex])
def test_negative_raises(fn):
    with pytest.raises(ValueError):
        fn(-1)


def test_aqi_is_max_of_subindices():
    # PM2.5 75 -> 150, PM10 390 -> 350
    assert cpcb_aqi(75, 390) == pytest.approx(350)
    assert cpcb_aqi(185, 50) == pytest.approx(350)


def test_aqi_with_one_pollutant():
    assert cpcb_aqi(pm25=60) == pytest.approx(100)
    assert cpcb_aqi(pm10=250) == pytest.approx(200)
    assert cpcb_aqi(math.nan, 250) == pytest.approx(200)


def test_aqi_both_missing():
    assert cpcb_aqi() is None
    assert cpcb_aqi(math.nan, None) is None
    assert label_pm(None, math.nan) is None


@pytest.mark.parametrize(
    "aqi, expected",
    [(0, "Good"), (50, "Good"), (50.01, "Satisfactory"), (100, "Satisfactory"),
     (100.01, "Moderate"), (200, "Moderate"), (200.01, "Poor"), (300, "Poor"),
     (300.01, "Very Poor"), (400, "Very Poor"), (400.01, "Severe"), (900, "Severe")],
)
def test_category_edges(aqi, expected):
    assert aqi_category(aqi) == expected


def test_category_missing():
    assert aqi_category(None) is None
    assert aqi_band(math.nan) is None


def test_all_categories_map_to_three_bands():
    assert [category_to_band(c) for c in CATEGORIES] == [0, 0, 1, 1, 2, 2]
    assert len(BANDS) == 3


@pytest.mark.parametrize("aqi, band", [(100, 0), (100.01, 1), (300, 1), (300.01, 2)])
def test_band_edges(aqi, band):
    assert aqi_band(aqi) == band


def test_label_pm():
    aqi, category, band = label_pm(pm25=185, pm10=None)
    assert aqi == pytest.approx(350)
    assert category == "Very Poor"
    assert band == 2
