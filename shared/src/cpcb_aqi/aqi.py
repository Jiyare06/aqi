"""CPCB-style AQI from PM2.5 / PM10 (CLAUDE.md §5).

Every dataset's labels are recomputed with these functions so they share one
definition. "CPCB-style": the official AQI uses 24-h averages and >= 3
pollutants; here we only use whichever of PM2.5 / PM10 is available.

Sub-index = linear interpolation within the breakpoint row. Upper row values
are treated as breakpoints (PM2.5 30 -> 50, 30.5 -> Satisfactory), so the
mapping is continuous with no gaps. Above the top row the last slope
continues; the AQI is not capped at 500.

CPCB's own formula interpolates within integer rows (e.g. PM2.5 31-60 ->
AQI 51-100), so near the bottom of a row our value can differ from it by
about 1 AQI (PM2.5 31: CPCB 51, ours 51.7). Categories only differ for
concentrations falling between the integer rows (e.g. 30.5).

Pure Python (no numpy/pandas) so the backend Lambda can import it as is.
"""

import math

# (concentration µg/m³, AQI) breakpoints, row by row from the CPCB table.
PM25_BREAKPOINTS = ((0, 0), (30, 50), (60, 100), (90, 200), (120, 300), (250, 400))
PM10_BREAKPOINTS = ((0, 0), (50, 50), (100, 100), (250, 200), (350, 300), (430, 400))

CATEGORIES = ("Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe")
# Inclusive upper AQI bound of each category; Severe is everything above 400.
CATEGORY_UPPER = (50, 100, 200, 300, 400, math.inf)

BANDS = ("Good/Satisfactory", "Moderate/Poor", "Very Poor/Severe")
CATEGORY_TO_BAND = {
    "Good": 0,
    "Satisfactory": 0,
    "Moderate": 1,
    "Poor": 1,
    "Very Poor": 2,
    "Severe": 2,
}


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _subindex(x, breakpoints) -> float | None:
    if _missing(x):
        return None
    x = float(x)
    if x < 0:
        raise ValueError(f"negative concentration: {x}")
    for (c_lo, i_lo), (c_hi, i_hi) in zip(breakpoints, breakpoints[1:]):
        if x <= c_hi:
            return i_lo + (x - c_lo) * (i_hi - i_lo) / (c_hi - c_lo)
    # Above the top breakpoint: keep the last row's slope.
    (c_lo, i_lo), (c_hi, i_hi) = breakpoints[-2], breakpoints[-1]
    return i_hi + (x - c_hi) * (i_hi - i_lo) / (c_hi - c_lo)


def pm25_subindex(pm25) -> float | None:
    return _subindex(pm25, PM25_BREAKPOINTS)


def pm10_subindex(pm10) -> float | None:
    return _subindex(pm10, PM10_BREAKPOINTS)


def cpcb_aqi(pm25=None, pm10=None) -> float | None:
    """max of the available PM sub-indices; None if both are missing."""
    subs = [s for s in (pm25_subindex(pm25), pm10_subindex(pm10)) if s is not None]
    return max(subs) if subs else None


def aqi_category(aqi) -> str | None:
    if _missing(aqi):
        return None
    for name, upper in zip(CATEGORIES, CATEGORY_UPPER):
        if aqi <= upper:
            return name
    raise AssertionError("unreachable")


def category_to_band(category: str) -> int:
    return CATEGORY_TO_BAND[category]


def aqi_band(aqi) -> int | None:
    category = aqi_category(aqi)
    return None if category is None else category_to_band(category)


def label_pm(pm25=None, pm10=None) -> tuple[float, str, int] | None:
    """(aqi, category, band) from raw PM; None if both PM values are missing."""
    aqi = cpcb_aqi(pm25, pm10)
    if aqi is None:
        return None
    category = aqi_category(aqi)
    return aqi, category, category_to_band(category)
