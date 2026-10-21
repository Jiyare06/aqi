"""How well does CAMS (via Open-Meteo) match ground stations? Daily PM2.5, 2023-2024.

    uv run python validate_cams.py

City-day station PM2.5 = mean over the city's stations of each station's daily mean
(station-day kept if >= 18 distinct hours have data). CAMS daily = mean of hourly values
(>= 18 hours) at the mean station location. Bands use the shared CPCB-style function on
PM2.5 only. CAMS is scored raw and x CORRECTION (backend combine.CAMS_CORRECTION).
Writes runs/cams_validation/{city}_daily.csv and summary.json.
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from cpcb_aqi import aqi_band, aqi_category, cpcb_aqi

from hazeml.paths import data_dir

OUT = Path(__file__).resolve().parent / "runs" / "cams_validation"
START, END = "2023-01-01", "2024-12-31"
MIN_HOURS = 18
CORRECTION = 1.25  # keep in sync with backend/src/air/combine.py CAMS_CORRECTION
STATE_DIRS = {"DL": "Delhi_NCT", "TG": "Telangana", "BR": "Bihar"}


def city_stations(stations: pd.DataFrame, city: str) -> pd.DataFrame:
    if city == "Delhi":
        return stations[stations["state_code"] == "DL"]
    return stations[stations["city"] == city]


def station_daily(path: Path) -> pd.Series:
    df = pd.read_csv(path, usecols=["pm2.5", "timestamp"], parse_dates=["timestamp"])
    df = df[(df["timestamp"] >= START) & (df["timestamp"] < pd.Timestamp(END) + pd.Timedelta(days=1))]
    df = df[df["pm2.5"].notna() & (df["pm2.5"] > 0)]
    if df.empty:
        return pd.Series(dtype=float)
    day = df["timestamp"].dt.normalize()
    hours = df.groupby(day)["timestamp"].agg(lambda s: s.dt.hour.nunique())
    mean = df.groupby(day)["pm2.5"].mean()
    return mean[hours >= MIN_HOURS]


def cams_daily(lat: float, lon: float) -> pd.Series:
    params = {"latitude": round(lat, 4), "longitude": round(lon, 4), "hourly": "pm2_5", "timezone": "Asia/Kolkata",
              "start_date": START, "end_date": END, "domains": "cams_global"}
    url = "https://air-quality-api.open-meteo.com/v1/air-quality?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hazebusters/0.1"}), timeout=60) as r:
        h = json.load(r)["hourly"]
    s = pd.Series(h["pm2_5"], index=pd.to_datetime(h["time"]), dtype=float).dropna()
    day = s.index.normalize()
    counts = s.groupby(day).size()
    return s.groupby(day).mean()[counts >= MIN_HOURS]


def band_of(pm25: float) -> int:
    return aqi_band(round(cpcb_aqi(pm25=pm25)))


def category_of(pm25: float) -> str:
    return aqi_category(round(cpcb_aqi(pm25=pm25)))


def score(station: pd.Series, cams: pd.Series) -> dict:
    sb, cb = station.map(band_of), cams.map(band_of)
    diff = cams - station
    confusion = pd.crosstab(sb, cb).reindex(index=range(3), columns=range(3), fill_value=0)
    return {
        "same_band_share": float((sb == cb).mean()),
        "same_category_share": float((station.map(category_of) == cams.map(category_of)).mean()),
        "vp_severe_station_days": int((sb == 2).sum()),
        "vp_severe_caught": int(((sb == 2) & (cb == 2)).sum()),
        "vp_severe_false_alarms": int(((sb != 2) & (cb == 2)).sum()),
        "bias_mean_ugm3": float(diff.mean()), "bias_median_ugm3": float(diff.median()),
        "ratio_median_cams_over_station": float((cams / station).median()),
        "mae_ugm3": float(diff.abs().mean()), "pearson_r": float(np.corrcoef(cams, station)[0, 1]),
        "cams_pm25_median": float(cams.median()),
        "band_confusion_rows_station_cols_cams": confusion.values.tolist(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    root = data_dir() / "stations"
    stations = pd.read_csv(root / "stations.csv")
    summary = {"period": [START, END], "min_hours_per_day": MIN_HOURS, "metric": "daily mean PM2.5 (µg/m³)", "correction": CORRECTION,
               "cams": "Open-Meteo air-quality API, domains=cams_global (0.4°), hourly pm2_5, Asia/Kolkata", "cities": {}}
    for city in ("Delhi", "Hyderabad", "Patna"):
        st = city_stations(stations, city)
        per_station = {}
        for code, state in zip(st["station_code"], st["state_code"]):
            path = root / STATE_DIRS[state] / f"{code}.csv"
            if path.exists():
                daily = station_daily(path)
                if len(daily):
                    per_station[code] = daily
        ground = pd.DataFrame(per_station)
        city_day = pd.DataFrame({"station_pm25": ground.mean(axis=1), "n_stations": ground.notna().sum(axis=1)})
        lat, lon = st["latitude"].mean(), st["longitude"].mean()
        cams = cams_daily(lat, lon).rename("cams_pm25")
        df = city_day.join(cams, how="inner").dropna()
        df["cams_pm25_x125"] = df["cams_pm25"] * CORRECTION
        df["station_band"] = df["station_pm25"].map(band_of)
        df["cams_band"] = df["cams_pm25"].map(band_of)
        df["cams_x125_band"] = df["cams_pm25_x125"].map(band_of)
        df.index.name = "date"
        df.to_csv(OUT / f"{city.lower()}_daily.csv")
        summary["cities"][city] = {
            "stations_in_index": int(len(st)), "stations_with_data": int(len(per_station)),
            "location": [round(lat, 4), round(lon, 4)], "days": int(len(df)),
            "station_pm25_median": float(df["station_pm25"].median()),
            "uncorrected": score(df["station_pm25"], df["cams_pm25"]),
            f"corrected_x{CORRECTION}": score(df["station_pm25"], df["cams_pm25_x125"]),
        }
        for name in ("uncorrected", f"corrected_x{CORRECTION}"):
            c = summary["cities"][city][name]
            print(f"{city} {name}: {len(df)} days, same band {c['same_band_share']:.1%}, VP/Severe caught "
                  f"{c['vp_severe_caught']}/{c['vp_severe_station_days']}, false alarms {c['vp_severe_false_alarms']}, "
                  f"bias median {c['bias_median_ugm3']:+.1f} µg/m³", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
