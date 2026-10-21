"""Build src/air/official_stations.json (code, city, state, lat, lon) from the stations dataset.

    uv run python build_station_index.py        # reads $DATA_DIR/stations/stations.csv (CLAUDE.md §6.3)

Used only to tell "no official station near you" apart from "official stations exist but aren't reporting to
WAQI" (WAQI's geo feed and map hide stations that stopped reporting). Source: Kaggle "India AQI 2010-2024"
(omsandeeppatil/indian-aqi-stations), CC BY-SA 4.0; the JSON carries the same licence and credit.
"""

import csv
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent / "src" / "air" / "official_stations.json"


def _data_dir() -> Path:
    if os.environ.get("DATA_DIR"):
        return Path(os.environ["DATA_DIR"])
    env = Path(__file__).resolve().parents[1] / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("DATA_DIR="):
            return Path(line.split("=", 1)[1].strip().strip("'\""))
    raise SystemExit("DATA_DIR not set (env or repo-root .env)")


def main() -> None:
    rows = list(csv.DictReader(open(_data_dir() / "stations" / "stations.csv")))
    stations = [{"code": r["station_code"], "city": r["city"], "state": r["state_code"],
                 "lat": round(float(r["latitude"]), 5), "lon": round(float(r["longitude"]), 5)}
                for r in rows if r["latitude"] and r["longitude"]]
    out = {"source": "Kaggle omsandeeppatil/indian-aqi-stations (India AQI 2010-2024), stations.csv",
           "source_url": "https://www.kaggle.com/datasets/omsandeeppatil/indian-aqi-stations",
           "licence": "CC BY-SA 4.0", "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/",
           "share_alike": "This file is a derivative and is shared under the same licence (CC BY-SA 4.0).",
           "changes": (f"Kept columns station_code, city, state_code, latitude, longitude (renamed code, city, state, "
                       f"lat, lon); coordinates rounded to 5 decimals; dropped {len(rows) - len(stations)} station(s) "
                       "without coordinates. Built by backend/build_station_index.py."),
           "fields": "CPCB station code, city, state code, lat, lon", "stations": stations}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{len(stations)} stations -> {OUT} ({OUT.stat().st_size / 1024:.0f} KB); skipped {len(rows) - len(stations)}")


if __name__ == "__main__":
    main()
