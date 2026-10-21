"""Nearest WAQI station, converted back to µg/m³ and then to the CPCB-style AQI.

WAQI field semantics (checked 9 Oct 2026):
- https://aqicn.org/api/ : the feed gives "Individual AQI for all pollutants"; raw concentrations
  are listed only as a planned feature. So `data.iaqi.pm25.v` / `pm10.v` are **US AQI sub-indices**,
  not µg/m³, and must be inverted before applying CPCB breakpoints.
- https://aqicn.org/faq/2013-09-09/revised-pm25-aqi-breakpoints/ : WAQI uses the 2012 US EPA PM2.5
  breakpoints (Good 0-12.0 µg/m³) for all countries. The EPA's 2024 revision (Good 0-9.0) is not
  mentioned there, so we invert with the 2012 table (PM25_US_2012). If WAQI switches, change it here.
- The geo feed (`/feed/geo:{lat};{lon}/`) returns the nearest station however far away it is,
  so we compute the distance ourselves from `data.city.geo` ([lat, lon]).
- `data.time.iso` is the station's last update with its UTC offset.
- WAQI values are real-time (hourly), not CPCB's 24-h averages; the result is "CPCB-style".
- Error statuses (checked live 9-10 Oct 2026, valid token): the geo feed returns `{"status": "nope",
  "data": "can not connect"}` for Ranchi (no station at all) **and for Hyderabad**, whose 9 WAQI stations
  all stopped reporting on 2026-06-23. WAQI's map (`/v2/map/bounds`) also hides stations that stopped
  reporting; only `/search/?keyword=` still lists them, with their last update. So "can not connect" does not
  mean "no station"; lookup() falls back to the map (stations still reporting within 25 km), then to our
  index of official stations (official_stations.json) + WAQI search for the stale case. A bad token gives
  `"Invalid key"` (InvalidKeyError). HTTP errors and timeouts propagate from http.get_json unchanged.
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

from cpcb_aqi import aqi_band, cpcb_aqi

from air import http

URL = "https://api.waqi.info/feed/geo:{lat};{lon}/"
UID_URL = "https://api.waqi.info/feed/@{uid}/"
BOUNDS_URL = "https://api.waqi.info/v2/map/bounds"
SEARCH_URL = "https://api.waqi.info/search/"
RADIUS_KM = 25.0  # same as combine.MAX_STATION_KM
STATIONS_FILE = Path(__file__).resolve().parent / "official_stations.json"
SOURCE = "WAQI"

# (AQI lo, AQI hi, conc lo, conc hi) — US EPA 2012 PM2.5 (µg/m³) and PM10 (µg/m³).
PM25_US_2012 = ((0, 50, 0.0, 12.0), (51, 100, 12.1, 35.4), (101, 150, 35.5, 55.4), (151, 200, 55.5, 150.4),
                (201, 300, 150.5, 250.4), (301, 400, 250.5, 350.4), (401, 500, 350.5, 500.4))
PM10_US = ((0, 50, 0, 54), (51, 100, 55, 154), (101, 150, 155, 254), (151, 200, 255, 354),
           (201, 300, 355, 424), (301, 400, 425, 504), (401, 500, 505, 604))


class WaqiError(RuntimeError):
    pass


class NoStationError(WaqiError):
    """No official station within 25 km (not an outage). `nearest`: the nearest official station anywhere
    ({code, city, dist_km, ...} from official_stations.json), or None."""

    def __init__(self, msg: str, nearest: dict | None = None):
        super().__init__(msg)
        self.nearest = nearest


class InvalidKeyError(WaqiError):
    pass


NO_STATION_MESSAGES = ("can not connect", "Unknown station")


def us_aqi_to_conc(aqi: float, table) -> float:
    """Invert the EPA piecewise-linear formula. Above 500 the top row's slope continues."""
    if aqi < 0:
        raise ValueError(f"negative AQI: {aqi}")
    for i_lo, i_hi, c_lo, c_hi in table:
        if aqi <= i_hi:
            return c_lo + (aqi - i_lo) * (c_hi - c_lo) / (i_hi - i_lo)
    i_lo, i_hi, c_lo, c_hi = table[-1]
    return c_hi + (aqi - i_hi) * (c_hi - c_lo) / (i_hi - i_lo)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def fetch(lat: float, lon: float, token: str) -> dict:
    return http.get_json(URL.format(lat=lat, lon=lon), {"token": token})


def _num(iaqi: dict, key: str) -> float | None:
    v = iaqi.get(key, {}).get("v")
    return float(v) if isinstance(v, (int, float)) else None  # offline stations report "-"


def parse(payload: dict, lat: float, lon: float) -> dict | None:
    """-> contract `air.measured`, or None when the station has neither PM2.5 nor PM10."""
    if payload.get("status") != "ok":
        msg = payload.get("data")
        if msg in NO_STATION_MESSAGES:
            raise NoStationError(f"WAQI: {msg}")
        if msg == "Invalid key":
            raise InvalidKeyError(f"WAQI error: {msg}")
        raise WaqiError(f"WAQI error: {msg}")
    data = payload["data"]
    iaqi = data.get("iaqi", {})
    us25, us10 = _num(iaqi, "pm25"), _num(iaqi, "pm10")
    if us25 is None and us10 is None:
        return None
    pm25 = None if us25 is None else us_aqi_to_conc(us25, PM25_US_2012)
    pm10 = None if us10 is None else us_aqi_to_conc(us10, PM10_US)
    aqi = round(cpcb_aqi(pm25, pm10))
    s_lat, s_lon = data["city"]["geo"]
    return {
        "source": SOURCE,
        "station": data["city"].get("name"),
        "dist_km": round(haversine_km(lat, lon, float(s_lat), float(s_lon)), 1),
        "updated_at": datetime.fromisoformat(data["time"]["iso"]).isoformat(),
        "pm25": None if pm25 is None else round(pm25, 1),
        "pm10": None if pm10 is None else round(pm10, 1),
        "aqi_cpcb": aqi,
        "band": aqi_band(aqi),
    }


# --- lookup: geo feed, then stations still reporting, then official stations that stopped reporting ------------


def official_by_distance(lat: float, lon: float) -> list[dict]:
    """All official (CPCB) stations from our index, nearest first, with dist_km."""
    stations = json.loads(STATIONS_FILE.read_text())["stations"]
    near = [{**s, "dist_km": round(haversine_km(lat, lon, s["lat"], s["lon"]), 1)} for s in stations]
    return sorted(near, key=lambda s: s["dist_km"])


def official_within(lat: float, lon: float, radius_km: float = RADIUS_KM) -> list[dict]:
    """Official stations within `radius_km`, nearest first."""
    return [s for s in official_by_distance(lat, lon) if s["dist_km"] <= radius_km]


def _box(lat: float, lon: float, km: float) -> str:
    dlat, dlon = km / 111.0, km / (111.0 * max(math.cos(math.radians(lat)), 0.1))
    return f"{lat - dlat},{lon - dlon},{lat + dlat},{lon + dlon}"


def reporting_within(lat: float, lon: float, token: str, radius_km: float = RADIUS_KM) -> list[dict]:
    """Stations on WAQI's map (i.e. still reporting) within `radius_km`, nearest first: [{uid, name, dist_km}]."""
    payload = http.get_json(BOUNDS_URL, {"latlng": _box(lat, lon, radius_km), "networks": "all", "token": token})
    if payload.get("status") != "ok":
        raise WaqiError(f"WAQI map error: {payload.get('data')}")
    out = [{"uid": s["uid"], "name": s["station"]["name"],
            "dist_km": round(haversine_km(lat, lon, float(s["lat"]), float(s["lon"])), 1)} for s in payload["data"]]
    return sorted((s for s in out if s["dist_km"] <= radius_km), key=lambda s: s["dist_km"])


def search_nearest(keyword: str, lat: float, lon: float, token: str, radius_km: float = RADIUS_KM) -> dict | None:
    """Nearest WAQI search hit for `keyword` within `radius_km` (includes stations that stopped reporting):
    {station, dist_km, updated_at (ISO with offset, or None)}."""
    payload = http.get_json(SEARCH_URL, {"keyword": keyword, "token": token})
    if payload.get("status") != "ok":
        raise WaqiError(f"WAQI search error: {payload.get('data')}")
    hits = []
    for h in payload["data"]:
        geo = (h.get("station") or {}).get("geo")
        if not geo:
            continue
        d = haversine_km(lat, lon, float(geo[0]), float(geo[1]))
        if d <= radius_km:
            t = h.get("time") or {}
            updated = None
            if t.get("stime") and t.get("tz"):
                updated = datetime.fromisoformat(f"{t['stime'].replace(' ', 'T')}{t['tz']}").isoformat()
            hits.append({"station": h["station"]["name"], "dist_km": round(d, 1), "updated_at": updated})
    return min(hits, key=lambda h: h["dist_km"]) if hits else None


def lookup(lat: float, lon: float, token: str) -> tuple[dict | None, dict | None]:
    """-> (measured, nearest). `measured`: contract air.measured from the nearest station with PM (may be far or
    stale; combine decides). `nearest`: when there's no reading, the nearest official station we know of
    ({station, dist_km, updated_at|None}). Raises NoStationError only if no official station is within 25 km."""
    try:
        return parse(fetch(lat, lon, token), lat, lon), None
    except NoStationError:
        pass
    for s in reporting_within(lat, lon, token)[:1]:  # geo feed failed but the map has a live station nearby
        m = parse(http.get_json(UID_URL.format(uid=s["uid"]), {"token": token}), lat, lon)
        if m is not None:
            return m, None
    official = official_by_distance(lat, lon)
    if not official or official[0]["dist_km"] > RADIUS_KM:
        raise NoStationError("no official station within 25 km", official[0] if official else None)
    o = official[0]
    found = search_nearest(o["city"], lat, lon, token)
    return None, found or {"station": f"{o['code']}, {o['city']}", "dist_km": o["dist_km"], "updated_at": None}
