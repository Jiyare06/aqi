"""Combine WAQI (measured) and CAMS (modelled) into the contract's `air`, `forecast_today`,
`forecast_tomorrow` and `decision`.

Rules (CLAUDE.md §3; rule table with CPCB health statements in decision_rules.md):
- Source: measured if the station is <= 25 km away and updated < 3 h ago, else modelled. Both are returned.
- Confidence: high = same band in both and station <= 25 km; medium = only one of those;
  low = neither, or modelled only.
- Decision window = the profile's outdoor hours (school 07-17, worker 06-19, general 06-21):
  before/inside today's hours -> today from now (decision.for = "today"); after today's hours ->
  tomorrow's hours (decision.for = "tomorrow").
- outdoor = rule on the WORST AQI in the window: the forecast slots overlapping it, plus the current
  chosen AQI when we're inside the hours. <= 100 yes, 101-200 limit, > 200 no (same rule for every
  profile; the profile sets the hours and the advice wording).
- best_window: earliest "yes" slot in the window, clipped to it; none -> null and
  reason_code "no_window_today" / "no_window_tomorrow".
- Whenever the final source is modelled (no station, or station > 25 km / >= 3 h old), CAMS PM is
  scaled by CAMS_CORRECTION before the AQI, the decision and the forecast, and a caveat says so.
"""

import json
import math
from datetime import datetime, timedelta

from air import openmeteo, waqi

MAX_STATION_KM = 25.0
OPENMETEO_ATTEMPTS = 2
MAX_STATION_AGE = timedelta(hours=3)

# CAMS daily PM2.5 ran ~0.8x the ground stations: median CAMS/station ratio 0.78 (Delhi), 0.79 (Hyderabad),
# 0.81 (Patna), 2023-24 (ml/runs/cams_validation/summary.json, from ml/validate_cams.py; ml/RESULTS.md §2).
CAMS_CORRECTION = 1.25

NO_STATION_NOTE = "No official monitoring station near this location"
CORRECTION_NOTE = (f"so this uses the CAMS model estimate with PM2.5/PM10 scaled x{CAMS_CORRECTION} (CAMS read about "
                   "0.8x ground stations in Delhi, Hyderabad and Patna, daily PM2.5, 2023-24).")

# Hours (local) in which a profile would go outdoors; best_window is clipped to these.
PROFILE_WINDOWS = {"school": (7, 17), "worker": (6, 19), "general": (6, 21)}

# (upper AQI bound inclusive, outdoor, reason_code)
DECISION_RULES = ((100, "yes", "aqi_le_100"), (200, "limit", "aqi_101_200"), (math.inf, "no", "aqi_gt_200"))


def decide(aqi: int) -> tuple[str, str]:
    for upper, outdoor, code in DECISION_RULES:
        if aqi <= upper:
            return outdoor, code
    raise AssertionError("unreachable")


def _fresh(measured: dict, now: datetime) -> bool:
    return now - datetime.fromisoformat(measured["updated_at"]) < MAX_STATION_AGE


def station_usable(measured: dict | None, now: datetime) -> bool:
    return bool(measured) and measured["dist_km"] <= MAX_STATION_KM and _fresh(measured, now)


def choose_source(measured: dict | None, modelled: dict | None, now: datetime) -> str:
    if station_usable(measured, now):
        return "measured"
    if modelled is None:
        if measured is None:
            raise ValueError("no air data: WAQI and Open-Meteo both unavailable")
        return "measured"  # CAMS failed; a far/stale station is better than nothing (flagged in caveats)
    return "modelled"


def confidence(measured: dict | None, modelled: dict | None) -> tuple[str, str]:
    if measured is None:
        return "low", "No station reading; CAMS model value only (~45 km grid)."
    if modelled is None:
        return "low", "CAMS model value unavailable; cannot cross-check the station."
    near = measured["dist_km"] <= MAX_STATION_KM
    agree = measured["band"] == modelled["band"]
    where = f"station {measured['dist_km']} km away"
    if near and agree:
        return "high", f"Station and CAMS agree on the band; {where}."
    if near:
        return "medium", f"Station and CAMS disagree on the band; {where}."
    if agree:
        return "medium", f"Station and CAMS agree on the band, but the {where} (> {MAX_STATION_KM:.0f} km)."
    return "low", f"Station and CAMS disagree on the band and the {where} (> {MAX_STATION_KM:.0f} km)."


def _profile_hours(midnight: datetime, profile: str) -> tuple[datetime, datetime]:
    lo, hi = PROFILE_WINDOWS[profile]
    return midnight + timedelta(hours=lo), midnight + timedelta(hours=hi)


def decision(current_aqi: int, forecast: list[dict], now: datetime, profile: str = "general") -> dict:
    """Contract `decision` for the profile's next outdoor hours (see module docstring).

    `forecast`: 3-hour slots for today and tomorrow (time = slot start, local)."""
    # Profile hours are Indian local time; without a forecast use IST, not `now`'s zone (UTC on Lambda).
    tz = datetime.fromisoformat(forecast[0]["time"]).tzinfo if forecast else openmeteo.IST
    local = now.astimezone(tz)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    start, end = _profile_hours(midnight, profile)
    if local < end:
        day, inside, start = "today", local >= start, max(start, local)
    else:
        day, inside = "tomorrow", False
        start, end = _profile_hours(midnight + timedelta(days=1), profile)

    slots = []
    for slot in forecast:
        s0 = datetime.fromisoformat(slot["time"])
        s, e = max(s0, start), min(s0 + timedelta(hours=3), end)
        if s < e:
            slots.append((s, e, slot["aqi_cpcb"]))
    aqis = [a for _, _, a in slots] + ([current_aqi] if inside else [])
    outdoor, code = decide(max(aqis) if aqis else current_aqi)
    window = next(({"start": s.isoformat(), "end": e.isoformat()} for s, e, a in slots if decide(a)[0] == "yes"), None)
    if window is None:
        code = f"no_window_{day}"
    return {"for": day, "outdoor": outdoor, "best_window": window, "reason_code": code}


def _day(forecast: list[dict], now: datetime, offset: int) -> list[dict]:
    """Slots on the local date `offset` days after `now` (0 = today, 1 = tomorrow)."""
    if not forecast:
        return []
    tz = datetime.fromisoformat(forecast[0]["time"]).tzinfo
    day = (now.astimezone(tz) + timedelta(days=offset)).date()
    return [s for s in forecast if datetime.fromisoformat(s["time"]).date() == day]


def _approx_km(d: float) -> str:
    return f"~{round(d, -1):.0f} km" if d >= 50 else f"~{d:.0f} km"


def no_station_note(nearest: dict | None) -> str:
    """No official station within 25 km, naming the nearest one from our index (no final full stop)."""
    if not nearest:
        return NO_STATION_NOTE
    return f"No official station within {MAX_STATION_KM:.0f} km; the nearest is {nearest['city']} ({_approx_km(nearest['dist_km'])})"


def _age(updated_at: str, now: datetime) -> str:
    seconds = (now - datetime.fromisoformat(updated_at)).total_seconds()
    if seconds < 3600:
        return f"{max(int(seconds // 60), 0)} min ago"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.0f} h ago"
    return f"{seconds / 86400:.0f} days ago ({datetime.fromisoformat(updated_at):%d %b %Y})"


def station_reason(measured: dict | None, nearest: dict | None, waqi_note: str | None, now: datetime) -> str | None:
    """Why there's no usable station reading (one clause, no final full stop), or None if the station is usable."""
    if measured is not None:
        far, stale = measured["dist_km"] > MAX_STATION_KM, not _fresh(measured, now)
        name, d = measured["station"], measured["dist_km"]
        if far and stale:
            return f"The nearest station, {name}, is {d} km away and last updated {_age(measured['updated_at'], now)}"
        if far:
            return f"The nearest station, {name}, is {d} km away (more than {MAX_STATION_KM:.0f} km)"
        if stale:
            return (f"The nearest station, {name} ({d} km away), last updated "
                    f"{_age(measured['updated_at'], now)} (more than 3 h)")
        return None
    if nearest is not None:
        where = f"The nearest official station, {nearest['station']} ({nearest['dist_km']} km away),"
        if nearest.get("updated_at"):
            return f"{where} last reported {_age(nearest['updated_at'], now)}"
        return f"{where} is not reporting to WAQI"
    return (waqi_note or "No WAQI station reading available").rstrip(".")


def combine(measured: dict | None, modelled: dict | None, forecast: list[dict], now: datetime,
            profile: str = "general", corrected: bool = False, waqi_note: str | None = None,
            nearest: dict | None = None) -> dict:
    """-> {"air", "forecast_today", "forecast_tomorrow", "decision", "caveats"} (contract fragments).

    `forecast`: today's and tomorrow's 3-hour slots. `corrected`: modelled/forecast PM were already
    scaled by CAMS_CORRECTION (air_for does this whenever the station isn't usable). `nearest`: the nearest
    official station when WAQI has no reading (waqi.lookup). The station situation and the CAMS correction are
    one caveat: "<why no usable station>, so this uses the CAMS model estimate with PM scaled x1.25 (...)"."""
    if profile not in PROFILE_WINDOWS:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILE_WINDOWS)}")
    used = choose_source(measured, modelled, now)
    chosen = measured if used == "measured" else modelled
    level, reason = confidence(measured, modelled)

    caveats = ["CPCB-style AQI from hourly PM (official CPCB AQI uses 24-h averages and >= 3 pollutants)."]
    if modelled is not None:
        caveats.append("CAMS is a model value for a ~45 km grid cell, not a measurement.")
    why = station_reason(measured, nearest, waqi_note, now)
    # The station/CAMS caveat goes first (the result page leads with the first caveat).
    if corrected:
        caveats.insert(0, f"{why}, {CORRECTION_NOTE}")
    elif why and used == "measured":  # CAMS unavailable, so a far/stale station is used anyway
        caveats.insert(0, f"{why}; it is used anyway because the CAMS model value is unavailable.")
    elif why:
        caveats.insert(0, f"{why}.")
    if not forecast:
        caveats.append("No CAMS forecast; the decision uses the current AQI only.")
    if measured is not None:
        caveats.append("WAQI US AQI sub-indices converted back to µg/m³ (US EPA 2012 breakpoints).")

    return {
        "air": {"measured": measured, "modelled": modelled, "aqi_cpcb": chosen["aqi_cpcb"], "band": chosen["band"],
                "used": used, "confidence": level, "confidence_reason": reason},
        "forecast_today": _day(forecast, now, 0),
        "forecast_tomorrow": _day(forecast, now, 1),
        "decision": decision(chosen["aqi_cpcb"], forecast, now, profile),
        "caveats": list(dict.fromkeys(caveats)),
    }


def air_for(lat: float, lon: float, token: str | None, now: datetime | None = None, profile: str = "general") -> dict:
    """Fetch both sources and combine. A failing source becomes None plus a caveat."""
    now = now or datetime.now().astimezone()
    measured, nearest, waqi_note = None, None, None
    if token:
        try:
            measured, nearest = waqi.lookup(lat, lon, token)
            if measured is None and nearest is None:
                waqi_note = "Nearest WAQI station reports no PM2.5 or PM10."
        except waqi.NoStationError as e:
            waqi_note = no_station_note(e.nearest)
        except Exception as e:  # invalid key, HTTP error, timeout: a real "unavailable"
            waqi_note = f"WAQI unavailable ({type(e).__name__}: {e})."
    else:
        waqi_note = "WAQI not queried (no WAQI_TOKEN)."
    scale = 1.0 if station_usable(measured, now) else CAMS_CORRECTION  # i.e. whenever `used` will be modelled
    modelled, forecast, notes = None, [], []
    for attempt in range(1, OPENMETEO_ATTEMPTS + 1):  # one retry: a single slow response (10 s timeout) seen live
        try:
            hours = openmeteo.parse_hourly(openmeteo.fetch(lat, lon))
            modelled, forecast = openmeteo.modelled_now(hours, now, scale), openmeteo.forecast_3h(hours, scale)
            break
        except Exception as e:
            print(json.dumps({"open_meteo": "failed", "attempt": attempt, "reason": f"{type(e).__name__}: {e}"}))
            if attempt == OPENMETEO_ATTEMPTS:
                notes.append(f"Open-Meteo unavailable ({type(e).__name__}: {e}).")
    out = combine(measured, modelled, forecast, now, profile,
                  corrected=scale != 1.0 and modelled is not None, waqi_note=waqi_note, nearest=nearest)
    out["caveats"] = list(dict.fromkeys(out["caveats"] + notes))
    return out
