"""Advice text (contract `advice`): Bedrock (llm.py) when BEDROCK_MODEL_ID is set, else / on any failure the
EN + HI template keyed by decision.outdoor (advice.source = "bedrock" | "template").

The model only phrases the facts built here (facts()); llm.validate rejects anything else.
Health wording follows the CPCB category statements in backend/src/air/decision_rules.md; no new numbers.
Times are written as spoken words at the source in both languages ("सुबह 9 बजे से दोपहर 12 बजे तक",
"from 9 AM to 12 noon"), so the same text reads naturally on screen and in Polly; spoken_times() converts
"HH:MM" left in free text (e.g. future Bedrock output). The CPCB-scale note lives in the result's caveats.
"""

import json
import re
from datetime import datetime

from cpcb_aqi import aqi_category

import llm

# outdoor -> (recommendation, health statement or None), each as (en, hi)
TEMPLATES = {
    "yes": (("Air is fine for outdoor activity.", "बाहर की गतिविधियों के लिए हवा ठीक है।"),
            None),
    "limit": (("Limit outdoor activity: keep it short and light.", "बाहर की गतिविधि सीमित रखें, कम समय और हल्की।"),
              ("People with lung or heart disease or asthma may feel breathing discomfort.",
               "फेफड़ों या दिल की बीमारी या दमा वाले लोगों को साँस लेने में तकलीफ़ हो सकती है।")),
    "no": (("Keep activities indoors.", "गतिविधियाँ घर के अंदर रखें।"),
           ("Most people may feel breathing discomfort on prolonged exposure.",
            "ज़्यादा देर बाहर रहने पर ज़्यादातर लोगों को साँस लेने में तकलीफ़ हो सकती है।")),
}
# The AQI is the current reading; when the decision is for tomorrow, say "right now" and prefix the advice.
AQI_SENTENCE = {"today": ("Today's AQI is {aqi}.", "आज हवा का AQI {aqi} है।"),
                "tomorrow": ("The AQI right now is {aqi}.", "अभी हवा का AQI {aqi} है।")}
TOMORROW = ("Tomorrow, ", "कल ")
WINDOW = ("Best time: {span}.", "सबसे अच्छा समय: {span}।")

# CPCB health statement per category (backend/src/air/decision_rules.md); the model may not go beyond it.
CPCB_HEALTH = {
    "Good": "Minimal impact",
    "Satisfactory": "Minor breathing discomfort to sensitive people",
    "Moderate": "Breathing discomfort to the people with lungs, asthma and heart diseases",
    "Poor": "Breathing discomfort to most people on prolonged exposure",
    "Very Poor": "Respiratory illness on prolonged exposure",
    "Severe": "Affects healthy people and seriously impacts those with existing diseases",
}

# Hedged cause line, appended in code whenever smoke/fire is present (the smoke/fire model is experimental),
# keyed by the labels present (in causes.LABELS order). Bedrock never writes about causes.
CAUSE_LINES = {
    ("smoke",): ("Our camera check suggests possible smoke in your clip (experimental). If you can see a smoke "
                 "source nearby, move away from it.",
                 "कैमरा जाँच के अनुसार आपकी क्लिप में धुआँ हो सकता है (प्रयोगात्मक)। अगर पास में धुएँ का स्रोत दिखे, "
                 "तो उससे दूर रहें।"),
    ("fire",): ("Our camera check suggests possible fire in your clip (experimental). If you can see a fire nearby, "
                "move away from it.",
                "कैमरा जाँच के अनुसार आपकी क्लिप में आग हो सकती है (प्रयोगात्मक)। अगर पास में आग दिखे, तो उससे दूर रहें।"),
    ("smoke", "fire"): ("Our camera check suggests possible smoke and fire in your clip (experimental). If you can "
                        "see a smoke source or a fire nearby, move away from it.",
                        "कैमरा जाँच के अनुसार आपकी क्लिप में धुआँ और आग हो सकते हैं (प्रयोगात्मक)। अगर पास में धुएँ "
                        "का स्रोत या आग दिखे, तो उससे दूर रहें।"),
}

# Period word by hour (24 h): 4-11 morning, 12-15 afternoon, 16-18 evening, 19-3 night.
PERIODS = ((4, 11, "सुबह"), (12, 15, "दोपहर"), (16, 18, "शाम"))


def hindi_period(hour: int) -> str:
    return next((word for lo, hi, word in PERIODS if lo <= hour <= hi), "रात")


def hindi_clock(hour: int, minute: int = 0) -> str:
    """Spoken clock time without the period: 9:00 -> "9 बजे", 9:30 -> "9 बजकर 30 मिनट", 0:00 -> "12 बजे"."""
    h12 = hour % 12 or 12
    return f"{h12} बजकर {minute} मिनट" if minute else f"{h12} बजे"


def hindi_time(hour: int, minute: int = 0) -> str:
    """09:00 -> "सुबह 9 बजे"; 12:00 -> "दोपहर 12 बजे"; 00:00 -> "रात 12 बजे"."""
    return f"{hindi_period(hour)} {hindi_clock(hour, minute)}"


def hindi_range(start: tuple[int, int], end: tuple[int, int]) -> str:
    """(9, 0), (12, 0) -> "सुबह 9 बजे से दोपहर 12 बजे तक"."""
    return f"{hindi_time(*start)} से {hindi_time(*end)} तक"


def english_time(hour: int, minute: int = 0) -> str:
    """09:00 -> "9 AM"; 09:30 -> "9:30 AM"; 12:00 -> "12 noon"; 00:00 -> "12 midnight"; 15:00 -> "3 PM"."""
    if minute == 0 and hour in (0, 12):
        return "12 midnight" if hour == 0 else "12 noon"
    h12 = hour % 12 or 12
    clock = f"{h12}:{minute:02d}" if minute else f"{h12}"
    return f"{clock} {'AM' if hour < 12 else 'PM'}"


def english_range(start: tuple[int, int], end: tuple[int, int]) -> str:
    """(9, 0), (12, 0) -> "from 9 AM to 12 noon"."""
    return f"from {english_time(*start)} to {english_time(*end)}"


SPEAK_TIME = {"hi": hindi_time, "en": english_time}
SPEAK_RANGE = {"hi": hindi_range, "en": english_range}
# One pass (range first, else a lone time), so spoken output like "9:30 AM" is never converted twice.
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})(?:\s*[–—\-:]\s*(\d{1,2}):(\d{2}))?\b")


def _speak(m: re.Match, lang: str) -> str:
    start = (int(m[1]), int(m[2]))
    return SPEAK_RANGE[lang](start, (int(m[3]), int(m[4]))) if m[3] else SPEAK_TIME[lang](*start)


def spoken_times(text: str, lang: str = "hi") -> str:
    """Replace "09:00–12:00" / "09:00:12:00" ranges and lone "09:30" with spoken times in `lang`."""
    return _CLOCK.sub(lambda m: _speak(m, lang), text)


def _hm(iso: str) -> tuple[int, int]:
    t = datetime.fromisoformat(iso)
    return t.hour, t.minute


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:]


def facts(air: dict, profile: str) -> dict:
    """The only input the model sees: computed numbers and decisions, windows already in spoken form. No causes:
    the hedged cause line is appended in code (cause_line)."""
    d, a = air["decision"], air["air"]
    category = aqi_category(a["aqi_cpcb"])
    w = d["best_window"] if d["outdoor"] != "yes" else None
    out = {
        "aqi": a["aqi_cpcb"], "category": category, "health_statement_en": CPCB_HEALTH[category],
        "outdoor": d["outdoor"], "for": d["for"], "profile": profile,
        "best_window": {"en": english_range(_hm(w["start"]), _hm(w["end"])),
                        "hi": hindi_range(_hm(w["start"]), _hm(w["end"]))} if w else None,
        "source": a["used"], "confidence": a["confidence"],
    }
    return out


def cause_info(causes: dict | None, available: bool) -> dict:
    """Which causes are present: {"available": bool, "smoke": bool, "fire": bool}. available is False unless the
    smoke/fire model ran (stub or skipped causes are never shown)."""
    if not (available and causes):
        return {"available": False}
    return {"available": True, **{label: bool(causes[label]["present"]) for label in ("smoke", "fire")}}


def cause_line(info: dict) -> tuple[str, str] | None:
    """Hedged cause line (en, hi) when smoke and/or fire is present, else None."""
    present = tuple(k for k in ("smoke", "fire") if info.get("available") and info[k])
    return CAUSE_LINES[present] if present else None


def build(air: dict, profile: str, causes: dict | None = None, causes_available: bool = False,
          bedrock=None) -> tuple[dict, dict]:
    """-> (contract `advice`, {"en": ..., "hi": ...} text or sections for speech).

    Bedrock first (if configured); the template on any failure, with the reason logged. Smoke/fire seen in the clip
    is always shown by appending the template cause sentence in code (Bedrock's advice too; source stays "bedrock"),
    so the model gets one sentence less to keep the total within llm.MAX_SENTENCES."""
    info = cause_info(causes, causes_available)
    template, sections = template_advice(air, profile, info)
    if not llm.model_id():
        return template, sections
    cause = cause_line(info)
    try:
        out = llm.advise(facts(air, profile), bedrock,
                         max_sentences=llm.MAX_SENTENCES - (1 if cause else 0))
    except Exception as e:
        print(json.dumps({"advice": "template_fallback", "reason": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
        return template, sections
    if not cause:
        return {"en": out["en"], "hi": out["hi"], "source": "bedrock"}, out
    speech = {lang: [out[lang], cause[i]] for i, lang in enumerate(("en", "hi"))}  # pause before the cause line
    return {"en": " ".join(speech["en"]), "hi": " ".join(speech["hi"]), "source": "bedrock"}, speech


def template_advice(air: dict, profile: str, causes: dict | None = None) -> tuple[dict, dict[str, list[str]]]:
    """-> (contract `advice`, {"en": sections, "hi": sections} for speech).

    Sections: AQI + recommendation, smoke/fire in the clip (if any), health statement, best window.
    `air`: output of air.combine.air_for; `causes`: cause_info()."""
    cause = cause_line(causes or {})
    d = air["decision"]
    aqi = air["air"]["aqi_cpcb"]
    rec, health = TEMPLATES[d["outdoor"]]
    sections = {}
    for i, lang in enumerate(("en", "hi")):
        r = rec[i] if d["for"] == "today" else TOMORROW[i] + (_lower_first(rec[i]) if lang == "en" else rec[i])
        parts = [f"{AQI_SENTENCE[d['for']][i].format(aqi=aqi)} {r}"]
        if cause:
            parts.append(cause[i])
        if health:
            parts.append(health[i])
        if (w := d["best_window"]) and d["outdoor"] != "yes":
            parts.append(WINDOW[i].format(span=SPEAK_RANGE[lang](_hm(w["start"]), _hm(w["end"]))))
        sections[lang] = parts
    return {"en": " ".join(sections["en"]), "hi": " ".join(sections["hi"]), "source": "template"}, sections


def generate(air: dict, profile: str) -> dict:
    return template_advice(air, profile)[0]
