"""Bedrock (Converse) advice: structured facts in, {"en", "hi"} out, validated before use.

The model only phrases facts our pipeline computed (CLAUDE.md §4): every number in its output must already be in the
facts, the AQI must appear in both languages, Hindi must be Devanagari, length is capped, and nothing extra is
allowed (no keys beyond en/hi, no text around the JSON, no markdown or line breaks). Any failure raises
AdviceRejected (or the boto error), and advice.build falls back to the template.
"""

import json
import os
import re
import time

import boto3
from botocore.config import Config

REGION = os.environ.get("AWS_REGION", "ap-south-1")
MAX_CHARS = 700
MAX_TOKENS = 700

SYSTEM = """You write short outdoor air-quality advice for one place in India, in English and in Hindi.

You get a JSON object of facts computed by our pipeline. Rules:
- Write 2-3 short sentences in English and 2-3 short sentences in Hindi with the same meaning. Never more than
  `max_sentences` sentences per language (the app adds its own sentences after yours).
- Start by stating the AQI number. Then the recommendation for the profile (school / outdoor worker / general public)
  that matches `outdoor`: yes = outdoor activity is fine; limit = keep outdoor activity short and light; no = keep
  activities indoors. If `for` is "tomorrow", the recommendation is for tomorrow and the AQI is the reading right now.
- Use only numbers that appear in the facts. Never invent numbers, pollutant values, durations or forecasts.
- Health: say nothing beyond `health_statement_en` (CPCB statement for this category). Do not add medical advice.
  Health effects are always possible, never certain: "may" in English, "हो सकती है" in Hindi; never "will" / "होगी".
- If `best_window` is given, mention it using exactly `best_window.en` in English and `best_window.hi` in Hindi
  (times are already written as spoken words; do not convert them to digits like 09:00).
- Do not say the reading is an estimate and do not explain sources, models, stations or confidence (the app shows
  those separately).
- Do not mention smoke or fire at all: the app adds its own sentence about anything seen in the clip. Never say
  what caused the AQI.

Hindi:
- Simple, natural spoken Hindi, the way an Indian news reader would say it on air: short sentences, everyday words,
  correct grammar and gender agreement. Write it fresh in Hindi; do not translate the English word by word.
- Devanagari script. Keep "AQI" in Latin letters. Use Western digits (0-9). End sentences with "।".
- Address people with "आप". Prefer common words (हवा, बाहर, घर के अंदर, साँस, बच्चे, काम) over formal Sanskrit words:
  दिल (not हृदय), ज़्यादातर (not अधिकांश), तकलीफ़ (not असुविधा).
- When mentioning an illness, always phrase it as "<illness> वाले लोगों को", e.g. "दमा या दिल की बीमारी वाले लोगों को".
- Example of the style (different facts): "आज हवा का AQI 85 है। बाहर खेलना और काम करना ठीक है।"

Output:
- Plain text inside the strings: no markdown, no asterisks, no emoji, no lists, no "|" characters, no line breaks.
- Return only the JSON object, nothing else: no code fences, no explanation before or after it.
  {"en": "...", "hi": "..."}"""

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_MARKUP = re.compile(r"[*#`|_\n]")
MAX_SENTENCES = 4
_SENTENCE = re.compile(r"[^.?!।]+[.?!।]*")
# A Hindi sentence about health effects must not use the certain future ("होगी" / "होगा"); CPCB says "may".
_HEALTH_HI = re.compile(r"साँस|सांस|तकलीफ़|तकलीफ|बीमारी|रोग|असुविधा|स्वास्थ्य|दमा|अस्थमा")
_CERTAIN_HI = re.compile(r"होगी|होगा")
# Smoke/fire mentions (Devanagari words need explicit edges: Python's \b breaks on vowel signs, and "आग" is inside
# "आगे"; "धुंध" (haze) is not smoke).
_DEV = "\u0900-\u097F"
CAUSE_WORDS = {
    "smoke": (re.compile(r"\bsmok", re.I), re.compile(f"(?<![{_DEV}])(धुआँ|धुआं|धुआ|धुएँ|धुएं|धुंआ|धुँआ)")),
    "fire": (re.compile(r"\b(fires?|flames?|blaze)\b", re.I), re.compile(f"(?<![{_DEV}])(आग|लपट)(?![{_DEV}])")),
}

_client = None


class AdviceRejected(ValueError):
    pass


def model_id() -> str | None:
    return os.environ.get("BEDROCK_MODEL_ID") or None


def client():
    global _client
    if _client is None:
        cfg = Config(connect_timeout=8, read_timeout=8, retries={"total_max_attempts": 2, "mode": "standard"})
        _client = boto3.client("bedrock-runtime", region_name=REGION, config=cfg)
    return _client


def parse(text: str) -> dict:
    try:
        out = json.loads(_FENCE.sub("", text.strip()))
    except json.JSONDecodeError as e:
        raise AdviceRejected(f"not JSON: {e}") from None
    if not isinstance(out, dict):
        raise AdviceRejected("JSON is not an object")
    if extra := sorted(set(out) - {"en", "hi"}):
        raise AdviceRejected(f"unexpected keys: {extra}")
    return out


def sentences(text: str) -> list[str]:
    """Sentences split on . ? ! । (a "." inside a number like 2.5 doesn't end one)."""
    text = re.sub(r"(\d)\.(\d)", r"\1<dot>\2", text)
    return [x.strip().replace("<dot>", ".") for x in _SENTENCE.findall(text) if x.strip(" .?!।")]


def check_causes(texts: dict) -> None:
    """The model must not mention smoke or fire at all: the hedged cause line is appended in code by advice.build."""
    for label, (en_re, hi_re) in CAUSE_WORDS.items():
        if en_re.search(texts["en"]) or hi_re.search(texts["hi"]):
            raise AdviceRejected(f"mentions {label} (causes come only from the app's own sentence)")


def validate(out: dict, facts: dict, max_sentences: int = MAX_SENTENCES) -> dict:
    """-> {"en", "hi"} stripped, or raise AdviceRejected with the reason."""
    texts = {}
    for lang in ("en", "hi"):
        t = out.get(lang)
        if not isinstance(t, str) or not t.strip():
            raise AdviceRejected(f"missing or empty {lang}")
        t = t.strip()
        if len(t) > MAX_CHARS:
            raise AdviceRejected(f"{lang} too long ({len(t)} > {MAX_CHARS} chars)")
        if m := _MARKUP.search(t):
            raise AdviceRejected(f"{lang} has markup or line breaks ({m[0]!r})")
        texts[lang] = t
    for lang, t in texts.items():
        if (n := len(sentences(t))) > max_sentences:
            raise AdviceRejected(f"{lang} has {n} sentences (> {max_sentences})")
    if bad := [x for x in sentences(texts["hi"]) if _HEALTH_HI.search(x) and _CERTAIN_HI.search(x)]:
        raise AdviceRejected(f"hi states a health effect as certain: {bad[0]!r}")
    check_causes(texts)
    if not _DEVANAGARI.search(texts["hi"]):
        raise AdviceRejected("hi has no Devanagari")
    aqi = str(facts["aqi"])
    allowed = set(_NUMBER.findall(json.dumps(facts, ensure_ascii=False)))
    for lang, t in texts.items():
        found = _NUMBER.findall(t)
        if aqi not in found:
            raise AdviceRejected(f"AQI {aqi} missing from {lang}")
        if extra := sorted(set(found) - allowed):
            raise AdviceRejected(f"{lang} has numbers not in the facts: {extra}")
    return texts


def advise(facts: dict, bedrock=None, max_sentences: int = MAX_SENTENCES) -> dict:
    """Facts -> validated {"en", "hi"}. Raises on any failure (caller falls back to the template)."""
    mid = model_id()
    if not mid:
        raise AdviceRejected("BEDROCK_MODEL_ID not set")
    facts = {**facts, "max_sentences": max_sentences}
    t0 = time.monotonic()
    resp = (bedrock or client()).converse(
        modelId=mid,
        system=[{"text": SYSTEM}],
        messages=[{"role": "user", "content": [{"text": json.dumps(facts, ensure_ascii=False)}]}],
        inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": 0.3},
    )
    ms = round((time.monotonic() - t0) * 1000)
    usage = resp.get("usage", {})
    print(json.dumps({"bedrock": "response", "ms": ms, "latency_ms": resp.get("metrics", {}).get("latencyMs"),
                      "stop": resp.get("stopReason"), "in_tokens": usage.get("inputTokens"),
                      "out_tokens": usage.get("outputTokens")}))
    if resp.get("stopReason") == "max_tokens":
        raise AdviceRejected("hit max_tokens")
    text = "".join(c.get("text", "") for c in resp["output"]["message"]["content"])
    return validate(parse(text), facts, max_sentences)
