"""Spoken advice (contract `audio`): Hindi + English advice -> Polly (Kajal, neural; hi-IN / en-IN)
-> s3://<bucket>/audio/{job_id}_{lang}.mp3.

Speech is SSML: rate 90%; the danda (and any ASCII "|") is replaced by "." plus a 450 ms break, as are other
sentence ends; 250 ms after colons, 200 ms after commas, 600 ms between advice sections (AQI + recommendation
-> health statement -> best window). Both languages are synthesized in parallel and fail independently: a failed
language gets a null key and a caveat; the job never fails. Stores S3 keys only; GetJob presigns fresh GET URLs
on each read. Never writes under uploads/ (that prefix triggers the processor).
"""

import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from xml.sax.saxutils import escape

import boto3

from advice import spoken_times

REGION = os.environ.get("AWS_REGION", "ap-south-1")
LANGS = {"hi": "hi-IN", "en": "en-IN"}
VOICE = {"VoiceId": "Kajal", "Engine": "neural", "OutputFormat": "mp3", "TextType": "ssml"}
CAVEATS = {"hi": "Hindi audio unavailable", "en": "English audio unavailable"}
RATE = "90%"
BREAK_SENTENCE, BREAK_COLON, BREAK_COMMA, BREAK_SECTION = "450ms", "250ms", "200ms", "600ms"

_DANDA = re.compile(r"\s*[।|]")
_SENTENCE_END = re.compile(r"([.?!])(?=\s|$)")
_COLON = re.compile(r":(?=\s)")
_COMMA = re.compile(r",(?=\s)")

_clients: dict = {}


def _client(name: str):
    if name not in _clients:
        _clients[name] = boto3.client(name, region_name=REGION)
    return _clients[name]


def key_for(job_id: str, lang: str) -> str:
    return f"audio/{job_id}_{lang}.mp3"


def _brk(t: str) -> str:
    return f'<break time="{t}"/>'


def _section_ssml(text: str, lang: str) -> str:
    text = _DANDA.sub(".", spoken_times(text.strip(), lang))  # Polly gets "." as the boundary, never "।" / "|"
    text = escape(text)
    text = _COMMA.sub("," + _brk(BREAK_COMMA), text)
    text = _COLON.sub(":" + _brk(BREAK_COLON), text)
    text = _SENTENCE_END.sub(lambda m: m[1] + _brk(BREAK_SENTENCE), text)
    return text.removesuffix(_brk(BREAK_SENTENCE))  # the section break replaces a trailing sentence break


def to_ssml(sections: str | list[str], lang: str = "hi") -> str:
    """Plain advice text (or its sections) -> SSML for Polly. Escapes &, <, > before adding tags."""
    parts = [sections] if isinstance(sections, str) else sections
    body = _brk(BREAK_SECTION).join(_section_ssml(p, lang) for p in parts if p.strip())
    return f'<speak><prosody rate="{RATE}">{body}</prosody></speak>'


def _one(lang: str, sections, job_id: str, polly, s3, bucket: str) -> str | None:
    t0 = time.monotonic()
    try:
        speech = polly.synthesize_speech(Text=to_ssml(sections, lang), LanguageCode=LANGS[lang], **VOICE)
        mp3 = speech["AudioStream"].read()
        if not mp3:
            raise ValueError("Polly returned empty audio")
        key = key_for(job_id, lang)
        s3.put_object(Bucket=bucket, Key=key, Body=mp3, ContentType="audio/mpeg")
        print(json.dumps({"audio": "ok", "lang": lang, "key": key, "bytes": len(mp3),
                          "ms": round((time.monotonic() - t0) * 1000)}))
        return key
    except Exception:
        print(json.dumps({"audio": "failed", "lang": lang}))
        traceback.print_exc()
        return None


def synthesize(texts: dict[str, str | list[str]], job_id: str, polly=None, s3=None,
               bucket: str | None = None) -> tuple[dict, list[str]]:
    """`texts`: {"hi": ..., "en": ...} advice text or sections (advice.build). -> (contract `audio`, caveats)."""
    t0 = time.monotonic()
    try:  # clients are created here, once, before the threads share them
        polly = polly or _client("polly")
        s3 = s3 or _client("s3")
        bucket = bucket or os.environ["BUCKET_NAME"]
    except Exception:
        traceback.print_exc()
        return {"hi_key": None, "en_key": None}, [CAVEATS[lang] for lang in LANGS]
    with ThreadPoolExecutor(max_workers=len(LANGS)) as pool:
        futures = {lang: pool.submit(_one, lang, texts.get(lang) or "", job_id, polly, s3, bucket) for lang in LANGS}
        keys = {lang: f.result() for lang, f in futures.items()}
    print(json.dumps({"audio_total_ms": round((time.monotonic() - t0) * 1000)}))
    return ({"hi_key": keys["hi"], "en_key": keys["en"]},
            [CAVEATS[lang] for lang in LANGS if keys[lang] is None])
