"""Polly + S3 audio (processor) and presigned audio URLs (GetJob), with fake clients: no AWS calls."""

import importlib.util
import io
import re

import pytest

import audio
from conftest import INFRA

TEXTS = {"hi": ["आज हवा का AQI 134 है। बाहर की गतिविधि सीमित रखें, कम समय और हल्की।",
                "सबसे अच्छा समय: सुबह 9 बजे से दोपहर 12 बजे तक।"],
         "en": ["Today's AQI is 134. Limit outdoor activity: keep it short and light.",
                "Best time: from 9 AM to 12 noon."]}


class FakePolly:
    def __init__(self, fail_langs=()):
        self.fail_langs, self.calls = set(fail_langs), []

    def synthesize_speech(self, **kw):
        self.calls.append(kw)
        if kw["LanguageCode"][:2] in self.fail_langs:
            raise RuntimeError("polly down")
        return {"AudioStream": io.BytesIO(b"ID3fake-" + kw["LanguageCode"].encode())}


class FakeS3:
    def __init__(self):
        self.puts = []

    def put_object(self, **kw):
        self.puts.append(kw)


def test_success_writes_both_keys():
    polly, s3 = FakePolly(), FakeS3()
    out, caveats = audio.synthesize(TEXTS, "job1", polly=polly, s3=s3, bucket="b")
    assert out == {"hi_key": "audio/job1_hi.mp3", "en_key": "audio/job1_en.mp3"} and caveats == []
    calls = {c["LanguageCode"]: c for c in polly.calls}
    assert set(calls) == {"hi-IN", "en-IN"}
    for lang, code in (("hi", "hi-IN"), ("en", "en-IN")):
        assert calls[code] == {"Text": audio.to_ssml(TEXTS[lang], lang), "TextType": "ssml", "VoiceId": "Kajal",
                               "Engine": "neural", "LanguageCode": code, "OutputFormat": "mp3"}
        assert calls[code]["Text"].startswith('<speak><prosody rate="90%">')
    puts = {p["Key"]: p for p in s3.puts}
    assert puts["audio/job1_hi.mp3"]["Body"] == b"ID3fake-hi-IN"
    assert puts["audio/job1_en.mp3"]["ContentType"] == "audio/mpeg"
    assert all(p["Bucket"] == "b" and p["Key"].startswith("audio/") for p in s3.puts)


@pytest.mark.parametrize("failing, keys, caveats", [
    (("hi",), {"hi_key": None, "en_key": "audio/job1_en.mp3"}, ["Hindi audio unavailable"]),
    (("en",), {"hi_key": "audio/job1_hi.mp3", "en_key": None}, ["English audio unavailable"]),
    (("hi", "en"), {"hi_key": None, "en_key": None}, ["Hindi audio unavailable", "English audio unavailable"]),
])
def test_languages_fail_independently(failing, keys, caveats):
    s3 = FakeS3()
    out, cav = audio.synthesize(TEXTS, "job1", polly=FakePolly(failing), s3=s3, bucket="b")
    assert out == keys and cav == caveats
    assert len(s3.puts) == 2 - len(failing)


# --- SSML ---------------------------------------------------------------------------------------


def test_ssml_hindi_breaks_and_no_danda():
    ssml = audio.to_ssml(TEXTS["hi"], "hi")
    assert ssml == ('<speak><prosody rate="90%">आज हवा का AQI 134 है.<break time="450ms"/> बाहर की गतिविधि सीमित '
                    'रखें,<break time="200ms"/> कम समय और हल्की.<break time="600ms"/>सबसे अच्छा समय:'
                    '<break time="250ms"/> सुबह 9 बजे से दोपहर 12 बजे तक.</prosody></speak>')


@pytest.mark.parametrize("text", ["एक | दो |", "एक। दो।", "एक ।दो", "समय: 09:00–12:00।"])
def test_ssml_never_contains_danda_or_pipe(text):
    ssml = audio.to_ssml(text, "hi")
    assert "।" not in ssml and "|" not in ssml


def test_ssml_english():
    ssml = audio.to_ssml(TEXTS["en"], "en")
    assert ssml == ('<speak><prosody rate="90%">Today\'s AQI is 134.<break time="450ms"/> Limit outdoor activity:'
                    '<break time="250ms"/> keep it short and light.<break time="600ms"/>Best time:'
                    '<break time="250ms"/> from 9 AM to 12 noon.</prosody></speak>')


def test_ssml_escapes_before_tags():
    ssml = audio.to_ssml("A & B <x> 5 > 3", "en")
    assert "A &amp; B &lt;x&gt; 5 &gt; 3" in ssml
    assert len(re.findall(r"<[^>]*>", ssml)) == 4  # speak, prosody, /prosody, /speak


def test_ssml_converts_leftover_clock_times_without_colon_breaks():
    hi = audio.to_ssml("सबसे अच्छा समय: 09:00–12:00।", "hi")
    assert "सुबह 9 बजे से दोपहर 12 बजे तक" in hi and hi.count("250ms") == 1
    assert "from 9:30 AM to 12 midnight" in audio.to_ssml("Best time: 09:30–00:00.", "en")


# --- GetJob -------------------------------------------------------------------------------------


@pytest.fixture
def get_job():
    spec = importlib.util.spec_from_file_location("get_job_app", INFRA / "functions" / "get_job" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_get_job_presigns_both_audio_keys(get_job, monkeypatch):
    seen = []
    monkeypatch.setattr(get_job.s3, "generate_presigned_url",
                        lambda op, Params, ExpiresIn: seen.append((op, Params, ExpiresIn)) or f"https://{Params['Key']}")
    urls = get_job.audio_urls({"audio": {"hi_key": "audio/job1_hi.mp3", "en_key": "audio/job1_en.mp3"}})
    assert urls == {"hi": "https://audio/job1_hi.mp3", "en": "https://audio/job1_en.mp3"}
    assert seen[0] == ("get_object", {"Bucket": "test-bucket", "Key": "audio/job1_hi.mp3"}, 3600)
    assert get_job.audio_urls(None) == {"hi": None, "en": None}
    assert get_job.audio_urls({"audio": {"hi_key": None, "en_key": "audio/j_en.mp3"}})["hi"] is None
