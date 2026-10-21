"""Bedrock advice with a fake Converse client: validated output is used, anything else falls back to the template."""

import json
import re
import xml.dom.minidom

import pytest
from botocore.exceptions import ReadTimeoutError

import advice
import audio
import llm

WINDOW = {"start": "2026-10-10T09:00:00+05:30", "end": "2026-10-10T12:00:00+05:30"}
AIR = {"air": {"aqi_cpcb": 134, "band": 1, "used": "modelled", "confidence": "low"},
       "decision": {"for": "today", "outdoor": "limit", "best_window": WINDOW, "reason_code": "aqi_101_200"}}
GOOD = {"en": "Today's AQI is 134, so keep outdoor activity short and light. People with lung or heart disease or "
              "asthma may feel breathing discomfort. The best time is from 9 AM to 12 noon.",
        "hi": "आज AQI 134 है, इसलिए बाहर की गतिविधि कम और हल्की रखें। फेफड़ों या दिल की बीमारी या दमा वाले लोगों को "
              "साँस लेने में तकलीफ़ हो सकती है। सबसे अच्छा समय सुबह 9 बजे से दोपहर 12 बजे तक है।"}


class FakeBedrock:
    def __init__(self, text=None, exc=None, stop="end_turn"):
        self.text, self.exc, self.stop, self.calls = text, exc, stop, []

    def converse(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return {"output": {"message": {"role": "assistant", "content": [{"text": self.text}]}},
                "stopReason": self.stop, "usage": {"inputTokens": 1, "outputTokens": 1}, "metrics": {"latencyMs": 5}}


@pytest.fixture(autouse=True)
def model(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "global.amazon.nova-2-lite-v1:0")


def run(fake):
    return advice.build(AIR, "school", bedrock=fake)


def test_good_json_is_used():
    fake = FakeBedrock(json.dumps(GOOD, ensure_ascii=False))
    out, speech = run(fake)
    assert out == {**GOOD, "source": "bedrock"} and speech == GOOD
    call = fake.calls[0]
    assert call["modelId"] == "global.amazon.nova-2-lite-v1:0"
    sent = json.loads(call["messages"][0]["content"][0]["text"])
    assert sent["aqi"] == 134 and sent["outdoor"] == "limit" and sent["profile"] == "school"
    assert sent["best_window"] == {"en": "from 9 AM to 12 noon", "hi": "सुबह 9 बजे से दोपहर 12 बजे तक"}
    assert sent["health_statement_en"] == advice.CPCB_HEALTH["Moderate"]
    assert "causes" not in sent  # causes never reach the model
    assert sent["max_sentences"] == llm.MAX_SENTENCES


def test_fenced_json_is_accepted():
    out, _ = run(FakeBedrock("```json\n" + json.dumps(GOOD, ensure_ascii=False) + "\n```"))
    assert out["source"] == "bedrock"


def _fallback(fake, capsys, reason):
    out, speech = run(fake)
    template, sections = advice.template_advice(AIR, "school")
    assert out == template and out["source"] == "template" and speech == sections
    log = capsys.readouterr().out
    assert '"template_fallback"' in log and reason in log


def test_bad_json_falls_back(capsys):
    _fallback(FakeBedrock("Sure! Here is the advice: AQI 134..."), capsys, "not JSON")


def test_commentary_around_json_falls_back(capsys):
    text = "Here is the advice:\n" + json.dumps(GOOD, ensure_ascii=False) + "\nLet me know if you need more."
    _fallback(FakeBedrock(text), capsys, "not JSON")


def test_extra_keys_fall_back(capsys):
    _fallback(FakeBedrock(json.dumps({**GOOD, "note": "estimate"}, ensure_ascii=False)), capsys, "unexpected keys")


@pytest.mark.parametrize("bad", ["**Today's AQI is 134.** Stay in.", "Today's AQI is 134.\n- Stay in."])
def test_markdown_falls_back(capsys, bad):
    _fallback(FakeBedrock(json.dumps({"en": bad, "hi": GOOD["hi"]}, ensure_ascii=False)), capsys, "markup")


def test_too_many_sentences_falls_back(capsys):
    bad = {"en": GOOD["en"] + " Masks help. Close windows.", "hi": GOOD["hi"]}  # 5 sentences
    _fallback(FakeBedrock(json.dumps(bad, ensure_ascii=False)), capsys, "en has 5 sentences (> 4)")
    bad = {"en": GOOD["en"], "hi": GOOD["hi"] + " पानी पिएँ। घर पर रहें।"}
    _fallback(FakeBedrock(json.dumps(bad, ensure_ascii=False)), capsys, "hi has 5 sentences (> 4)")


@pytest.mark.parametrize("word", ["होगी", "होगा"])
def test_certain_health_effect_in_hindi_falls_back(capsys, word):
    bad = {"en": GOOD["en"], "hi": GOOD["hi"].replace("तकलीफ़ हो सकती है", f"तकलीफ़ {word}")}
    assert word in bad["hi"]
    _fallback(FakeBedrock(json.dumps(bad, ensure_ascii=False)), capsys, "health effect as certain")


def test_future_tense_outside_health_is_fine():
    ok = {"en": GOOD["en"], "hi": GOOD["hi"] + " कल फिर से जानकारी दी जाएगी।"}
    assert llm.validate(ok, advice.facts(AIR, "school")) == ok


def test_sentence_split():
    assert llm.sentences("AQI is 134. PM2.5 is high! Ok?") == ["AQI is 134.", "PM2.5 is high!", "Ok?"]
    assert len(llm.sentences(GOOD["hi"])) == 3 and len(llm.sentences(GOOD["en"])) == 3


def test_hindi_word_edges():
    ok = {"en": GOOD["en"], "hi": GOOD["hi"] + " आगे भी धुंध रह सकती है।"}  # "आगे" isn't fire, "धुंध" isn't smoke
    assert llm.validate(ok, advice.facts(AIR, "school")) == ok


def test_missing_hindi_falls_back(capsys):
    _fallback(FakeBedrock(json.dumps({"en": GOOD["en"]})), capsys, "missing or empty hi")


def test_wrong_number_falls_back(capsys):
    bad = {"en": GOOD["en"].replace("134", "143"), "hi": GOOD["hi"]}
    _fallback(FakeBedrock(json.dumps(bad, ensure_ascii=False)), capsys, "AQI 134 missing from en")


def test_invented_number_falls_back(capsys):
    bad = {"en": GOOD["en"] + " Stay out for at most 30 minutes.", "hi": GOOD["hi"]}
    _fallback(FakeBedrock(json.dumps(bad, ensure_ascii=False)), capsys, "numbers not in the facts: ['30']")


def test_non_devanagari_hindi_falls_back(capsys):
    bad = {"en": GOOD["en"], "hi": "Aaj AQI 134 hai."}
    _fallback(FakeBedrock(json.dumps(bad)), capsys, "no Devanagari")


def test_too_long_falls_back(capsys):
    bad = {"en": GOOD["en"] + " x" * 400, "hi": GOOD["hi"]}
    _fallback(FakeBedrock(json.dumps(bad, ensure_ascii=False)), capsys, "too long")


def test_timeout_falls_back(capsys):
    _fallback(FakeBedrock(exc=ReadTimeoutError(endpoint_url="https://bedrock-runtime")), capsys, "ReadTimeoutError")


def test_max_tokens_falls_back(capsys):
    _fallback(FakeBedrock(json.dumps(GOOD, ensure_ascii=False), stop="max_tokens"), capsys, "max_tokens")


def test_no_model_id_uses_template_without_calling(monkeypatch):
    monkeypatch.delenv("BEDROCK_MODEL_ID")
    fake = FakeBedrock(json.dumps(GOOD, ensure_ascii=False))
    out, _ = run(fake)
    assert out["source"] == "template" and fake.calls == []


def test_client_timeouts_and_retries():
    cfg = llm.client()._client_config
    assert cfg.connect_timeout == 8 and cfg.read_timeout == 8
    assert cfg.retries == {"total_max_attempts": 2, "mode": "standard"}


@pytest.mark.parametrize("text", [
    GOOD["hi"],
    "AQI 134 है | बाहर <कम> जाएँ & ध्यान रखें। समय: 09:00–12:00।",
    "**AQI 134** है।\nबाहर \"कम\" जाएँ।।",
])
def test_ssml_safe_for_model_output(text):
    ssml = audio.to_ssml(text, "hi")
    assert "।" not in ssml and "|" not in ssml
    body = ssml.removeprefix('<speak><prosody rate="90%">').removesuffix("</prosody></speak>")
    assert re.sub(r'<break time="\d+ms"/>', "", body).count("<") == 0  # only our own tags
    xml.dom.minidom.parseString(ssml)  # well-formed


# --- smoke/fire: cause sentence appended in code to Bedrock's advice ------------------------------

LIKELY_SMOKE = {"smoke": {"present": True, "prob": 0.95}, "fire": {"present": False, "prob": 0.1}}
BOTH = {"smoke": {"present": True, "prob": 0.7}, "fire": {"present": True, "prob": 0.8}}
FIRE = {"smoke": {"present": False, "prob": 0.1}, "fire": {"present": True, "prob": 0.9}}


def test_no_cause_line_when_clear_or_model_did_not_run():
    clear = {"smoke": {"present": False, "prob": 0.1}, "fire": {"present": False, "prob": 0.1}}
    for causes, ran in ((clear, True), (LIKELY_SMOKE, False)):
        out, speech = advice.build(AIR, "school", causes, ran, bedrock=FakeBedrock(json.dumps(GOOD, ensure_ascii=False)))
        assert out == {**GOOD, "source": "bedrock"} and speech == GOOD


def test_sentence_limit_leaves_room_for_cause_line(capsys):
    four = {"en": GOOD["en"] + " Drink water.", "hi": GOOD["hi"] + " पानी पिएँ।"}  # 4 sentences each
    out, _ = advice.build(AIR, "school", None, False, bedrock=FakeBedrock(json.dumps(four, ensure_ascii=False)))
    assert out["source"] == "bedrock"  # 4 allowed without a cause line
    out, _ = advice.build(AIR, "school", LIKELY_SMOKE, True, bedrock=FakeBedrock(json.dumps(four, ensure_ascii=False)))
    assert out["source"] == "template" and "has 4 sentences (> 3)" in capsys.readouterr().out
    assert advice.CAUSE_LINES[("smoke",)][0] in out["en"]  # the template shows it too


def test_sentence_budget_sent_to_model():
    fake = FakeBedrock(json.dumps(GOOD, ensure_ascii=False))
    advice.build(AIR, "school", LIKELY_SMOKE, True, bedrock=fake)
    assert json.loads(fake.calls[0]["messages"][0]["content"][0]["text"])["max_sentences"] == llm.MAX_SENTENCES - 1


@pytest.mark.parametrize("causes, key", [(LIKELY_SMOKE, ("smoke",)), (FIRE, ("fire",)), (BOTH, ("smoke", "fire"))])
def test_hedged_cause_line_appended_to_bedrock_advice(causes, key):
    en, hi = advice.CAUSE_LINES[key]
    out, speech = advice.build(AIR, "school", causes, True, bedrock=FakeBedrock(json.dumps(GOOD, ensure_ascii=False)))
    assert out == {"en": f"{GOOD['en']} {en}", "hi": f"{GOOD['hi']} {hi}", "source": "bedrock"}
    assert speech == {"en": [GOOD["en"], en], "hi": [GOOD["hi"], hi]}


def test_cause_lines_are_hedged():
    assert advice.CAUSE_LINES[("smoke",)] == (
        "Our camera check suggests possible smoke in your clip (experimental). If you can see a smoke source nearby, "
        "move away from it.",
        "कैमरा जाँच के अनुसार आपकी क्लिप में धुआँ हो सकता है (प्रयोगात्मक)। अगर पास में धुएँ का स्रोत दिखे, तो उससे दूर रहें।")
    assert advice.CAUSE_LINES[("fire",)][1] == (
        "कैमरा जाँच के अनुसार आपकी क्लिप में आग हो सकती है (प्रयोगात्मक)। अगर पास में आग दिखे, तो उससे दूर रहें।")
    for en, hi in advice.CAUSE_LINES.values():
        assert "possible" in en and "(experimental)" in en and "हो सकत" in hi and "(प्रयोगात्मक)" in hi


@pytest.mark.parametrize("causes", [None, LIKELY_SMOKE])
def test_model_mentioning_smoke_is_always_rejected(capsys, causes):
    bad = {"en": GOOD["en"].replace(" The best time", " Away from the smoke, the best time"), "hi": GOOD["hi"]}
    out, _ = advice.build(AIR, "school", causes, causes is not None,
                          bedrock=FakeBedrock(json.dumps(bad, ensure_ascii=False)))
    assert out["source"] == "template" and "mentions smoke" in capsys.readouterr().out


def test_model_mentioning_fire_in_hindi_is_rejected():
    with pytest.raises(llm.AdviceRejected, match="mentions fire"):
        llm.validate({"en": GOOD["en"], "hi": GOOD["hi"] + " पास में आग है।"}, advice.facts(AIR, "school"))
