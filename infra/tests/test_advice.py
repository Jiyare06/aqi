"""Hindi spoken times and advice sections (pure functions)."""

import pytest

import advice


@pytest.mark.parametrize("h, m, expected", [
    (9, 0, "सुबह 9 बजे"),
    (4, 0, "सुबह 4 बजे"),
    (12, 0, "दोपहर 12 बजे"),   # noon
    (15, 0, "दोपहर 3 बजे"),
    (17, 0, "शाम 5 बजे"),
    (19, 0, "रात 7 बजे"),
    (0, 0, "रात 12 बजे"),      # midnight
    (3, 0, "रात 3 बजे"),
    (9, 30, "सुबह 9 बजकर 30 मिनट"),
    (7, 5, "सुबह 7 बजकर 5 मिनट"),
])
def test_hindi_time(h, m, expected):
    assert advice.hindi_time(h, m) == expected


def test_hindi_range():
    assert advice.hindi_range((9, 0), (12, 0)) == "सुबह 9 बजे से दोपहर 12 बजे तक"
    assert advice.hindi_range((9, 30), (12, 15)) == "सुबह 9 बजकर 30 मिनट से दोपहर 12 बजकर 15 मिनट तक"
    assert advice.hindi_range((21, 0), (0, 0)) == "रात 9 बजे से रात 12 बजे तक"


@pytest.mark.parametrize("text, expected", [
    ("समय: 09:00–12:00।", "समय: सुबह 9 बजे से दोपहर 12 बजे तक।"),
    ("समय: 09:00:12:00।", "समय: सुबह 9 बजे से दोपहर 12 बजे तक।"),
    ("समय: 16:30 - 18:00", "समय: शाम 4 बजकर 30 मिनट से शाम 6 बजे तक"),
    ("अभी 00:00 है", "अभी रात 12 बजे है"),
    ("AQI 122 है", "AQI 122 है"),
])
def test_spoken_times(text, expected):
    assert advice.spoken_times(text) == expected


@pytest.mark.parametrize("h, m, expected", [
    (9, 0, "9 AM"),
    (9, 30, "9:30 AM"),
    (12, 0, "12 noon"),
    (12, 30, "12:30 PM"),
    (15, 0, "3 PM"),
    (0, 0, "12 midnight"),
    (0, 15, "12:15 AM"),
    (23, 0, "11 PM"),
])
def test_english_time(h, m, expected):
    assert advice.english_time(h, m) == expected


def test_english_range():
    assert advice.english_range((9, 0), (12, 0)) == "from 9 AM to 12 noon"
    assert advice.english_range((21, 0), (0, 0)) == "from 9 PM to 12 midnight"
    assert advice.spoken_times("Best time: 16:30-18:00.", "en") == "Best time: from 4:30 PM to 6 PM."


def _air(outdoor, window, day="today", aqi=134):
    return {"air": {"aqi_cpcb": aqi},
            "decision": {"for": day, "outdoor": outdoor, "best_window": window, "reason_code": "x"}}


WINDOW = {"start": "2026-10-10T09:00:00+05:30", "end": "2026-10-10T12:00:00+05:30"}


def test_build_limit_today():
    out, sections = advice.build(_air("limit", WINDOW), "school")
    assert out["source"] == "template"
    assert sections["hi"] == ["आज हवा का AQI 134 है। बाहर की गतिविधि सीमित रखें, कम समय और हल्की।",
                              "फेफड़ों या दिल की बीमारी या दमा वाले लोगों को साँस लेने में तकलीफ़ हो सकती है।",
                              "सबसे अच्छा समय: सुबह 9 बजे से दोपहर 12 बजे तक।"]
    assert sections["en"] == ["Today's AQI is 134. Limit outdoor activity: keep it short and light.",
                              "People with lung or heart disease or asthma may feel breathing discomfort.",
                              "Best time: from 9 AM to 12 noon."]
    assert out["hi"] == " ".join(sections["hi"]) and out["en"] == " ".join(sections["en"])


def test_build_has_no_parenthetical_or_double_colon():
    for outdoor in ("yes", "limit", "no"):
        out, _ = advice.build(_air(outdoor, WINDOW), "school")
        for text in (out["en"], out["hi"]):
            assert "(" not in text and "CPCB" not in text and "|" not in text
            assert text.count(":") <= 2  # at most "Limit outdoor activity:" and "Best time:"
        assert out["hi"].count(":") <= 1


def test_build_tomorrow_says_now_for_the_aqi():
    out, _ = advice.build(_air("no", None, day="tomorrow"), "school")
    assert out["en"].startswith("The AQI right now is 134. Tomorrow, keep activities indoors.")
    assert out["hi"].startswith("अभी हवा का AQI 134 है। कल गतिविधियाँ घर के अंदर रखें।")


def test_build_yes_has_one_section():
    out, sections = advice.build(_air("yes", None), "general")
    assert len(sections["hi"]) == len(sections["en"]) == 1
    assert out["en"] == "Today's AQI is 134. Air is fine for outdoor activity."


def test_template_no_cause_line_when_model_did_not_run_or_clear():
    c = {"smoke": {"present": True, "prob": 0.95}, "fire": {"present": False, "prob": 0.1}}
    for info in (advice.cause_info(c, False), advice.cause_info(
            {"smoke": {"present": False, "prob": 0.1}, "fire": {"present": False, "prob": 0.1}}, True)):
        _, sections = advice.template_advice(_air("limit", WINDOW), "school", info)
        assert len(sections["en"]) == 3 and not any("clip" in s for s in sections["en"])


@pytest.mark.parametrize("smoke, fire, key", [(0.95, None, ("smoke",)), (None, 0.9, ("fire",)), (0.65, 0.8, ("smoke", "fire"))])
def test_template_cause_line(smoke, fire, key):
    c = {"smoke": {"present": smoke is not None, "prob": smoke or 0.1},
         "fire": {"present": fire is not None, "prob": fire or 0.1}}
    en, hi = advice.CAUSE_LINES[key]
    out, sections = advice.template_advice(_air("limit", WINDOW), "school", advice.cause_info(c, True))
    assert sections["en"][1] == en and sections["hi"][1] == hi  # right after AQI + recommendation
    assert en in out["en"] and hi in out["hi"]
