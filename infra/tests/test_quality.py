"""Quality gate on synthetic frames (BGR uint8), and causes.analyse skipping the models when too few frames pass."""

import cv2
import numpy as np
import pytest

import causes
import quality

RNG = np.random.default_rng(0)


def outdoor(h=480, w=640):
    """Blue sky (top half) over a sharp, textured street (bottom half)."""
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2] = (220, 160, 90)  # BGR: blue sky
    img[h // 2:] = RNG.integers(40, 200, (h - h // 2, w, 3), dtype=np.uint8)
    return img


def dark():
    return (outdoor() * 0.1).astype(np.uint8)


def blurred():
    return cv2.GaussianBlur(outdoor(), (0, 0), 12)


def indoor(h=480, w=640):
    """Beige wall with some texture: no sky-like pixels anywhere."""
    img = np.full((h, w, 3), (110, 140, 160), np.uint8)  # BGR beige, V ~ 160 but S ~ 80+
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[..., 1] = 120  # clearly saturated -> not "bright low-saturation"
    img[..., 2] = np.clip(img[..., 2].astype(int) - 30 + RNG.integers(0, 60, (h, w)), 0, 255)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def ground():
    """Camera pointed down at soil/road: earth-toned texture everywhere, no sky in the top 40%."""
    hsv = np.stack([RNG.integers(10, 26, (480, 640)), RNG.integers(100, 200, (480, 640)),
                    RNG.integers(60, 140, (480, 640))], axis=2).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@pytest.mark.parametrize("make, expected", [
    (outdoor, []), (dark, ["dark", "no_sky"]),  # a night sky isn't sky-like either
    (blurred, ["blur"]), (indoor, ["no_sky"]), (ground, ["no_sky"]),
])
def test_reasons(make, expected):
    f = quality.features(make())
    assert quality.reasons(f) == expected, f


def test_features_resolution_independent():
    small, big = outdoor(360, 640), None
    big = cv2.resize(small, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    a, b = quality.features(small), quality.features(big)
    assert a["luma"] == pytest.approx(b["luma"], abs=1) and a["sky"] == pytest.approx(b["sky"], abs=0.02)
    assert quality.reasons(a) == quality.reasons(b) == []


def test_gate_counts_and_reasons():
    kept, why, log = quality.gate([outdoor(), dark(), outdoor(), blurred(), indoor()])
    assert len(kept) == 2 and why == ["dark", "blur", "no_sky"]
    assert [f["reject"] for f in log] == [[], ["dark", "no_sky"], [], ["blur"], ["no_sky"]]


def test_analyse_skips_models_when_too_few_frames_pass(monkeypatch):
    monkeypatch.setattr(causes, "detect", lambda *a, **k: pytest.fail("models must not run"))
    out, cav, ran = causes.analyse([indoor(), indoor(), dark(), outdoor(), blurred()], "job1", None, "b")
    assert not ran and cav == [causes.QUALITY_SKIPPED]
    assert out["smoke"] == {"present": False, "prob": 0.0} and out["fire"]["present"] is False
    assert out["vehicle_count"] is None and out["gradcam_key"] is None
    assert (out["frames_used"], out["frames_rejected"], out["reject_reasons"]) == (1, 4, ["dark", "blur", "no_sky"])


def test_analyse_runs_models_on_passing_frames_only(monkeypatch):
    seen = []

    def fake_detect(frames, job_id, s3, bucket):
        seen.append(len(frames))
        return {**causes.stub(len(frames)), "vehicle_count": 2}, [], True
    monkeypatch.setattr(causes, "detect", fake_detect)
    out, cav, ran = causes.analyse([outdoor(), dark(), outdoor(), outdoor(), dark()], "job1", None, "b")
    assert seen == [3] and ran and cav == []
    assert (out["frames_used"], out["frames_rejected"], out["reject_reasons"]) == (3, 2, ["dark", "no_sky"])
    assert out["vehicle_count"] == 2


def test_skipped_causes_never_reach_bedrock():
    import advice
    skipped = causes.stub(1)
    assert advice.cause_info(skipped, False) == {"available": False}
