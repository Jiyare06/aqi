import numpy as np
import pandas as pd
import pytest

from hazeml import cam
from hazeml.train_smokefire import combo, dfire_report, label_metrics, pick_thresholds, pyro_report


def test_cam_gap_identity():
    rng = np.random.default_rng(0)
    f, w, b = rng.normal(size=(8, 7, 7)), rng.normal(size=(2, 8)), rng.normal(size=2)
    logits = w @ f.mean(axis=(1, 2)) + b
    for c in range(2):
        assert cam.cam_raw(f, w, c).mean() + b[c] == pytest.approx(logits[c])
    m = cam.cam(f, w, 0)
    assert m.min() >= 0 and m.max() == pytest.approx(1.0)


def test_resize_and_overlay_shapes():
    a = np.arange(4.0).reshape(2, 2)
    r = cam.resize_bilinear(a, 4, 4)
    assert r.shape == (4, 4) and r.min() == pytest.approx(0) and r.max() == pytest.approx(3)
    out = cam.overlay(np.zeros((10, 12, 3), dtype=np.uint8), np.ones((7, 7)))
    assert out.shape == (10, 12, 3) and out.dtype == np.uint8 and out[..., 0].max() > 0


def test_thresholds_maximise_f1():
    y = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    p = np.array([[0.1, 0.2], [0.3, 0.9], [0.8, 0.7], [0.6, 0.1]])
    thr = pick_thresholds(y, p)
    assert thr[0] == pytest.approx(0.6) and thr[1] == pytest.approx(0.7)
    m = label_metrics(y[:, 0], p[:, 0], thr[0])
    assert (m["precision"], m["recall"], m["f1"], m["fpr"]) == (1.0, 1.0, 1.0, 0.0)


def test_combo_confusion():
    y = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    assert combo(y).tolist() == [0, 1, 2, 3]
    rep = dfire_report(y, np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.9, 0.9]]), [0.5, 0.5])
    assert rep["combo_confusion"]["rows_true_cols_pred"][2] == [0, 0, 0, 1]  # fire-only predicted as both
    assert rep["exact_match"] == pytest.approx(0.75)


def test_pyro_report_by_camera():
    df = pd.DataFrame({"camera": ["a", "a", "b", "b"], "smoke": [1, 0, 1, 1]})
    rep = pyro_report(df, np.array([0.9, 0.8, 0.2, 0.7]), 0.5)
    assert rep["overall"]["fpr"] == 1.0 and rep["overall"]["recall"] == pytest.approx(2 / 3)
    cams = {c["camera"]: c for c in rep["per_camera"]}
    assert cams["b"]["fpr"] is None and cams["b"]["recall"] == 0.5
    assert rep["by_camera_macro"]["cameras_with_neg"] == 1
