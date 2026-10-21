import numpy as np
import pytest

from hazeml.metrics import accuracy, confusion, macro_f1, precision_recall_f1, summarize, within_one

Y_TRUE = [0, 0, 0, 1, 1, 2, 2, 2]
Y_PRED = [0, 0, 1, 1, 2, 2, 0, 2]


def test_confusion():
    assert confusion(Y_TRUE, Y_PRED).tolist() == [[2, 1, 0], [0, 1, 1], [1, 0, 2]]


def test_precision_recall_f1():
    p, r, f1 = precision_recall_f1(confusion(Y_TRUE, Y_PRED))
    assert p == pytest.approx([2 / 3, 1 / 2, 2 / 3])
    assert r == pytest.approx([2 / 3, 1 / 2, 2 / 3])
    assert f1 == pytest.approx([2 / 3, 1 / 2, 2 / 3])


def test_summary_numbers():
    cm = confusion(Y_TRUE, Y_PRED)
    assert accuracy(cm) == pytest.approx(5 / 8)
    assert macro_f1(cm) == pytest.approx((2 / 3 + 1 / 2 + 2 / 3) / 3)
    assert within_one(cm) == pytest.approx(7 / 8)  # only 2 -> 0 is two bands off


def test_undefined_classes_are_zero_not_nan():
    cm = confusion([0, 0, 1], [0, 0, 0])
    p, r, f1 = precision_recall_f1(cm)
    assert not np.isnan(p).any() and not np.isnan(f1).any()
    assert r[2] == 0
    # Band 2 never occurs in y_true, so it is left out of macro-F1.
    assert macro_f1(cm) == pytest.approx((0.8 + 0) / 2)


def test_summarize_keys():
    s = summarize(Y_TRUE, Y_PRED)
    assert s["n"] == 8
    assert s["band2_recall"] == pytest.approx(2 / 3)
    assert s["confusion"][2] == [1, 0, 2]
