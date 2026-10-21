"""Classification metrics for the 3 AQI bands (numpy only)."""

import numpy as np

N_BANDS = 3


def confusion(y_true, y_pred, n: int = N_BANDS) -> np.ndarray:
    """Rows = true band, columns = predicted band."""
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return cm


def _safe_div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


def precision_recall_f1(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class arrays; 0 where undefined (no predictions / no true rows)."""
    tp = np.diag(cm)
    precision = _safe_div(tp, cm.sum(axis=0))
    recall = _safe_div(tp, cm.sum(axis=1))
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def macro_f1(cm: np.ndarray) -> float:
    """Mean F1 over classes that occur in y_true."""
    present = cm.sum(axis=1) > 0
    return float(precision_recall_f1(cm)[2][present].mean()) if present.any() else 0.0


def accuracy(cm: np.ndarray) -> float:
    return float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0


def within_one(cm: np.ndarray) -> float:
    """Share of predictions at most one band away from the truth."""
    i, j = np.indices(cm.shape)
    return float(cm[np.abs(i - j) <= 1].sum() / cm.sum()) if cm.sum() else 0.0


def summarize(y_true, y_pred) -> dict:
    cm = confusion(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(cm)
    return {
        "n": int(cm.sum()),
        "accuracy": accuracy(cm),
        "macro_f1": macro_f1(cm),
        "within_one": within_one(cm),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "band2_recall": float(recall[2]),
        "confusion": cm.tolist(),
    }
