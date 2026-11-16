from __future__ import annotations

import numpy as np


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:

    conf = np.maximum(prob, 1.0 - prob)
    pred = (prob >= 0.5).astype(np.float32)
    correct = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(conf[mask].mean()) - float(correct[mask].mean()))
    return float(ece)


def overconfidence_rate(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.8) -> float:

    conf = np.maximum(prob, 1.0 - prob)
    pred = (prob >= 0.5).astype(np.float32)
    wrong = pred != y_true
    return float(((conf >= threshold) & wrong).mean())


def confidence_error_correlation(y_true: np.ndarray, prob: np.ndarray) -> float:

    conf = np.maximum(prob, 1.0 - prob)
    err = np.abs(prob - y_true)
    if np.std(conf) < 1e-8 or np.std(err) < 1e-8:
        return 0.0
    return float(np.corrcoef(conf, err)[0, 1])
