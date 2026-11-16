from __future__ import annotations

from typing import Dict

import numpy as np


def temporal_split_indices(
    timestamps: np.ndarray,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Dict[str, np.ndarray]:

    order = np.argsort(timestamps, kind="mergesort")
    n = len(order)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = max(1, n - n_val - n_test)
    train = order[:n_train]
    val = order[n_train:n_train + n_val]
    test = order[n_train + n_val:]
    return {"train": train, "val": val, "test": test}
