from __future__ import annotations

import numpy as np


def apply_missing_rate(arrays: dict[str, np.ndarray], missing_rate: float, seed: int) -> dict[str, np.ndarray]:

    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in arrays.items()}
    n = out["observed_mask"].shape[0]
    out["observed_mask"] = (rng.random(n) > missing_rate).astype(np.float32)
    out["delta_t"] = out["delta_t"].astype(np.float32) * (1.0 + missing_rate)
    return out
