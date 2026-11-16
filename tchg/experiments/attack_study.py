from __future__ import annotations

import numpy as np


def _copy(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:

    return {k: v.copy() for k, v in arrays.items()}


def _choose_events(arrays: dict[str, np.ndarray], ratio: float, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)
    n = arrays["src"].shape[0]
    count = max(1, int(n * ratio))
    return rng.choice(np.arange(n), size=count, replace=False)


def _choose_event_prefix(arrays: dict[str, np.ndarray], ratio: float, seed: int) -> np.ndarray:

    candidates = np.arange(arrays["src"].shape[0])
    if "split" in arrays:
        test = np.where(arrays["split"].astype(np.int64) == 2)[0]
        if len(test):
            candidates = test
    count = max(1, int(len(candidates) * ratio))
    rng = np.random.default_rng(seed)
    return rng.permutation(candidates)[:count]


def on_off_attack(arrays: dict[str, np.ndarray], ratio: float, seed: int) -> dict[str, np.ndarray]:

    out = _copy(arrays)
    idx = _choose_events(out, ratio, seed)
    bad = idx[np.arange(len(idx)) % 2 == 1]
    out["behavior_features"][bad, 5] = np.maximum(out["behavior_features"][bad, 5], 0.95)
    out["behavior_features"][bad, 6] = np.maximum(out["behavior_features"][bad, 6], 0.95)
    out["observed_mask"][bad] = 1.0
    return out


def good_mouthing(arrays: dict[str, np.ndarray], ratio: float, seed: int) -> dict[str, np.ndarray]:

    out = _copy(arrays)
    idx = _choose_events(out, ratio, seed)
    out["entity_features"][idx, :2] = np.maximum(out["entity_features"][idx, :2], 0.95)
    out["behavior_features"][idx, :2] = np.maximum(out["behavior_features"][idx, :2], 0.95)
    return out


def bad_mouthing(arrays: dict[str, np.ndarray], ratio: float, seed: int) -> dict[str, np.ndarray]:

    out = _copy(arrays)
    idx = _choose_events(out, ratio, seed)
    out["entity_features"][idx, :2] = np.minimum(out["entity_features"][idx, :2], 0.05)
    out["behavior_features"][idx, :2] = np.minimum(out["behavior_features"][idx, :2], 0.05)
    return out


def collusion(arrays: dict[str, np.ndarray], ratio: float, seed: int) -> dict[str, np.ndarray]:

    out = good_mouthing(arrays, ratio / 2.0, seed)
    return bad_mouthing(out, ratio / 2.0, seed + 17)


def stale_trust_replay(arrays: dict[str, np.ndarray], ratio: float, seed: int) -> dict[str, np.ndarray]:

    out = _copy(arrays)
    idx = _choose_events(out, ratio, seed)
    out["observed_mask"][idx] = 1.0
    out["delta_t"][idx] = out["delta_t"][idx] * 5.0 + 10.0
    out["entity_features"][idx, :2] = np.maximum(out["entity_features"][idx, :2], 0.9)
    return out


def trust_oriented_poison(arrays: dict[str, np.ndarray], ratio: float, seed: int) -> dict[str, np.ndarray]:

    out = _copy(arrays)
    idx = _choose_event_prefix(out, ratio, seed)
    out["entity_features"][idx, :2] = np.minimum(out["entity_features"][idx, :2], 0.02)
    out["entity_features"][idx, 2:6] = np.maximum(out["entity_features"][idx, 2:6], 0.98)
    out["behavior_features"][idx, :2] = np.minimum(out["behavior_features"][idx, :2], 0.02)
    out["behavior_features"][idx, 5:8] = np.maximum(out["behavior_features"][idx, 5:8], 0.98)
    out["observed_mask"][idx] = 1.0
    return out


def gnn_oriented_rewire(arrays: dict[str, np.ndarray], ratio: float, seed: int) -> dict[str, np.ndarray]:

    out = _copy(arrays)
    idx = _choose_event_prefix(out, ratio, seed)
    rng = np.random.default_rng(seed + 7919)
    out["delta_t"][idx] = out["delta_t"][idx].astype(np.float32) * 8.0 + 25.0
    if len(idx) > 1:
        perm = rng.permutation(idx)
        out["context_features"][idx] = out["context_features"][perm]
        out["edge_features"][idx] = out["edge_features"][perm[::-1]]
    out["context_features"][idx, 4] = 1.0
    out["context_features"][idx, 5] = np.minimum(out["context_features"][idx, 5], 0.02)
    out["edge_features"][idx, 1] = 1.0
    out["edge_features"][idx, 3] = np.minimum(out["edge_features"][idx, 3], 0.02)
    return out


ATTACKS = {
    "on_off": on_off_attack,
    "good_mouthing": good_mouthing,
    "bad_mouthing": bad_mouthing,
    "collusion": collusion,
    "stale_trust_replay": stale_trust_replay,
    "trust_oriented": trust_oriented_poison,
    "gnn_oriented": gnn_oriented_rewire,
}
