from __future__ import annotations

import numpy as np


def sample_negative_edges(
    num_nodes: int,
    positive_edges: set[tuple[int, int]],
    count: int,
    rng: np.random.Generator,
    candidate_nodes: np.ndarray | None = None,
) -> np.ndarray:

    candidates = np.arange(num_nodes, dtype=np.int64) if candidate_nodes is None else np.asarray(candidate_nodes, dtype=np.int64)
    if len(candidates) < 2:
        raise ValueError("Negative sampling requires at least two candidate nodes")
    samples: list[tuple[int, int]] = []
    while len(samples) < count:
        src = int(candidates[rng.integers(0, len(candidates))])
        dst = int(candidates[rng.integers(0, len(candidates))])
        if src == dst or (src, dst) in positive_edges:
            continue
        samples.append((src, dst))
    return np.asarray(samples, dtype=np.int64)
