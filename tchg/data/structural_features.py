"""Pair evidence constructed only from admissible historical trust edges."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def causal_structural_features(arrays, cache_path=None):
    """Return reciprocal/recency/two-hop/common-neighbor evidence per query.

    Only observed positive edges supply structure. Train reads strictly earlier
    timestamp groups. Validation reads a frozen training graph. Test reads
    training plus observed validation events, with
    no updates from test queries or sampled negatives. Test labels are never
    consulted. No statistics are fitted on validation or test features.
    """
    src, dst, times, split = (arrays[k] for k in ('src', 'dst', 'timestamp', 'split'))
    n = len(src)
    visible = arrays.get('history_visible_mask', np.ones(n))
    past = np.flatnonzero((split <= 1) & (arrays['sample_type'] == 0) &
                         (arrays['observed_mask'] > .5) & (visible > .5))
    eligible = np.zeros(n, dtype=bool)
    eligible[past] = arrays['label'][past] > .5
    digest = hashlib.sha256(b'causal-structural-v1;recency=180days;fixed-log-scale=10')
    for value in (src, dst, times, split, eligible):
        digest.update(np.ascontiguousarray(value).tobytes())
    fingerprint = digest.hexdigest()
    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved['fingerprint'].item()) == fingerprint:
                return saved['features'].copy()
    parts = [np.flatnonzero(split == k) for k in (0, 1, 2)]
    for left, right in zip(parts, parts[1:]):
        if len(left) and len(right) and times[left].max() >= times[right].min():
            raise ValueError('Structural evidence requires strictly separated chronological splits')
    outgoing = [set() for _ in range(int(arrays['num_nodes'][0]))]
    incoming = [set() for _ in outgoing]
    last = {}
    result = np.zeros((n, 5), dtype=np.float32)

    def write(i):
        if eligible[i]:
            s, d = int(src[i]), int(dst[i])
            outgoing[s].add(d)
            incoming[d].add(s)
            last[s, d] = float(times[i])

    def read(i):
        s, d, t = int(src[i]), int(dst[i]), float(times[i])
        reverse_time = last.get((d, s))
        if reverse_time is not None:
            result[i, 0] = 1.
            result[i, 1] = np.exp(-max(0., t - reverse_time) / (180. * 86400.))
        result[i, 2] = np.log1p(len(outgoing[s] & incoming[d])) / 10.
        common = len(outgoing[s] & outgoing[d])
        result[i, 3] = common / max(1, len(outgoing[s]) + len(outgoing[d]) - common)
        common = len(incoming[s] & incoming[d])
        result[i, 4] = common / max(1, len(incoming[s]) + len(incoming[d]) - common)

    for k, indices in enumerate(parts):
        indices = indices[np.argsort(times[indices], kind='stable')]
        start = 0
        while start < len(indices):
            end = start + 1
            while end < len(indices) and times[indices[end]] == times[indices[start]]:
                end += 1
            for i in indices[start:end]:
                read(i)
            if k == 0:
                for i in indices[start:end]:
                    write(i)
            start = end
        if k == 1:
            for i in indices:
                write(i)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, features=result, fingerprint=np.asarray(fingerprint))
    return result
