from __future__ import annotations

import numpy as np
import pandas as pd


def collaborative_bad_mouthing(arrays, rate, seed=59, target_fraction=.1, max_rate=.2,
                               colluder_fraction=.1):
    if (not 0. <= rate <= max_rate <= 1. or not 0. < target_fraction < 1. or
            not 0. < colluder_fraction < .5):
        raise ValueError('Require valid edge rates, target fraction and a minority colluder fraction')
    keep = (arrays['split'] < 2) & (arrays['sample_type'] == 0)
    hist = pd.DataFrame({key: arrays[key][keep] for key in ['src', 'dst', 'timestamp', 'label', 'split']})
    count = int(arrays['num_nodes'][0])
    degree = np.bincount(np.r_[hist.src, hist.dst], minlength=count)
    positive = hist[hist.label > .5]
    positive_degree = np.bincount(np.r_[positive.src, positive.dst], minlength=count)
    good = np.flatnonzero(positive_degree > degree - positive_degree)
    original = {k: int((hist['split'] == k).sum()) for k in [0, 1]}
    requested_max = {k: int(round(max_rate * original[k])) for k in [0, 1]}
    max_count = sum(requested_max.values())
    rng = np.random.default_rng(seed)
    pool = rng.permutation(good)
    minimum = max(1, int(round(target_fraction * len(good))))
    required = int(np.searchsorted(np.cumsum(degree[pool]), max_count, side='left')) + 1
    targets = pool[:max(minimum, required)]
    history_users = np.unique(np.r_[hist.src, hist.dst])
    eligible_colluders = np.setdiff1d(history_users, targets)
    colluder_count = max(1, int(round(colluder_fraction * len(history_users))))
    if colluder_count > len(eligible_colluders):
        raise ValueError('Insufficient non-target users for the fixed colluder pool')
    colluders = rng.permutation(eligible_colluders)[:colluder_count]
    destinations = rng.permutation(np.repeat(targets, degree[targets]))[:max_count]
    available = len(destinations)
    train_count = min(requested_max[0], int(round(available * original[0] / max(1, len(hist)))))
    train_count = max(train_count, available - requested_max[1])
    split_ids = np.r_[np.zeros(train_count, dtype=int), np.ones(available - train_count, dtype=int)]
    frames = []
    for split in [0, 1]:
        indices = np.flatnonzero(split_ids == split)
        part = hist[hist['split'] == split]
        if not len(indices):
            continue
        src = rng.choice(colluders, size=len(indices), replace=True)
        dst = destinations[indices]
        times = rng.choice(part.timestamp.to_numpy(), size=len(indices), replace=True)
        frames.append(pd.DataFrame({'src': src, 'dst': dst, 'timestamp': times,
                                    'label': 0., 'split': split}))
    columns = ['src', 'dst', 'timestamp', 'label', 'split']
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    edges = pd.concat([full[full['split'] == split].iloc[:int(round(rate * original[split]))]
                       for split in [0, 1]], ignore_index=True)
    if len(edges):
        additions = np.bincount(edges.dst, minlength=count)
        if not np.all(additions <= degree):
            raise RuntimeError('Attack exceeded the per-target degree cap')
    audit = {
        'attack': 'TrustGuard collaborative bad-mouthing with CAT timestamp sampling',
        'implementation': 'published-description reproduction; CAT Epinions attack code is not released',
        'references': ['https://arxiv.org/html/2306.13339',
                       'https://www.ndss-symposium.org/wp-content/uploads/2026-f2171-paper.pdf'],
        'seed': int(seed), 'requested_rate': float(rate),
        'max_rate': float(max_rate),
        'minimum_target_fraction': float(target_fraction),
        'actual_target_fraction': len(targets) / len(good) if len(good) else 0.,
        'target_count': len(targets),
        'target_reference': 'good nodes in clean train/validation input; no test labels or candidates',
        'target_selection_adaptation': 'TrustGuard uses good test targets on signed Bitcoin; CAT does not release its Epinions targets. Use pre-test input nodes.',
        'budget_adaptation': 'expand the initial target set only as needed for max_rate under the original-degree cap; no model scores used',
        'colluders': 'fixed minority of clean history users, disjoint from honest targets',
        'colluder_fraction': float(colluder_fraction), 'colluder_count': len(colluders),
        'colluder_policy_adaptation': f'CAT does not publish the Epinions colluder count; declare a fixed {100 * colluder_fraction:g}% minority independently of victim scores',
        'timestamps': 'empirical original timestamps sampled separately in train and validation',
        'label': 0, 'rating': -1,
        'budget_basis': 'original genuine edges in each history split',
        'target_degree_cap': 'original total incoming plus outgoing degree',
        'test_data_used_for_attack': False,
        'by_split': {str(split): {'original_edges': original[split],
                                'requested_edges': int(round(rate * original[split])),
                                'injected_edges': int((edges['split'] == split).sum())}
                     for split in [0, 1]},
    }
    return edges[columns], audit
