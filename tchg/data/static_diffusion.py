from __future__ import annotations

from collections import Counter

import numpy as np
import torch


def static_diffusion_features(arrays, damping=(.5, .85), device='cpu'):
    if np.any(arrays['timestamp'] != 0.):
        raise ValueError('Static diffusion requires zero native trust timestamps')
    damping = tuple(float(value) for value in damping)
    if not damping or any(not 0. < value < 1. for value in damping):
        raise ValueError('Diffusion damping values must be in (0, 1)')
    src, dst = arrays['src'], arrays['dst']
    n, nodes = len(src), int(arrays['num_nodes'][0])
    train = arrays['split'] == 0
    eligible = np.zeros(n, dtype=bool)
    eligible[train] = ((arrays['label'][train] > .5) & (arrays['sample_type'][train] == 0)
                       & (arrays['observed_mask'][train] > .5))
    if 'history_visible_mask' in arrays:
        eligible &= arrays['history_visible_mask'] > .5
    pairs = Counter(zip(src[eligible].tolist(), dst[eligible].tolist()))
    behavior = np.zeros((n, 4 * len(damping)), dtype=np.float32)
    context = np.zeros((n, 8 * len(damping)), dtype=np.float32)
    if not pairs:
        return {'behavior_features': behavior, 'context_features': context}
    edge_src, edge_dst = np.array(list(pairs), dtype=np.int64).T
    active = np.unique(np.concatenate([edge_src, edge_dst]))
    mapping = np.full(nodes, -1, dtype=np.int64)
    mapping[active] = np.arange(len(active))
    valid = (mapping[src] >= 0) & (mapping[dst] >= 0)
    indices = np.flatnonzero(valid)
    s, d = mapping[src[valid]], mapping[dst[valid]]
    delete = np.array([eligible[i] and pairs[int(src[i]), int(dst[i])] == 1
                       for i in indices], dtype=bool)
    adjacency = np.zeros((len(active), len(active)), dtype=np.float64)
    adjacency[mapping[edge_src], mapping[edge_dst]] = 1.
    dtype = torch.float64
    for orientation in range(2):
        graph = adjacency if orientation == 0 else adjacency.T
        degree = graph.sum(axis=1)
        transition = graph / np.maximum(degree[:, None], 1.)
        matrix = torch.as_tensor(transition, dtype=dtype, device=device)
        identity = torch.eye(len(active), dtype=dtype, device=device)
        r, c = (s, d) if orientation == 0 else (d, s)
        k = degree[r]
        for column, alpha in enumerate(damping):
            inverse = torch.linalg.inv(identity - alpha * matrix).cpu().numpy()
            def score(i, j):
                value = inverse[i, j].astype(np.float64)
                delta = inverse[r, j] - (r == j) - alpha * inverse[c, j]
                denominator = k - inverse[r, r] + alpha * inverse[c, r]
                many = value + inverse[i, r] * delta / np.where(k > 1., denominator, 1.)
                one = value - alpha * inverse[i, r] * inverse[c, j] / (1. + alpha * inverse[c, r])
                return np.where(delete, np.where(k > 1., many, one), value)
            sd, ds, ss, dd = score(s, d), score(d, s), score(s, s), score(d, d)
            forward, backward = np.maximum(sd - (s == d), 0.), np.maximum(ds - (s == d), 0.)
            offset = orientation * len(damping) + column
            behavior[indices, 2 * offset:2 * offset + 2] = np.log1p(
                100. * np.column_stack([forward, backward])) / 10.
            context[indices, 4 * offset:4 * offset + 4] = np.log1p(100. * np.column_stack([
                np.maximum(ss - 1., 0.), np.maximum(dd - 1., 0.),
                forward / np.maximum(dd, 1.), backward / np.maximum(ss, 1.)])) / 10.
    return {'behavior_features': behavior, 'context_features': context}
