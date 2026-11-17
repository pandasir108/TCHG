from __future__ import annotations

from collections import Counter

import numpy as np


def static_training_features(arrays):
    if np.any(arrays['timestamp'] != 0.):
        raise ValueError('Static trust features require zero native trust timestamps')
    src, dst = arrays['src'], arrays['dst']
    n, nodes = len(src), int(arrays['num_nodes'][0])
    train = arrays['split'] == 0
    eligible = np.zeros(n, dtype=bool)
    eligible[train] = ((arrays['sample_type'][train] == 0) & (arrays['label'][train] > .5)
                       & (arrays['observed_mask'][train] > .5))
    if 'history_visible_mask' in arrays:
        eligible &= arrays['history_visible_mask'] > .5
    outgoing = np.bincount(src[eligible], minlength=nodes)
    incoming = np.bincount(dst[eligible], minlength=nodes)
    pair_counts = Counter(zip(src[eligible].tolist(), dst[eligible].tolist()))
    removal = eligible.astype(np.int64)
    out_src, in_dst = outgoing[src] - removal, incoming[dst] - removal
    activity = outgoing + incoming
    activity_src = activity[src] - removal * (1 + (src == dst))
    activity_dst = activity[dst] - removal * (1 + (src == dst))
    count = np.fromiter((pair_counts.get((int(s), int(d)), 0) for s, d in zip(src, dst)),
                        dtype=np.int64, count=n) - removal
    def norm(value, scale=10.):
        return np.clip(np.log1p(value) / np.log1p(scale), 0., 1.)
    ratio_src = (out_src + 1.) / (out_src + 2.)
    ratio_dst = (in_dst + 1.) / (in_dst + 2.)
    ratio_pair = (count + 1.) / (count + 2.)
    last = np.where(count > 0, 1., .5)
    recent_negative = np.where(int(eligible.sum()) - removal > 0, 0., .5)
    entity = np.column_stack([ratio_src, ratio_dst, norm(activity_src), norm(activity_dst),
                              norm(out_src), norm(in_dst), np.zeros(n), np.zeros(n)]).astype(np.float32)
    behavior = np.column_stack([last, ratio_pair, norm(count), np.zeros(n), np.zeros(n),
                                1. - ratio_pair, np.abs(last - ratio_src), recent_negative]).astype(np.float32)
    context = np.zeros_like(arrays['context_features'])
    context[:, 0] = arrays['edge_type'] / 4.
    context[:, 5] = norm(activity_src + activity_dst, 20.)
    context[:, 6] = recent_negative
    context[:, 8:] = arrays['context_features'][:, 8:]
    edge = np.column_stack([arrays['edge_type'] / 4., np.zeros(n), np.zeros(n), norm(count),
                            arrays['observed_mask'], recent_negative]).astype(np.float32)
    return {'entity_features': entity, 'behavior_features': behavior,
            'context_features': context, 'edge_features': edge,
            'delta_t': np.zeros(n, dtype=np.float32)}


def static_structural_features(arrays):
    src, dst = arrays['src'], arrays['dst']
    n, nodes = len(src), int(arrays['num_nodes'][0])
    train = arrays['split'] == 0
    eligible = np.zeros(n, dtype=bool)
    eligible[train] = ((arrays['sample_type'][train] == 0) & (arrays['label'][train] > .5)
                       & (arrays['observed_mask'][train] > .5))
    if 'history_visible_mask' in arrays:
        eligible &= arrays['history_visible_mask'] > .5
    pairs = Counter(zip(src[eligible].tolist(), dst[eligible].tolist()))
    outgoing, incoming = [set() for _ in range(nodes)], [set() for _ in range(nodes)]
    for s, d in pairs:
        outgoing[s].add(d)
        incoming[d].add(s)
    entity = np.zeros((n, 4), dtype=np.float32)
    behavior = np.zeros((n, 2), dtype=np.float32)
    context = np.zeros((n, 4), dtype=np.float32)
    for i, (s, d) in enumerate(zip(src, dst)):
        s, d = int(s), int(d)
        remove = eligible[i] and pairs[s, d] == 1
        out_s = outgoing[s] - {d} if remove else outgoing[s]
        in_d = incoming[d] - {s} if remove else incoming[d]
        out_d = out_s if s == d else outgoing[d]
        in_s = in_d if s == d else incoming[s]
        reciprocal = s in out_d
        two_hop = len(out_s & in_d)
        common_out, common_in = len(out_s & out_d), len(in_s & in_d)
        entity[i] = np.log1p([len(in_s), len(out_d),
                              len(out_s) + len(in_s), len(out_d) + len(in_d)]) / 10.
        behavior[i] = [float(reciprocal), np.log1p(two_hop) / 10.]
        context[i] = [common_out / max(1, len(out_s | out_d)),
                       common_in / max(1, len(in_s | in_d)),
                       np.log1p(common_out) / 10., np.log1p(common_in) / 10.]
    return {'entity_features': entity, 'behavior_features': behavior, 'context_features': context}
