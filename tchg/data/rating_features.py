"""Pair evidence from timestamped rating history, without trust query labels."""
from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

from tchg.data.preprocessors import read_epinions_raw
from tchg.utils.io import resolve_path


def causal_rating_features(arrays, ratings, history_policy='split_frozen', profile_dim=0,
                           profile_window_days=365., temporal_activity=False, profile_seed=42):
    """Build entity/activity, behavioral agreement, and category evidence.

    Rating columns are mapped user, item, category, rating, helpfulness, time.
    Training reads strictly earlier ratings. With split_frozen, validation
    freezes at the training boundary and test at the validation boundary.
    pre_event instead permits all ratings strictly earlier than each query.
    Neither policy consults trust labels, candidate kinds, or event masks.
    """
    if history_policy not in {'split_frozen', 'pre_event'}:
        raise ValueError('Unknown rating history policy')
    if profile_dim and (profile_dim < 4 or profile_dim % 2):
        raise ValueError('rating profile_dim must be zero or an even integer >= 4')
    if profile_window_days <= 0.:
        raise ValueError('profile_window_days must be positive')
    src, dst = arrays['src'], arrays['dst']
    times = np.asarray(arrays['timestamp'], dtype=np.float64)
    limits = times.copy()
    if history_policy == 'split_frozen':
        split = arrays['split']
        for k in (1, 2):
            previous = times[split < k]
            if len(previous):
                cutoff = np.nextafter(previous.max(), np.inf)
                limits[split == k] = np.minimum(limits[split == k], cutoff)
    n, nodes = len(src), int(arrays['num_nodes'][0])
    result = {'entity_features': np.zeros((n, 8), np.float32),
              'behavior_features': np.zeros((n, 3), np.float32),
              'context_features': np.zeros((n, 2), np.float32)}
    if temporal_activity:
        result['entity_features'] = np.zeros((n, 16), np.float32)
        result['context_features'] = np.zeros((n, 4), np.float32)
    if profile_dim:
        result['rating_profile_src'] = np.zeros((n, profile_dim), np.float32)
        result['rating_profile_dst'] = np.zeros((n, profile_dim), np.float32)
    ratings = np.asarray(ratings, dtype=np.float64)
    ratings = ratings[np.argsort(ratings[:, 5], kind='stable')]
    count = np.zeros(nodes, dtype=np.int64)
    total = np.zeros(nodes); square = np.zeros(nodes); helpful = np.zeros(nodes)
    item_scores = [{} for _ in range(nodes)]
    categories = [{} for _ in range(nodes)]
    category_norm2 = np.zeros(nodes)
    history_times = [[] for _ in range(nodes)] if profile_dim or temporal_activity else None
    profile_state = np.zeros((nodes, profile_dim), np.float32) if profile_dim else None
    profile_prefix = [[] for _ in range(nodes)] if profile_dim else None
    item_vectors = {}
    half = profile_dim // 2
    pointer = 0
    for i in np.lexsort((times, limits)):
        while pointer < len(ratings) and ratings[pointer, 5] < limits[i]:
            u, item, category, rating, help_value, _ = ratings[pointer]
            u, item, category = int(u), int(item), int(category)
            if not 0 <= u < nodes:
                raise ValueError('Rating user ID is outside the processed trust-user mapping')
            count[u] += 1; total[u] += rating; square[u] += rating * rating
            helpful[u] += max(0., help_value)
            item_scores[u][item] = rating
            old = categories[u].get(category, 0)
            category_norm2[u] += 2 * old + 1
            categories[u][category] = old + 1
            if history_times is not None:
                history_times[u].append(float(ratings[pointer, 5]))
            if profile_dim:
                if item not in item_vectors:
                    rng = np.random.default_rng(np.random.SeedSequence([int(profile_seed), item]))
                    item_vectors[item] = (2 * rng.integers(0, 2, size=half) - 1).astype(np.float32) / np.sqrt(half)
                profile_state[u, :half] += ((rating - 3.) / 2.) * item_vectors[item]
                profile_state[u, half + category % half] += 1.
                profile_prefix[u].append(profile_state[u].copy())
            pointer += 1
        s, d = int(src[i]), int(dst[i])
        ids = np.array([s, d])
        mean = np.divide(total[ids], count[ids], out=np.full(2, 2.5), where=count[ids] > 0)
        second = np.divide(square[ids], count[ids], out=np.square(mean), where=count[ids] > 0)
        deviation = np.sqrt(np.maximum(0., second - mean * mean))
        help_mean = helpful[ids] / np.maximum(1, count[ids])
        result['entity_features'][i, :8] = np.r_[np.log1p(count[ids]) / 10., mean / 5.,
                                                   deviation / 5., np.log1p(help_mean) / 10.]
        left, right = item_scores[s], item_scores[d]
        shared = left.keys() & right.keys()
        agreement = 1. - sum(abs(left[x] - right[x]) for x in shared) / (4. * len(shared)) if shared else .5
        result['behavior_features'][i] = [np.log1p(len(shared)) / 10.,
            len(shared) / max(1, len(left) + len(right) - len(shared)), agreement]
        left_cat, right_cat = categories[s], categories[d]
        common_cat = left_cat.keys() & right_cat.keys()
        dot = sum(left_cat[x] * right_cat[x] for x in common_cat)
        norm = np.sqrt(category_norm2[s] * category_norm2[d])
        result['context_features'][i, :2] = [dot / norm if norm else 0.,
            len(common_cat) / max(1, len(left_cat) + len(right_cat) - len(common_cat))]
        if temporal_activity:
            for role, user in enumerate((s, d)):
                history = history_times[user]
                recent = [len(history) - bisect_left(history, times[i] - day * 86400.) for day in (30., 180., 365.)]
                result['entity_features'][i, 8 + role * 4:12 + role * 4] = [
                    *list(np.log1p(recent) / 10.), recent[-1] / max(1, len(history))]
                result['context_features'][i, 2 + role] = np.log1p(max(0., times[i] - history[-1]) / 86400.) / 10. if history else 0.
        if profile_dim:
            for key, user in [('rating_profile_src', s), ('rating_profile_dst', d)]:
                cutoff = bisect_left(history_times[user], times[i] - profile_window_days * 86400.)
                value = profile_state[user].copy()
                if cutoff:
                    value -= profile_prefix[user][cutoff-1]
                recent_count = len(history_times[user]) - cutoff
                for piece in (slice(0, half), slice(half, profile_dim)):
                    value[piece] /= max(1e-8, float(np.linalg.norm(value[piece])))
                result[key][i] = value * np.sqrt(recent_count / (recent_count + 5.))
    return result


def load_causal_rating_features(arrays, raw_root='data/raw', cache_path=None, history_policy='split_frozen',
                               profile_dim=0, profile_window_days=365., temporal_activity=False, profile_seed=42):
    root = resolve_path(raw_root) / 'epinions'
    trust_path, rating_path = root / 'user_user_time.txt', root / 'user_item_time.txt'
    digest = hashlib.sha256(('causal-ratings-v1;' + history_policy).encode())
    if profile_dim or temporal_activity:
        digest.update(f';recent-profile-v1;dim={profile_dim};days={profile_window_days};activity={temporal_activity};seed={profile_seed}'.encode())
    source_hashes = {}
    for path in (trust_path, rating_path):
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(source_hashes[path.name].encode())
    for key in ('src', 'dst', 'timestamp', 'split', 'num_nodes'):
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    if 'node_to_raw' in arrays:
        digest.update(np.ascontiguousarray(arrays['node_to_raw']).tobytes())
    fingerprint = digest.hexdigest()
    trust = read_epinions_raw(resolve_path(raw_root))
    mapping = trust.attrs['raw_to_node']
    if len(mapping) != int(arrays['num_nodes'][0]):
        raise ValueError('Raw trust-user mapping does not match the processed dataset')
    if 'node_to_raw' in arrays:
        if not np.array_equal(arrays['node_to_raw'], trust.attrs['node_to_raw']):
            raise ValueError('Processed node_to_raw does not match the raw trust-user mapping')
    elif 'sample_type' in arrays:
        real = np.asarray(arrays['sample_type']) == 0
        ordered = trust.sort_values('timestamp', kind='stable')
        if (int(real.sum()) != len(ordered) or
                not np.array_equal(arrays['src'][real], ordered['src'].to_numpy()) or
                not np.array_equal(arrays['dst'][real], ordered['dst'].to_numpy()) or
                not np.array_equal(arrays['timestamp'][real], ordered['timestamp'].to_numpy().astype(np.float32))):
            raise ValueError('Legacy processed relations do not certify the raw trust-user mapping; provide node_to_raw')
    else:
        raise ValueError('Rating evidence requires a verified node_to_raw mapping')
    cache = resolve_path(cache_path) if cache_path else None
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved['fingerprint'].item()) == fingerprint:
                metadata = json.loads(str(saved['metadata'].item()))
                metadata['user_id_mapping_verified'] = True
                keys = ['entity_features', 'behavior_features', 'context_features']
                if profile_dim: keys += ['rating_profile_src', 'rating_profile_dst']
                return ({key: saved[key].copy() for key in keys}, metadata)
    names = ['user_raw', 'item', 'category', 'rating', 'helpfulness', 'timestamp']
    ratings = pd.read_csv(rating_path, header=None, names=names, usecols=list(range(6)), index_col=False).dropna()
    ratings['user_raw'] = ratings['user_raw'].astype(np.int64)
    mapped = ratings['user_raw'].map(mapping)
    ratings = ratings.loc[mapped.notna()].copy()
    ratings['user_raw'] = mapped.loc[mapped.notna()].astype(np.int64)
    values = ratings.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not ((values[:, 3] >= 1.) & (values[:, 3] <= 5.)).all():
        raise ValueError('Invalid numeric rating records or unsupported rating scale')
    features = causal_rating_features(arrays, values, history_policy, profile_dim, profile_window_days, temporal_activity, profile_seed)
    metadata = {'history_policy': history_policy, 'source_sha256': source_hashes,
                'mapped_rating_records': len(values), 'user_id_mapping': 'same raw_to_node mapping as trust preprocessing',
                'user_id_mapping_verified': True,
                'trust_labels_used': False, 'trust_event_missing_mask_used': False,
                'rating_observation_policy': 'all timestamped rating records within the permitted history prefix',
                'profile_dim': profile_dim, 'profile_window_days': profile_window_days,
                'temporal_activity': temporal_activity, 'profile_seed': profile_seed,
                'profile_basis': 'fixed item hash and category counts; no fitting on future interactions',
                'feature_dims': {k: v.shape[1] for k, v in features.items()}}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **features, fingerprint=np.asarray(fingerprint), metadata=np.asarray(json.dumps(metadata)))
    return features, metadata
