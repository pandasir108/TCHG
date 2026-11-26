from __future__ import annotations

import numpy as np
import pandas as pd

from tchg.data.feature_builder import build_event_features_cat_rolling
from tchg.data.user_regimes import training_reference_users, with_user_regime_masks


POISONING_PROTOCOL = {
    'name': 'cat_paper_poisoning',
    'stage': 'training_validation_edge_injection',
    'retraining': True,
    'budget_basis': 'original_genuine_edges_in_each_history_split',
    'query_edges_and_labels': 'fixed_clean_test_candidates',
    'user_regimes': 'fixed_clean_training_reference',
    'selection': 'validation_only',
    'reference': 'https://www.ndss-symposium.org/wp-content/uploads/2026-f2171-paper.pdf',
}


def materialize_poisoning(clean, injections, data_cfg, seed):
    clean = with_user_regime_masks(clean)
    if 'node_to_raw' not in clean and data_cfg.get('causal_rating_features', False):
        from tchg.experiments.cat_style_history import _verified_node_mapping

        clean = {**clean, 'node_to_raw': _verified_node_mapping(clean, data_cfg)}
    columns = ['src', 'dst', 'timestamp', 'label', 'split']
    if not set(columns).issubset(injections.columns):
        raise ValueError(f'Injected edges require columns: {columns}')
    injections = injections[columns].copy()
    count = int(clean['num_nodes'][0])
    if len(injections):
        if not np.isfinite(injections.to_numpy()).all():
            raise ValueError('Injected edges must be finite')
        for key in ['src', 'dst', 'split']:
            if not np.equal(injections[key], injections[key].astype(np.int64)).all():
                raise ValueError(f'{key} must contain integer identifiers')
            injections[key] = injections[key].astype(np.int64)
        if not injections['split'].isin([0, 1]).all():
            raise ValueError('Poisoning cannot inject test edges')
        if not injections['label'].isin([0., 1.]).all():
            raise ValueError('Injected labels must be binary')
        if ((injections.src < 0) | (injections.dst < 0) |
                (injections.src >= count) | (injections.dst >= count) |
                (injections.src == injections.dst)).any():
            raise ValueError('Injected edges require distinct valid endpoints')
        for split in [0, 1]:
            original = clean['timestamp'][clean['split'] == split]
            added = injections.loc[injections['split'] == split, 'timestamp']
            if len(added) and (added.min() < original.min() or added.max() > original.max()):
                raise ValueError('Injected timestamps must remain inside their history split')
    n = len(clean['src'])
    events = pd.DataFrame({key: clean[key] for key in ['src', 'dst', 'timestamp', 'label', 'split', 'edge_type', 'sample_type']})
    events['generated_observed'] = clean['observed_mask']
    events['history_visible_mask'] = clean.get('history_visible_mask', np.ones(n))
    events['source_event_id'] = np.arange(n)
    events['poison_mask'] = 0
    injections['edge_type'] = 0
    injections['sample_type'] = 0
    injections['generated_observed'] = 1.
    injections['history_visible_mask'] = 1.
    injections['source_event_id'] = -1
    injections['poison_mask'] = 1
    events = pd.concat([events, injections], ignore_index=True)
    events['rating'] = np.where(events['label'] > .5, 1., -1.)
    ordered = pd.concat([events[events['split'] == split].sort_values('timestamp', kind='stable')
                         for split in [0, 1, 2]], ignore_index=True)
    if len(injections):
        arrays = build_event_features_cat_rolling(
            events, count, float(data_cfg.get('p_missing', 0.)), seed,
            respect_observed_history=bool(data_cfg.get('respect_observed_history', False)),
            zero_time_features=bool(data_cfg.get('zero_trust_time_features', False)),
            node_features_override=clean['node_features'],
        )
    else:
        arrays = {key: value.copy() for key, value in clean.items()}
    arrays['split'] = ordered['split'].to_numpy(dtype=np.int64)
    arrays['source_event_id'] = ordered['source_event_id'].to_numpy(dtype=np.int64)
    arrays['poison_mask'] = ordered['poison_mask'].to_numpy(dtype=np.int64)
    if 'node_to_raw' in clean:
        arrays['node_to_raw'] = clean['node_to_raw'].copy()
    original_users = training_reference_users(clean)
    if not np.array_equal(training_reference_users(arrays), original_users):
        raise ValueError('Injected training endpoints changed the clean training user reference')
    arrays = with_user_regime_masks(arrays)
    test = arrays['split'] == 2
    original_test = clean['split'] == 2
    for key in ['src', 'dst', 'timestamp', 'label', 'sample_type', 'observed_mask',
                'observed_user_mask', 'unobserved_user_mask', 'is_new_node']:
        if not np.array_equal(arrays[key][test], clean[key][original_test]):
            raise ValueError(f'Poisoning changed clean test candidates: {key}')
    audit = {**POISONING_PROTOCOL, 'seed': int(seed), 'injected_edges': len(injections),
             'test_candidates': int(test.sum()), 'test_candidates_preserved': True,
             'clean_training_reference_preserved': True, 'by_split': {}}
    for split in [0, 1]:
        original = int(np.count_nonzero((clean['split'] == split) & (clean['sample_type'] == 0)))
        added = int(np.count_nonzero(injections['split'] == split))
        audit['by_split'][str(split)] = {'original_edges': original, 'injected_edges': added,
                                        'actual_ratio': added / original if original else 0.}
    return arrays, audit
