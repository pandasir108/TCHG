from __future__ import annotations

from copy import copy

import numpy as np
import pandas as pd

from tchg.data.feature_builder import build_event_features_cat_rolling


HISTORY_SCENARIOS = {"trust_oriented", "gnn_oriented"}
HISTORY_PROTOCOL = {
    "name": "cat_style_history",
    "stage": "inference_history_perturbation",
    "scope": "train_val_positive_history",
    "budget_basis": "test_positive_edges",
    "retraining": False,
    "query_edges_and_labels": "fixed_clean_test_candidates",
    "user_regimes": "fixed_clean_training_reference",
    "calibration_and_threshold": "fixed_clean_validation",
}


def perturb_history(arrays, scenario, rate, seed):
    if scenario not in HISTORY_SCENARIOS:
        raise ValueError(f"Unknown history attack: {scenario}")
    if not np.isfinite(rate) or not 0. <= rate <= 1.:
        raise ValueError("Attack rate must be finite and in [0, 1].")
    out = {key: value.copy() for key, value in arrays.items()}
    real_positive = (arrays['sample_type'] == 0) & (arrays['label'] > .5)
    candidates = np.flatnonzero((arrays['split'] < 2) & real_positive)
    test_positive = int(np.count_nonzero((arrays['split'] == 2) & real_positive))
    count = max(1, int(round(test_positive * rate))) if rate > 0. and test_positive else 0
    chosen = np.random.default_rng(seed).permutation(candidates)[:count]
    audit = {
        **HISTORY_PROTOCOL,
        'scenario': scenario,
        'requested_ratio': float(rate),
        'test_positive_edges': test_positive,
        'history_positive_edges': len(candidates),
        'selected_history_edges': len(chosen),
        'actual_ratio_vs_test_positive': len(chosen) / test_positive if test_positive else 0.,
        'test_edges_changed': 0,
        'test_labels_changed': 0,
    }
    if not len(chosen):
        audit['changed_history_edges'] = 0
        return out, audit
    rng = np.random.default_rng(seed + 7919)
    if scenario == 'trust_oriented':
        nodes = int(arrays['num_nodes'][0])
        if nodes < 2:
            raise ValueError("History rewiring requires at least two users.")
        src = rng.integers(0, nodes, size=len(chosen), endpoint=False)
        dst = rng.integers(0, nodes, size=len(chosen), endpoint=False)
        same = src == dst
        dst[same] = (dst[same] + 1) % nodes
        out['src'][chosen], out['dst'][chosen] = src, dst
        audit['perturbation'] = 'rewire_history_endpoints_timestamps_fixed'
    else:
        if len(chosen) > 1:
            shuffled = chosen[rng.permutation(len(chosen))]
            out['src'][chosen] = arrays['src'][shuffled]
            out['dst'][chosen] = arrays['dst'][shuffled[::-1]]
        out['timestamp'][chosen] = np.quantile(arrays['timestamp'][candidates], .05)
        audit['perturbation'] = 'shuffle_history_endpoints_and_shift_history_time'
    changed = np.zeros(len(arrays['src']), dtype=bool)
    for key in ('src', 'dst', 'timestamp'):
        changed |= out[key] != arrays[key]
    audit['changed_history_edges'] = int(changed.sum())
    return out, audit


def rebuild_history_features(clean_arrays, attacked, data_cfg, seed):
    split = np.asarray(attacked['split'])
    if not np.isin(split, [0, 1, 2]).all():
        raise ValueError("History attacks require explicit train/validation/test splits.")
    history = split < 2
    test = split == 2
    if not history.any() or not test.any():
        raise ValueError("History attacks require nonempty history and test sets.")
    if attacked['timestamp'][history].max() >= attacked['timestamp'][test].min():
        raise ValueError("Perturbed history must precede the test queries.")
    widths = {'entity_features': 8, 'behavior_features': 8,
              'context_features': 8, 'edge_features': 6}
    if any(attacked[key].shape[1] != width for key, width in widths.items()):
        raise ValueError("Rebuild history from the base Epinions event arrays before feature augmentation.")
    n = len(split)
    events = pd.DataFrame({
        key: attacked[key] for key in ('src', 'dst', 'timestamp', 'label', 'sample_type', 'edge_type')
    })
    events['rating'] = np.where(events['label'] > .5, 1., -1.)
    events['split'] = split
    events['generated_observed'] = clean_arrays['observed_mask']
    events['history_visible_mask'] = clean_arrays.get('history_visible_mask', np.ones(n))
    events['row_id'] = np.arange(n)
    ordered = pd.concat([
        events[events['split'] == k].sort_values('timestamp', kind='mergesort')
        for k in (0, 1, 2)
    ], ignore_index=True)
    rebuilt = build_event_features_cat_rolling(
        events,
        num_nodes=int(attacked['num_nodes'][0]),
        p_missing=float(data_cfg.get('p_missing', 0.)),
        seed=seed,
        respect_observed_history=bool(data_cfg.get('respect_observed_history', False)),
        zero_time_features=bool(data_cfg.get('zero_trust_time_features', False)),
        node_features_override=clean_arrays['node_features'],
    )
    ids = ordered['row_id'].to_numpy(dtype=np.int64)
    event_keys = {'split', 'is_new_node', 'observed_user_mask', 'unobserved_user_mask'}
    for key, value in clean_arrays.items():
        if key not in rebuilt:
            rebuilt[key] = value[ids].copy() if key in event_keys else value.copy()
    rebuilt['split'] = ordered['split'].to_numpy(dtype=np.int64)
    return rebuilt


def _verified_node_mapping(arrays, data_cfg):
    from tchg.data.preprocessors import read_epinions_raw

    raw = read_epinions_raw(data_cfg.get('raw_dir', 'data/raw'))
    mapping = np.asarray(raw.attrs['node_to_raw'], dtype=np.int64)
    if len(mapping) != int(arrays['num_nodes'][0]):
        raise ValueError("Raw and processed Epinions user mappings differ.")
    if 'node_to_raw' in arrays:
        if not np.array_equal(arrays['node_to_raw'], mapping):
            raise ValueError("Raw and processed Epinions user mappings differ.")
    else:
        real = np.flatnonzero(arrays['sample_type'] == 0)
        real = real[np.argsort(arrays['timestamp'][real], kind='stable')]
        ordered = raw.sort_values('timestamp', kind='mergesort')
        for key in ('src', 'dst', 'timestamp'):
            if not np.array_equal(arrays[key][real], ordered[key].to_numpy().astype(arrays[key].dtype)):
                raise ValueError("Cannot certify the clean Epinions user mapping.")
    return mapping


def history_attack_batches(trainer, clean_arrays, scenario, rate, seed):
    data_cfg = trainer.cfg['dataset']
    if trainer.static_graph or not str(data_cfg.get('name', '')).startswith('epinions'):
        raise ValueError("The CAT-style history protocol currently supports temporal Epinions only.")
    attacked, audit = perturb_history(clean_arrays, scenario, rate, seed)
    if not audit['selected_history_edges']:
        return trainer.batches, audit
    arrays = rebuild_history_features(clean_arrays, attacked, data_cfg, seed)
    if data_cfg.get('causal_structure_features', False):
        from tchg.data.structural_features import causal_structural_features

        features = causal_structural_features(arrays, require_separated_splits=False)
        arrays['behavior_features'] = np.concatenate([arrays['behavior_features'], features[:, :2]], axis=1)
        arrays['context_features'] = np.concatenate([arrays['context_features'], features[:, 2:]], axis=1)
    if data_cfg.get('causal_rating_features', False):
        from tchg.data.rating_features import load_causal_rating_features

        arrays['node_to_raw'] = _verified_node_mapping(clean_arrays, data_cfg)
        features, _ = load_causal_rating_features(
            arrays, data_cfg.get('raw_dir', 'data/raw'), None,
            data_cfg.get('rating_history_policy', 'split_frozen'),
            int(data_cfg.get('rating_profile_dim', 0)),
            float(data_cfg.get('rating_profile_window_days', 365.)),
            bool(data_cfg.get('rating_temporal_activity', False)),
            int(data_cfg.get('rating_profile_seed', 42)),
        )
        for key, value in features.items():
            arrays[key] = np.concatenate([arrays[key], value], axis=1) if key in arrays else value
    arrays['split'] = np.where(arrays['split'] == 2, 2, 0).astype(np.int64)
    dataset = copy(trainer.dataset)
    dataset.arrays = arrays
    if dataset.feature_dims != trainer.dataset.feature_dims:
        raise ValueError("Perturbed and clean model input dimensions differ.")
    batches = {
        name: dataset.to_torch(dataset.subset(np.flatnonzero(arrays['split'] == k)), trainer.device)
        for name, k in (('train', 0), ('val', 1), ('test', 2))
    }
    for key in ('src', 'dst', 'timestamp', 'label', 'sample_type',
                'observed_user_mask', 'unobserved_user_mask', 'is_new_node'):
        if not np.array_equal(batches['test'][key].cpu().numpy(), trainer.batches['test'][key].cpu().numpy()):
            raise ValueError(f"History perturbation changed test candidates or user regimes: {key}")
    return batches, audit


def evaluate_history_attack(trainer, clean_arrays, scenario, rate, seed, threshold):
    batches, audit = history_attack_batches(trainer, clean_arrays, scenario, rate, seed)
    original = trainer.batches
    try:
        trainer.batches = batches
        metrics = trainer.evaluate('test', threshold=threshold)
    finally:
        trainer.batches = original
        trainer.model.reset_memory()
    return metrics, audit
