from __future__ import annotations

import numpy as np


HISTORY_PROTOCOL = {
    'name': 'fixed_history_coverage',
    'stage': 'inference_history_perturbation',
    'scope': 'visible_train_val_positive_history',
    'budget_basis': 'test_positive_edges',
    'retraining': False,
    'query_edges_and_labels': 'fixed_clean_test_candidates',
    'user_regimes': 'fixed_clean_training_reference',
    'calibration_and_threshold': 'fixed_clean_validation',
    'endpoint_policy': 'clean_visible_role_history_strictly_before_event',
    'coverage_policy': 'preserve_first_visible_source_and_target_times',
    'protected_events': 'first_visible_source_or_target_occurrences',
    'timestamp_policy': 'same_split_and_after_clean_endpoint_role_births',
}


def visible_history_mask(arrays):
    n = len(arrays['src'])
    return ((arrays['split'] < 2) & (arrays['sample_type'] == 0) &
            (arrays['observed_mask'] > .5) &
            (arrays.get('history_visible_mask', np.ones(n)) > .5))


def first_visible_role_times(arrays):
    count = int(arrays['num_nodes'][0])
    visible = visible_history_mask(arrays)
    first = {}
    for role in ('src', 'dst'):
        first[role] = np.full(count, np.inf, dtype=np.float64)
        np.minimum.at(first[role], arrays[role][visible], arrays['timestamp'][visible])
    return first


def _next_timestamp(value, dtype):
    if np.issubdtype(dtype, np.floating):
        return float(np.nextafter(dtype.type(value), dtype.type(np.inf)))
    return float(int(value) + 1)


def _previous_timestamp(value, dtype):
    if np.issubdtype(dtype, np.floating):
        return float(np.nextafter(dtype.type(value), dtype.type(-np.inf)))
    return float(int(value) - 1)


def _coverage_audit(clean, attacked):
    before, after = first_visible_role_times(clean), first_visible_role_times(attacked)
    audit = {}
    for role in ('src', 'dst'):
        audit[f'new_visible_{role}_users'] = int(np.count_nonzero(
            np.isinf(before[role]) & np.isfinite(after[role])))
        audit[f'earlier_first_visible_{role}_times'] = int(np.count_nonzero(
            after[role] < before[role]))
        if not np.array_equal(before[role], after[role]):
            raise RuntimeError(f'History attack changed first visible {role} times.')
    audit['first_visible_role_times_preserved'] = True
    return audit


def perturb_history_preserving_coverage(arrays, scenario, rate, seed):
    if scenario not in {'trust_oriented', 'gnn_oriented'}:
        raise ValueError(f'Unknown history attack: {scenario}')
    if not np.isfinite(rate) or not 0. <= rate <= 1.:
        raise ValueError('Attack rate must be finite and in [0, 1].')
    if not np.isin(arrays['split'], [0, 1, 2]).all():
        raise ValueError('History attacks require train/validation/test split identifiers.')
    out = {key: value.copy() for key, value in arrays.items()}
    first = first_visible_role_times(arrays)
    times = arrays['timestamp']
    positive = (arrays['sample_type'] == 0) & (arrays['label'] > .5)
    visible = visible_history_mask(arrays)
    eligible = visible & positive & (first['src'][arrays['src']] < times) & (
        first['dst'][arrays['dst']] < times)
    lower_by_split = {}
    if scenario == 'gnn_oriented':
        birth = np.maximum(first['src'][arrays['src']], first['dst'][arrays['dst']])
        if np.issubdtype(times.dtype, np.floating):
            next_birth = np.nextafter(birth.astype(times.dtype), times.dtype.type(np.inf))
        else:
            next_birth = birth + 1
        eligible &= next_birth < times
        for split in (0, 1):
            indices = np.flatnonzero(arrays['split'] == split)
            if len(indices):
                lower_by_split[split] = float(times[indices].min())
                eligible[indices[times[indices] <= lower_by_split[split]]] = False
    candidates = np.flatnonzero(eligible)
    test_positive = int(np.count_nonzero((arrays['split'] == 2) & positive))
    requested = max(1, int(round(test_positive * rate))) if rate > 0. and test_positive else 0
    chosen = np.random.default_rng(seed).permutation(candidates)[:requested]
    audit = {
        **HISTORY_PROTOCOL,
        'scenario': scenario,
        'requested_ratio': float(rate),
        'attack_seed': int(seed),
        'test_positive_edges': test_positive,
        'history_positive_edges': int(np.count_nonzero((arrays['split'] < 2) & positive)),
        'visible_history_positive_edges': int(np.count_nonzero(visible & positive)),
        'eligible_history_positive_edges': len(candidates),
        'requested_history_edges': requested,
        'selected_history_edges': len(chosen),
        'budget_shortfall': max(0, requested - len(chosen)),
        'actual_ratio_vs_test_positive': len(chosen) / test_positive if test_positive else 0.,
        'test_edges_changed': 0,
        'test_labels_changed': 0,
    }
    rng = np.random.default_rng(seed + 7919)
    if scenario == 'trust_oriented' and len(chosen):
        ordered = {role: np.argsort(first[role], kind='stable') for role in first}
        ordered_times = {role: first[role][ordered[role]] for role in first}
        skipped = 0
        for i in chosen:
            limits = {role: int(np.searchsorted(ordered_times[role], times[i], side='left'))
                      for role in first}
            for _ in range(64):
                src = int(ordered['src'][rng.integers(limits['src'])])
                dst = int(ordered['dst'][rng.integers(limits['dst'])])
                if src != dst and (src != arrays['src'][i] or dst != arrays['dst'][i]):
                    out['src'][i], out['dst'][i] = src, dst
                    break
            else:
                skipped += 1
        audit['unrewirable_selected_edges'] = skipped
        audit['perturbation'] = 'rewire_endpoints_with_clean_visible_role_prefixes'
    elif scenario == 'gnn_oriented' and len(chosen):
        order = chosen[np.argsort(times[chosen], kind='stable')]
        starts = np.r_[0, np.flatnonzero(np.diff(times[order]) != 0) + 1, len(order)]
        swaps = 0
        for start, end in zip(starts[:-1], starts[1:]):
            group = order[start:end]
            shuffled = group[rng.permutation(len(group))]
            for left, right in zip(shuffled[::2], shuffled[1::2]):
                a, b = int(out['dst'][left]), int(out['dst'][right])
                if a != b and out['src'][left] != b and out['src'][right] != a:
                    out['dst'][left], out['dst'][right] = b, a
                    swaps += 1
        quantiles = {
            split: float(np.quantile(times[visible & positive & (arrays['split'] == split)], .05))
            for split in lower_by_split
            if np.any(visible & positive & (arrays['split'] == split))
        }
        for i in chosen:
            split = int(arrays['split'][i])
            birth = max(first['src'][out['src'][i]], first['dst'][out['dst'][i]])
            lower = max(lower_by_split[split], _next_timestamp(birth, times.dtype))
            upper = _previous_timestamp(times[i], times.dtype)
            if lower > upper:
                raise RuntimeError('Selected event has no legal earlier timestamp.')
            out['timestamp'][i] = max(lower, min(upper, quantiles[split]))
        audit['degree_preserving_swaps'] = swaps
        audit['perturbation'] = 'same_timestamp_endpoint_swaps_and_bounded_history_time_shift'
    changed = np.zeros(len(times), dtype=bool)
    for key in ('src', 'dst', 'timestamp'):
        changed |= out[key] != arrays[key]
    audit['changed_history_edges'] = int(changed.sum())
    audit['unchanged_selected_edges'] = len(chosen) - audit['changed_history_edges']
    if np.any(changed & ~eligible):
        raise RuntimeError('History attack modified a protected or invisible event.')
    for i in np.flatnonzero(changed):
        t = out['timestamp'][i]
        if first['src'][out['src'][i]] >= t or first['dst'][out['dst'][i]] >= t:
            raise RuntimeError('History attack used an endpoint without a strict clean history prefix.')
    for split in (0, 1):
        mask = arrays['split'] == split
        if mask.any() and (np.any(out['timestamp'][mask] < times[mask].min()) or
                           np.any(out['timestamp'][mask] > times[mask].max())):
            raise RuntimeError('History attack moved an event outside its clean split interval.')
    audit.update(_coverage_audit(arrays, out))
    return out, audit
