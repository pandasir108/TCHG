from __future__ import annotations

import numpy as np


USER_REGIME_DEFINITIONS = {
    "reference": "Endpoints of genuine positive training relations (split=0, sample_type=0, label>0.5).",
    "observed_user": "Both candidate endpoints belong to the training reference users.",
    "unobserved_user": "At least one candidate endpoint does not belong to the training reference users.",
    "event_missingness": "observed_mask controls event evidence availability, independently of the user regime.",
    "reference_is_fixed": "Validation replay and changes to observed_mask do not reclassify users.",
}


def validate_user_regime_masks(observed, unobserved, legacy_new=None):
    observed, unobserved = np.asarray(observed), np.asarray(unobserved)
    if observed.ndim != 1 or observed.shape != unobserved.shape:
        raise ValueError("User-regime masks must be one-dimensional and have identical shapes")
    if not np.isin(observed, [0, 1]).all() or not np.isin(unobserved, [0, 1]).all():
        raise ValueError("User-regime masks must contain only 0 or 1")
    observed, unobserved = observed.astype(bool), unobserved.astype(bool)
    if not np.all(observed ^ unobserved):
        raise ValueError("Observed-user and Unobserved-user must partition every candidate exactly once")
    if legacy_new is not None and not np.array_equal(np.asarray(legacy_new), unobserved):
        raise ValueError("Legacy is_new_node must equal unobserved_user_mask")
    return observed, unobserved


def training_reference_users(arrays, train_indices=None):
    src, dst = np.asarray(arrays['src']), np.asarray(arrays['dst'])
    if train_indices is None:
        if 'split' not in arrays:
            raise ValueError("User regimes require a training split or explicit train_indices")
        train_indices = np.flatnonzero(np.asarray(arrays['split']) == 0)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    real_positive = ((np.asarray(arrays['sample_type'])[train_indices] == 0) &
                     (np.asarray(arrays['label'])[train_indices] > .5))
    reference_indices = train_indices[real_positive]
    return np.unique(np.concatenate([src[reference_indices], dst[reference_indices]]))


def derive_user_regime_masks(arrays, train_indices=None):
    src, dst = np.asarray(arrays['src']), np.asarray(arrays['dst'])
    reference_users = training_reference_users(arrays, train_indices)
    observed = np.isin(src, reference_users) & np.isin(dst, reference_users)
    return {
        'observed_user_mask': observed.astype(np.int64),
        'unobserved_user_mask': (~observed).astype(np.int64),
        'is_new_node': (~observed).astype(np.int64),
    }


def with_user_regime_masks(arrays, train_indices=None):
    expected = derive_user_regime_masks(arrays, train_indices)
    has_observed = 'observed_user_mask' in arrays
    has_unobserved = 'unobserved_user_mask' in arrays
    if has_observed != has_unobserved:
        raise ValueError("Store both observed_user_mask and unobserved_user_mask together")
    if has_observed:
        validate_user_regime_masks(arrays['observed_user_mask'], arrays['unobserved_user_mask'], arrays.get('is_new_node'))
    for key, value in expected.items():
        if key in arrays and not np.array_equal(arrays[key], value):
            raise ValueError(f"{key} disagrees with the training-only user-regime definition")
    return {**arrays, **expected}
