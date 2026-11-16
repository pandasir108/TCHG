from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score

from tchg.training.calibration import confidence_error_correlation, expected_calibration_error, overconfidence_rate
from tchg.data.user_regimes import validate_user_regime_masks


USER_REGIME_REPORT_FIELDS = ('num_events', 'num_positive', 'num_negative', 'mrr', 'ap', 'auc', 'f1', 'accuracy', 'global_mrr')


def _np(x: torch.Tensor) -> np.ndarray:

    return x.detach().float().cpu().numpy()


def _safe(fn, *args, default: float = 0.0, **kwargs) -> float:

    try:
        if fn in {roc_auc_score, average_precision_score} and len(np.unique(args[0])) < 2:
            return default
        return float(fn(*args, **kwargs))
    except Exception:
        return default


def parameter_count(model: torch.nn.Module) -> int:

    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def mean_reciprocal_rank(labels: np.ndarray, scores: np.ndarray) -> float:

    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels >= 0.5
    if not np.any(pos):
        return 0.0
    return float(np.mean(1.0 / ranks[pos]))


def sampled_mean_reciprocal_rank(labels: np.ndarray, scores: np.ndarray, num_negatives: int = 100, seed: int = 42) -> float:

    pos_scores = scores[labels >= 0.5]
    neg_scores = scores[labels < 0.5]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    sampled = neg_scores[rng.integers(0, len(neg_scores), size=(len(pos_scores), num_negatives))]
    pos_scores = pos_scores.reshape(-1, 1)
    optimistic_rank = (sampled > pos_scores).sum(axis=1)
    pessimistic_rank = (sampled >= pos_scores).sum(axis=1)
    rank = 0.5 * (optimistic_rank + pessimistic_rank) + 1.0
    return float(np.mean(1.0 / rank))


def _group_auc(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray, default: float = 0.5) -> float:

    if not np.any(mask):
        return default
    return _safe(roc_auc_score, labels[mask], scores[mask], default=default)


def _binary_group_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> Dict[str, float]:

    if not np.any(mask):
        empty = {f'{prefix}_{key}': 0.0 for key in USER_REGIME_REPORT_FIELDS}
        empty[f'{prefix}_auc'] = .5
        return empty
    y = labels[mask]
    prob = scores[mask]
    yhat = predictions[mask]
    return {
        f"{prefix}_num_events": float(mask.sum()),
        f"{prefix}_num_positive": float((y >= .5).sum()),
        f"{prefix}_num_negative": float((y < .5).sum()),
        f"{prefix}_auc": _safe(roc_auc_score, y, prob, default=0.5),
        f"{prefix}_ap": _safe(average_precision_score, y, prob, default=0.0),
        f"{prefix}_f1": _safe(f1_score, y, yhat, default=0.0, zero_division=0),
        f"{prefix}_accuracy": _safe(accuracy_score, y, yhat, default=0.0),
        # Sample negatives from THIS user regime, never the combined pool.
        f"{prefix}_mrr": sampled_mean_reciprocal_rank(y, prob),
        f"{prefix}_global_mrr": mean_reciprocal_rank(y, prob),
    }


def compute_metrics(pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], threshold: float = 0.5) -> Dict[str, float]:

    y = _np(batch["label"]).astype(np.float32)
    prob = _np(pred["trust_pred"]).clip(1e-6, 1.0 - 1e-6)
    yhat = (prob >= threshold).astype(np.float32)
    pos_mask = y >= 0.5
    neg_mask = ~pos_mask
    negative_recall = float(np.mean(yhat[neg_mask] < 0.5)) if np.any(neg_mask) else 0.0
    positive_rate = float(np.mean(yhat >= 0.5)) if len(yhat) else 0.0
    delta_t = _np(batch["delta_t"])
    event_observed = _np(batch["observed_mask"]) > 0.5
    short_mask = delta_t <= np.quantile(delta_t, 0.25) if len(delta_t) else np.zeros_like(delta_t, dtype=bool)
    long_mask = delta_t >= np.quantile(delta_t, 0.75) if len(delta_t) else np.zeros_like(delta_t, dtype=bool)
    reappearing = event_observed & long_mask
    metrics = {
        "num_events": float(len(y)),
        "num_positive": float(pos_mask.sum()),
        "num_negative": float(neg_mask.sum()),
        "auc": _safe(roc_auc_score, y, prob, default=0.5),
        "ap": _safe(average_precision_score, y, prob, default=0.0),
        "f1": _safe(f1_score, y, yhat, default=0.0, zero_division=0),
        "accuracy": _safe(accuracy_score, y, yhat, default=0.0),
        "balanced_accuracy": _safe(balanced_accuracy_score, y, yhat, default=0.0),
        "precision": _safe(precision_score, y, yhat, default=0.0, zero_division=0),
        "recall": _safe(recall_score, y, yhat, default=0.0, zero_division=0),
        "negative_recall": negative_recall,
        "positive_prediction_rate": positive_rate,
        "mean_trust_prob": float(np.mean(prob)) if len(prob) else 0.0,
        "mrr": sampled_mean_reciprocal_rank(y, prob),
        "global_mrr": mean_reciprocal_rank(y, prob),
        "brier_score": float(np.mean((prob - y) ** 2)),
        "ece": expected_calibration_error(y, prob),
        "nll": _safe(log_loss, y, prob, default=0.0, labels=[0, 1]),
        "overconfidence_rate": overconfidence_rate(y, prob),
        "confidence_error_corr": confidence_error_correlation(y, prob),
        "long_gap_auc": _group_auc(y, prob, long_mask),
        "short_gap_auc": _group_auc(y, prob, short_mask),
        "reappearing_edge_auc": _group_auc(y, prob, reappearing),
    }
    # Explicit overall names keep combined results distinct from paper rows.
    metrics.update({f'overall_{key}': metrics[key] for key in USER_REGIME_REPORT_FIELDS})
    has_observed = 'observed_user_mask' in batch
    has_unobserved = 'unobserved_user_mask' in batch
    if has_observed != has_unobserved:
        raise ValueError('Both user-regime masks are required; observed_mask is an event mask')
    legacy = _np(batch['is_new_node']) if 'is_new_node' in batch else None
    if has_observed:
        observed_user, unobserved_user = validate_user_regime_masks(
            _np(batch['observed_user_mask']), _np(batch['unobserved_user_mask']), legacy)
    elif legacy is not None:
        observed_user, unobserved_user = validate_user_regime_masks(1. - legacy, legacy, legacy)
    else:
        return metrics
    if observed_user.shape != y.shape:
        raise ValueError('User-regime masks must have one entry per candidate')
    for regime, mask, alias in [('observed_user', observed_user, 'old'), ('unobserved_user', unobserved_user, 'new')]:
        group = _binary_group_metrics(y, prob, yhat, mask, regime)
        metrics.update(group)
        # Preserve existing consumers without using ambiguous names in reports.
        metrics.update({f'{alias}_{key}': group[f'{regime}_{key}'] for key in USER_REGIME_REPORT_FIELDS})
    # Auxiliary validation-selection score; paper tasks retain separate pools.
    metrics['macro_user_mrr'] = .5 * (metrics['observed_user_mrr'] + metrics['unobserved_user_mrr'])
    return metrics


def user_regime_results(metrics):
    """Extract named report rows, including from legacy saved metric files.

    Missing class counts in older files are omitted, never inferred. Scores
    are copied verbatim; this function does not re-evaluate any predictions.
    """
    grouped = {}
    for regime, legacy_prefix in [('observed_user', 'old_'), ('unobserved_user', 'new_'), ('overall', '')]:
        row = {}
        for key in USER_REGIME_REPORT_FIELDS:
            canonical, legacy_key = f'{regime}_{key}', legacy_prefix + key
            if canonical in metrics:
                value = metrics[canonical]
            elif legacy_key in metrics:
                value = metrics[legacy_key]
            else:
                if key in {'num_positive', 'num_negative', 'global_mrr'}:
                    continue
                raise ValueError(f'Missing {canonical}; user regimes must be evaluated explicitly')
            row[key] = int(value) if key.startswith('num_') else float(value)
        grouped[regime] = row
    return grouped


def validation_target_metrics(pred, batch, observed_target, unobserved_target, confirmation_fraction=.5):
    """Additional validation diagnostics; paper/test metrics are unchanged.

    Both periods have previously been used for model selection in this study.
    Splitting them provides a temporal stability check, not a fresh holdout.
    """
    if not (0. < observed_target <= 1. and 0. < unobserved_target <= 1.):
        raise ValueError('MRR targets must be in (0, 1]')
    if not 0. < confirmation_fraction < 1.:
        raise ValueError('confirmation_fraction must be in (0, 1)')
    times = _np(batch['timestamp'])
    if len(times) < 2 or np.any(times[1:] < times[:-1]):
        raise ValueError('Validation target checks require chronological predictions')
    index = min(len(times)-2, max(0, int(len(times) * (1. - confirmation_fraction))-1))
    boundary = int(np.searchsorted(times, times[index], side='right'))
    if boundary == len(times):
        boundary = int(np.searchsorted(times, times[index], side='left'))
    if boundary == 0 or boundary == len(times):
        raise ValueError('Validation needs at least two distinct timestamp groups')
    targets = {'observed_user': observed_target, 'unobserved_user': unobserved_target}
    result = {'tuning_num_events': float(boundary), 'confirmation_num_events': float(len(times)-boundary),
              'confirmation_start_time': float(times[boundary])}
    all_metrics = compute_metrics(pred, batch)
    ratios = [all_metrics[name + '_mrr'] / target for name, target in targets.items()]
    result['whole_target_ratio'] = min(ratios)
    for name, selection in [('tuning', slice(0, boundary)), ('confirmation', slice(boundary, len(times)))]:
        p = {k: v[selection] for k, v in pred.items() if v.ndim > 0 and v.shape[0] == len(times)}
        b = {k: v[selection] for k, v in batch.items() if v.ndim > 0 and v.shape[0] == len(times)}
        values = compute_metrics(p, b)
        for regime, target in targets.items():
            for metric in ['mrr', 'ap', 'auc', 'num_events', 'num_positive', 'num_negative']:
                result[f'{name}_{regime}_{metric}'] = values[f'{regime}_{metric}']
            ratios.append(values[regime + '_mrr'] / target)
    result['robust_target_ratio'] = min(ratios)
    return result
