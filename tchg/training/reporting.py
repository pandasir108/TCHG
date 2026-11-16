"""Explicit paper-task reports and the protocol used to produce them."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from tchg.data.user_regimes import USER_REGIME_DEFINITIONS
from tchg.training.metrics import user_regime_results
from tchg.utils.io import resolve_path, write_csv


USER_REGIME_LABELS = {
    'observed_user': 'Observed-user',
    'unobserved_user': 'Unobserved-user',
    'overall': 'Overall (auxiliary)',
}


def write_user_regime_report(metrics, output_dir, stem='test'):
    grouped = user_regime_results(metrics)
    write_csv([
        {'user_regime': regime, 'task': USER_REGIME_LABELS[regime], **values}
        for regime, values in grouped.items()
    ], Path(output_dir) / f'{stem}_user_regimes.csv')
    return grouped


def format_user_regime_table(grouped):
    lines = [f"{'Task':<23} {'Candidates':>10} {'Positive':>9} {'Negative':>9} {'MRR':>8} {'AP':>8} {'AUC':>8}"]
    for regime, label in USER_REGIME_LABELS.items():
        row = grouped[regime]
        lines.append(f"{label:<23} {row['num_events']:>10} {row.get('num_positive', '-'):>9} "
                     f"{row.get('num_negative', '-'):>9} {row['mrr']:>8.4f} {row['ap']:>8.4f} {row['auc']:>8.4f}")
    return '\n'.join(lines)


def evaluation_protocol(trainer, fit_metrics, epochs=None):
    arrays = trainer.dataset.arrays
    counts = {}
    for split, indices in trainer.indices.items():
        counts[split] = {}
        for regime, mask_key in [('overall', None), ('observed_user', 'observed_user_mask'),
                                 ('unobserved_user', 'unobserved_user_mask')]:
            mask = np.ones(len(indices), dtype=bool) if mask_key is None else arrays[mask_key][indices].astype(bool)
            selected = indices[mask]
            label = arrays['label'][selected]
            event_observed = arrays['observed_mask'][selected] > .5
            counts[split][regime] = {
                'num_events': int(len(selected)), 'num_positive': int((label > .5).sum()),
                'num_negative': int((label <= .5).sum()),
                'event_observed': int(event_observed.sum()), 'event_unobserved': int((~event_observed).sum()),
            }
    train_cfg, data_cfg = trainer.cfg['training'], trainer.cfg['dataset']
    source = resolve_path(trainer.dataset.path)
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
    return {
        'schema_version': 1,
        'dataset': data_cfg.get('name'), 'dataset_file': str(trainer.dataset.path), 'dataset_sha256': fingerprint,
        'split': data_cfg.get('cat_table_split_label'), 'seed': int(trainer.cfg.get('seed', 42)),
        'seeds_in_this_run': 1, 'aggregation': 'none; one run is not a multi-seed mean',
        'device': str(trainer.device),
        'user_regime_definitions': USER_REGIME_DEFINITIONS,
        'mask_fields': {'observed_user': 'observed_user_mask', 'unobserved_user': 'unobserved_user_mask',
                        'event_evidence_available': 'observed_mask', 'legacy_is_new_node': 'unobserved_user_mask'},
        'sample_counts': counts,
        'ranking': {'metric': 'sampled MRR', 'negative_pool': 'separate pool within each evaluated user regime',
                    'num_negatives_per_positive': 100, 'replacement': True, 'seed': 42,
                    'ties': 'average of optimistic and pessimistic ranks',
                    'probability_clip': [1e-6, 1. - 1e-6]},
        'selection': {'shared_model_for_both_regimes': True,
                      'monitor': fit_metrics.get('early_stop_monitor', train_cfg.get('early_stop_monitor', 'val_loss')),
                      'best_epoch': fit_metrics.get('best_epoch'), 'stopped_epoch': fit_metrics.get('stopped_epoch'),
                      'max_epochs': int(epochs if epochs is not None else train_cfg.get('epochs', 20)),
                      'warm_start_checkpoint': train_cfg.get('warm_start_checkpoint'),
                      'checkpoint_weights': 'ema' if train_cfg.get('ema_decay', 0.) else 'raw',
                      'ema_decay': float(train_cfg.get('ema_decay', 0.)),
                      'selection_split': 'val', 'test_used_for_selection': False},
        'calibration': {'enabled': bool(train_cfg.get('posthoc_calibration', True)),
                        'fit_subset': 'overall validation' if train_cfg.get('posthoc_calibration', True) else None,
                        'shared_across_user_regimes': True},
        'decision_threshold': {'tuned': bool(train_cfg.get('auto_tune_threshold', True)),
                               'fit_subset': 'overall validation' if train_cfg.get('auto_tune_threshold', True) else None,
                               'shared_across_user_regimes': True},
        'event_missingness': {'p_missing': data_cfg.get('p_missing', 0.),
                              'changes_user_regime': False, 'training_loss': 'all labeled training candidates'},
        'history': ('fixed training graph and memories for both validation and test; no invented trust chronology'
                    if trainer.model.operator_stack.static_mode else
                    'validation replays training; test replays training plus observed validation; evaluated queries do not write'),
        'evaluation_group': data_cfg.get('evaluation_group', 'observed_and_unobserved_users'),
        'static_training_evidence': {
            'enabled': bool(data_cfg.get('static_training_features', False)),
            'directed_structure': bool(data_cfg.get('static_structural_features', False)),
            'directed_diffusion': bool(data_cfg.get('static_diffusion_features', False)),
            'diffusion_damping': data_cfg.get('static_diffusion_damping', [.5, .85])
                                 if data_cfg.get('static_diffusion_features', False) else None,
            'pair_interactions': bool(trainer.cfg['model'].get('static_pair_interactions', False)),
            'training_history_dropout': float(train_cfg.get('static_history_dropout', 0.)),
            'training_query_edges_excluded_from_graph': bool(train_cfg.get('exclude_static_query_edges', False)),
            'bidirectional_propagation': bool(trainer.cfg['model'].get('static_bidirectional', False)),
            'feature_history': 'training trust only; current training relation removed from its trust features',
            'graph_history': 'observed genuine training relations only',
            'auxiliary_context': 'existing static rating summaries',
        },
        'negative_exclusion_scope': data_cfg.get('negative_exclusion_scope'),
        'causal_rating_evidence': getattr(trainer.dataset, '_causal_rating_metadata', None),
        'user_identity': {
            'enabled': bool(trainer.cfg['model'].get('use_user_identity', False)),
            'reference_users': 'endpoints of genuine positive training relations only',
            'unknown_identity': 'zero; held-out users have no learned lookup contribution',
            'training_dropout': float(trainer.cfg['model'].get('user_identity_dropout', .3)),
            'changes_user_regime': False,
        },
        'stationary_time_features': bool(trainer.cfg['model'].get('stationary_time_features', False)),
        'memory_time': {
            'input_unit': 'seconds',
            'seconds_per_memory_unit': trainer.model.memory.time_scale_seconds,
            'rate_unit': 'per memory time unit',
            'applies_to': ['decay', 'uncertainty discount', 'update gate', 'reconnect gate'],
        },
        'validation_temporal_stability': {
            'enabled': train_cfg.get('observed_user_mrr_target') is not None,
            'observed_user_mrr_target': train_cfg.get('observed_user_mrr_target'),
            'unobserved_user_mrr_target': train_cfg.get('unobserved_user_mrr_target'),
            'later_fraction': float(train_cfg.get('validation_confirmation_fraction', .5)),
            'timestamp_ties': 'kept together',
            'score': 'minimum MRR/target across both tasks in whole, early, and late validation',
            'negative_pools': 'separate within each task and validation window',
            'interpretation': 'stability check within validation; does not create an independent holdout',
        },
        'class_balance': 'overall candidates are sampled first; user subsets retain their actual class counts',
        'legacy_metric_aliases': {'old_*': 'observed_user_*', 'new_*': 'unobserved_user_*',
                                  'unprefixed metrics': 'overall_*'},
    }
