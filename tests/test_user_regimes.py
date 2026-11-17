import hashlib
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from tchg.data.event_dataset import EventGraphDataset
from tchg.data.user_regimes import derive_user_regime_masks, with_user_regime_masks
from tchg.data.preprocessors import preprocess_dataset
from tchg.runner import run_training
from tchg.training.metrics import compute_metrics, user_regime_results, validation_target_metrics
from tchg.training.trainer import Trainer


def example_arrays():
    rng = np.random.default_rng(23)
    return {
        'src': np.array([0, 0, 2, 1, 0, 2, 1, 1]),
        'dst': np.array([1, 2, 3, 0, 1, 0, 3, 0]),
        'split': np.array([0, 0, 1, 1, 2, 2, 2, 2]),
        'label': np.array([1., 0., 1., 0., 1., 1., 0., 0.], dtype=np.float32),
        'sample_type': np.array([0, 1, 0, 1, 0, 0, 1, 1]),
        'timestamp': np.arange(8, dtype=np.float32),
        'observed_mask': np.array([0., 1., 1., 0., 1., 1., 0., 0.], dtype=np.float32),
        'delta_t': np.ones(8, dtype=np.float32), 'edge_type': np.zeros(8, dtype=np.int64),
        'entity_features': rng.random((8, 8), dtype=np.float32),
        'behavior_features': rng.random((8, 8), dtype=np.float32),
        'context_features': rng.random((8, 8), dtype=np.float32),
        'edge_features': rng.random((8, 4), dtype=np.float32),
        'node_features': rng.random((4, 16), dtype=np.float32),
        'num_nodes': np.array([4]), 'num_edge_types': np.array([1]),
    }


def test_training_negatives_and_validation_relations_do_not_make_users_observed():
    masks = derive_user_regime_masks(example_arrays())
    np.testing.assert_array_equal(masks['observed_user_mask'], [1, 0, 0, 1, 1, 0, 0, 1])
    np.testing.assert_array_equal(masks['observed_user_mask'] + masks['unobserved_user_mask'], 1)
    np.testing.assert_array_equal(masks['is_new_node'], masks['unobserved_user_mask'])


def test_event_missingness_and_held_out_labels_do_not_reclassify_users():
    arrays = example_arrays()
    before = derive_user_regime_masks(arrays)
    arrays['observed_mask'] = 1. - arrays['observed_mask']
    held_out = arrays['split'] > 0
    arrays['label'][held_out] = 1. - arrays['label'][held_out]
    arrays['sample_type'][held_out] = 1 - arrays['sample_type'][held_out]
    after = derive_user_regime_masks(arrays)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_legacy_dataset_gains_explicit_masks_without_rewriting_the_file(tmp_path):
    arrays = example_arrays()
    arrays['is_new_node'] = derive_user_regime_masks(arrays)['is_new_node']
    path = tmp_path / 'legacy.npz'
    np.savez_compressed(path, **arrays)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset = EventGraphDataset(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with np.load(path) as stored:
        assert 'observed_user_mask' not in stored.files
    subset = dataset.to_torch(dataset.subset(np.array([7, 5, 4, 6])), torch.device('cpu'))
    assert subset['timestamp'].tolist() == [4., 5., 6., 7.]
    assert subset['observed_user_mask'].tolist() == [1, 0, 0, 1]
    assert subset['unobserved_user_mask'].tolist() == [0, 1, 1, 0]


def test_loader_rejects_inconsistent_legacy_and_explicit_group_metadata():
    arrays = with_user_regime_masks(example_arrays())
    bad = deepcopy(arrays)
    bad['is_new_node'][4] = 1
    with pytest.raises(ValueError, match='is_new_node'):
        EventGraphDataset.from_arrays(bad)
    bad = deepcopy(arrays)
    for key in ('observed_user_mask', 'unobserved_user_mask', 'is_new_node'):
        bad[key][4] = 1 - bad[key][4]
    with pytest.raises(ValueError, match='training-only'):
        EventGraphDataset.from_arrays(bad)
    del arrays['unobserved_user_mask']
    with pytest.raises(ValueError, match='both'):
        EventGraphDataset.from_arrays(arrays)


def metric_example():
    scores = torch.tensor([.9, .8, .7, .6, .4, .3, .2, .1])
    batch = {
        'label': torch.tensor([1., 0., 1., 0., 1., 0., 1., 0.]),
        'delta_t': torch.ones(8), 'observed_mask': torch.tensor([0., 1.] * 4),
        'observed_user_mask': torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]),
        'unobserved_user_mask': torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
    }
    return {'trust_pred': scores}, batch


def test_each_user_regime_has_its_own_ranking_pool_and_event_masks_are_independent():
    pred, batch = metric_example()
    before = compute_metrics(pred, batch)
    pred['trust_pred'][[1, 3]] = torch.tensor([.01, .02])
    after = compute_metrics(pred, batch)
    for key in ('mrr', 'ap', 'auc'):
        assert before['unobserved_user_' + key] == after['unobserved_user_' + key]
    assert before['observed_user_mrr'] != after['observed_user_mrr']
    assert before['overall_mrr'] != after['overall_mrr']
    batch['observed_mask'] = 1. - batch['observed_mask']
    flipped = compute_metrics(pred, batch)
    assert user_regime_results(after) == user_regime_results(flipped)


def test_metric_aliases_preserve_legacy_scores_and_partition_counts():
    pred, batch = metric_example()
    explicit = compute_metrics(pred, batch)
    legacy_batch = {k: v for k, v in batch.items() if k not in {'observed_user_mask', 'unobserved_user_mask'}}
    legacy_batch['is_new_node'] = batch['unobserved_user_mask']
    assert explicit == compute_metrics(pred, legacy_batch)
    grouped = user_regime_results(explicit)
    assert grouped['observed_user']['num_events'] + grouped['unobserved_user']['num_events'] == grouped['overall']['num_events']
    assert grouped['observed_user']['num_positive'] == grouped['unobserved_user']['num_positive'] == 2
    for task, alias in [('observed_user', 'old'), ('unobserved_user', 'new')]:
        for key, value in grouped[task].items():
            assert explicit[f'{task}_{key}'] == explicit[f'{alias}_{key}'] == value
    assert explicit['overall_mrr'] == explicit['mrr']


def test_empty_user_regime_has_consistent_zero_rank_and_count_fields():
    pred, batch = metric_example()
    batch['observed_user_mask'].fill_(1)
    batch['unobserved_user_mask'].zero_()
    empty = user_regime_results(compute_metrics(pred, batch))['unobserved_user']
    assert empty['num_events'] == empty['num_positive'] == empty['num_negative'] == empty['mrr'] == 0


def test_validation_confirmation_is_temporal_and_does_not_change_tuning_scores():
    pred,batch=metric_example();batch['timestamp']=torch.arange(8,dtype=torch.float32)
    batch['observed_user_mask']=torch.tensor([1,1,0,0,1,1,0,0])
    batch['unobserved_user_mask']=1-batch['observed_user_mask']
    before=validation_target_metrics(pred,batch,.7,.45)
    assert before['tuning_num_events']==before['confirmation_num_events']==4
    changed={k:v.clone() for k,v in pred.items()};changed['trust_pred'][4:]=.5
    after=validation_target_metrics(changed,batch,.7,.45)
    for key,value in before.items():
        if key.startswith('tuning_'):assert after[key]==value
    assert after['robust_target_ratio']<=before['robust_target_ratio']
    batch['timestamp']=torch.tensor([0.,0.,0.,1.,1.,1.,2.,2.])
    tied=validation_target_metrics(pred,batch,.7,.45)
    assert tied['tuning_num_events']==6 and tied['confirmation_num_events']==2


def test_misspelled_user_regime_monitor_fails_instead_of_selecting_on_another_metric():
    pred, batch = metric_example()
    metrics = compute_metrics(pred, batch)
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = {'model': {'name': 'tchgt'}, 'dataset': {},
                   'training': {'epochs': 1, 'early_stop_monitor': 'val_obseved_user_mrr'}}
    trainer.device = torch.device('cpu')
    trainer._step = lambda split, train: (0., {}, metrics)
    with pytest.raises(ValueError, match='Unknown early_stop_monitor'):
        trainer.fit()


def test_main_pipeline_exports_named_tasks_and_protocol(tmp_path, monkeypatch):
    arrays = example_arrays()
    arrays['observed_mask'][0] = 1.
    path = tmp_path / 'events.npz'
    np.savez_compressed(path, **arrays)
    cfg = {
        'seed': 42, 'device': 'cpu',
        'dataset': {'name': 'synthetic', 'processed_file': str(path), 'p_missing': .2},
        'model': {'name': 'tchgt', 'hidden_dim': 16, 'trust_dim': 8, 'memory_dim': 8,
                  'num_layers': 1, 'num_operator_candidates': 2, 'dropout': 0., 'memory_key': 'src_node'},
        'training': {'epochs': 2, 'batch_mode': 'event_stream', 'stream_batch_size': 2,
                     'early_stop_monitor': 'val_overall_mrr', 'early_stop_mode': 'max',
                     'posthoc_calibration': False, 'auto_tune_threshold': False},
    }
    trainer, result = run_training(cfg, tmp_path / 'run', device='cpu')
    output = tmp_path / 'run'
    frame = pd.read_csv(output / 'test_user_regimes.csv')
    assert frame['task'].tolist() == ['Observed-user', 'Unobserved-user', 'Overall (auxiliary)']
    assert frame['num_events'].tolist() == [2, 2, 4]
    assert frame['num_positive'].tolist() == [1, 1, 2]
    protocol = json.loads((output / 'evaluation_protocol.json').read_text())
    assert protocol['selection']['shared_model_for_both_regimes']
    assert protocol['selection']['monitor'] == 'val_overall_mrr'
    assert protocol['ranking']['negative_pool'] == 'separate pool within each evaluated user regime'
    assert protocol['dataset_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert protocol['calibration']['fit_subset'] is None
    assert protocol['sample_counts']['test']['observed_user']['num_events'] == 2
    assert protocol['sample_counts']['test']['unobserved_user']['event_observed'] == 1
    assert result['test_by_user_regime']['observed_user']['mrr'] == result['test']['old_mrr']
    assert result['test_by_user_regime']['unobserved_user']['mrr'] == result['test']['new_mrr']
    assert trainer.history[-1]['val_overall_mrr'] == trainer.history[-1]['val_mrr']
    assert 'val_observed_user_mrr' in pd.read_csv(output / 'history.csv')
    expected={k:v.clone() for k,v in trainer.model.state_dict().items()}
    continuation=deepcopy(cfg);continuation['seed']=43
    continuation['training'].update({'epochs':1,'lr':0.,'ema_decay':.99,'warm_start_checkpoint':str(output/'best.pt')})
    resumed,resumed_result=run_training(continuation,tmp_path/'continued',device='cpu')
    for key,value in expected.items():
        torch.testing.assert_close(resumed.model.state_dict()[key],value)
    assert resumed_result['evaluation_protocol']['selection']['warm_start_checkpoint']==str(output/'best.pt')
    original_evaluate = Trainer.evaluate
    evaluated_splits = []
    def guarded_evaluate(self, split, **kwargs):
        evaluated_splits.append(split)
        assert split != 'test'
        return original_evaluate(self, split, **kwargs)
    monkeypatch.setattr(Trainer, 'evaluate', guarded_evaluate)
    _, validation_result = run_training(cfg, tmp_path/'validation_only', device='cpu',
                                        epochs=1, evaluate_test=False)
    assert evaluated_splits == ['val']
    assert not validation_result['test_evaluated'] and 'test' not in validation_result
    assert validation_result['evaluation_protocol']['evaluation']['reported_split'] == 'val'
    assert (tmp_path/'validation_only'/'validation_user_regimes.csv').exists()
    assert not (tmp_path/'validation_only'/'test_user_regimes.csv').exists()


def test_raw_preparation_persists_both_user_masks_and_matches_legacy_grouping(tmp_path):
    raw = tmp_path / 'raw' / 'epinions'
    raw.mkdir(parents=True)
    (raw / 'user_user_time.txt').write_text('0 1 1\n1 0 2\n2 0 3\n0 2 4\n3 0 5\n4 0 6\n')
    cfg = {'dataset': {'raw_dir': str(tmp_path / 'raw'), 'processed_dir': str(tmp_path / 'processed'),
                       'val_time_ratio': .5, 'test_time_ratio': .75, 'negative_exclusion_scope': 'full',
                       'negative_sampling_ratio': 1., 'p_missing': .2, 'respect_observed_history': True}}
    path = preprocess_dataset('epinions_full_cat_rolling', cfg, seed=42)
    with np.load(path) as stored:
        assert 'observed_user_mask' in stored.files and 'unobserved_user_mask' in stored.files
        np.testing.assert_array_equal(stored['observed_user_mask'] + stored['unobserved_user_mask'], 1)
        np.testing.assert_array_equal(stored['is_new_node'], stored['unobserved_user_mask'])
    dataset = EventGraphDataset(path)
    assert set(dataset.arrays['split']) == {0, 1, 2}
    metadata = json.loads((path.parent / 'metadata.json').read_text())
    assert metadata['user_regime_definitions']['reference_is_fixed']
