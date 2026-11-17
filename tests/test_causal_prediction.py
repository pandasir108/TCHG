from copy import deepcopy

import pytest
import torch
import numpy as np

from test_model import make_batch, make_tiny_model
from tchg.models.conditioned_operator import ConditionedHeterogeneousOperatorStack
from tchg.models.masked_memory import ComponentDecoupledMaskedTrustMemory
from tchg.data.event_dataset import EventGraphDataset
from tchg.training.trainer import Trainer


def slice_events(batch, start, end):
    globals_ = {'node_features', 'metapath_node_features', 'num_nodes', 'num_edge_types'}
    return {k: v if k in globals_ else v[start:end] for k, v in batch.items()}


@pytest.mark.parametrize('same_time', [False, True])
def test_future_or_simultaneous_queries_cannot_change_first_prediction(same_time):
    torch.manual_seed(23)
    model = make_tiny_model().eval()
    model.use_masked_memory = False
    batch = make_batch()
    if same_time:
        batch['timestamp'].zero_()
    model.reset_memory()
    prefix = model(slice_events(batch, 0, 1))['trust_pred']
    model.reset_memory()
    full = model(batch)['trust_pred'][:1]
    torch.testing.assert_close(prefix, full, rtol=1e-6, atol=1e-7)


def test_future_evidence_cannot_change_past_prediction():
    torch.manual_seed(23)
    model = make_tiny_model().eval()
    batch = make_batch()
    model.reset_memory()
    before = model(batch)['trust_pred']
    other = deepcopy(batch)
    other['entity_features'][2:] += 10.
    other['behavior_features'][2:] += 10.
    other['context_features'][2:] += 10.
    model.reset_memory()
    after = model(other)['trust_pred']
    torch.testing.assert_close(before[:2], after[:2], rtol=1e-6, atol=1e-7)


def test_memory_writers_receive_gradients_from_later_prediction():
    torch.manual_seed(23)
    model = make_tiny_model()
    batch = make_batch()
    batch['src'].zero_()
    model.reset_memory()
    prediction = model(batch)
    prediction['trust_logit'][1:].sum().backward()
    for component in ('E', 'B', 'C'):
        cell = model.memory.cells[component]
        for module in (cell.project, cell.gru, cell.gamma):
            grads = [p.grad for p in module.parameters()]
            assert any(g is not None and g.abs().sum() > 0 for g in grads), (component, module)


def test_reconnect_writer_receives_gradients():
    torch.manual_seed(23)
    model = make_tiny_model()
    batch = make_batch()
    batch['src'].zero_()
    batch['sample_type'].zero_()
    batch['observed_mask'] = torch.tensor([0., 1., 1., 1.])
    model.reset_memory()
    model(batch)['trust_logit'][2:].sum().backward()
    for cell in model.memory.cells.values():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in cell.kappa.parameters())


def test_one_layer_matches_slow_history_only_graph_reference():
    torch.manual_seed(23)
    batch = make_batch()
    batch['history_write_mask'] = (batch['sample_type'] == 0).float()
    stack = ConditionedHeterogeneousOperatorStack(num_layers=1, hidden_dim=4, trust_dim=3,
        edge_dim=4, num_edge_types=1, num_operator_candidates=2, dropout=0.).eval()
    h = torch.randn(3, 4)
    enc = {**{f'z_{q}': torch.randn(4, 3) for q in ('B', 'C')},
           **{f'tau_{q}': torch.rand(4) for q in ('E', 'B', 'C')}}
    actual_src, actual_dst = stack(h, batch, enc)
    expected_src, expected_dst = [], []
    layer = stack.layers[0]
    for i in range(4):
        history = ((batch['timestamp'] < batch['timestamp'][i]) &
                   (batch['history_write_mask'] > .5) & (batch['observed_mask'] > .5))
        node_h = layer(h, batch['src'][history], batch['dst'][history], batch['edge_type'][history],
            batch['edge_features'][history], enc['z_B'][history], enc['z_C'][history],
            enc['tau_E'][history], enc['tau_B'][history], enc['tau_C'][history])
        expected_src.append(node_h[batch['src'][i]])
        expected_dst.append(node_h[batch['dst'][i]])
    torch.testing.assert_close(actual_src, torch.stack(expected_src), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_dst, torch.stack(expected_dst), rtol=1e-5, atol=1e-6)


def test_predictions_match_across_batch_boundaries():
    torch.manual_seed(23)
    model = make_tiny_model().eval()
    batch = make_batch()
    batch['src'].zero_()
    model.reset_memory()
    whole = model(batch)['trust_pred']
    model.reset_memory()
    sequential = [model(slice_events(batch, i, i+1))['trust_pred'] for i in range(4)]
    torch.testing.assert_close(whole, torch.cat(sequential), rtol=1e-5, atol=1e-6)


def test_simultaneous_memory_reads_and_writes_are_order_independent():
    torch.manual_seed(23)
    model = make_tiny_model().eval()
    batch = make_batch()
    batch['timestamp'].zero_()
    batch['src'].zero_()
    model.reset_memory()
    before = model(batch)['trust_pred']
    bank = {k: v.clone() for k, v in model.memory._tensor_state.items()}
    model.reset_memory()
    other = {k: v if k in {'node_features', 'num_nodes', 'num_edge_types'} else v.flip(0) for k, v in batch.items()}
    after = model(other)['trust_pred'].flip(0)
    torch.testing.assert_close(before, after)
    for key, value in bank.items():
        torch.testing.assert_close(value, model.memory._tensor_state[key])


def test_detach_boundary_allows_multiple_optimizer_steps():
    torch.manual_seed(23)
    model = make_tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    batch = make_batch()
    batch['src'].zero_()
    model.reset_memory()
    initial = model.memory.cells['E'].gru.weight_ih.detach().clone()
    for step in range(2):
        chunk = deepcopy(batch)
        chunk['timestamp'] += step * 4
        optimizer.zero_grad()
        model(chunk)['trust_logit'].sum().backward()
        optimizer.step()
        model.detach_memory()
    assert not torch.equal(initial, model.memory.cells['E'].gru.weight_ih)


def test_evaluation_replays_training_and_validation_but_freezes_test_queries():
    torch.manual_seed(23)
    seed = make_batch()
    arrays = {k: v.numpy() for k, v in seed.items()}
    globals_ = {'node_features', 'num_nodes', 'num_edge_types'}
    for key in list(arrays):
        if key not in globals_:
            arrays[key] = np.concatenate([arrays[key]] * 3)
    arrays['timestamp'] = np.arange(12, dtype=np.float32)
    arrays['split'] = np.repeat([0, 1, 2], 4)
    dataset = EventGraphDataset.from_arrays(arrays)
    cfg = {'device': 'cpu', 'dataset': {},
        'model': {'name':'tchgt', 'hidden_dim':16, 'trust_dim':8, 'memory_dim':8,
            'num_layers':1, 'num_operator_candidates':2, 'dropout':0.,
            'memory_key':'src_node', 'fast_batch_memory':True},
        'training': {'batch_mode':'event_stream', 'stream_batch_size':2}}
    trainer = Trainer(cfg, dataset, device='cpu')
    prediction, _ = trainer.collect_predictions('test', apply_calibration=False)
    count = trainer.model.operator_stack._history[0][1].sum()
    assert count == 4
    assert trainer._history_batch(trainer.batches['test'], False)['history_write_mask'].sum() == 0
    trainer.batches['val']['label'] = 1. - trainer.batches['val']['label']
    other, _ = trainer.collect_predictions('test', apply_calibration=False)
    torch.testing.assert_close(prediction['trust_pred'], other['trust_pred'])
    for size in (1, 20):
        trainer.cfg['training']['stream_batch_size'] = size
        other, _ = trainer.collect_predictions('test', apply_calibration=False)
        torch.testing.assert_close(prediction['trust_pred'], other['trust_pred'], rtol=1e-5, atol=1e-6)


def test_trainer_keeps_equal_timestamp_groups_together():
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = {'training': {'stream_batch_size': 2}}
    batch = make_batch()
    batch['timestamp'] = torch.tensor([0., 0., 0., 1.])
    assert [len(x['src']) for x in trainer._chunks(batch)] == [3, 1]


@pytest.mark.parametrize('key', ['src_node', 'dst_node', 'edge'])
@pytest.mark.parametrize('component_wise', [False, True])
def test_memory_key_and_shared_state_variants_remain_differentiable(key, component_wise):
    torch.manual_seed(23)
    memory = ComponentDecoupledMaskedTrustMemory(4, 4, memory_key=key,
        component_wise_memory=component_wise, fast_batch_memory=True)
    z = torch.randn(3, 4)
    out = memory(3, torch.zeros(3, dtype=torch.long), torch.ones(3, dtype=torch.long),
        torch.ones(3), torch.ones(3), z, z, z, timestamp=torch.arange(3.))
    out['memory'][1:].sum().backward()
    assert memory.cells['E'].gru.weight_ih.grad.abs().sum() > 0
    memory.detach_state()
    assert all(v.grad_fn is None for v in memory._tensor_state.values())


def test_stable_evidence_options_preserve_causality_and_memory_learning():
    torch.manual_seed(23)
    model = make_tiny_model(use_absolute_time_features=False, entity_log_odds=True, memory_strength=.1).eval()
    batch = make_batch()
    batch['src'].zero_()
    model.reset_memory()
    before = model(batch)['trust_pred']
    changed = deepcopy(batch)
    changed['entity_features'][2:] = .99
    model.reset_memory()
    after = model(changed)['trust_pred']
    torch.testing.assert_close(before[:2], after[:2], rtol=1e-6, atol=1e-7)
    model.reset_memory()
    model(batch)['trust_logit'][1:].sum().backward()
    assert model.memory.cells['E'].gru.weight_ih.grad.abs().sum() > 0


def test_absolute_clock_can_be_removed_without_removing_temporal_memory():
    torch.manual_seed(23)
    model = make_tiny_model(use_absolute_time_features=False).eval()
    batch = make_batch()
    model.reset_memory()
    before = model(batch)
    changed = deepcopy(batch)
    changed['context_features'][:, [1, 2, 3, 7]] += 100.
    changed['edge_features'][:, 2] += 100.
    model.reset_memory()
    after = model(changed)
    torch.testing.assert_close(before['trust_pred'], after['trust_pred'])
    changed['delta_t'] += 100.
    model.reset_memory()
    later = model(changed)
    assert not torch.allclose(before['memory_uncertainty'], later['memory_uncertainty'])


def test_evidence_statistics_are_fixed_before_prediction():
    torch.manual_seed(23)
    model = make_tiny_model(standardize_evidence=True, entity_log_odds=True,
        use_absolute_time_features=False, memory_strength=.1).eval()
    training = make_batch()
    model.fit_input_statistics(training)
    stats = {k: v.clone() for k, v in model.named_buffers() if k.endswith(('_mean','_std'))}
    changed = deepcopy(training)
    changed['entity_features'] += 100.
    changed['behavior_features'] += 100.
    model.reset_memory()
    model(changed)
    for key, value in model.named_buffers():
        if key in stats:
            torch.testing.assert_close(value, stats[key])
    model.reset_memory()
    before = model(training)['trust_pred']
    changed = deepcopy(training)
    changed['context_features'][2:] += 100.
    model.reset_memory()
    after = model(changed)['trust_pred']
    torch.testing.assert_close(before[:2], after[:2], rtol=1e-6, atol=1e-7)


def test_normalizer_fits_training_examples_only():
    seed = make_batch()
    globals_ = {'node_features', 'num_nodes', 'num_edge_types'}
    arrays = {k: v.numpy() if k in globals_ else np.concatenate([v.numpy()] * 3)
              for k, v in seed.items()}
    arrays['timestamp'] = np.arange(12, dtype=np.float32)
    arrays['split'] = np.repeat([0, 1, 2], 4)
    first = EventGraphDataset.from_arrays(arrays)
    other = EventGraphDataset.from_arrays(arrays)
    for key in ('entity_features', 'behavior_features', 'context_features', 'edge_features'):
        other.arrays[key][4:] += 100.
    cfg = {'device': 'cpu', 'dataset': {},
           'model': {'name': 'tchgt', 'standardize_evidence': True}, 'training': {}}
    a, b = Trainer(cfg, first), Trainer(cfg, other)
    for key in a.model.evidence_keys:
        torch.testing.assert_close(getattr(a.model, key + '_mean'), getattr(b.model, key + '_mean'))
        torch.testing.assert_close(getattr(a.model, key + '_std'), getattr(b.model, key + '_std'))
        np.testing.assert_array_equal(first.arrays[key], arrays[key])


def test_zero_initialized_evidence_residual_preserves_logits_and_remains_causal():
    torch.manual_seed(23)
    base=make_tiny_model().eval()
    extra=make_tiny_model(evidence_residual=True).eval()
    missing=extra.load_state_dict(base.state_dict(),strict=False)
    assert all(k.startswith('evidence_residual.') for k in missing.missing_keys)
    assert not missing.unexpected_keys
    batch=make_batch();batch['src'].zero_()
    base.reset_memory();extra.reset_memory()
    torch.testing.assert_close(base(batch)['trust_pred'],extra(batch)['trust_pred'])
    with torch.no_grad():extra.evidence_residual[-1].weight.fill_(.1)
    extra.reset_memory();before=extra(batch)['trust_pred']
    changed=deepcopy(batch);changed['entity_features'][2:]+=100.
    extra.reset_memory();after=extra(changed)['trust_pred']
    torch.testing.assert_close(before[:2],after[:2],rtol=1e-6,atol=1e-7)
    extra.reset_memory();extra(batch)['trust_logit'][1:].sum().backward()
    for cell in extra.memory.cells.values():
        assert cell.gru.weight_ih.grad is not None and cell.gru.weight_ih.grad.abs().sum()>0
    assert extra.evidence_residual[-1].weight.grad.abs().sum()>0


def test_unseen_id_parameters_cannot_change_predictions_or_receive_gradients():
    torch.manual_seed(23)
    model=make_tiny_model(use_user_identity=True,user_identity_dropout=0.).eval()
    model.set_identity_reference_users([0,1])
    batch=make_batch()
    model.reset_memory();before=model(batch)['trust_pred']
    with torch.no_grad():
        model.user_source_embedding.weight[2].fill_(100.)
        model.user_target_embedding.weight[2].fill_(100.)
        model.user_source_bias.weight[2].fill_(100.)
        model.user_target_bias.weight[2].fill_(100.)
    model.reset_memory();after=model(batch)['trust_pred']
    torch.testing.assert_close(before,after)
    model.reset_memory();model(batch)['trust_logit'].sum().backward()
    for parameter in model.identity_parameters():
        assert parameter.grad[2].abs().sum()==0
    assert model.user_source_embedding.weight.grad[:2].abs().sum()>0


def test_learned_user_identity_does_not_allow_future_queries_into_past_prediction():
    torch.manual_seed(23)
    model=make_tiny_model(use_user_identity=True).eval();model.set_identity_reference_users([0,1,2])
    batch=make_batch();model.reset_memory();before=model(batch)['trust_pred']
    changed=deepcopy(batch);changed['dst'][2:]=0;changed['entity_features'][2:]+=100.
    model.reset_memory();after=model(changed)['trust_pred']
    torch.testing.assert_close(before[:2],after[:2],rtol=1e-6,atol=1e-7)


def test_stationary_time_view_ignores_global_horizon_age_and_gap_scaling():
    torch.manual_seed(23)
    model=make_tiny_model(stationary_time_features=True,use_absolute_time_features=False).eval()
    batch=make_batch();model.reset_memory();before=model(batch)['trust_pred']
    changed=deepcopy(batch)
    changed['entity_features'][:,6:8]+=100.
    changed['context_features'][:,4]+=100.
    changed['edge_features'][:,1]+=100.
    model.reset_memory();after=model(changed)['trust_pred']
    torch.testing.assert_close(before,after)


def test_profile_identity_branch_is_differentiable_and_prefix_causal():
    torch.manual_seed(23)
    model=make_tiny_model(use_user_identity=True,use_rating_profiles=True).eval()
    model.set_identity_reference_users([0,1])
    batch=make_batch();batch['rating_profile_src']=torch.randn(4,16);batch['rating_profile_dst']=torch.randn(4,16)
    model.reset_memory();before=model(batch)['trust_pred']
    changed=deepcopy(batch);changed['rating_profile_dst'][2:]+=100.
    model.reset_memory();after=model(changed)['trust_pred']
    torch.testing.assert_close(before[:2],after[:2],rtol=1e-6,atol=1e-7)
    model.reset_memory();model(batch)['trust_logit'].sum().backward()
    assert model.user_profile_source_projection.weight.grad.abs().sum()>0
    assert model.user_profile_target_projection.weight.grad.abs().sum()>0
    assert model.user_source_embedding.weight.grad[2].abs().sum()==0
