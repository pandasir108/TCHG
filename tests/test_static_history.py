from copy import deepcopy

import numpy as np
import torch
import pytest

from tchg.data.event_dataset import EventGraphDataset
from tchg.data.static_features import static_training_features, static_structural_features
from tchg.training.trainer import Trainer
from test_model import make_batch
from tchg.models.conditioned_operator import ConditionedHeterogeneousOperatorStack
from test_model import make_tiny_model


def static_example():
    base = {k: v.numpy() for k, v in make_batch().items()}
    globals_ = {'node_features', 'num_nodes', 'num_edge_types'}
    arrays = {k: v if k in globals_ else np.concatenate([v] * 3) for k, v in base.items()}
    arrays['split'] = np.repeat([0, 1, 2], 4)
    arrays['timestamp'][:] = 0.
    arrays['delta_t'][:] = 0.
    return arrays


def test_static_features_exclude_current_training_edge_and_heldout_relations():
    arrays = static_example()
    expected = static_training_features(arrays)
    assert expected['entity_features'][0, 0] == .5
    assert expected['entity_features'][0, 1] == .5
    assert expected['behavior_features'][0, 0] == .5
    changed = deepcopy(arrays)
    changed['label'][4:] = 1. - changed['label'][4:]
    changed['sample_type'][4:] = 1 - changed['sample_type'][4:]
    result = static_training_features(changed)
    for key, value in expected.items():
        np.testing.assert_array_equal(result[key], value)


def test_directed_structure_matches_edge_deleted_graph_and_ignores_heldout_labels():
    edges = [(0, 1), (1, 0), (0, 2), (2, 1), (3, 0), (1, 1), (0, 3), (2, 3)]
    arrays = {'src': np.array([s for s, d in edges]), 'dst': np.array([d for s, d in edges]),
              'num_nodes': np.array([4]), 'split': np.array([0] * 6 + [1, 2]),
              'sample_type': np.zeros(8, dtype=np.int64), 'label': np.ones(8),
              'observed_mask': np.ones(8)}
    result = static_structural_features(arrays)
    for i, (s, d) in enumerate(edges):
        graph = set(edges[:6])
        if i < 6:
            graph.remove((s, d))
        out_s = {v for u, v in graph if u == s}
        in_d = {u for u, v in graph if v == d}
        assert result['behavior_features'][i, 0] == float((d, s) in graph)
        np.testing.assert_allclose(result['behavior_features'][i, 1], np.log1p(len(out_s & in_d)) / 10.)
    changed = deepcopy(arrays)
    changed['label'][6:] = 0.
    changed['sample_type'][6:] = 1
    after = static_structural_features(changed)
    for key, value in result.items():
        np.testing.assert_array_equal(value, after[key])


@pytest.mark.parametrize('refined', [False, True])
def test_zero_time_static_queries_use_train_only_and_memory_writers_learn(refined):
    torch.manual_seed(23)
    cfg = {'device': 'cpu', 'dataset': {'static_training_features': True,
                                      'static_diffusion_features': refined},
           'model': {'name': 'tchgt', 'hidden_dim': 16, 'trust_dim': 8, 'memory_dim': 8,
                     'num_layers': 2, 'num_operator_candidates': 2, 'dropout': 0.,
                     'memory_key': 'src_node', 'graph_history_mode': 'static_train',
                     'memory_time_scale_seconds': 86400., 'static_pair_interactions': refined},
           'training': {'batch_mode': 'event_stream', 'stream_batch_size': 2, 'lr': .001,
                        'exclude_static_query_edges': refined}}
    trainer = Trainer(cfg, EventGraphDataset.from_arrays(static_example()), device='cpu')
    assert trainer.static_history['src'].numel() == 2
    initial = {q: cell.gru.weight_ih.detach().clone() for q, cell in trainer.model.memory.cells.items()}
    trainer._step('train', train=True)
    for q, cell in trainer.model.memory.cells.items():
        assert (cell.gru.weight_ih - initial[q]).abs().sum() > 0.
    pred, _ = trainer.collect_predictions('test', apply_calibration=False)
    trainer.batches['val']['entity_features'] += 100.
    trainer.batches['val']['label'] = 1. - trainer.batches['val']['label']
    trainer.batches['val']['sample_type'] = 1 - trainer.batches['val']['sample_type']
    other, _ = trainer.collect_predictions('test', apply_calibration=False)
    torch.testing.assert_close(pred['trust_pred'], other['trust_pred'])
    trainer.cfg['training']['stream_batch_size'] = 20
    whole, _ = trainer.collect_predictions('test', apply_calibration=False)
    torch.testing.assert_close(pred['trust_pred'], whole['trust_pred'], rtol=1e-5, atol=1e-6)
    assert trainer._history_batch(trainer.batches['test'], True)['history_write_mask'].sum() == 0.


def test_bidirectional_static_graph_reads_outgoing_neighbors_and_trains_reverse_bank():
    torch.manual_seed(23)
    stack = ConditionedHeterogeneousOperatorStack(num_layers=1, static_mode=True,
        bidirectional=True, hidden_dim=4, trust_dim=3, edge_dim=4,
        num_edge_types=1, num_operator_candidates=2, dropout=0.,
        use_entity_gate=False, use_behavior_modulation=False).eval()
    batch = {'src': torch.tensor([0]), 'dst': torch.tensor([1]),
             'edge_type': torch.zeros(1, dtype=torch.long), 'edge_features': torch.zeros(1, 4)}
    enc = {'z_B': torch.zeros(1, 3), 'z_C': torch.zeros(1, 3),
           **{f'tau_{q}': torch.ones(1) for q in ['E', 'B', 'C']}}
    h = torch.randn(3, 4)
    stack.prime_static_history(h, batch, enc)
    original = stack._static_h[0].clone()
    modified = h.clone(); modified[1] += torch.tensor([3., -2., 1., 4.])
    stack.prime_static_history(modified, batch, enc)
    assert not torch.allclose(original, stack._static_h[0])
    (stack._static_h[0] * torch.tensor([1., 2., 3., 4.])).sum().backward()
    assert stack.layers[0].reverse_operators.grad.abs().sum() > 0.


def test_static_bidirectional_mode_cannot_silently_change_temporal_execution():
    with pytest.raises(ValueError, match='fixed training-graph'):
        make_tiny_model(static_bidirectional=True)
    with pytest.raises(ValueError, match='fixed training-graph'):
        make_tiny_model(static_pair_interactions=True)


def test_static_query_exclusion_uses_endpoints_only_and_is_training_only(monkeypatch):
    cfg = {'device': 'cpu', 'dataset': {'static_training_features': True},
           'model': {'name': 'tchgt', 'hidden_dim': 16, 'trust_dim': 8, 'memory_dim': 8,
                     'num_layers': 1, 'num_operator_candidates': 2, 'dropout': 0.,
                     'memory_key': 'src_node', 'graph_history_mode': 'static_train'},
           'training': {'batch_mode': 'event_stream', 'stream_batch_size': 2,
                        'exclude_static_query_edges': True}}
    t = Trainer(cfg, EventGraphDataset.from_arrays(static_example()), device='cpu')
    original, recorded = t.model.prime_static_history, []
    def observe(batch):
        recorded.append(set(zip(batch['src'].tolist(), batch['dst'].tolist())))
        original(batch)
    monkeypatch.setattr(t.model, 'prime_static_history', observe)
    query = t._slice_batch(t.batches['train'], 0, 2)
    t.model.train(); t._reset_and_prime('train', query=query)
    assert recorded[-1] == {(2, 0)}
    altered = dict(query)
    altered['label'] = 1. - query['label']; altered['sample_type'] = 1 - query['sample_type']
    t._reset_and_prime('train', query=altered)
    assert recorded[-1] == {(2, 0)}
    t.cfg['training']['static_history_dropout'] = .99
    t.model.eval(); t._reset_and_prime('val', query=query)
    assert recorded[-1] == {(0, 1), (2, 0)}
