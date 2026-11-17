from copy import deepcopy

import numpy as np
import pytest

from tchg.data.static_diffusion import static_diffusion_features


def graph_example():
    pairs = [(0, 1), (1, 2), (2, 0), (0, 2), (2, 3), (3, 3),
             (0, 1), (4, 0), (3, 1), (2, 1), (1, 3)]
    n = len(pairs)
    return {'src': np.array([s for s, d in pairs]), 'dst': np.array([d for s, d in pairs]),
            'num_nodes': np.array([5]), 'timestamp': np.zeros(n),
            'split': np.array([0] * 7 + [1, 2, 1, 2]),
            'label': np.ones(n), 'sample_type': np.zeros(n, dtype=np.int64),
            'observed_mask': np.ones(n)}


def test_diffusion_agrees_with_direct_inverse_of_each_edge_deleted_graph():
    arrays = graph_example()
    damping = [.3, .8, .99]
    result = static_diffusion_features(arrays, damping=damping)
    adjacency = np.zeros((5, 5))
    for s, d in zip(arrays['src'][:7], arrays['dst'][:7]):
        adjacency[s, d] += 1.
    for row, (s, d) in enumerate(zip(arrays['src'], arrays['dst'])):
        graph = adjacency.copy()
        if row < 7:
            graph[s, d] -= 1.
        graph = (graph > 0).astype(float)
        for orientation in range(2):
            g = graph if orientation == 0 else graph.T
            transition = g / np.maximum(g.sum(1, keepdims=True), 1.)
            for column, alpha in enumerate(damping):
                inverse = np.linalg.inv(np.eye(5) - alpha * transition)
                offset = orientation * len(damping) + column
                if s == 4 or d == 4:
                    np.testing.assert_array_equal(result['behavior_features'][row], 0.)
                    continue
                sd, ds, ss, dd = inverse[s, d], inverse[d, s], inverse[s, s], inverse[d, d]
                forward, backward = max(sd - (s == d), 0.), max(ds - (s == d), 0.)
                np.testing.assert_allclose(result['behavior_features'][row, 2 * offset:2 * offset + 2],
                    np.log1p(100. * np.array([forward, backward])) / 10., atol=1e-7)
                np.testing.assert_allclose(result['context_features'][row, 4 * offset:4 * offset + 4],
                    np.log1p(100. * np.array([max(ss - 1., 0.), max(dd - 1., 0.),
                        forward / max(dd, 1.), backward / max(ss, 1.)])) / 10., atol=1e-7)


def test_diffusion_ignores_heldout_outcomes_and_is_query_order_invariant():
    arrays = graph_example()
    expected = static_diffusion_features(arrays)
    changed = deepcopy(arrays)
    changed['label'][7:] = 0.
    changed['sample_type'][7:] = 1
    actual = static_diffusion_features(changed)
    permutation = np.random.default_rng(2).permutation(len(arrays['src']))
    permuted = {k: v if k == 'num_nodes' else v[permutation] for k, v in changed.items()}
    reordered = static_diffusion_features(permuted)
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
        np.testing.assert_array_equal(expected[key][permutation], reordered[key])


def test_diffusion_has_no_evidence_without_observed_training_relations():
    arrays = graph_example()
    arrays['history_visible_mask'] = np.zeros(len(arrays['src']))
    result = static_diffusion_features(arrays)
    for value in result.values():
        np.testing.assert_array_equal(value, 0.)
    with pytest.raises(ValueError, match='damping'):
        static_diffusion_features(arrays, damping=[1.])
