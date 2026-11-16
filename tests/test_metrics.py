import numpy as np

from tchg.training.metrics import sampled_mean_reciprocal_rank


def test_sampled_mrr_is_one_when_positives_rank_first() -> None:
    labels = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
    assert sampled_mean_reciprocal_rank(labels, scores) == 1.0
