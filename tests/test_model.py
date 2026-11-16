import torch

from tchg.models.factory import make_model


def make_batch() -> dict[str, torch.Tensor]:
    return {
        "src": torch.tensor([0, 1, 2, 0]),
        "dst": torch.tensor([1, 2, 0, 2]),
        "edge_type": torch.zeros(4, dtype=torch.long),
        "timestamp": torch.arange(4, dtype=torch.float32),
        "observed_mask": torch.ones(4),
        "delta_t": torch.ones(4),
        "entity_features": torch.rand(4, 8),
        "behavior_features": torch.rand(4, 8),
        "context_features": torch.rand(4, 8),
        "edge_features": torch.rand(4, 4),
        "label": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "sample_type": torch.tensor([0, 1, 0, 1]),
        "node_features": torch.rand(3, 16),
        "num_nodes": torch.tensor([3]),
        "num_edge_types": torch.tensor([1]),
    }


def make_tiny_model(**overrides) -> torch.nn.Module:
    dims = {
        "num_nodes": 3,
        "rating_profile_dim": 16,
        "node_dim": 16,
        "entity_dim": 8,
        "behavior_dim": 8,
        "context_dim": 8,
        "edge_dim": 4,
    }
    cfg = {
        "hidden_dim": 16,
        "trust_dim": 8,
        "memory_dim": 8,
        "num_layers": 1,
        "num_operator_candidates": 2,
        "dropout": 0.0,
        "memory_key": "src_node",
        "fast_batch_memory": True,
    }
    cfg.update(overrides)
    return make_model("tchgt", dims, 1, cfg)


def test_model_forward_shapes() -> None:
    output = make_tiny_model()(make_batch())
    assert output["trust_pred"].shape == (4,)
    assert output["uncertainty_pred"].shape == (4,)
    assert torch.isfinite(output["trust_pred"]).all()
