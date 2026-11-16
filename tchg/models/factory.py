from __future__ import annotations

from torch import nn

from tchg.models.tchgt_model import TCHGModel


def make_model(name: str, dims: dict[str, int], num_edge_types: int, cfg: dict) -> nn.Module:
    if name.lower() not in {"tchg", "tchgt", "full"}:
        raise ValueError(f"Unknown model name: {name}")
    return TCHGModel(dims, num_edge_types, cfg)
