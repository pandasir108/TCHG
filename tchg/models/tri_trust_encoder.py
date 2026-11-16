from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class _ComponentEncoder(nn.Module):


    def __init__(self, base_dim: int, meta_dim: int, trust_dim: int, dropout: float):
        super().__init__()
        self.base = nn.Linear(base_dim, trust_dim)
        self.meta = nn.Linear(meta_dim, trust_dim) if meta_dim > 0 else None
        self.gate = nn.Linear(2 * trust_dim, trust_dim) if meta_dim > 0 else None
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trust_dim, trust_dim),
            nn.ReLU(),
            nn.LayerNorm(trust_dim),
        )
        self.tau = nn.Sequential(nn.Linear(trust_dim, 1), nn.Sigmoid())

    def forward(self, x_base: torch.Tensor, x_meta: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:

        h_base = self.base(x_base)
        if self.meta is not None and self.gate is not None and x_meta is not None:
            h_meta = self.meta(x_meta)
            gate = torch.sigmoid(self.gate(torch.cat([h_base, h_meta], dim=-1)))
            h = gate * h_meta + (1.0 - gate) * h_base
        else:
            h = h_base
        z = self.mlp(h)
        tau = self.tau(z).squeeze(-1)
        return z, tau


class TriTrustEncoder(nn.Module):


    def __init__(
        self,
        entity_dim: int,
        behavior_dim: int,
        context_dim: int,
        trust_dim: int,
        dropout: float = 0.1,
        entity_meta_dim: int = 0,
        behavior_meta_dim: int = 0,
        context_meta_dim: int = 0,
    ):
        super().__init__()
        self.entity = _ComponentEncoder(entity_dim, entity_meta_dim, trust_dim, dropout)
        self.behavior = _ComponentEncoder(behavior_dim, behavior_meta_dim, trust_dim, dropout)
        self.context = _ComponentEncoder(context_dim, context_meta_dim, trust_dim, dropout)

    def forward(
        self,
        entity_features: torch.Tensor,
        behavior_features: torch.Tensor,
        context_features: torch.Tensor,
        entity_meta: torch.Tensor | None = None,
        behavior_meta: torch.Tensor | None = None,
        context_meta: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:

        z_e, tau_e = self.entity(entity_features, entity_meta)
        z_b, tau_b = self.behavior(behavior_features, behavior_meta)
        z_c, tau_c = self.context(context_features, context_meta)
        return {"z_E": z_e, "z_B": z_b, "z_C": z_c, "tau_E": tau_e, "tau_B": tau_b, "tau_C": tau_c}
