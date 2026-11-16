from __future__ import annotations

import torch
from torch import nn


class UncertaintyHead(nn.Module):


    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.trust_logit = nn.Linear(hidden_dim, 1)
        self.uncertainty = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:

        h = self.shared(x)
        logit = self.trust_logit(h).squeeze(-1)
        trust_pred = torch.sigmoid(logit)
        uncertainty = self.uncertainty(h).squeeze(-1)
        return {"trust_logit": logit, "trust_pred": trust_pred, "uncertainty_pred": uncertainty}
