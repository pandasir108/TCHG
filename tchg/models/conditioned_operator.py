from __future__ import annotations

import torch
from torch import nn


class ConditionedHeterogeneousOperatorLayer(nn.Module):


    def __init__(
        self,
        hidden_dim: int,
        trust_dim: int,
        edge_dim: int,
        num_edge_types: int,
        num_operator_candidates: int,
        dropout: float,
        theta_e: float = 0.5,
        beta: float = 0.1,
        entity_gate_mode: str = "suppressive",
        entity_gate_scale: float = 0.5,
        entity_gate_cold_start_identity: bool = False,
        behavior_modulation_mode: str = "suppressive",
        behavior_modulation_scale: float = 0.5,
        use_entity_gate: bool = True,
        use_behavior_modulation: bool = True,
        use_context_operator: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_operator_candidates = num_operator_candidates
        self.theta_e = theta_e
        self.beta = beta
        self.entity_gate_mode = entity_gate_mode
        self.entity_gate_scale = entity_gate_scale
        self.entity_gate_cold_start_identity = entity_gate_cold_start_identity
        self.behavior_modulation_mode = behavior_modulation_mode
        self.behavior_modulation_scale = behavior_modulation_scale
        self.use_entity_gate = use_entity_gate
        self.use_behavior_modulation = use_behavior_modulation
        self.use_context_operator = use_context_operator
        self.bidirectional = bidirectional
        self.operators = nn.Parameter(torch.empty(num_edge_types, num_operator_candidates, hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.operators)
        self.reverse_operators = None
        if bidirectional:
            self.reverse_operators = nn.Parameter(torch.empty_like(self.operators))
            nn.init.xavier_uniform_(self.reverse_operators)
        self.edge_type_embedding = nn.Embedding(num_edge_types, trust_dim)
        self.behavior_modulator = nn.Sequential(nn.Linear(1 + trust_dim + edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.context_selector = nn.Sequential(nn.Linear(2 * trust_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_operator_candidates))
        self.update = nn.Sequential(nn.Linear((3 if bidirectional else 2) * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def entity_gate(
        self,
        tau_e: torch.Tensor,
        is_new_node: torch.Tensor | None = None,
        apply_cold_start_identity: bool = True,
    ) -> torch.Tensor:

        if not self.use_entity_gate:
            return torch.ones_like(tau_e).unsqueeze(-1)
        gate = torch.sigmoid((tau_e - self.theta_e) / max(self.beta, 1e-6))
        if self.entity_gate_mode == "residual":
            gate = 1.0 + self.entity_gate_scale * (2.0 * gate - 1.0)
        if apply_cold_start_identity and self.entity_gate_cold_start_identity and is_new_node is not None:
            gate = torch.where(is_new_node.to(dtype=torch.bool), torch.ones_like(gate), gate)
        return gate.clamp_min(0.0).unsqueeze(-1)

    def behavior_reliability(self, tau_b: torch.Tensor, z_b: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:

        if not self.use_behavior_modulation:
            return torch.ones_like(tau_b).unsqueeze(-1)
        rho = self.behavior_modulator(torch.cat([tau_b.unsqueeze(-1), z_b, edge_features], dim=-1))
        if self.behavior_modulation_mode == "residual":
            rho = 1.0 + self.behavior_modulation_scale * (2.0 * rho - 1.0)
        return rho.clamp_min(0.0)

    def context_mixture(self, z_c: torch.Tensor, tau_c: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:

        if not self.use_context_operator:
            return torch.full((z_c.shape[0], self.num_operator_candidates), 1.0 / self.num_operator_candidates, device=z_c.device, dtype=z_c.dtype)
        emb = self.edge_type_embedding(edge_type)
        logits = self.context_selector(torch.cat([z_c, tau_c.unsqueeze(-1), emb], dim=-1))
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_type: torch.Tensor,
        edge_features: torch.Tensor,
        z_b: torch.Tensor,
        z_c: torch.Tensor,
        tau_e: torch.Tensor,
        tau_b: torch.Tensor,
        tau_c: torch.Tensor,
        is_new_node: torch.Tensor | None = None,
    ) -> torch.Tensor:

        h_src = h[src]
        if self.operators.shape[0] == 1:
            candidate_msg = torch.einsum('ed,kdo->eko', h_src, self.operators[0])
        else:
            candidate_w = self.operators[edge_type]
            candidate_msg = torch.einsum("ed,ekdo->eko", h_src, candidate_w)
        pi_c = self.context_mixture(z_c, tau_c, edge_type)
        w_context_h = (candidate_msg * pi_c.unsqueeze(-1)).sum(dim=1)
        reliability = self.entity_gate(tau_e, is_new_node) * self.behavior_reliability(tau_b, z_b, edge_features)
        message = reliability * w_context_h
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, message)
        deg = torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=h.device, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0)
        parts = [h, agg]
        if self.bidirectional:
            if self.reverse_operators.shape[0] == 1:
                reverse = torch.einsum('ed,kdo->eko', h[dst], self.reverse_operators[0])
            else:
                reverse = torch.einsum('ed,ekdo->eko', h[dst], self.reverse_operators[edge_type])
            reverse = reliability * (reverse * pi_c.unsqueeze(-1)).sum(1)
            outgoing = torch.zeros_like(h).index_add(0, src, reverse)
            degree_out = h.new_zeros((len(h), 1)).index_add(0, src, h.new_ones((len(src), 1)))
            parts.append(outgoing / degree_out.clamp_min(1.))
        update = self.update(torch.cat(parts, dim=-1))
        return self.norm(h + self.dropout(update))

    def diversity_loss(self) -> torch.Tensor:

        operators = self.operators if self.reverse_operators is None else torch.cat([self.operators, self.reverse_operators], 0)
        flat = operators.flatten(start_dim=2)
        flat = torch.nn.functional.normalize(flat, dim=-1)
        sim = torch.matmul(flat, flat.transpose(1, 2))
        eye = torch.eye(self.num_operator_candidates, device=sim.device).unsqueeze(0)
        return ((sim - eye) ** 2).mean()


class ConditionedHeterogeneousOperatorStack(nn.Module):


    def __init__(self, num_layers: int, static_mode: bool = False, **kwargs):
        super().__init__()
        self.static_mode = static_mode
        self.layers = nn.ModuleList([ConditionedHeterogeneousOperatorLayer(**kwargs) for _ in range(num_layers)])
        self.reset_state()

    def reset_state(self) -> None:
        self._history: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._last_timestamp: float | None = None
        self._static_h = None

    def detach_state(self) -> None:
        self._history = [(s.detach(), c.detach()) for s, c in self._history]
        if self._static_h is not None:
            self._static_h = self._static_h.detach()

    def prime_static_history(self, h, batch, enc):
        if not self.static_mode:
            raise RuntimeError('Static priming requires static_mode')
        for layer in self.layers:
            h = layer(h, batch['src'], batch['dst'], batch['edge_type'],
                      batch['edge_features'], enc['z_B'], enc['z_C'],
                      enc['tau_E'], enc['tau_B'], enc['tau_C'])
        self._static_h = h

    @staticmethod
    def _historical_means(message, batch, history):
        src, dst, timestamp = batch['src'], batch['dst'], batch['timestamp']
        sums, counts = history
        weight = batch['history_write_mask'] * (batch['observed_mask'] > .5).to(message.dtype)
        weighted = message * weight[:, None]
        _, group = torch.unique_consecutive(timestamp, return_inverse=True)
        stride = int(group[-1]) + 2
        order = torch.argsort(dst, stable=True)
        nodes, values = dst[order], weighted[order].double()
        weights = weight[order, None]
        prefix, prefix_count = values.cumsum(0), weights.cumsum(0)
        first = torch.ones_like(nodes, dtype=torch.bool)
        first[1:] = nodes[1:] != nodes[:-1]
        offsets = sums.new_zeros(sums.shape, dtype=torch.float64).index_copy(0, nodes[first], (prefix - values)[first])
        count_offsets = torch.zeros_like(counts).index_copy(0, nodes[first], (prefix_count - weights)[first])
        prefix, prefix_count = (prefix - offsets[nodes]).to(message.dtype), prefix_count - count_offsets[nodes]
        edge_keys = nodes * stride + group[order] + 1

        def read(query):
            pos = torch.searchsorted(edge_keys, query * stride + group + 1, right=False) - 1
            safe = pos.clamp_min(0)
            valid = ((pos >= 0) & (nodes[safe] == query)).to(message.dtype)[:, None]
            return (sums[query] + prefix[safe] * valid) / (counts[query] + prefix_count[safe] * valid).clamp_min(1.)

        means = read(src), read(dst)
        updated = sums.index_add(0, dst, weighted), counts.index_add(0, dst, weight[:, None])
        return means, updated

    def forward(self, h: torch.Tensor, batch: dict[str, torch.Tensor], enc: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst, timestamp = batch['src'], batch['dst'], batch['timestamp']
        if self.static_mode:
            if self._static_h is None:
                raise RuntimeError('Prime the fixed training graph before static queries')
            return self._static_h[src], self._static_h[dst]
        if not src.numel():
            return h[src], h[dst]
        if bool((timestamp[1:] < timestamp[:-1]).any()):
            raise ValueError('Graph queries must be chronological')
        if self._last_timestamp is not None and float(timestamp[0]) <= self._last_timestamp:
            raise ValueError('Do not split simultaneous events across graph calls')
        if not self._history:
            self._history = [(torch.zeros_like(h), h.new_zeros((h.shape[0], 1))) for _ in self.layers]
        h_src, h_dst = h[src], h[dst]
        for i, layer in enumerate(self.layers):
            if layer.operators.shape[0] == 1:
                candidate = torch.einsum('ed,kdo->eko', h_src, layer.operators[0])
            else:
                candidate = torch.einsum('ed,ekdo->eko', h_src, layer.operators[batch['edge_type']])
            mixture = layer.context_mixture(enc['z_C'], enc['tau_C'], batch['edge_type'])
            message = (candidate * mixture.unsqueeze(-1)).sum(1)
            unobserved_user = batch.get('unobserved_user_mask', batch.get('is_new_node'))
            message = message * layer.entity_gate(enc['tau_E'], unobserved_user)
            message = message * layer.behavior_reliability(enc['tau_B'], enc['z_B'], batch['edge_features'])
            (a_src, a_dst), self._history[i] = self._historical_means(message, batch, self._history[i])
            h_src = layer.norm(h_src + layer.dropout(layer.update(torch.cat([h_src, a_src], -1))))
            h_dst = layer.norm(h_dst + layer.dropout(layer.update(torch.cat([h_dst, a_dst], -1))))
        self._last_timestamp = float(timestamp[-1])
        return h_src, h_dst

    def diversity_loss(self) -> torch.Tensor:

        if not self.layers:
            return torch.tensor(0.0)
        return torch.stack([layer.diversity_loss() for layer in self.layers]).mean()
