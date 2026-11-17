from __future__ import annotations

from typing import Dict
import math

import numpy as np
import torch
from torch import nn


class _MemoryCell(nn.Module):


    def __init__(self, trust_dim: int, memory_dim: int, raw_dim: int = 0):
        super().__init__()
        self.raw_dim = raw_dim
        self.project = nn.Linear(trust_dim + raw_dim, memory_dim)
        self.gru = nn.GRUCell(memory_dim, memory_dim)
        self.gamma = nn.Sequential(nn.Linear(2 * memory_dim + 2, memory_dim), nn.ReLU(), nn.Linear(memory_dim, 1), nn.Sigmoid())
        self.kappa = nn.Sequential(nn.Linear(memory_dim + 2, memory_dim), nn.ReLU(), nn.Linear(memory_dim, 1), nn.Sigmoid())
        self.to_trust = nn.Linear(memory_dim, trust_dim)
        self.norm = nn.LayerNorm(trust_dim)
        self.tau = nn.Sequential(nn.Linear(trust_dim, 1), nn.Sigmoid())

    def input(self, z: torch.Tensor, raw: torch.Tensor | None = None) -> torch.Tensor:

        if self.raw_dim > 0 and raw is not None:
            z = torch.cat([z, raw], dim=-1)
        return torch.tanh(self.project(z))


class ComponentDecoupledMaskedTrustMemory(nn.Module):

    def __init__(self, trust_dim, memory_dim, component_wise_memory=True,
                 reconnect_correction=True, memory_key='edge', fast_batch_memory=False,
                 lambdas=None, etas=None, raw_dim=0, residual_scale=1.0,
                 time_scale_seconds=1.0):
        super().__init__()
        self.trust_dim, self.memory_dim, self.raw_dim = trust_dim, memory_dim, raw_dim
        self.component_wise_memory, self.reconnect_correction = component_wise_memory, reconnect_correction
        self.memory_key, self.fast_batch_memory = memory_key, fast_batch_memory
        self.residual_scale = float(residual_scale)
        self.time_scale_seconds = float(time_scale_seconds)
        if not math.isfinite(self.time_scale_seconds) or self.time_scale_seconds <= 0.:
            raise ValueError('memory time_scale_seconds must be finite and positive')
        self.cells = nn.ModuleDict({q: _MemoryCell(trust_dim, memory_dim, raw_dim) for q in ('E', 'B', 'C')})
        self.register_buffer('lambdas', torch.tensor(lambdas or ((.015, .045, .090) if component_wise_memory else (.045,) * 3)))
        self.register_buffer('etas', torch.tensor(etas or ((.025, .060, .120) if component_wise_memory else (.060,) * 3)))
        self.reset_state()

    def reset_state(self):
        self._tensor_state = None
        self._edge_rows = {}

    def detach_state(self):
        if self._tensor_state is not None:
            self._tensor_state = {k: v.detach() for k, v in self._tensor_state.items()}

    def _keys(self, num_nodes, src, dst):
        if self.memory_key in {'src', 'source', 'src_node', 'trustor'}:
            return src, num_nodes
        if self.memory_key in {'dst', 'target', 'dst_node', 'trustee'}:
            return dst, num_nodes
        rows = []
        for s, d in zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist()):
            key = (s, d)
            if key not in self._edge_rows:
                self._edge_rows[key] = len(self._edge_rows)
            rows.append(self._edge_rows[key])
        return torch.tensor(rows, device=src.device, dtype=torch.long), len(self._edge_rows)

    def _ensure_state(self, count, z):
        if self._tensor_state is None:
            self._tensor_state = {}
            for q in ('E', 'B', 'C'):
                self._tensor_state[f'M_{q}'] = z.new_zeros((count, self.memory_dim))
                self._tensor_state[f'U_{q}'] = z.new_zeros(count)
                self._tensor_state[f'obs_{q}'] = z.new_ones(count)
        else:
            old = self._tensor_state['M_E'].shape[0]
            if count > old:
                for k, v in self._tensor_state.items():
                    shape = (count - old,) + v.shape[1:]
                    pad = v.new_ones(shape) if k.startswith('obs_') else v.new_zeros(shape)
                    self._tensor_state[k] = torch.cat([v, pad])
        return self._tensor_state

    def _group_plan(self, keys, timestamp, update_mask, n):
        mask = np.ones(n, dtype=bool) if update_mask is None else update_mask.detach().cpu().numpy() > .5
        has_writes = bool(mask.any())
        if timestamp is None:
            ends = np.arange(1, n + 1)
        else:
            times = timestamp.detach().cpu().numpy()
            if np.any(times[1:] < times[:-1]):
                raise ValueError('Memory events must be chronological')
            ends = np.r_[np.flatnonzero(times[1:] != times[:-1]) + 1, n] if n else np.empty(0, dtype=np.int64)
        if not has_writes:
            ends = [n] if n else []
        key_values = keys.detach().cpu().numpy() if has_writes else None
        positions, selections, inverses, denominators, plan = [], [], [], [], []
        start = pstart = sstart = 0
        for end in ends:
            end = int(end)
            pos = np.flatnonzero(mask[start:end])
            if pos.size:
                selected, inverse, counts = np.unique(key_values[start:end][pos], return_inverse=True, return_counts=True)
            else:
                selected = inverse = counts = np.empty(0, dtype=np.int64)
            positions.append(pos); selections.append(selected); inverses.append(inverse); denominators.append(counts)
            plan.append((start, end, pstart, pstart + len(pos), sstart, sstart + len(selected)))
            pstart += len(pos); sstart += len(selected); start = end
        def pack(rows, dtype):
            values = np.concatenate(rows) if rows else np.empty(0)
            return torch.as_tensor(values, dtype=dtype, device=keys.device)
        return (has_writes, plan, pack(positions, torch.long), pack(selections, torch.long),
                pack(inverses, torch.long), pack(denominators, torch.float32))

    def forward(self, num_nodes, src, dst, observed_mask, delta_t, z_e, z_b, z_c,
                raw_event=None, timestamp=None, update_mask=None):
        n = src.numel()
        keys, count = self._keys(num_nodes, src, dst)
        state = self._ensure_state(count, z_e)
        has_writes, plan, positions, selections, inverses, denominators = self._group_plan(keys, timestamp, update_mask, n)
        z_map = {'E': z_e, 'B': z_b, 'C': z_c}
        components = ('E', 'B', 'C') if self.component_wise_memory else ('E',)
        projected = {}
        if has_writes:
            for q in components:
                z = z_map[q] if self.component_wise_memory else (z_e + z_b + z_c) / 3.
                projected[q] = self.cells[q].input(z, raw_event)
        memories, uncertainties = {q: [] for q in components}, {q: [] for q in components}
        for start, end, pstart, pend, sstart, send in plan:
            ids = keys[start:end]
            obs = observed_mask[start:end].to(z_e.dtype)
            dt = delta_t[start:end].to(z_e.dtype).clamp_min(0.) / self.time_scale_seconds
            eligible = positions[pstart:pend]
            selected, inverse = selections[sstart:send], inverses[pstart:pend]
            denom = denominators[sstart:send].to(z_e.dtype)
            for idx, q in enumerate(components):
                cell = self.cells[q]
                prev_m, prev_u = state[f'M_{q}'][ids], state[f'U_{q}'][ids]
                prev_obs = state[f'obs_{q}'][ids]
                decayed = torch.exp(-self.lambdas[idx] * dt)[:, None] * prev_m
                u = 1. - torch.exp(-self.etas[idx] * dt)
                memories[q].append(decayed)
                uncertainties[q].append(u)
                if selected.numel() == 0:
                    continue
                x = projected[q][start:end]
                candidate = cell.gru(x, decayed)
                gamma = cell.gamma(torch.cat([x, decayed, u[:, None], dt[:, None]], -1))
                updated = gamma * candidate + (1. - gamma) * decayed
                if self.reconnect_correction:
                    kappa = cell.kappa(torch.cat([x, prev_u[:, None], dt[:, None]], -1))
                    reconnect = ((prev_obs <= .5) & (obs > .5))[:, None]
                    updated = torch.where(reconnect, kappa * candidate + (1. - kappa) * decayed, updated)
                new_m = torch.where((obs > .5)[:, None], updated, decayed)
                merged_m = z_e.new_zeros((selected.numel(), self.memory_dim)).index_add(0, inverse, new_m[eligible]) / denom[:, None]
                merged_u = z_e.new_zeros(selected.numel()).index_add(0, inverse, u[eligible]) / denom
                merged_obs = z_e.new_zeros(selected.numel()).scatter_reduce(0, inverse, obs[eligible], reduce='amax')
                state[f'M_{q}'] = state[f'M_{q}'].index_copy(0, selected, merged_m)
                state[f'U_{q}'] = state[f'U_{q}'].index_copy(0, selected, merged_u)
                state[f'obs_{q}'] = state[f'obs_{q}'].index_copy(0, selected, merged_obs.detach())
        out = {}
        for q in ('E', 'B', 'C'):
            bank_q = q if self.component_wise_memory else 'E'
            m = torch.cat(memories[bank_q]) if n else z_e.new_zeros((0, self.memory_dim))
            u = torch.cat(uncertainties[bank_q]) if n else z_e.new_zeros(0)
            cell = self.cells[q]
            z = cell.norm(z_map[q] + self.residual_scale * (1. - u)[:, None] * cell.to_trust(m))
            out[f'z_{q}'], out[f'tau_{q}'] = z, cell.tau(z).squeeze(-1)
            out[f'M_{q}'], out[f'U_{q}'] = m, u
        out['memory'] = torch.cat([out[f'M_{q}'] for q in ('E', 'B', 'C')], -1)
        out['memory_uncertainty'] = torch.stack([out[f'U_{q}'] for q in ('E', 'B', 'C')], -1)
        return out
