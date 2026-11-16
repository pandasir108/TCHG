from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from tchg.models.conditioned_operator import ConditionedHeterogeneousOperatorStack
from tchg.models.masked_memory import ComponentDecoupledMaskedTrustMemory
from tchg.models.tri_trust_encoder import TriTrustEncoder
from tchg.models.uncertainty_head import UncertaintyHead


class MetaPathTemporalHead(nn.Module):


    def __init__(self, meta_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.pair = nn.Sequential(
            nn.Linear(3 * meta_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.time = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.evidence_time_token = nn.Parameter(torch.zeros(hidden_dim))
        self.gate = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(
        self,
        src_meta: torch.Tensor,
        dst_meta: torch.Tensor,
        timestamp: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:

        meta_product = src_meta * dst_meta
        dot = meta_product.sum(dim=-1, keepdim=True)
        denom = src_meta.norm(dim=-1, keepdim=True).clamp_min(1e-6) * dst_meta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cosine = dot / denom
        pair_h = self.pair(torch.cat([src_meta, dst_meta, meta_product, dot, cosine], dim=-1))

        timestamp = timestamp.float()
        delta_t = delta_t.float()
        time_norm = timestamp - timestamp.min()
        time_norm = time_norm / time_norm.max().clamp_min(1.0)
        delta_norm = delta_t / delta_t.max().clamp_min(1.0)
        phase = 2.0 * torch.pi * time_norm
        time_features = torch.stack([time_norm, delta_norm, torch.sin(phase), torch.cos(phase)], dim=-1)
        time_h = self.time(time_features) + self.evidence_time_token.unsqueeze(0)

        gate = self.gate(torch.cat([pair_h, time_h], dim=-1))
        return self.out(gate * pair_h + (1.0 - gate) * time_h).squeeze(-1)


class TCHGModel(nn.Module):


    def __init__(self, dims: dict[str, int], num_edge_types: int, cfg: Dict):
        super().__init__()
        hidden_dim = int(cfg.get("hidden_dim", 64))
        trust_dim = int(cfg.get("trust_dim", 24))
        memory_dim = int(cfg.get("memory_dim", 24))
        dropout = float(cfg.get("dropout", 0.1))
        component_wise = bool(cfg.get("component_wise_memory", True))
        lambdas = tuple(float(x) for x in cfg.get("memory_lambdas", (0.015, 0.045, 0.09))) if component_wise else (0.045, 0.045, 0.045)
        etas = tuple(float(x) for x in cfg.get("uncertainty_etas", (0.025, 0.06, 0.12))) if component_wise else (0.06, 0.06, 0.06)
        self.use_masked_memory = bool(cfg.get("use_masked_memory", True))
        self.use_absolute_time_features = bool(cfg.get('use_absolute_time_features', True))
        self.stationary_time_features = bool(cfg.get('stationary_time_features', False))
        self.entity_log_odds = bool(cfg.get('entity_log_odds', False))
        self.memory_strength = float(cfg.get('memory_strength', 1.0))
        self.standardize_evidence = bool(cfg.get('standardize_evidence', False))
        self.use_user_identity = bool(cfg.get('use_user_identity', False))
        self.use_rating_profiles = bool(cfg.get('use_rating_profiles', False))
        self.user_identity_dropout = float(cfg.get('user_identity_dropout', .3))
        self.evidence_keys = ('entity_features', 'behavior_features', 'context_features', 'edge_features')
        if self.standardize_evidence:
            for key, dim in zip(self.evidence_keys, ('entity_dim', 'behavior_dim', 'context_dim', 'edge_dim')):
                self.register_buffer(key + '_mean', torch.zeros(dims[dim]))
                self.register_buffer(key + '_std', torch.ones(dims[dim]))
        self.memory_components = {
            "E": bool(cfg.get("use_entity_memory", True)),
            "B": bool(cfg.get("use_behavior_memory", True)),
            "C": bool(cfg.get("use_context_memory", True)),
        }
        self.memory_raw_event = bool(cfg.get("memory_raw_event", True))
        self.use_metapath_head_features = bool(cfg.get("use_metapath_head_features", False))
        self.use_metapath_bilinear_head = bool(cfg.get("metapath_head_bilinear", False))
        self.metapath_bilinear_weight = float(cfg.get("metapath_bilinear_weight", 0.5))
        self.use_metapath_temporal_head = bool(cfg.get("metapath_temporal_head", False))
        self.metapath_temporal_weight = float(cfg.get("metapath_temporal_weight", 0.5))
        memory_edge_type_emb_dim = int(cfg.get("memory_edge_type_emb_dim", 8))
        self.node_proj = nn.Sequential(nn.Linear(dims["node_dim"], hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.encoder = TriTrustEncoder(
            dims["entity_dim"],
            dims["behavior_dim"],
            dims["context_dim"],
            trust_dim,
            dropout,
            entity_meta_dim=int(dims.get("entity_meta_dim", 0)),
            behavior_meta_dim=int(dims.get("behavior_meta_dim", 0)),
            context_meta_dim=int(dims.get("context_meta_dim", 0)),
        )
        if self.memory_raw_event:
            self.memory_edge_type_embedding = nn.Embedding(num_edge_types, memory_edge_type_emb_dim)
            memory_raw_dim = dims["edge_dim"] + memory_edge_type_emb_dim
        else:
            self.memory_edge_type_embedding = None
            memory_raw_dim = 0
        self.memory = ComponentDecoupledMaskedTrustMemory(
            trust_dim=trust_dim,
            memory_dim=memory_dim,
            component_wise_memory=component_wise,
            reconnect_correction=bool(cfg.get("reconnect_correction", True)),
            memory_key=str(cfg.get("memory_key", "edge")),
            fast_batch_memory=bool(cfg.get("fast_batch_memory", False)),
            lambdas=lambdas,
            etas=etas,
            raw_dim=memory_raw_dim,
            residual_scale=self.memory_strength,
            time_scale_seconds=float(cfg.get('memory_time_scale_seconds', 1.0)),
        )
        static_bidirectional = bool(cfg.get('static_bidirectional', False))
        if static_bidirectional and cfg.get('graph_history_mode') != 'static_train':
            raise ValueError('static_bidirectional requires the fixed training-graph mode')
        self.operator_stack = ConditionedHeterogeneousOperatorStack(
            num_layers=int(cfg.get("num_layers", 2)),
            static_mode=cfg.get('graph_history_mode') == 'static_train',
            bidirectional=static_bidirectional,
            hidden_dim=hidden_dim,
            trust_dim=trust_dim,
            edge_dim=dims["edge_dim"],
            num_edge_types=num_edge_types,
            num_operator_candidates=int(cfg.get("num_operator_candidates", 4)),
            dropout=dropout,
            theta_e=float(cfg.get("theta_e", 0.5)),
            beta=float(cfg.get("beta", 0.1)),
            entity_gate_mode=str(cfg.get("entity_gate_mode", "suppressive")),
            entity_gate_scale=float(cfg.get("entity_gate_scale", 0.5)),
            entity_gate_cold_start_identity=bool(cfg.get("entity_gate_cold_start_identity", False)),
            behavior_modulation_mode=str(cfg.get("behavior_modulation_mode", "suppressive")),
            behavior_modulation_scale=float(cfg.get("behavior_modulation_scale", 0.5)),
            use_entity_gate=bool(cfg.get("use_entity_gate", True)),
            use_behavior_modulation=bool(cfg.get("use_behavior_modulation", True)),
            use_context_operator=bool(cfg.get("use_context_operator", True)),
        )
        self.head_metapath_dim = int(dims.get("metapath_node_dim", 0)) or int(dims["node_dim"])
        self.static_pair_interactions = bool(cfg.get('static_pair_interactions', False))
        if self.static_pair_interactions and cfg.get('graph_history_mode') != 'static_train':
            raise ValueError('static_pair_interactions requires the fixed training-graph mode')
        head_dim = 2 * hidden_dim + 3 * trust_dim + 3 * memory_dim + 3 + dims["edge_dim"]
        if self.static_pair_interactions:
            head_dim += 2 * hidden_dim
        if self.use_metapath_head_features:
            head_dim += self.head_metapath_dim + 1
        self.head = UncertaintyHead(head_dim, hidden_dim, dropout)
        self.evidence_residual = None
        if bool(cfg.get('evidence_residual', False)):
            evidence_dim = sum(dims[k] for k in ('entity_dim', 'behavior_dim', 'context_dim', 'edge_dim'))
            self.evidence_residual = nn.Sequential(
                nn.Linear(evidence_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
            # The added path initially preserves the original model's logits.
            nn.init.zeros_(self.evidence_residual[-1].weight)
            nn.init.zeros_(self.evidence_residual[-1].bias)
        self.meta_branch = (
            nn.Sequential(
                nn.Linear(3 * self.head_metapath_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            if self.use_metapath_bilinear_head
            else None
        )
        self.meta_temporal_branch = (
            MetaPathTemporalHead(self.head_metapath_dim, hidden_dim, dropout)
            if self.use_metapath_temporal_head
            else None
        )
        self.user_source_embedding = self.user_target_embedding = None
        self.user_source_bias = self.user_target_bias = None
        self.user_identity_projection = None
        if self.use_user_identity:
            nodes = int(dims.get('num_nodes', 0))
            width = int(cfg.get('user_identity_dim', 32))
            if nodes <= 0 or not 0. <= self.user_identity_dropout < 1.:
                raise ValueError('User identity requires num_nodes and dropout in [0, 1)')
            self.register_buffer('identity_reference_mask', torch.zeros(nodes, dtype=torch.bool))
            self.user_source_embedding = nn.Embedding(nodes, width)
            self.user_target_embedding = nn.Embedding(nodes, width)
            self.user_source_bias = nn.Embedding(nodes, 1)
            self.user_target_bias = nn.Embedding(nodes, 1)
            self.user_identity_projection = nn.Linear(2 * width, hidden_dim, bias=False)
            nn.init.normal_(self.user_source_embedding.weight, std=.05)
            nn.init.normal_(self.user_target_embedding.weight, std=.05)
            nn.init.zeros_(self.user_source_bias.weight)
            nn.init.zeros_(self.user_target_bias.weight)
        self.user_profile_source_projection = self.user_profile_target_projection = None
        self.user_profile_source_bias = self.user_profile_target_bias = None
        if self.use_rating_profiles:
            profile_width = int(dims.get('rating_profile_dim', 0))
            width = int(cfg.get('user_identity_dim', 32))
            if profile_width <= 0:
                raise ValueError('use_rating_profiles requires causal rating-profile inputs')
            self.user_profile_source_projection = nn.Linear(profile_width, width, bias=False)
            self.user_profile_target_projection = nn.Linear(profile_width, width, bias=False)
            self.user_profile_source_bias = nn.Linear(profile_width, 1, bias=False)
            self.user_profile_target_bias = nn.Linear(profile_width, 1, bias=False)
            nn.init.zeros_(self.user_profile_source_bias.weight)
            nn.init.zeros_(self.user_profile_target_bias.weight)

    @torch.no_grad()
    def set_identity_reference_users(self, users):
        if self.use_user_identity:
            self.identity_reference_mask.zero_()
            self.identity_reference_mask[torch.as_tensor(users, dtype=torch.long, device=self.identity_reference_mask.device)] = True

    def identity_parameters(self):
        if not self.use_user_identity:
            return []
        return [module.weight for module in (self.user_source_embedding, self.user_target_embedding,
                                             self.user_source_bias, self.user_target_bias)]

    def _identity_state(self):
        available = self.identity_reference_mask[:, None]
        if self.training and self.user_identity_dropout:
            # Hide known identities during training to teach anonymous-user
            # prediction. Fixed paper user-regime metadata is not changed.
            available = available & (torch.rand(available.shape, device=available.device) >= self.user_identity_dropout)
        return (self.user_source_embedding.weight * available, self.user_target_embedding.weight * available,
                self.user_source_bias.weight * available, self.user_target_bias.weight * available)

    def reset_memory(self) -> None:

        self.memory.reset_state()
        self.operator_stack.reset_state()

    def prime_static_history(self, history_batch):
        """Encode observed training relations for an untimestamped trust task.

        Query labels and validation/test relations never enter this method.
        During training the priming graph remains differentiable, including
        the memory writers. Native zero gaps are retained.
        """
        if not self.operator_stack.static_mode:
            raise RuntimeError('Static priming is disabled for this model')
        batch = self._prepare_evidence(history_batch)
        enc = self.encoder(batch['entity_features'], batch['behavior_features'],
                           batch['context_features'], batch.get('entity_meta'),
                           batch.get('behavior_meta'), batch.get('context_meta'))
        if self.use_masked_memory:
            raw = None
            if self.memory_raw_event and self.memory_edge_type_embedding is not None:
                raw = torch.cat([batch['edge_features'], self.memory_edge_type_embedding(batch['edge_type'])], -1)
            memory_enc = self.memory(int(batch['num_nodes'][0]), batch['src'], batch['dst'],
                                     batch['observed_mask'], batch['delta_t'],
                                     enc['z_E'], enc['z_B'], enc['z_C'], raw_event=raw,
                                     timestamp=batch['timestamp'],
                                     update_mask=torch.ones_like(batch['observed_mask']))
            enc = self._apply_memory_component_mask(enc, memory_enc)
        h = self.node_proj(batch['node_features'])
        if self.use_user_identity:
            identity = self._identity_state()
            h = h + self.user_identity_projection(torch.cat(identity[:2], -1))
        self.operator_stack.prime_static_history(h, batch, enc)

    def detach_memory(self) -> None:

        self.memory.detach_state()
        self.operator_stack.detach_state()

    def _apply_memory_component_mask(
        self,
        base_enc: dict[str, torch.Tensor],
        memory_enc: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        masked = dict(memory_enc)
        for component, enabled in self.memory_components.items():
            if enabled:
                continue
            masked[f"z_{component}"] = base_enc[f"z_{component}"]
            masked[f"tau_{component}"] = base_enc[f"tau_{component}"]
            masked[f"M_{component}"] = torch.zeros_like(memory_enc[f"M_{component}"])
            masked[f"U_{component}"] = torch.zeros_like(memory_enc[f"U_{component}"])
        masked["memory"] = torch.cat([masked[f"M_{q}"] for q in ("E", "B", "C")], dim=-1)
        masked["memory_uncertainty"] = torch.stack([masked[f"U_{q}"] for q in ("E", "B", "C")], dim=-1)
        return masked

    def _prepare_evidence(self, batch, apply_statistics=True):
        batch = dict(batch)
        if not self.use_absolute_time_features:
            # Retain real timestamps and relative gaps for causal state updates;
            # omit the arbitrary global-horizon clock from neural predictors.
            batch['context_features'] = batch['context_features'].clone()
            batch['context_features'][:, [1, 2, 3, 7]] = 0.
            batch['edge_features'] = batch['edge_features'].clone()
            batch['edge_features'][:, 2] = 0.
        if self.stationary_time_features:
            batch['entity_features'] = batch['entity_features'].clone()
            batch['entity_features'][:, 6:8] = 0.
            batch['context_features'] = batch['context_features'].clone()
            batch['edge_features'] = batch['edge_features'].clone()
            gap = torch.log1p(batch['delta_t'].clamp_min(0.) / 86400.) / 10.
            batch['context_features'][:, 4] = gap
            batch['edge_features'][:, 1] = gap
        if self.entity_log_odds:
            entity_input = batch['entity_features']
            odds = torch.logit(entity_input[:, :2].clamp(1e-5, 1. - 1e-5)) / 10.
            batch['entity_features'] = torch.cat([odds, entity_input[:, 2:]], dim=-1)
        if self.standardize_evidence and apply_statistics:
            for key in self.evidence_keys:
                batch[key] = (batch[key] - getattr(self, key + '_mean')) / getattr(self, key + '_std')
        return batch

    @torch.no_grad()
    def fit_input_statistics(self, training_batch):
        """Fit once on training evidence; inference only reads these buffers."""
        if self.standardize_evidence:
            transformed = self._prepare_evidence(training_batch, apply_statistics=False)
            for key in self.evidence_keys:
                value = transformed[key]
                mean = value.mean(dim=0)
                std = value.std(dim=0, unbiased=False)
                std = torch.where(std > 1e-6, std, torch.ones_like(std))
                getattr(self, key + '_mean').copy_(mean)
                getattr(self, key + '_std').copy_(std)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = self._prepare_evidence(batch)
        if 'history_write_mask' not in batch:
            write = torch.ones_like(batch['observed_mask'])
            if 'sample_type' in batch:
                write = write * (batch['sample_type'] == 0)
            if 'history_visible_mask' in batch:
                write = write * (batch['history_visible_mask'] > .5)
            batch['history_write_mask'] = write
        num_nodes = int(batch["num_nodes"][0].detach().item())
        enc = self.encoder(
            batch['entity_features'],
            batch["behavior_features"],
            batch["context_features"],
            batch.get("entity_meta"),
            batch.get("behavior_meta"),
            batch.get("context_meta"),
        )
        if self.use_masked_memory:
            raw_event = None
            if self.memory_raw_event and self.memory_edge_type_embedding is not None:
                edge_type_emb = self.memory_edge_type_embedding(batch["edge_type"].long())
                raw_event = torch.cat([batch["edge_features"], edge_type_emb], dim=-1)
            memory_enc = self.memory(
                num_nodes,
                batch["src"],
                batch["dst"],
                batch["observed_mask"],
                batch["delta_t"],
                enc["z_E"],
                enc["z_B"],
                enc["z_C"],
                raw_event=raw_event,
                timestamp=batch['timestamp'],
                update_mask=batch['history_write_mask'],
            )
            enc = self._apply_memory_component_mask(enc, memory_enc)
            mem = self.memory_strength * enc["memory"]
            mem_u = enc["memory_uncertainty"]
        else:
            n = batch["src"].shape[0]
            device, dtype = batch["edge_features"].device, batch["edge_features"].dtype
            mem = torch.zeros(n, 3 * self.memory.memory_dim, device=device, dtype=dtype)
            mem_u = torch.zeros(n, 3, device=device, dtype=dtype)
        h = self.node_proj(batch["node_features"])
        identity = self._identity_state() if self.use_user_identity else None
        if identity is not None:
            h = h + self.user_identity_projection(torch.cat(identity[:2], dim=-1))
        src_h, dst_h = self.operator_stack(h, batch, enc)
        head_parts = [src_h, dst_h, enc["z_E"], enc["z_B"], enc["z_C"], mem, mem_u, batch["edge_features"]]
        if self.static_pair_interactions:
            head_parts.extend([src_h * dst_h, (src_h - dst_h).abs()])
        metapath_table = batch.get("metapath_node_features", batch["node_features"])
        if self.use_metapath_head_features:
            src_meta = metapath_table[batch["src"]]
            dst_meta = metapath_table[batch["dst"]]
            meta_product = src_meta * dst_meta
            meta_dot = meta_product.sum(dim=-1, keepdim=True)
            head_parts.extend([meta_product, meta_dot])
        head_in = torch.cat(head_parts, dim=-1)
        out = self.head(head_in)
        if identity is not None or self.use_rating_profiles:
            pair_source = pair_target = bias_source = bias_target = None
            if self.use_rating_profiles:
                pair_source = self.user_profile_source_projection(batch['rating_profile_src'])
                pair_target = self.user_profile_target_projection(batch['rating_profile_dst'])
                bias_source = self.user_profile_source_bias(batch['rating_profile_src']).squeeze(-1)
                bias_target = self.user_profile_target_bias(batch['rating_profile_dst']).squeeze(-1)
            if identity is not None:
                source, target, source_bias, target_bias = identity
                source, target = source[batch['src']], target[batch['dst']]
                sb, tb = source_bias[batch['src']].squeeze(-1), target_bias[batch['dst']].squeeze(-1)
                pair_source = source if pair_source is None else pair_source + source
                pair_target = target if pair_target is None else pair_target + target
                bias_source = sb if bias_source is None else bias_source + sb
                bias_target = tb if bias_target is None else bias_target + tb
            pair = (pair_source * pair_target).sum(-1) / pair_source.shape[1] ** .5
            pair = pair + bias_source + bias_target
            out['trust_logit'] = out['trust_logit'] + pair
            out['trust_pred'] = torch.sigmoid(out['trust_logit'])
        if self.evidence_residual is not None:
            evidence = torch.cat([batch[k] for k in self.evidence_keys], dim=-1)
            out['trust_logit'] = out['trust_logit'] + self.evidence_residual(evidence).squeeze(-1)
            out['trust_pred'] = torch.sigmoid(out['trust_logit'])
        if self.meta_branch is not None:
            src_meta = metapath_table[batch["src"]]
            dst_meta = metapath_table[batch["dst"]]
            meta_logit = self.meta_branch(torch.cat([src_meta, dst_meta, src_meta * dst_meta], dim=-1)).squeeze(-1)
            out["trust_logit"] = out["trust_logit"] + self.metapath_bilinear_weight * meta_logit
            out["trust_pred"] = torch.sigmoid(out["trust_logit"])
        if self.meta_temporal_branch is not None:
            src_meta = metapath_table[batch["src"]]
            dst_meta = metapath_table[batch["dst"]]
            meta_logit = self.meta_temporal_branch(src_meta, dst_meta, batch["timestamp"], batch["delta_t"])
            out["trust_logit"] = out["trust_logit"] + self.metapath_temporal_weight * meta_logit
            out["trust_pred"] = torch.sigmoid(out["trust_logit"])
        out.update(
            {
                "tau_E": enc["tau_E"],
                "tau_B": enc["tau_B"],
                "tau_C": enc["tau_C"],
                "memory_uncertainty": mem_u,
                "operator_diversity": self.operator_stack.diversity_loss(),
            }
        )
        return out
