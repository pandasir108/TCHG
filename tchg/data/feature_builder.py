from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict
import numpy as np
import pandas as pd

@dataclass
class NodeHistory:
    pos_out: int = 0
    neg_out: int = 0
    pos_in: int = 0
    neg_in: int = 0
    first_time: float | None = None
    last_time: float | None = None

    @property
    def activity(self) -> int:
        return self.pos_out + self.neg_out + self.pos_in + self.neg_in

@dataclass
class PairHistory:
    count: int = 0
    pos: int = 0
    neg: int = 0
    last_time: float | None = None
    last_label: float = 0.5
    ratings: list[float] = field(default_factory=list)

def _ratio(pos: int, neg: int) -> float:
    return (pos + 1.0) / (pos + neg + 2.0)

def _log_norm(x: float, scale: float=10.0) -> float:
    return float(np.clip(np.log1p(max(0.0, x)) / np.log1p(scale), 0.0, 1.0))

def _safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)

def _metapath_enabled(cfg: dict | None, key: str) -> bool:
    if not cfg:
        return False
    return bool(cfg.get(key, False))

def _prepare_metapath_user_emb(emb: np.ndarray | None, cfg: dict | None, seed: int) -> np.ndarray | None:
    if emb is None:
        return None
    out = np.asarray(emb, dtype=np.float32)
    dim = int((cfg or {}).get('per_channel_dim', 0))
    if dim <= 0 or dim == out.shape[1]:
        return out
    rng = np.random.default_rng(int(seed) + 15485863)
    projection = rng.normal(0.0, 1.0 / np.sqrt(out.shape[1]), size=(out.shape[1], dim)).astype(np.float32)
    return (out @ projection).astype(np.float32)

def _attach_history_visible_mask(events: pd.DataFrame, history_invisible_rate: float, seed: int) -> pd.DataFrame:
    events = events.copy()
    if 'history_visible_mask' in events.columns:
        events['history_visible_mask'] = events['history_visible_mask'].astype(float)
        return events
    rate = float(np.clip(history_invisible_rate, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    events['history_visible_mask'] = (rng.random(len(events)) > rate).astype(float)
    return events

def build_event_features_cat_rolling(events: pd.DataFrame, num_nodes: int, p_missing: float, seed: int, respect_observed_history: bool=False, history_invisible_rate: float=0.0, zero_time_features: bool=False, node_feature_policy: str='constant', node_feature_dim: int=16, node_features_override: np.ndarray | None=None, metapath_context_features: np.ndarray | None=None, metapath_user_emb: np.ndarray | None=None, metapath_cfg: dict | None=None) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    metapath_emb = _prepare_metapath_user_emb(metapath_user_emb, metapath_cfg, seed)
    use_entity_metapath = metapath_emb is not None and _metapath_enabled(metapath_cfg, 'inject_entity')
    use_behavior_metapath = metapath_emb is not None and _metapath_enabled(metapath_cfg, 'inject_behavior')
    use_context_metapath = metapath_emb is not None and _metapath_enabled(metapath_cfg, 'inject_context')
    events = events.sort_values('timestamp', kind='mergesort').reset_index(drop=True)
    events = _attach_history_visible_mask(events, history_invisible_rate, seed + 1009)
    if 'generated_observed' in events.columns:
        events['generated_observed'] = events['generated_observed'].astype(float)
    else:
        events['generated_observed'] = (rng.random(len(events)) > p_missing).astype(float)
    extra_context_cols = [col for col in events.columns if col.startswith('ctx_')]
    extra_edge_cols = [col for col in events.columns if col.startswith('edge_extra_')]
    extra_entity_meta_cols = [col for col in events.columns if col.startswith('entity_meta_extra_')]
    extra_behavior_meta_cols = [col for col in events.columns if col.startswith('behavior_meta_extra_')]
    extra_context_meta_cols = [col for col in events.columns if col.startswith('context_meta_extra_')]
    use_entity_meta = use_entity_metapath or bool(extra_entity_meta_cols)
    use_behavior_meta = use_behavior_metapath or bool(extra_behavior_meta_cols)
    use_context_meta = use_context_metapath or bool(extra_context_meta_cols)
    min_t = float(events['timestamp'].min())
    max_t = float(events['timestamp'].max())
    time_span = max(max_t - min_t, 1.0)
    node_hist = [NodeHistory() for _ in range(num_nodes)]
    pair_hist: dict[tuple[int, int], PairHistory] = {}
    global_recent: list[float] = []
    src_arr: list[int] = []
    dst_arr: list[int] = []
    edge_types: list[int] = []
    timestamps: list[float] = []
    observed: list[float] = []
    delta_ts: list[float] = []
    entity_features: list[list[float]] = []
    behavior_features: list[list[float]] = []
    context_features: list[list[float]] = []
    entity_meta: list[list[float]] = []
    behavior_meta: list[list[float]] = []
    context_meta: list[list[float]] = []
    edge_features: list[list[float]] = []
    labels: list[float] = []
    sample_types: list[int] = []
    history_visible_masks: list[float] = []

    def snapshot() -> tuple[list[NodeHistory], dict[tuple[int, int], PairHistory], list[float]]:
        return (deepcopy(node_hist), deepcopy(pair_hist), list(global_recent))

    def restore(state: tuple[list[NodeHistory], dict[tuple[int, int], PairHistory], list[float]]) -> None:
        nonlocal node_hist, pair_hist, global_recent
        node_hist, pair_hist, global_recent = (deepcopy(state[0]), deepcopy(state[1]), list(state[2]))

    def handle(row, append: bool, allow_update: bool) -> None:
        src = int(row.src)
        dst = int(row.dst)
        timestamp = float(row.timestamp)
        label = float(row.label)
        rating = float(getattr(row, 'rating', 1.0 if label > 0.5 else -1.0))
        edge_type = int(getattr(row, 'edge_type', 0))
        sample_type = int(getattr(row, 'sample_type', 0))
        obs = float(getattr(row, 'generated_observed'))
        history_visible = float(getattr(row, 'history_visible_mask', 1.0))
        node_s = node_hist[src]
        node_d = node_hist[dst]
        pair = pair_hist.setdefault((src, dst), PairHistory())
        dt = 0.0 if zero_time_features else timestamp - pair.last_time if pair.last_time is not None else 1.0
        dt = 0.0 if zero_time_features else max(float(dt), 1.0)
        time_norm = 0.0 if zero_time_features else (timestamp - min_t) / time_span
        src_ratio = _ratio(node_s.pos_out, node_s.neg_out)
        dst_ratio = _ratio(node_d.pos_in, node_d.neg_in)
        src_span = 0.0 if zero_time_features or node_s.first_time is None else (timestamp - node_s.first_time) / time_span
        dst_span = 0.0 if zero_time_features or node_d.first_time is None else (timestamp - node_d.first_time) / time_span
        pair_ratio = _ratio(pair.pos, pair.neg)
        variance = float(np.var(pair.ratings)) if len(pair.ratings) > 1 else 0.0
        recent_neg = 1.0 - float(np.mean(global_recent[-100:])) if global_recent else 0.5
        anomaly = abs(pair.last_label - src_ratio)
        historical_rating_drift = abs((pair.ratings[-1] if pair.ratings else 0.0) - (np.mean(pair.ratings) if pair.ratings else 0.0))
        bucket = 0 if zero_time_features else int(10 * time_norm)
        time_sin = 0.0 if zero_time_features else 0.5 + 0.5 * np.sin(2.0 * np.pi * time_norm)
        time_cos = 0.0 if zero_time_features else 0.5 + 0.5 * np.cos(2.0 * np.pi * time_norm)
        entity = [src_ratio, dst_ratio, _log_norm(node_s.activity), _log_norm(node_d.activity), _log_norm(node_s.pos_out + node_s.neg_out), _log_norm(node_d.pos_in + node_d.neg_in), float(np.clip(src_span, 0.0, 1.0)), float(np.clip(dst_span, 0.0, 1.0))]
        behavior = [pair.last_label, pair_ratio, _log_norm(pair.count), float(np.clip(variance / 100.0, 0.0, 1.0)), float(np.clip(historical_rating_drift / 20.0, 0.0, 1.0)), 1.0 - pair_ratio, float(np.clip(anomaly, 0.0, 1.0)), recent_neg]
        context = [edge_type / 4.0, time_norm, time_sin, time_cos, float(np.clip(dt / (time_span + 1.0), 0.0, 1.0)), _log_norm(node_s.activity + node_d.activity, 20.0), recent_neg, bucket / 10.0]
        for col in extra_context_cols:
            context.append(float(getattr(row, col)))
        if metapath_context_features is not None:
            src_meta = metapath_context_features[src]
            dst_meta = metapath_context_features[dst]
            context.extend(src_meta.tolist())
            context.extend(dst_meta.tolist())
            context.extend((src_meta * dst_meta).tolist())
        event_entity_meta: list[float] | None = None
        event_behavior_meta: list[float] | None = None
        event_context_meta: list[float] | None = None
        if metapath_emb is not None:
            m_src = metapath_emb[src]
            m_dst = metapath_emb[dst]
            m_had = m_src * m_dst
            if use_entity_metapath:
                event_entity_meta = m_src.tolist()
                event_entity_meta.extend((float(getattr(row, col)) for col in extra_entity_meta_cols))
            elif extra_entity_meta_cols:
                event_entity_meta = [float(getattr(row, col)) for col in extra_entity_meta_cols]
            if use_context_metapath:
                event_context_meta = np.concatenate([m_src, m_dst]).tolist()
                event_context_meta.extend((float(getattr(row, col)) for col in extra_context_meta_cols))
            elif extra_context_meta_cols:
                event_context_meta = [float(getattr(row, col)) for col in extra_context_meta_cols]
            if use_behavior_metapath:
                event_behavior_meta = m_had.tolist() + [float(m_had.sum()), _safe_cos(m_src, m_dst)]
                event_behavior_meta.extend((float(getattr(row, col)) for col in extra_behavior_meta_cols))
            elif extra_behavior_meta_cols:
                event_behavior_meta = [float(getattr(row, col)) for col in extra_behavior_meta_cols]
        else:
            if extra_entity_meta_cols:
                event_entity_meta = [float(getattr(row, col)) for col in extra_entity_meta_cols]
            if extra_behavior_meta_cols:
                event_behavior_meta = [float(getattr(row, col)) for col in extra_behavior_meta_cols]
            if extra_context_meta_cols:
                event_context_meta = [float(getattr(row, col)) for col in extra_context_meta_cols]
        edge = [edge_type / 4.0, float(np.clip(dt / (time_span + 1.0), 0.0, 1.0)), time_norm, _log_norm(pair.count), obs, recent_neg]
        for col in extra_edge_cols:
            edge.append(float(getattr(row, col)))
        if append:
            src_arr.append(src)
            dst_arr.append(dst)
            edge_types.append(edge_type)
            timestamps.append(timestamp)
            observed.append(obs)
            delta_ts.append(dt)
            entity_features.append(entity)
            behavior_features.append(behavior)
            context_features.append(context)
            if event_entity_meta is not None:
                entity_meta.append(event_entity_meta)
            if event_behavior_meta is not None:
                behavior_meta.append(event_behavior_meta)
            if event_context_meta is not None:
                context_meta.append(event_context_meta)
            edge_features.append(edge)
            labels.append(label)
            sample_types.append(sample_type)
            history_visible_masks.append(history_visible)
        should_update = sample_type == 0 and allow_update and (history_visible > 0.5) and (not respect_observed_history or obs > 0.5)
        if should_update:
            if node_s.first_time is None:
                node_s.first_time = timestamp
            if node_d.first_time is None:
                node_d.first_time = timestamp
            if label > 0.5:
                node_s.pos_out += 1
                node_d.pos_in += 1
                pair.pos += 1
            else:
                node_s.neg_out += 1
                node_d.neg_in += 1
                pair.neg += 1
            node_s.last_time = timestamp
            node_d.last_time = timestamp
            pair.count += 1
            pair.last_time = timestamp
            pair.last_label = label
            pair.ratings.append(rating)
            global_recent.append(label)
    if 'split' not in events.columns:
        for row in events.itertuples(index=False):
            handle(row, append=True, allow_update=True)
    else:
        train = events[events['split'] == 0].sort_values('timestamp', kind='mergesort')
        val = events[events['split'] == 1].sort_values('timestamp', kind='mergesort')
        test = events[events['split'] == 2].sort_values('timestamp', kind='mergesort')
        for row in train.itertuples(index=False):
            handle(row, append=True, allow_update=True)
        train_state = snapshot()
        for row in val.itertuples(index=False):
            handle(row, append=True, allow_update=False)
        restore(train_state)
        for row in val.itertuples(index=False):
            handle(row, append=False, allow_update=True)
        train_val_state = snapshot()
        restore(train_val_state)
        for row in test.itertuples(index=False):
            handle(row, append=True, allow_update=False)
    if node_features_override is not None:
        node_features = np.asarray(node_features_override, dtype=np.float32)
        if node_features.shape[0] != num_nodes:
            raise ValueError(f'node_features_override has {node_features.shape[0]} rows, expected {num_nodes}')
    else:
        node_features = build_node_features(node_hist, num_nodes, policy=node_feature_policy, seed=seed, dim=node_feature_dim)
    out = {'src': np.asarray(src_arr, dtype=np.int64), 'dst': np.asarray(dst_arr, dtype=np.int64), 'edge_type': np.asarray(edge_types, dtype=np.int64), 'timestamp': np.asarray(timestamps, dtype=np.float32), 'observed_mask': np.asarray(observed, dtype=np.float32), 'delta_t': np.asarray(delta_ts, dtype=np.float32), 'entity_features': np.asarray(entity_features, dtype=np.float32), 'behavior_features': np.asarray(behavior_features, dtype=np.float32), 'context_features': np.asarray(context_features, dtype=np.float32), 'edge_features': np.asarray(edge_features, dtype=np.float32), 'label': np.asarray(labels, dtype=np.float32), 'sample_type': np.asarray(sample_types, dtype=np.int64), 'history_visible_mask': np.asarray(history_visible_masks, dtype=np.float32), 'node_features': node_features.astype(np.float32), 'num_nodes': np.asarray([num_nodes], dtype=np.int64), 'num_edge_types': np.asarray([max(edge_types) + 1 if edge_types else 1], dtype=np.int64)}
    if metapath_emb is not None:
        out['metapath_node_features'] = metapath_emb.astype(np.float32)
    if use_entity_meta:
        out['entity_meta'] = np.asarray(entity_meta, dtype=np.float32)
    if use_behavior_meta:
        out['behavior_meta'] = np.asarray(behavior_meta, dtype=np.float32)
    if use_context_meta:
        out['context_meta'] = np.asarray(context_meta, dtype=np.float32)
    return out

def build_node_features(node_hist: list[NodeHistory], num_nodes: int, policy: str='constant', seed: int=42, dim: int=16) -> np.ndarray:
    del node_hist
    dim = max(2, int(dim))
    if policy == 'random_identity':
        rng = np.random.default_rng(int(seed) + 7919)
        base = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(num_nodes, dim)).astype(np.float32)
        base[:, -1] = 1.0
        return base
    if policy == 'one_hot':
        return np.eye(num_nodes, dtype=np.float32)
    if policy != 'constant':
        raise ValueError(f'Unsupported node feature policy: {policy}')
    base = np.zeros((num_nodes, dim), dtype=np.float32)
    base[:, -1] = 1.0
    return base
