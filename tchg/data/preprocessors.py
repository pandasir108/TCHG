from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tchg.data.feature_builder import build_event_features_cat_rolling
from tchg.data.negative_sampling import sample_negative_edges
from tchg.data.user_regimes import USER_REGIME_DEFINITIONS, derive_user_regime_masks
from tchg.utils.io import ensure_dir, resolve_path, save_json

def _node_mapping(src_values: pd.Series, dst_values: pd.Series) -> tuple[dict[int, int], list[int]]:
    raw = pd.concat([src_values, dst_values], ignore_index=True).astype(np.int64)
    raw_nodes = pd.Index(raw.unique())
    raw_to_node = {int(node): idx for idx, node in enumerate(raw_nodes)}
    node_to_raw = [int(node) for node in raw_nodes]
    return (raw_to_node, node_to_raw)

def read_epinions_raw(raw_root: str | Path='data/raw') -> pd.DataFrame:
    raw_dir = resolve_path(raw_root) / 'epinions'
    trust_path = raw_dir / 'user_user_time.txt'
    if not trust_path.exists():
        raise FileNotFoundError(f'Missing Epinions trust file: {trust_path}')
    df = pd.read_csv(trust_path, sep='\\s+', header=None, names=['src_raw', 'dst_raw', 'timestamp'], engine='python')
    df = df.dropna()
    df['src_raw'] = df['src_raw'].astype(np.int64)
    df['dst_raw'] = df['dst_raw'].astype(np.int64)
    df['timestamp'] = df['timestamp'].astype(np.int64)
    df['rating'] = 1.0
    df['label'] = 1.0
    df['edge_type'] = 0
    df['sample_type'] = 0
    raw_to_node, node_to_raw = _node_mapping(df['src_raw'], df['dst_raw'])
    df['src'] = df['src_raw'].map(raw_to_node).astype(np.int64)
    df['dst'] = df['dst_raw'].map(raw_to_node).astype(np.int64)
    num_nodes = len(node_to_raw)
    df.attrs['num_nodes'] = num_nodes
    df.attrs['raw_to_node'] = raw_to_node
    df.attrs['node_to_raw'] = node_to_raw
    return df[['src', 'dst', 'src_raw', 'dst_raw', 'rating', 'timestamp', 'label', 'edge_type', 'sample_type']]

def _add_cat_protocol_negatives(df: pd.DataFrame, num_nodes: int, node_to_raw: list[int], negative_ratio: float, seed: int, exclusion_scope: str='cat_context', candidate_node_scope: str='all') -> pd.DataFrame:
    if negative_ratio <= 0.0:
        return df
    rng = np.random.default_rng(seed)
    frames = [df]
    if exclusion_scope == 'full':
        full_positive = df.loc[df['sample_type'] == 0, ['src', 'dst']]
        split_exclusion = {0: full_positive, 1: full_positive, 2: full_positive}
    else:
        split_exclusion = {0: df.loc[df['split'] == 0, ['src', 'dst']], 1: df.loc[df['split'] == 0, ['src', 'dst']], 2: df.loc[df['split'].isin([0, 1]), ['src', 'dst']]}
    candidate_nodes = None
    if candidate_node_scope == 'trust_active':
        candidate_nodes = np.unique(np.concatenate([df['src'].to_numpy(), df['dst'].to_numpy()])).astype(np.int64)
    elif candidate_node_scope != 'all':
        raise ValueError(f'Unsupported negative candidate node scope: {candidate_node_scope}')
    for split_id in (0, 1, 2):
        positives = df[df['split'] == split_id]
        count = int(round(len(positives) * negative_ratio))
        if count == 0:
            continue
        blocked = {(int(s), int(d)) for s, d in zip(split_exclusion[split_id]['src'], split_exclusion[split_id]['dst'])}
        neg_edges = sample_negative_edges(num_nodes, blocked, count, rng, candidate_nodes=candidate_nodes)
        matched_idx = rng.integers(0, len(positives), size=count)
        timestamps = positives['timestamp'].to_numpy()[matched_idx]
        src_raw = [node_to_raw[int(node)] for node in neg_edges[:, 0]]
        dst_raw = [node_to_raw[int(node)] for node in neg_edges[:, 1]]
        neg = pd.DataFrame({'src': neg_edges[:, 0], 'dst': neg_edges[:, 1], 'src_raw': src_raw, 'dst_raw': dst_raw, 'rating': -1.0, 'timestamp': timestamps, 'label': 0.0, 'edge_type': 0, 'sample_type': 1, 'split': split_id})
        frames.append(neg)
    return pd.concat(frames, ignore_index=True)

def prepare_cat_full_epinions(df: pd.DataFrame, cfg: dict, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    val_ratio = float(cfg.get('cat_val_time_ratio', cfg.get('val_ratio', 0.7)))
    test_ratio = float(cfg.get('cat_test_time_ratio', 0.85))
    negative_ratio = float(cfg.get('negative_sampling_ratio', 1.0))
    num_nodes = int(df.attrs['num_nodes'])
    node_to_raw = list(df.attrs['node_to_raw'])
    ts = df['timestamp'].to_numpy()
    val_time, test_time = list(np.quantile(ts, [val_ratio, test_ratio]))
    df = df.copy()
    df['split'] = np.where(df['timestamp'] <= val_time, 0, np.where(df['timestamp'] <= test_time, 1, 2)).astype(np.int64)
    train_nodes = set(df.loc[df['split'] == 0, 'src'].astype(int)).union(set(df.loc[df['split'] == 0, 'dst'].astype(int)))
    df = _add_cat_protocol_negatives(df, num_nodes, node_to_raw, negative_ratio, seed, exclusion_scope=str(cfg.get('negative_exclusion_scope', 'cat_context')), candidate_node_scope=str(cfg.get('negative_candidate_node_scope', 'all')))
    for key, mask in derive_user_regime_masks(df).items():
        df[key] = mask
    metadata = {'cat_val_time': float(val_time), 'cat_test_time': float(test_time), 'cat_train_positive_edges': int((df['split'].eq(0) & df['sample_type'].eq(0)).sum()), 'cat_val_positive_edges': int((df['split'].eq(1) & df['sample_type'].eq(0)).sum()), 'cat_test_positive_edges': int((df['split'].eq(2) & df['sample_type'].eq(0)).sum()), 'cat_new_nodes': int(num_nodes - len(train_nodes)), 'negative_sampling_ratio': float(negative_ratio), 'negative_candidate_node_scope': str(cfg.get('negative_candidate_node_scope', 'all'))}
    return (df, metadata)

def prepare_epinions_full_protocol(df: pd.DataFrame, cfg: dict, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    protocol_cfg = {**cfg, 'cat_val_time_ratio': cfg.get('val_time_ratio', cfg.get('cat_val_time_ratio', 0.7)), 'cat_test_time_ratio': cfg.get('test_time_ratio', cfg.get('cat_test_time_ratio', 0.85)), 'negative_exclusion_scope': cfg.get('negative_exclusion_scope', 'full')}
    df, metadata = prepare_cat_full_epinions(df, protocol_cfg, seed)
    metadata = {'protocol': 'unified_full_epinions', 'val_time': metadata['cat_val_time'], 'test_time': metadata['cat_test_time'], 'train_positive_edges': metadata['cat_train_positive_edges'], 'val_positive_edges': metadata['cat_val_positive_edges'], 'test_positive_edges': metadata['cat_test_positive_edges'], 'new_nodes': metadata['cat_new_nodes'], 'negative_sampling_ratio': metadata['negative_sampling_ratio'], 'negative_exclusion_scope': str(protocol_cfg.get('negative_exclusion_scope', 'full'))}
    return (df, metadata)

def prepare_epinions_full_cat_rolling_protocol(df: pd.DataFrame, cfg: dict, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    rolling_cfg = {**cfg, 'negative_exclusion_scope': cfg.get('negative_exclusion_scope', 'full')}
    df, metadata = prepare_epinions_full_protocol(df, rolling_cfg, seed)
    metadata = {**metadata, 'protocol': 'cat_rolling_full_epinions', 'history_update_policy': 'val_uses_train_test_uses_train_val_no_eval_internal_update'}
    return (df, metadata)

def preprocess_dataset(dataset: str, cfg: dict, seed: int = 42) -> Path:
    name = dataset.lower()
    supported = {"epinions_full_cat_rolling", "epinions_full_cat_rolling_80_10_10"}
    if name not in supported:
        raise ValueError(f"Unsupported dataset: {dataset}")
    data_cfg = dict(cfg.get("dataset", {}))
    raw_root = data_cfg.get("raw_dir", "data/raw")
    output_dir = ensure_dir(Path(data_cfg.get("processed_dir", "data/processed")) / name)
    output_path = output_dir / "events.npz"
    df = read_epinions_raw(raw_root)
    node_to_raw = np.asarray(df.attrs['node_to_raw'], dtype=np.int64)
    df, metadata = prepare_epinions_full_cat_rolling_protocol(df, data_cfg, seed)
    df = df.sort_values("timestamp", kind="mergesort")
    max_events = data_cfg.get("max_events")
    if max_events:
        df = df.head(int(max_events))
    num_nodes = len(node_to_raw)
    arrays = build_event_features_cat_rolling(
        df,
        num_nodes=num_nodes,
        p_missing=float(data_cfg.get("p_missing", 0.0)),
        seed=seed,
        respect_observed_history=bool(data_cfg.get("respect_observed_history", False)),
        history_invisible_rate=float(data_cfg.get("history_invisible_rate", 0.0)),
        zero_time_features=bool(data_cfg.get("zero_trust_time_features", False)),
        node_feature_policy=str(data_cfg.get("node_feature_policy", "constant")),
        node_feature_dim=int(data_cfg.get("node_feature_dim", 16)),
    )
    ordered_parts = []
    for split_id in (0, 1, 2):
        part = df[df["split"] == split_id].sort_values("timestamp", kind="mergesort")
        part.attrs = {}
        ordered_parts.append(part)
    ordered = pd.concat(ordered_parts, ignore_index=True)
    for optional in ("split", "is_new_node", "observed_user_mask", "unobserved_user_mask"):
        arrays[optional] = ordered[optional].to_numpy(dtype=np.int64)
    arrays['node_to_raw'] = node_to_raw
    np.savez_compressed(output_path, **arrays)
    save_json(
        {
            "dataset": name,
            "num_events": int(arrays["src"].shape[0]),
            "num_nodes": int(arrays["num_nodes"][0]),
            "num_edge_types": int(arrays["num_edge_types"][0]),
            "node_dim": int(arrays["node_features"].shape[1]),
            "entity_dim": int(arrays["entity_features"].shape[1]),
            "behavior_dim": int(arrays["behavior_features"].shape[1]),
            "context_dim": int(arrays["context_features"].shape[1]),
            "edge_dim": int(arrays["edge_features"].shape[1]),
            "user_regime_definitions": USER_REGIME_DEFINITIONS,
            **metadata,
        },
        output_dir / "metadata.json",
    )
    return output_path
