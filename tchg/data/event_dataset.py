from __future__ import annotations

from dataclasses import dataclass
from copy import copy
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from tchg.data.temporal_split import temporal_split_indices
from tchg.data.user_regimes import with_user_regime_masks
from tchg.utils.io import resolve_path


@dataclass
class EventGraphDataset:


    path: str | Path

    def __post_init__(self) -> None:

        with np.load(resolve_path(self.path), allow_pickle=False) as loaded:
            self.arrays: Dict[str, np.ndarray] = {k: loaded[k] for k in loaded.files}
        if 'split' in self.arrays:
            self.arrays = with_user_regime_masks(self.arrays)

    @classmethod
    def from_arrays(cls, arrays: Dict[str, np.ndarray]) -> "EventGraphDataset":
        instance = cls.__new__(cls)
        instance.path = Path()
        instance.arrays = {key: value.copy() for key, value in arrays.items()}
        if 'split' in instance.arrays:
            instance.arrays = with_user_regime_masks(instance.arrays)
        return instance

    def with_user_regimes(self, train_indices) -> "EventGraphDataset":
        instance = copy(self)
        instance.arrays = with_user_regime_masks(self.arrays, train_indices)
        return instance

    @property
    def num_nodes(self) -> int:

        return int(self.arrays["num_nodes"][0])

    @property
    def num_edge_types(self) -> int:

        return int(self.arrays["num_edge_types"][0])

    @property
    def feature_dims(self) -> dict[str, int]:

        return {
            "num_nodes": self.num_nodes,
            "node_dim": int(self.arrays["node_features"].shape[1]),
            "entity_dim": int(self.arrays["entity_features"].shape[1]),
            "behavior_dim": int(self.arrays["behavior_features"].shape[1]),
            "context_dim": int(self.arrays["context_features"].shape[1]),
            "edge_dim": int(self.arrays["edge_features"].shape[1]),
            "entity_meta_dim": int(self.arrays["entity_meta"].shape[1]) if "entity_meta" in self.arrays else 0,
            "behavior_meta_dim": int(self.arrays["behavior_meta"].shape[1]) if "behavior_meta" in self.arrays else 0,
            "context_meta_dim": int(self.arrays["context_meta"].shape[1]) if "context_meta" in self.arrays else 0,
            "rating_profile_dim": int(self.arrays['rating_profile_src'].shape[1]) if 'rating_profile_src' in self.arrays else 0,
            "metapath_node_dim": int(self.arrays["metapath_node_features"].shape[1]) if "metapath_node_features" in self.arrays else 0,
        }

    def split_indices(self, val_ratio: float, test_ratio: float) -> dict[str, np.ndarray]:

        if "split" in self.arrays:
            split = self.arrays["split"].astype(np.int64)
            return {
                "train": np.where(split == 0)[0],
                "val": np.where(split == 1)[0],
                "test": np.where(split == 2)[0],
            }
        return temporal_split_indices(self.arrays["timestamp"], val_ratio=val_ratio, test_ratio=test_ratio)

    def subset(self, indices: np.ndarray) -> dict[str, np.ndarray]:

        event_keys = {
            "src",
            "dst",
            "edge_type",
            "timestamp",
            "observed_mask",
            "delta_t",
            "entity_features",
            "behavior_features",
            "context_features",
            "edge_features",
            "label",
            "sample_type",
        }
        optional_event_keys = {"split", "is_new_node", "observed_user_mask", "unobserved_user_mask", "history_visible_mask", "entity_meta", "behavior_meta", "context_meta", 'rating_profile_src', 'rating_profile_dst'}
        event_keys |= {key for key in optional_event_keys if key in self.arrays}
        out = {"node_features": self.arrays["node_features"], "num_nodes": self.arrays["num_nodes"], "num_edge_types": self.arrays["num_edge_types"]}
        if "metapath_node_features" in self.arrays:
            out["metapath_node_features"] = self.arrays["metapath_node_features"]
        for key in event_keys:
            out[key] = self.arrays[key][indices]
        order = np.argsort(out["timestamp"], kind="mergesort")
        for key in event_keys:
            out[key] = out[key][order]
        return out

    def to_torch(self, arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:

        int_keys = {"src", "dst", "edge_type", "sample_type", "split", "is_new_node", "observed_user_mask", "unobserved_user_mask", "num_nodes", "num_edge_types"}
        batch: dict[str, torch.Tensor] = {}
        for key, value in arrays.items():
            dtype = torch.long if key in int_keys else torch.float32
            batch[key] = torch.as_tensor(value, dtype=dtype, device=device)
        return batch

    def full_batch(self, device: torch.device) -> dict[str, torch.Tensor]:

        return self.to_torch(self.subset(np.arange(len(self.arrays["src"]))), device)
