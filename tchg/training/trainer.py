from __future__ import annotations

import time
from copy import copy, deepcopy
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm.auto import tqdm

from tchg.data.event_dataset import EventGraphDataset
from tchg.data.user_regimes import training_reference_users
from tchg.models.factory import make_model
from tchg.training.early_stopping import EarlyStopping
from tchg.training.losses import trust_calibration_loss
from tchg.training.metrics import compute_metrics, parameter_count, validation_target_metrics
from tchg.utils.device import get_device, gpu_memory_mb
from tchg.utils.io import ensure_dir, resolve_path


class Trainer:


    def __init__(self, cfg: Dict, dataset: EventGraphDataset, model_name: str | None = None, device: str | None = None):
        self.cfg = deepcopy(cfg)
        if model_name is not None:
            self.cfg["model"]["name"] = model_name
        data_cfg = self.cfg['dataset']
        self.indices = dataset.split_indices(float(data_cfg.get("val_ratio", 0.2)), float(data_cfg.get("test_ratio", 0.2)))
        dataset = dataset.with_user_regimes(self.indices['train'])
        if data_cfg.get('static_training_features', False):
            from tchg.data.static_features import static_training_features
            dataset = copy(dataset)
            dataset.arrays = {**dataset.arrays, **static_training_features(dataset.arrays)}
        if data_cfg.get('static_structural_features', False):
            from tchg.data.static_features import static_structural_features
            features = static_structural_features(dataset.arrays)
            dataset = copy(dataset)
            dataset.arrays = dict(dataset.arrays)
            for key, value in features.items():
                dataset.arrays[key] = np.concatenate([dataset.arrays[key], value], axis=1)
        if data_cfg.get('static_diffusion_features', False):
            from tchg.data.static_diffusion import static_diffusion_features
            features = static_diffusion_features(dataset.arrays,
                damping=data_cfg.get('static_diffusion_damping', [.5, .85]),
                device=get_device(device or self.cfg.get('device', 'auto')))
            dataset = copy(dataset)
            dataset.arrays = dict(dataset.arrays)
            for key, value in features.items():
                dataset.arrays[key] = np.concatenate([dataset.arrays[key], value], axis=1)
        if data_cfg.get('causal_structure_features', False) and not getattr(dataset, '_has_causal_structure', False):
            from tchg.data.structural_features import causal_structural_features
            cache = data_cfg.get('structural_feature_cache')
            features = causal_structural_features(dataset.arrays, resolve_path(cache) if cache else None)
            dataset = copy(dataset)
            dataset.arrays = dict(dataset.arrays)
            dataset.arrays['behavior_features'] = np.concatenate([dataset.arrays['behavior_features'], features[:, :2]], axis=1)
            dataset.arrays['context_features'] = np.concatenate([dataset.arrays['context_features'], features[:, 2:]], axis=1)
            dataset._has_causal_structure = True
        if data_cfg.get('causal_rating_features', False) and not getattr(dataset, '_has_causal_ratings', False):
            from tchg.data.rating_features import load_causal_rating_features
            features, metadata = load_causal_rating_features(dataset.arrays, data_cfg.get('raw_dir', 'data/raw'),
                data_cfg.get('rating_feature_cache'), data_cfg.get('rating_history_policy', 'split_frozen'),
                int(data_cfg.get('rating_profile_dim', 0)), float(data_cfg.get('rating_profile_window_days', 365.)),
                bool(data_cfg.get('rating_temporal_activity', False)), int(data_cfg.get('rating_profile_seed', 42)))
            dataset = copy(dataset); dataset.arrays = dict(dataset.arrays)
            for key, value in features.items():
                dataset.arrays[key] = np.concatenate([dataset.arrays[key], value], axis=1) if key in dataset.arrays else value
            dataset._has_causal_ratings = True
            dataset._causal_rating_metadata = metadata
        self.dataset = dataset
        self.device = get_device(device or self.cfg.get("device", "auto"))
        data_cfg = self.cfg["dataset"]
        self.batches = {k: dataset.to_torch(dataset.subset(v), self.device) for k, v in self.indices.items()}
        self.model = make_model(self.cfg["model"]["name"], dataset.feature_dims, dataset.num_edge_types, self.cfg["model"]).to(self.device)
        self.static_graph = self.model.operator_stack.static_mode
        if self.cfg['training'].get('exclude_static_query_edges', False) and (
                not self.static_graph or self.cfg['training'].get('batch_mode') != 'event_stream'):
            raise ValueError('Static query exclusion requires static_train with event_stream minibatches')
        if self.static_graph:
            train = self.batches['train']
            eligible = (train['sample_type'] == 0) & (train['label'] > .5) & (train['observed_mask'] > .5)
            if 'history_visible_mask' in train:
                eligible &= train['history_visible_mask'] > .5
            globals_ = {'node_features', 'metapath_node_features', 'num_nodes', 'num_edge_types'}
            self.static_history = {k: v if k in globals_ else v[eligible] for k, v in train.items()}
        if self.model.use_user_identity:
            self.model.set_identity_reference_users(training_reference_users(dataset.arrays, self.indices['train']))
        self.model.fit_input_statistics(self.batches['train'])
        self.posthoc_calibration: dict[str, float] | None = None
        identity_params = self.model.identity_parameters()
        identity_ids = {id(p) for p in identity_params}
        groups = [{'params': [p for p in self.model.parameters() if id(p) not in identity_ids]}]
        if identity_params:
            groups.append({'params': identity_params,
                           'lr': float(self.cfg['training'].get('lr', 1e-3)) * float(self.cfg['training'].get('user_identity_lr_multiplier', 1.))})
        self.optim = torch.optim.AdamW(
            groups,
            lr=float(self.cfg["training"].get("lr", 1e-3)),
            weight_decay=float(self.cfg["training"].get("weight_decay", 1e-4)),
        )
        self.rank_hard_negative_masks: dict[str, torch.Tensor] = {}

    def _start_ema(self):
        decay = float(self.cfg['training'].get('ema_decay', 0.))
        if not 0. <= decay < 1.:
            raise ValueError('ema_decay must be in [0, 1)')
        self._ema_decay = decay
        self._ema_state = deepcopy(self.model.state_dict()) if decay else None

    @torch.no_grad()
    def _update_ema(self):
        if getattr(self, '_ema_state', None) is None:
            return
        for key, value in self.model.state_dict().items():
            if value.is_floating_point():
                self._ema_state[key].mul_(self._ema_decay).add_(value, alpha=1. - self._ema_decay)
            else:
                self._ema_state[key].copy_(value)

    @contextmanager
    def _validation_weights(self):
        """Score/save EMA weights while retaining raw weights for optimization."""
        if getattr(self, '_ema_state', None) is None:
            yield
            return
        raw = deepcopy(self.model.state_dict())
        try:
            self.model.load_state_dict(self._ema_state)
            yield
        finally:
            self.model.load_state_dict(raw)

    def _event_count(self, batch: dict[str, torch.Tensor]) -> int:

        return int(batch["src"].shape[0])

    def _slice_batch(self, batch: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:

        n = self._event_count(batch)
        out: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if key in {"node_features", "metapath_node_features", "num_nodes", "num_edge_types"}:
                out[key] = value
            elif value.ndim > 0 and value.shape[0] == n:
                out[key] = value[start:end]
            else:
                out[key] = value
        return out

    def _chunks(self, batch):
        """Never split a simultaneous timestamp group between model calls."""
        size = int(self.cfg['training'].get('stream_batch_size', 4096))
        if size <= 0:
            raise ValueError('stream_batch_size must be positive')
        times = batch['timestamp'].detach().cpu().numpy()
        start, n = 0, self._event_count(batch)
        while start < n:
            end = min(start + size, n)
            if end < n and not getattr(self, 'static_graph', False):
                end = int(np.searchsorted(times, times[end - 1], side='right'))
            yield self._slice_batch(batch, start, end)
            start = end

    def _history_batch(self, batch, write):
        out = dict(batch)
        if getattr(self, 'static_graph', False):
            write = False  # The training graph is primed separately.
        mask = torch.ones_like(batch['observed_mask']) if write else torch.zeros_like(batch['observed_mask'])
        if 'sample_type' in batch:
            mask = mask * (batch['sample_type'] == 0)
        if 'history_visible_mask' in batch:
            mask = mask * (batch['history_visible_mask'] > .5)
        out['history_write_mask'] = mask
        return out

    def _reset_and_prime(self, split, query=None):
        self.model.reset_memory()
        if getattr(self, 'static_graph', False):
            history = self.static_history
            if self.model.training and split == 'train':
                cfg = self.cfg['training']
                probability = float(cfg.get('static_history_dropout', 0.))
                if not 0. <= probability < 1.:
                    raise ValueError('static_history_dropout must be in [0, 1)')
                exclude = bool(cfg.get('exclude_static_query_edges', False)) and query is not None
                if probability or exclude:
                    keep = torch.ones(len(history['src']), dtype=torch.bool, device=self.device)
                    if exclude:
                        nodes = int(history['num_nodes'][0])
                        historical = history['src'] * nodes + history['dst']
                        candidates = query['src'] * nodes + query['dst']
                        keep &= ~torch.isin(historical, candidates)
                    if probability:
                        keep &= torch.rand(keep.shape, device=self.device) >= probability
                    globals_ = {'node_features', 'metapath_node_features', 'num_nodes', 'num_edge_types'}
                    history = {k: v if k in globals_ else v[keep] for k, v in history.items()}
            with torch.set_grad_enabled(self.model.training and split == 'train'):
                self.model.prime_static_history(history)
            return
        if split == 'train':
            return
        # Match the existing prepared-data protocol: validation sees training
        # history; test sees training plus observed validation events. Query
        # outcomes within the evaluated split never enter the history graph.
        history_splits = ['train'] + (['val'] if split == 'test' else [])
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for name in history_splits:
                for chunk in self._chunks(self._history_batch(self.batches[name], True)):
                    self.model(chunk)
                    self.model.detach_memory()
        self.model.train(was_training)

    def _concat_metric_batches(self, rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:

        keys = {key for row in rows for key in row}
        merged: dict[str, torch.Tensor] = {}
        for key in keys:
            values = [row[key] for row in rows if key in row]
            merged[key] = torch.cat(values, dim=0)
        return merged

    def _rank_hard_negative_enabled(self) -> bool:

        train_cfg = self.cfg["training"]
        mode = str(train_cfg.get("rank_negative_mining", "")).lower()
        return mode in {"previous_topk", "topk", "hard_topk"} or bool(train_cfg.get("rank_hard_negative_mining", False))

    def _attach_rank_hard_negatives(self, split: str, batch: dict[str, torch.Tensor], train: bool) -> dict[str, torch.Tensor]:

        if not train or not self._rank_hard_negative_enabled() or split not in self.rank_hard_negative_masks:
            return batch
        mask = self.rank_hard_negative_masks[split].to(batch["label"].device)
        if mask.shape[0] != batch["label"].shape[0]:
            return batch
        out = dict(batch)
        out["rank_negative_mask"] = mask
        return out

    def _update_rank_hard_negatives(self, split: str, labels: torch.Tensor, logits: torch.Tensor) -> None:

        if split != "train" or not self._rank_hard_negative_enabled():
            return
        labels = labels.detach()
        logits = logits.detach()
        neg_idx = torch.nonzero(labels <= 0.5, as_tuple=False).flatten()
        if neg_idx.numel() == 0:
            self.rank_hard_negative_masks.pop(split, None)
            return
        train_cfg = self.cfg["training"]
        topk = int(train_cfg.get("rank_hard_negative_topk", 0))
        if topk <= 0:
            fraction = float(train_cfg.get("rank_hard_negative_fraction", 0.25))
            topk = int(round(float(neg_idx.numel()) * fraction))
        topk = max(1, min(int(topk), int(neg_idx.numel())))
        hard_local = torch.topk(logits[neg_idx], k=topk, largest=True).indices
        hard_idx = neg_idx[hard_local]
        mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
        mask[hard_idx] = True
        self.rank_hard_negative_masks[split] = mask.detach().cpu()

    def _metric_threshold(self, threshold: float | None = None) -> float:

        if threshold is not None:
            return float(threshold)
        return float(self.cfg["training"].get("threshold", 0.5))

    def _metric_batch(self, chunk: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        metric_batch = {
            "label": chunk["label"].detach().cpu(),
            "delta_t": chunk["delta_t"].detach().cpu(),
            "observed_mask": chunk["observed_mask"].detach().cpu(),
            "timestamp": chunk['timestamp'].detach().cpu(),
        }
        for key in ('observed_user_mask', 'unobserved_user_mask', 'is_new_node'):
            if key in chunk:
                metric_batch[key] = chunk[key].detach().cpu()
        return metric_batch

    def _validation_targets(self, metrics, pred, batch, split):
        cfg = self.cfg['training']
        if split == 'val' and cfg.get('observed_user_mrr_target') is not None:
            metrics.update(validation_target_metrics(pred, batch,
                float(cfg['observed_user_mrr_target']), float(cfg['unobserved_user_mrr_target']),
                float(cfg.get('validation_confirmation_fraction', .5))))
        return metrics

    def _paper_score(self, metrics: dict[str, float]) -> float:

        return (
            float(metrics.get("auc", 0.0))
            + 0.75 * float(metrics.get("f1", 0.0))
            + 0.25 * float(metrics.get("accuracy", 0.0))
            + 0.25 * float(metrics.get("mrr", 0.0))
            - 0.25 * float(metrics.get("ece", 0.0))
            - 0.10 * float(metrics.get("brier_score", 0.0))
        )

    def _apply_posthoc_calibration(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        if self.posthoc_calibration is None or "trust_logit" not in pred:
            return pred
        scale = float(self.posthoc_calibration.get("scale", 1.0))
        bias = float(self.posthoc_calibration.get("bias", 0.0))
        out = dict(pred)
        logit = out["trust_logit"] * scale + bias
        out["trust_logit"] = logit
        out["trust_pred"] = torch.sigmoid(logit)
        return out

    def _step_stream(self, split: str, train: bool, threshold: float | None = None) -> tuple[float, dict[str, float], dict[str, float]]:

        batch = self._history_batch(self._attach_rank_hard_negatives(split, self.batches[split], train), train)
        self.model.train(train)
        if not (getattr(self, 'static_graph', False) and train):
            self._reset_and_prime(split)
        n = self._event_count(batch)
        total_loss = 0.0
        total_weight = 0
        parts_sum: dict[str, float] = {}
        pred_rows: list[dict[str, torch.Tensor]] = []
        batch_rows: list[dict[str, torch.Tensor]] = []
        with torch.set_grad_enabled(train):
            for chunk in self._chunks(batch):
                if getattr(self, 'static_graph', False) and train:
                    # Rebuild after each optimizer step; never reuse an old
                    # autograd graph or turn minibatch order into history.
                    self._reset_and_prime(split, query=chunk)
                pred = self.model(chunk)
                if not train:
                    pred = self._apply_posthoc_calibration(pred)
                loss, parts = trust_calibration_loss(pred, chunk, self.cfg["training"], self.cfg["model"])
                if train:
                    self.optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg["training"].get("grad_clip", 2.0)))
                    self.optim.step()
                    self._update_ema()
                    if hasattr(self.model, "detach_memory"):
                        self.model.detach_memory()
                weight = self._event_count(chunk)
                total_loss += float(loss.detach().cpu()) * weight
                total_weight += weight
                for key, value in parts.items():
                    parts_sum[key] = parts_sum.get(key, 0.0) + value * weight
                pred_rows.append(
                    {
                        "trust_logit": pred["trust_logit"].detach().cpu(),
                        "trust_pred": pred["trust_pred"].detach().cpu(),
                        "uncertainty_pred": pred["uncertainty_pred"].detach().cpu(),
                    }
                )
                batch_rows.append(self._metric_batch(chunk))
        pred_all = self._concat_metric_batches(pred_rows)
        batch_all = self._concat_metric_batches(batch_rows)
        if train:
            self._update_rank_hard_negatives(split, batch_all["label"], pred_all["trust_logit"])
        metrics = compute_metrics(pred_all, batch_all, threshold=self._metric_threshold(threshold))
        metrics = self._validation_targets(metrics, pred_all, batch_all, split)
        denom = max(1, total_weight)
        parts_avg = {key: value / denom for key, value in parts_sum.items()}
        return total_loss / denom, parts_avg, metrics

    def _step(self, split: str, train: bool, threshold: float | None = None) -> tuple[float, dict[str, float], dict[str, float]]:

        if str(self.cfg["training"].get("batch_mode", "full_temporal")) == "event_stream":
            return self._step_stream(split, train, threshold=threshold)
        batch = self._history_batch(self._attach_rank_hard_negatives(split, self.batches[split], train), train)
        self.model.train(train)
        self._reset_and_prime(split)
        with torch.set_grad_enabled(train):
            pred = self.model(batch)
            if not train:
                pred = self._apply_posthoc_calibration(pred)
            loss, parts = trust_calibration_loss(pred, batch, self.cfg["training"], self.cfg["model"])
            if train:
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg["training"].get("grad_clip", 2.0)))
                self.optim.step()
                self._update_ema()
                self._update_rank_hard_negatives(split, batch["label"], pred["trust_logit"])
                self.model.detach_memory()
        metrics = compute_metrics(pred, batch, threshold=self._metric_threshold(threshold))
        metrics = self._validation_targets(metrics, pred, self._metric_batch(batch), split)
        return float(loss.detach().cpu()), parts, metrics

    def fit(self, epochs: int | None = None, checkpoint_path: str | Path | None = None) -> dict[str, float]:

        epochs = int(epochs or self.cfg["training"].get("epochs", 20))
        train_cfg = self.cfg["training"]
        monitor = str(train_cfg.get("early_stop_monitor", "val_loss"))
        default_mode = "min" if monitor.endswith("loss") else "max"
        stopper = EarlyStopping(
            int(train_cfg.get("early_stop_patience", 5)),
            mode=str(train_cfg.get("early_stop_mode", default_mode)),
            min_delta=float(train_cfg.get("early_stop_min_delta", 0.0)),
        )
        best_state = None
        best_metrics: dict[str, float] = {}
        last_metrics: dict[str, float] = {}
        self.history: list[dict[str, float]] = []
        self._start_ema()
        start = time.perf_counter()
        dataset_name = self.cfg.get("dataset", {}).get("name", "dataset")
        desc = f"{self.cfg['model']['name']} | {dataset_name} | {self.device}"
        progress = tqdm(range(1, epochs + 1), desc=desc, unit="epoch", dynamic_ncols=True)
        for epoch in progress:
            epoch_start = time.perf_counter()
            train_loss, parts, train_metrics = self._step("train", train=True)
            with self._validation_weights():
                val_loss, _, val_metrics = self._step("val", train=False)
                current_metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_paper_score": self._paper_score(train_metrics),
                    "val_paper_score": self._paper_score(val_metrics),
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **parts,
                }
                if monitor not in current_metrics:
                    raise ValueError(f"Unknown early_stop_monitor: {monitor}")
                monitor_value = float(current_metrics[monitor])
                last_metrics = current_metrics
                self.history.append(current_metrics)
                improved = stopper.is_improvement(monitor_value)
                if improved:
                    best_metrics = dict(current_metrics)
                    best_state = deepcopy(self.model.state_dict())
                    if checkpoint_path is not None:
                        self.save_checkpoint(checkpoint_path)
            elapsed = time.perf_counter() - epoch_start
            progress.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_overall_auc": f"{val_metrics['overall_auc']:.4f}",
                    "val_observed_user_mrr": f"{val_metrics['observed_user_mrr']:.4f}",
                    "val_unobserved_user_mrr": f"{val_metrics['unobserved_user_mrr']:.4f}",
                    monitor: f"{monitor_value:.4f}",
                    "sec": f"{elapsed:.1f}",
                    "best": "*" if improved else "",
                }
            )
            if stopper.step(monitor_value):
                best_text = "nan" if stopper.best is None else f"{stopper.best:.4f}"
                tqdm.write(f"Early stopping at epoch {epoch}; best_{monitor}={best_text}")
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if not best_metrics:
            best_metrics = dict(last_metrics)
        if last_metrics:
            best_metrics["best_epoch"] = best_metrics["epoch"]
            best_metrics["early_stop_monitor"] = monitor
            best_metrics["best_monitor_value"] = float(best_metrics.get(monitor, best_metrics.get("val_loss", 0.0)))
            best_metrics["best_val_loss"] = best_metrics["val_loss"]
            best_metrics["stopped_epoch"] = last_metrics["epoch"]
            best_metrics["last_val_loss"] = last_metrics["val_loss"]
        best_metrics["training_time_sec"] = time.perf_counter() - start
        best_metrics['checkpoint_weights'] = 'ema' if self._ema_state is not None else 'raw'
        best_metrics['ema_decay'] = self._ema_decay
        tqdm.write(f"Training finished in {best_metrics['training_time_sec']:.1f}s")
        return best_metrics

    @torch.no_grad()
    def collect_predictions(self, split: str = "val", apply_calibration: bool = True) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

        batch = self._history_batch(self.batches[split], split == 'train')
        self.model.eval()
        self._reset_and_prime(split)
        if str(self.cfg["training"].get("batch_mode", "full_temporal")) != "event_stream":
            pred = self.model(batch)
            if apply_calibration:
                pred = self._apply_posthoc_calibration(pred)
            pred_cpu = {
                "trust_logit": pred["trust_logit"].detach().cpu(),
                "trust_pred": pred["trust_pred"].detach().cpu(),
                "uncertainty_pred": pred["uncertainty_pred"].detach().cpu(),
            }
            return pred_cpu, self._metric_batch(batch)

        pred_rows: list[dict[str, torch.Tensor]] = []
        batch_rows: list[dict[str, torch.Tensor]] = []
        for chunk in self._chunks(batch):
            pred = self.model(chunk)
            if apply_calibration:
                pred = self._apply_posthoc_calibration(pred)
            pred_rows.append(
                {
                    "trust_logit": pred["trust_logit"].detach().cpu(),
                    "trust_pred": pred["trust_pred"].detach().cpu(),
                    "uncertainty_pred": pred["uncertainty_pred"].detach().cpu(),
                }
            )
            batch_rows.append(self._metric_batch(chunk))
        return self._concat_metric_batches(pred_rows), self._concat_metric_batches(batch_rows)

    @torch.no_grad()
    def clear_posthoc_calibration(self) -> None:

        self.posthoc_calibration = None

    def fit_posthoc_calibrator(self, split: str = "val") -> dict[str, float]:

        saved = self.posthoc_calibration
        self.posthoc_calibration = None
        pred, batch = self.collect_predictions(split, apply_calibration=False)
        self.posthoc_calibration = saved
        logits = pred["trust_logit"].to(self.device).float()
        labels = batch["label"].to(self.device).float()
        log_scale = torch.zeros((), device=self.device, requires_grad=True)
        bias = torch.zeros((), device=self.device, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [log_scale, bias],
            lr=float(self.cfg["training"].get("calibration_lr", 0.5)),
            max_iter=int(self.cfg["training"].get("calibration_max_iter", 80)),
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            scale = torch.exp(log_scale).clamp(0.05, 20.0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits * scale + bias, labels)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        scale = float(torch.exp(log_scale.detach()).clamp(0.05, 20.0).cpu())
        offset = float(bias.detach().cpu())
        self.posthoc_calibration = {"scale": scale, "bias": offset}
        calibrated_pred, calibrated_batch = self.collect_predictions(split, apply_calibration=True)
        calibrated_metrics = compute_metrics(calibrated_pred, calibrated_batch, threshold=self._metric_threshold())
        return {
            "calibration_scale": scale,
            "calibration_bias": offset,
            "calibration_val_loss": float(loss.detach().cpu()),
            "calibration_val_ece": float(calibrated_metrics.get("ece", 0.0)),
            "calibration_val_brier": float(calibrated_metrics.get("brier_score", 0.0)),
        }

    @torch.no_grad()
    def tune_threshold(self, split: str = "val") -> dict[str, float]:

        pred, batch = self.collect_predictions(split)
        y = batch["label"].float().cpu().numpy()
        prob = pred["trust_pred"].float().cpu().numpy()
        train_cfg = self.cfg["training"]
        lo = float(train_cfg.get("threshold_min", 0.01))
        hi = float(train_cfg.get("threshold_max", 0.99))
        steps = int(train_cfg.get("threshold_steps", 199))
        metric_name = str(train_cfg.get("threshold_metric", "f1")).lower()
        thresholds = np.linspace(lo, hi, max(2, steps))
        best_threshold = float(self._metric_threshold())
        best_score = -1.0
        for value in thresholds:
            yhat = (prob >= value).astype(np.float32)
            if metric_name == "accuracy":
                score = float(accuracy_score(y, yhat))
            elif metric_name == "balanced_accuracy":
                score = float(balanced_accuracy_score(y, yhat))
            else:
                score = float(f1_score(y, yhat, zero_division=0))
            if score > best_score:
                best_score = score
                best_threshold = float(value)
        tuned_metrics = compute_metrics(pred, batch, threshold=best_threshold)
        keys = ["f1", "accuracy", "balanced_accuracy", "precision", "recall", "positive_prediction_rate"]
        out = {
            "best_threshold": best_threshold,
            "threshold_score": best_score,
            "threshold_metric": metric_name,
        }
        out.update({f"{split}_tuned_{key}": float(tuned_metrics.get(key, 0.0)) for key in keys})
        return out

    @torch.no_grad()
    def evaluate(self, split: str = "test", threshold: float | None = None) -> dict[str, float]:

        batch = self._history_batch(self.batches[split], split == 'train')
        self.model.eval()
        if str(self.cfg["training"].get("batch_mode", "full_temporal")) == "event_stream":
            start = time.perf_counter()
            loss_value, parts, metrics = self._step_stream(split, train=False, threshold=threshold)
            elapsed = time.perf_counter() - start
            metrics.update(parts)
            metrics["eval_loss"] = loss_value
            metrics["inference_time_ms"] = elapsed * 1000.0
            metrics["parameter_count"] = parameter_count(self.model)
            metrics["gpu_memory_mb"] = gpu_memory_mb(self.device)
            metrics["num_events"] = int(batch["src"].shape[0])
            return metrics
        self._reset_and_prime(split)
        start = time.perf_counter()
        pred = self.model(batch)
        pred = self._apply_posthoc_calibration(pred)
        elapsed = time.perf_counter() - start
        loss, parts = trust_calibration_loss(pred, batch, self.cfg["training"], self.cfg["model"])
        metrics = compute_metrics(pred, batch, threshold=self._metric_threshold(threshold))
        metrics.update(parts)
        metrics["eval_loss"] = float(loss.detach().cpu())
        metrics["inference_time_ms"] = elapsed * 1000.0
        metrics["parameter_count"] = parameter_count(self.model)
        metrics["gpu_memory_mb"] = gpu_memory_mb(self.device)
        metrics["num_events"] = int(batch["src"].shape[0])
        return metrics

    def save_checkpoint(self, path: str | Path) -> Path:

        p = resolve_path(path)
        ensure_dir(p.parent)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "cfg": self.cfg,
                "dims": self.dataset.feature_dims,
                "num_edge_types": self.dataset.num_edge_types,
            },
            p,
        )
        return p

    def load_checkpoint(self, path: str | Path) -> None:

        ckpt = torch.load(resolve_path(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
