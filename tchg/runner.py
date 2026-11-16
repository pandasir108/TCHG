from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from tchg.data.event_dataset import EventGraphDataset
from tchg.data.preprocessors import preprocess_dataset
from tchg.training.trainer import Trainer
from tchg.training.reporting import evaluation_protocol, write_user_regime_report
from tchg.utils.io import ensure_dir, resolve_path, save_json, write_csv
from tchg.utils.seed import set_seed


def prepare(cfg: dict[str, Any], force: bool = False) -> Path:
    path = resolve_path(cfg["dataset"]["processed_file"])
    if force or not path.exists():
        path = preprocess_dataset(cfg["dataset"]["name"], cfg, int(cfg.get("seed", 42)))
    return path


def run_training(
    cfg: dict[str, Any],
    output_dir: str | Path,
    device: str | None = None,
    epochs: int | None = None,
    evaluate_test: bool = True,
) -> tuple[Trainer, dict[str, Any]]:
    cfg = deepcopy(cfg)
    if epochs is not None:
        cfg['training']['epochs'] = int(epochs)
    if device is not None:
        cfg['device'] = device
    set_seed(int(cfg.get("seed", 42)))
    dataset = EventGraphDataset(prepare(cfg))
    trainer = Trainer(cfg, dataset, device=device)
    warm_start = cfg['training'].get('warm_start_checkpoint')
    if warm_start:
        trainer.load_checkpoint(warm_start)
    out = ensure_dir(output_dir)
    checkpoint = out / "best.pt"
    fit_metrics = trainer.fit(epochs=epochs, checkpoint_path=checkpoint)
    calibration_metrics: dict[str, Any] = {}
    if bool(cfg["training"].get("posthoc_calibration", True)):
        calibration_metrics = trainer.fit_posthoc_calibrator("val")
    threshold_metrics: dict[str, Any] = {}
    threshold = float(cfg["training"].get("threshold", 0.5))
    if bool(cfg["training"].get("auto_tune_threshold", True)):
        threshold_metrics = trainer.tune_threshold("val")
        threshold = float(threshold_metrics["best_threshold"])
    protocol = evaluation_protocol(trainer, fit_metrics, epochs)
    results = {
        "fit": fit_metrics,
        "calibration": calibration_metrics,
        "threshold": threshold_metrics,
        "test_evaluated": evaluate_test,
        "evaluation_protocol": protocol,
    }
    split = 'test' if evaluate_test else 'val'
    metrics = trainer.evaluate(split, threshold=threshold)
    label = 'test' if evaluate_test else 'validation'
    results[label] = metrics
    results[label + '_by_user_regime'] = write_user_regime_report(metrics, out, stem=label)
    protocol['evaluation'] = {'reported_split': split, 'test_evaluated': evaluate_test}
    save_json(results, out / "metrics.json")
    write_csv(trainer.history, out / "history.csv")
    save_json(protocol, out / 'evaluation_protocol.json')
    save_json(trainer.cfg, out / 'resolved_config.json')
    save_json({'posthoc_calibration': trainer.posthoc_calibration, 'threshold': threshold},
              out / 'inference_state.json')
    trainer.save_checkpoint(checkpoint)
    return trainer, results


def variant_dir_name(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return value or "variant"


def with_overrides(cfg: dict[str, Any], epochs: int | None = None) -> dict[str, Any]:
    out = deepcopy(cfg)
    if epochs is not None:
        out["training"]["epochs"] = int(epochs)
    return out
