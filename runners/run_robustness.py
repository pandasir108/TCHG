from __future__ import annotations

import argparse
from copy import deepcopy

from tchg.data.event_dataset import EventGraphDataset
from tchg.experiments.attack_study import ATTACKS
from tchg.experiments.missing_study import apply_missing_rate
from tchg.runner import prepare
from tchg.training.trainer import Trainer
from tchg.training.metrics import user_regime_results
from tchg.training.reporting import USER_REGIME_LABELS, evaluation_protocol, format_user_regime_table
from tchg.utils.io import ensure_dir, load_config, resolve_path, save_json, write_csv
from tchg.utils.seed import set_seed


def evaluate_arrays(
    cfg: dict,
    arrays: dict,
    checkpoint: str,
    device: str | None,
    calibration: dict[str, float] | None,
    threshold: float,
) -> dict[str, float]:
    dataset = EventGraphDataset.from_arrays(arrays)
    trainer = Trainer(cfg, dataset, device=device)
    trainer.load_checkpoint(checkpoint)
    trainer.posthoc_calibration = calibration
    return trainer.evaluate("test", threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/epinions_70_15_15.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rates", nargs="+", type=float, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    dataset = EventGraphDataset(prepare(cfg))
    checkpoint = args.checkpoint or f"{cfg['paths']['output_dir']}/best.pt"
    if not resolve_path(checkpoint).exists():
        raise FileNotFoundError(f"Train the main model first: {checkpoint}")
    clean_trainer = Trainer(cfg, dataset, device=args.device)
    clean_trainer.load_checkpoint(checkpoint)
    calibration = None
    if bool(cfg["training"].get("posthoc_calibration", True)):
        clean_trainer.fit_posthoc_calibrator("val")
        calibration = deepcopy(clean_trainer.posthoc_calibration)
    threshold = float(cfg["training"].get("threshold", 0.5))
    if bool(cfg["training"].get("auto_tune_threshold", True)):
        threshold = float(clean_trainer.tune_threshold("val")["best_threshold"])
    experiment_cfg = cfg.get("robustness", {})
    rates = args.rates or experiment_cfg.get("rates", [0.1, 0.2, 0.3, 0.4, 0.5])
    scenarios = args.scenarios or experiment_cfg.get("scenarios", ["missing", "trust_oriented", "gnn_oriented"])
    output = ensure_dir(args.output or f"{cfg['paths']['output_dir']}/robustness")
    rows = []
    regime_rows = []

    def record(scenario, rate, metrics):
        rows.append({'scenario': scenario, 'rate': rate, **metrics})
        grouped = user_regime_results(metrics)
        for regime, values in grouped.items():
            regime_rows.append({'scenario': scenario, 'rate': rate, 'user_regime': regime,
                                'task': USER_REGIME_LABELS[regime], **values})
        write_csv(rows, output / 'summary.csv')
        write_csv(regime_rows, output / 'summary_user_regimes.csv')
        print(f'{scenario}, rate={rate}')
        print(format_user_regime_table(grouped))

    clean_metrics = clean_trainer.evaluate("test", threshold=threshold)
    record('clean', 0., clean_metrics)
    protocol = evaluation_protocol(clean_trainer, {})
    protocol['selection'] = {'source_checkpoint': str(checkpoint), 'selected_during_this_run': False,
                             'shared_model_for_both_regimes': True, 'test_used_for_selection': False}
    protocol['sample_count_reference'] = 'Clean data; user regimes stay fixed across event perturbations.'
    save_json(protocol, output / 'evaluation_protocol.json')
    seed = int(cfg.get("seed", 42))
    for scenario in scenarios:
        for rate in rates:
            if scenario == "missing":
                arrays = apply_missing_rate(dataset.arrays, float(rate), seed)
            elif scenario in ATTACKS:
                arrays = ATTACKS[scenario](dataset.arrays, float(rate), seed)
            else:
                raise ValueError(f"Unknown robustness scenario: {scenario}")
            metrics = evaluate_arrays(cfg, arrays, checkpoint, args.device, calibration, threshold)
            record(scenario, float(rate), metrics)
    save_json({"rows": rows, 'by_user_regime': regime_rows, 'evaluation_protocol': protocol}, output / "results.json")


if __name__ == "__main__":
    main()
