from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy

from tchg.data.event_dataset import EventGraphDataset
from tchg.experiments.attack_study import ATTACKS
from tchg.experiments.cat_style_history import HISTORY_PROTOCOLS, HISTORY_SCENARIOS, evaluate_history_attack
from tchg.experiments.missing_study import apply_missing_rate
from tchg.runner import prepare
from tchg.training.trainer import Trainer
from tchg.training.metrics import user_regime_results
from tchg.training.reporting import USER_REGIME_LABELS, evaluation_protocol, format_user_regime_table
from tchg.utils.io import ensure_dir, load_config, resolve_path, save_json, write_csv
from tchg.utils.seed import set_seed


def resolve_checkpoint(cfg, checkpoint=None, seed=None):
    configured_seed = int(cfg.get('seed', 42))
    selected_seed = configured_seed if seed is None else int(seed)
    if not 0 <= selected_seed < 2**32:
        raise ValueError('Seed must be in [0, 2**32).')
    if checkpoint is not None:
        candidates = [resolve_path(checkpoint)]
    else:
        root = resolve_path(cfg['paths']['output_dir'])
        candidates = [root / f'seed_{selected_seed}' / 'best.pt']
        if selected_seed == configured_seed:
            candidates.append(root / 'best.pt')
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        attempted = ', '.join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f'No checkpoint found. Checked: {attempted}')
    saved_config = path.parent / 'resolved_config.json'
    recorded_seed = None
    if saved_config.is_file():
        recorded_seed = json.loads(saved_config.read_text()).get('seed')
    if recorded_seed is None:
        match = re.fullmatch(r'seed_(\d+)', path.parent.name)
        if match:
            recorded_seed = int(match.group(1))
    if recorded_seed is not None:
        recorded_seed = int(recorded_seed)
        if (seed is not None or checkpoint is None) and recorded_seed != selected_seed:
            raise ValueError(f'Checkpoint seed {recorded_seed} does not match requested seed {selected_seed}.')
        selected_seed = recorded_seed
    return path, selected_seed


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
    parser = argparse.ArgumentParser(description='Evaluate fixed-weight history perturbations. Training-time poisoning uses runners/run_poisoning.py.')
    parser.add_argument("--config", default="configs/epinions_70_15_15_memory_days.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Training seed of the checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--rates", nargs="+", type=float, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--attack-protocol", choices=[*HISTORY_PROTOCOLS, "feature_corruption"], default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkpoint, training_seed = resolve_checkpoint(cfg, args.checkpoint, args.seed)
    dataset = EventGraphDataset(prepare(cfg))
    cfg['seed'] = training_seed
    set_seed(training_seed)
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
    attack_protocol = args.attack_protocol or experiment_cfg.get('attack_protocol', 'fixed_history_coverage')
    if attack_protocol not in {*HISTORY_PROTOCOLS, 'feature_corruption'}:
        raise ValueError(f'Unknown attack protocol: {attack_protocol}')
    rates = args.rates or experiment_cfg.get("rates", [0.1, 0.2, 0.3, 0.4, 0.5])
    scenarios = args.scenarios or experiment_cfg.get("scenarios", ["missing", "trust_oriented", "gnn_oriented"])
    output = ensure_dir(args.output or checkpoint.parent / 'robustness' / attack_protocol)
    rows = []
    regime_rows = []

    def record(scenario, rate, metrics, audit=None):
        row_protocol = attack_protocol if scenario in HISTORY_SCENARIOS or scenario == 'clean' else (
            'event_missingness' if scenario == 'missing' else 'feature_corruption')
        rows.append({'scenario': scenario, 'rate': rate, 'attack_protocol': row_protocol, **metrics})
        grouped = user_regime_results(metrics)
        for regime, values in grouped.items():
            regime_rows.append({'scenario': scenario, 'rate': rate, 'attack_protocol': row_protocol, 'user_regime': regime,
                                'task': USER_REGIME_LABELS[regime], **values})
        if audit is not None:
            save_json(audit, output / scenario / f'rate_{float(rate):g}' / 'attack_protocol.json')
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
    protocol['robustness'] = dict(HISTORY_PROTOCOLS[attack_protocol]) if attack_protocol in HISTORY_PROTOCOLS else {
        'name': 'feature_corruption', 'stage': 'test_feature_corruption', 'retraining': False,
    }
    if attack_protocol in HISTORY_PROTOCOLS:
        protocol['robustness']['attack_seeds'] = {
            'trust_oriented': training_seed + 17, 'gnn_oriented': training_seed + 97,
        }
    save_json(protocol, output / 'evaluation_protocol.json')
    seed = int(cfg.get("seed", 42))
    for scenario in scenarios:
        for rate in rates:
            if scenario in HISTORY_SCENARIOS and attack_protocol in HISTORY_PROTOCOLS:
                if float(rate) == 0.:
                    metrics = clean_metrics
                    audit = {**HISTORY_PROTOCOLS[attack_protocol], 'scenario': scenario, 'requested_ratio': 0.,
                             'selected_history_edges': 0, 'changed_history_edges': 0}
                else:
                    metrics, audit = evaluate_history_attack(
                        clean_trainer, dataset.arrays, scenario, float(rate),
                        seed + {'trust_oriented': 17, 'gnn_oriented': 97}[scenario], threshold,
                        attack_protocol,
                    )
                record(scenario, float(rate), metrics, audit)
                continue
            if scenario == "missing":
                arrays = apply_missing_rate(dataset.arrays, float(rate), seed)
            elif scenario in HISTORY_SCENARIOS:
                key = {'trust_oriented': 'trust_feature_corruption',
                       'gnn_oriented': 'gnn_feature_corruption'}[scenario]
                arrays = ATTACKS[key](dataset.arrays, float(rate), seed)
            elif scenario in ATTACKS:
                arrays = ATTACKS[scenario](dataset.arrays, float(rate), seed)
            else:
                raise ValueError(f"Unknown robustness scenario: {scenario}")
            metrics = evaluate_arrays(cfg, arrays, checkpoint, args.device, calibration, threshold)
            record(scenario, float(rate), metrics)
    save_json({"rows": rows, 'by_user_regime': regime_rows, 'evaluation_protocol': protocol}, output / "results.json")


if __name__ == "__main__":
    main()
