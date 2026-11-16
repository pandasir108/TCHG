from __future__ import annotations

import argparse

from tchg.experiments.ablation import ablation_configs
from tchg.runner import run_training, variant_dir_name
from tchg.training.reporting import format_user_regime_table, USER_REGIME_LABELS
from tchg.utils.io import ensure_dir, load_config, save_json, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/epinions_70_15_15.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = ensure_dir(args.output or f"{cfg['paths']['output_dir']}/ablation")
    variants = ablation_configs(cfg)
    if args.variants:
        requested = set(args.variants)
        variants = {name: value for name, value in variants.items() if name in requested}
        missing = requested.difference(variants)
        if missing:
            raise ValueError(f"Unknown variants: {sorted(missing)}")
    rows = []
    regime_rows = []
    all_results = {}
    for name, variant_cfg in variants.items():
        _, result = run_training(variant_cfg, root / variant_dir_name(name), args.device, args.epochs)
        row = {"variant": name, **result["test"]}
        rows.append(row)
        for regime, values in result['test_by_user_regime'].items():
            regime_rows.append({'variant': name, 'user_regime': regime, 'task': USER_REGIME_LABELS[regime], **values})
        all_results[name] = result
        write_csv(rows, root / "summary.csv")
        write_csv(regime_rows, root / 'summary_user_regimes.csv')
        save_json(all_results, root / "results.json")
        print(name)
        print(format_user_regime_table(result['test_by_user_regime']))


if __name__ == "__main__":
    main()
