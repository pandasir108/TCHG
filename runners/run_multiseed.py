from __future__ import annotations

import argparse

from tchg.experiments.seed_sweep import run_seed_sweep
from tchg.utils.io import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/epinions_70_15_15_memory_days.yaml')
    parser.add_argument('--device', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    summary = run_seed_sweep(
        cfg, args.output or cfg['paths']['output_dir'], args.device, args.epochs,
        seeds=args.seeds, evaluate_test=not args.validation_only,
    )
    print(f"Seeds: {', '.join(str(seed) for seed in summary['seeds'])}")
    print(f"{'Task':<23} {'MRR (mean +/- std)':>22} {'AP (mean +/- std)':>22} {'AUC (mean +/- std)':>22}")
    for row in summary['by_user_regime'].values():
        values = [f"{row[metric + '_mean']:.4f} +/- {row[metric + '_std']:.4f}"
                  for metric in ('mrr', 'ap', 'auc')]
        print(f"{row['task']:<23} {values[0]:>22} {values[1]:>22} {values[2]:>22}")


if __name__ == '__main__':
    main()
