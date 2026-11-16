from __future__ import annotations

import argparse

from tchg.runner import run_training
from tchg.training.reporting import format_user_regime_table
from tchg.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/epinions_70_15_15.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--validation-only", action="store_true",
                        help="Select and report on validation without evaluating test")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg['seed'] = args.seed
    output = args.output or cfg["paths"]["output_dir"]
    _, results = run_training(cfg, output, args.device, args.epochs,
                              evaluate_test=not args.validation_only)
    key = 'validation_by_user_regime' if args.validation_only else 'test_by_user_regime'
    print(format_user_regime_table(results[key]))


if __name__ == "__main__":
    main()
