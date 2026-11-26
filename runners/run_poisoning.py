from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from tchg.experiments.poisoning import materialize_poisoning
from tchg.runner import prepare, run_training
from tchg.training.reporting import format_user_regime_table
from tchg.utils.io import ensure_dir, load_config, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--injected-edges', required=True)
    parser.add_argument('--attack-metadata', default=None)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    cfg = deepcopy(load_config(args.config))
    if args.seed is not None:
        cfg['seed'] = args.seed
    if cfg['training'].get('warm_start_checkpoint'):
        raise ValueError('Poisoning experiments retrain each victim from scratch')
    out = ensure_dir(Path(args.output).resolve())
    with np.load(prepare(cfg), allow_pickle=False) as loaded:
        clean = {key: loaded[key] for key in loaded.files}
    injected = pd.read_csv(args.injected_edges)
    arrays, audit = materialize_poisoning(clean, injected, cfg['dataset'], int(cfg.get('seed', 42)))
    if args.attack_metadata:
        import json
        audit['attack_generation'] = json.loads(Path(args.attack_metadata).read_text())
    save_json(audit, out / 'poisoning_protocol.json')
    path = out / 'events.npz'
    np.savez_compressed(path, **arrays)
    cfg['dataset']['processed_file'] = str(path)
    cfg['dataset']['structural_feature_cache'] = str(out / 'causal_structure.npz')
    cfg['dataset']['rating_feature_cache'] = str(out / 'causal_ratings.npz')
    cfg['paths']['output_dir'] = str(out)
    save_json(cfg, out / 'resolved_poisoning_config.json')
    del arrays, clean
    if args.prepare_only:
        print(audit)
        return
    _, result = run_training(cfg, out, args.device, args.epochs)
    result['poisoning_protocol'] = audit
    save_json(result, out / 'metrics.json')
    print(format_user_regime_table(result['test_by_user_regime']))


if __name__ == '__main__':
    main()
