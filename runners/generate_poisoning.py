from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tchg.experiments.trustguard_poisoning import collaborative_bad_mouthing
from tchg.runner import prepare
from tchg.utils.io import ensure_dir, load_config, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rates', nargs='+', type=float, default=[.05, .1, .15, .2])
    parser.add_argument('--attack-seed', type=int, default=59)
    parser.add_argument('--target-fraction', type=float, default=.1)
    parser.add_argument('--colluder-fraction', type=float, default=.1)
    parser.add_argument('--max-rate', type=float, default=.2)
    args = parser.parse_args()
    cfg = load_config(args.config)
    with np.load(prepare(cfg), allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    for rate in args.rates:
        edges, audit = collaborative_bad_mouthing(arrays, rate, args.attack_seed,
                                                 args.target_fraction, args.max_rate, args.colluder_fraction)
        out = ensure_dir(Path(args.output) / f'rate_{rate:g}')
        edges.to_csv(out / 'injected_edges.csv', index=False)
        save_json(audit, out / 'attack_generation.json')
        print(rate, len(edges))


if __name__ == '__main__':
    main()
