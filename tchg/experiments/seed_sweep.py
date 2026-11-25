from __future__ import annotations

from copy import deepcopy
from numbers import Integral
import numpy as np

from tchg.runner import prepare, run_training
from tchg.training.reporting import USER_REGIME_LABELS
from tchg.utils.io import ensure_dir, save_json, write_csv


def select_seeds(cfg, seeds=None):
    chosen = list(cfg.get('seeds', [42, 43, 44, 45, 46]) if seeds is None else seeds)
    if not chosen or any(isinstance(seed, bool) or not isinstance(seed, Integral)
                         or not 0 <= seed < 2**32 for seed in chosen):
        raise ValueError('Seeds must be integers in [0, 2**32).')
    chosen = [int(seed) for seed in chosen]
    if len(set(chosen)) != len(chosen):
        raise ValueError('Seeds must be distinct.')
    return chosen


def summarize_seed_runs(runs, reported_split):
    if not runs:
        raise ValueError('No completed seed runs to aggregate.')
    seeds = [run['seed'] for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Each seed must contribute exactly once.')
    fingerprints = {run['dataset_sha256'] for run in runs}
    if len(fingerprints) != 1 or not next(iter(fingerprints)):
        raise ValueError('All seeds must use the same prepared candidate dataset.')
    grouped = {}
    for regime, label in USER_REGIME_LABELS.items():
        rows = [run['metrics'][regime] for run in runs]
        if any(set(row) != set(rows[0]) for row in rows):
            raise ValueError('Metric fields differ between seeds.')
        summary = {'task': label}
        for key in sorted(rows[0]):
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite metric: {regime}/{key}')
            if key.startswith('num_'):
                if not np.equal(values, values[0]).all():
                    raise ValueError(f'Candidate counts differ between seeds: {regime}/{key}')
                summary[key] = int(values[0])
            else:
                summary[key + '_mean'] = float(values.mean())
                summary[key + '_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        grouped[regime] = summary
    return {
        'seeds': seeds,
        'n_seeds': len(seeds),
        'dataset_sha256': next(iter(fingerprints)),
        'reported_split': reported_split,
        'aggregation': 'arithmetic mean and sample standard deviation',
        'std_ddof': 1,
        'by_user_regime': grouped,
        'runs': runs,
    }


def run_seed_sweep(cfg, output_dir, device=None, epochs=None, seeds=None, evaluate_test=True):
    chosen = select_seeds(cfg, seeds)
    prepare(cfg)
    root = ensure_dir(output_dir)
    label = 'test' if evaluate_test else 'validation'
    summary_json = root / f'{label}_seed_summary.json'
    summary_csv = root / f'{label}_seed_summary.csv'
    summary_json.unlink(missing_ok=True)
    summary_csv.unlink(missing_ok=True)
    runs = []
    for seed in chosen:
        run_cfg = deepcopy(cfg)
        run_cfg['seed'] = seed
        output = root / f'seed_{seed}'
        run_cfg['paths']['output_dir'] = str(output)
        trainer, result = run_training(run_cfg, output, device, epochs, evaluate_test=evaluate_test)
        runs.append({
            'seed': seed,
            'output_dir': str(output),
            'dataset_sha256': result['evaluation_protocol']['dataset_sha256'],
            'metrics': result[label + '_by_user_regime'],
        })
        del trainer
    summary = summarize_seed_runs(runs, label)
    save_json(summary, summary_json)
    write_csv([
        {'user_regime': regime, **values}
        for regime, values in summary['by_user_regime'].items()
    ], summary_csv)
    return summary
