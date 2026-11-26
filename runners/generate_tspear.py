from pathlib import Path
from types import SimpleNamespace
import argparse
import itertools
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--sampler-root', required=True)
parser.add_argument('--attack-seed', type=int, default=139)
parser.add_argument('--rates', nargs='+', type=float, default=[.05, .1, .15, .2])
cli = parser.parse_args()
ROOT = Path(cli.output).resolve()
sys.path[:0] = [str(ROOT / 'tspear_runtime'), str(Path(cli.sampler_root).resolve())]
from tatk.utils import parse_config, set_random_seed, to_dgl_blocks, prepare_input
from tatk.utils_train import create_model_mailbox_sampler
from tatk.adapter_helpers import score_pairs, use_double_precision
from tatk.attacker import TemporalAttack

work = ROOT
os.chdir(work)
report = json.loads((work / 'surrogate_training.json').read_text())
with (work / 'clean_input.pkl').open('rb') as f:
    clean = pickle.load(f)
graph, df = clean['graph'], clean['events']
assert not np.any(df.ext_roll == 2)
torch.set_num_threads(1)
sample, memory, gnn, training = parse_config('SURROGATE')
set_random_seed(cli.attack_seed)
model, mailbox, sampler = create_model_mailbox_sampler(None, None, graph, df, sample, memory, gnn, training)
model.load_state_dict(torch.load(report['checkpoint'], map_location='cuda'))
model, mailbox = use_double_precision(model, mailbox)
model.eval()
prefix = df.iloc[3000:3600]
users = np.unique(np.r_[prefix.src, prefix.dst])
sources, destinations = users[:16], users[-16:]
timestamp = float(df.iloc[6000].time)
pairs = np.array(list(itertools.product(sources, destinations)), dtype=np.int32)
roots = pairs.T.reshape(-1)
times = np.full(len(roots), timestamp, dtype=np.float32)
sampler.sample(roots, times)
blocks = prepare_input(to_dgl_blocks(sampler.get_ret(), sample['history']), None, None)
assert blocks[0][0].num_edges() > 0
mailbox.prep_input_mails(blocks[0])
with torch.no_grad():
    reference, _ = model(blocks, neg_samples=0)
    optimized = score_pairs(model, mailbox, sampler, sample, gnn, None, None, sources, destinations, timestamp)
torch.testing.assert_close(optimized, reference, rtol=1e-5, atol=1e-6)
equivalence = {'pairs': len(pairs), 'maximum_absolute_logit_difference': float((optimized - reference).abs().max()),
               'rtol': 1e-5, 'atol': 1e-6, 'passed': True}
(work / 'score_equivalence.json').write_text(json.dumps(equivalence, indent=2))
print('PAIR_SCORE_EQUIVALENCE', equivalence, flush=True)
del model, mailbox, sampler, blocks, optimized, reference
torch.cuda.empty_cache()

for rate in cli.rates:
    set_random_seed(cli.attack_seed)
    args = SimpleNamespace(attack='proposed', surrogate='SURROGATE', data='EPINIONS',
                           batch_size=600, use_hungarian=True, xpool=0,
                           surrogate_checkpoint=report['checkpoint'])
    attacker = TemporalAttack(args)
    _, _, _, attacked = attacker.attack(None, None, graph, df, rate, args, seed=cli.attack_seed)
    added = attacked[attacked['adv'] == 1].copy()
    edges = pd.DataFrame({'src': added.src.astype(np.int64), 'dst': added.dst.astype(np.int64),
                          'timestamp': added.time.astype(np.float64), 'label': 1.,
                          'split': added.ext_roll.astype(np.int64)})
    assert not (edges.src == edges.dst).any()
    assert edges['split'].isin([0, 1]).all()
    out = ROOT / f'rate_{rate:g}'
    out.mkdir(parents=True, exist_ok=True)
    edges.to_csv(out / 'injected_edges.csv', index=False)
    metadata = {
        'attack': 'T-Spear proposed attack', 'upstream': 'https://github.com/wooner49/T-spear-shield',
        'reference': 'https://www.ndss-symposium.org/wp-content/uploads/2026-f2171-paper.pdf',
        'implementation': 'upstream TemporalAttack with preserved score and Hungarian selection algorithms; compatibility patch recorded separately',
        'requested_rate': rate, 'attack_seed': cli.attack_seed, 'training_seed': report['training_seed'],
        'budget_basis': 'original genuine edges in each original history batch',
        'window_size': 600, 'kde_bandwidth': .1, 'hungarian': True, 'xpool': 0,
        'data_adaptation': 'Epinions treated as unipartite, using the upstream Bitcoin/UCI node-pool branch',
        'victim_access': False, 'test_data_used_for_attack': False,
        'surrogate_checkpoint': report['checkpoint'], 'surrogate_training': report,
        'pair_score_equivalence': equivalence,
        'attack_scoring_precision': 'float64 CUDA; learned parameter values unchanged; surrogate training uses upstream float32',
        'by_split': {str(k): {'original_edges': int((df.ext_roll == k).sum()),
                             'injected_edges': int((edges['split'] == k).sum()),
                             'actual_ratio': int((edges['split'] == k).sum()) / int((df.ext_roll == k).sum())}
                     for k in [0, 1]},
        'partial_batch_policy': 'upstream batch flooring; skip zero-budget partial batches; no top-up using test edges',
    }
    (out / 'attack_generation.json').write_text(json.dumps(metadata, indent=2))
    print('ATTACK_COMPLETED', ROOT.name, rate, len(edges), flush=True)
    torch.cuda.empty_cache()
print('ALL_TSPEAR_RATES_COMPLETED', ROOT.name, flush=True)
