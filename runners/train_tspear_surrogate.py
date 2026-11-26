from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--events', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--sampler-root', required=True)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
ROOT = Path(args.output).resolve()
events_path = Path(args.events).resolve()
sys.path[:0] = [str(ROOT / 'tspear_runtime'), str(Path(args.sampler_root).resolve())]
from tatk.utils import parse_config, set_random_seed
from tatk.utils_train import create_model_mailbox_sampler, train_model_link_pred

work = ROOT
work.mkdir(exist_ok=True)
config = ROOT / 'tspear_runtime/config'
if not (work / 'config').exists():
    (work / 'config').symlink_to(config, target_is_directory=True)
os.chdir(work)
with np.load(events_path, allow_pickle=False) as loaded:
    keep = (loaded['split'] < 2) & (loaded['sample_type'] == 0)
    df = pd.DataFrame({'src': loaded['src'][keep].astype(np.int32),
                       'dst': loaded['dst'][keep].astype(np.int32),
                       'time': loaded['timestamp'][keep].astype(np.float64),
                       'ext_roll': loaded['split'][keep].astype(np.int32)})
    num_nodes = int(loaded['num_nodes'][0])
df = df.sort_values('time', kind='stable').reset_index(drop=True)
df['Unnamed: 0'] = np.arange(len(df))
df['adv'] = 0
indices, times, eids = [[] for _ in range(num_nodes)], [[] for _ in range(num_nodes)], [[] for _ in range(num_nodes)]
for eid, row in enumerate(df.itertuples(index=False)):
    s, d = int(row.src), int(row.dst)
    for left, right in [(s, d), (d, s)]:
        indices[left].append(right);times[left].append(float(row.time));eids[left].append(eid)
indptr = np.r_[0, np.cumsum([len(x) for x in indices])].astype(np.int32)
graph = {'indptr': indptr, 'indices': np.concatenate(indices).astype(np.int32),
         'ts': np.concatenate(times).astype(np.float32), 'eid': np.concatenate(eids).astype(np.int32),
         'ext_full_indices': indices, 'ext_full_ts': times, 'ext_full_eid': eids}
with (work / 'clean_input.pkl').open('wb') as f:
    pickle.dump({'graph': graph, 'events': df, 'num_nodes': num_nodes}, f)
set_random_seed(args.seed)
torch.set_num_threads(1)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('surrogate')
logger.setLevel(logging.DEBUG)
model_dir = work / 'MODEL/AAAI/NON_ROBUST/EPINIONS/none'
model_dir.mkdir(parents=True, exist_ok=True)
checkpoint = model_dir / f'SURROGATE_seed_{args.seed}_best_from51.pt'
run_args = SimpleNamespace(data='EPINIONS', attack='none', robust='none', model='SURROGATE',
                           exp_id='AAAI', exp_file='NON_ROBUST/EPINIONS/none/SURROGATE',
                           logger=logger, eval_neg_samples=100,
                           surrogate_checkpoint=str(checkpoint), best_val_ap=-float('inf'),
                           best_epoch=0, validation_history=[])
sample, memory, gnn, train = parse_config('SURROGATE')
assert train['epoch'] == 100 and train['batch_size'] == 600
model, mailbox, sampler = create_model_mailbox_sampler(None, None, graph, df, sample, memory, gnn, train)
print('START_SURROGATE', str(events_path), len(df), 'edges', 'cuda', flush=True)
train_model_link_pred(None, None, graph, df, model, mailbox, sampler, sample, memory, gnn, train, run_args, seed=args.seed)
report = {'training_seed': args.seed, 'input_events': str(events_path), 'epochs': 100, 'selection_epochs': [51, 100],
          'best_epoch': run_args.best_epoch, 'best_validation_ap': run_args.best_val_ap,
          'checkpoint': str(checkpoint), 'test_used': False,
          'validation_history': run_args.validation_history,
          'config': {'sampling': sample, 'memory': memory, 'gnn': gnn, 'training': train}}
(work / 'surrogate_training.json').write_text(json.dumps(report, indent=2))
print('SURROGATE_COMPLETED', run_args.best_epoch, run_args.best_val_ap, flush=True)
