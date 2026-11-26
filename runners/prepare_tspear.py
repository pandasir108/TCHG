from pathlib import Path
import difflib
import json
import shutil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--upstream', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
ROOT = Path(args.output).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
source = Path(args.upstream).resolve()
runtime = ROOT / 'tspear_runtime'
runtime.mkdir(exist_ok=True)
for name in ['tatk', 'config']:
    shutil.copytree(source / name, runtime / name, dirs_exist_ok=True)
(runtime / 'tatk/__init__.py').touch()
changes = []
for name in ['modules.py', 'memorys.py', 'utils_train.py', 'attacker.py']:
    path = runtime / 'tatk' / name
    before = path.read_text()
    after = before.replace('from memorys import ', 'from .memorys import ').replace('from layers import ', 'from .layers import ')
    if name == 'utils_train.py':
        after = after.replace('from deeprobust.graph.defense.pgd import PGD, prox_operators\n', '').replace('from pympler.tracker import SummaryTracker\n', '')
        needle = '        time_val = time.time() - t_val_s\n'
        replacement = needle + "        args.validation_history.append({'epoch': e, 'ap': float(val_ap), 'auc': float(val_auc)})\n        if e >= 51 and val_ap > args.best_val_ap:\n            args.best_val_ap = float(val_ap)\n            args.best_epoch = e\n            torch.save(model.state_dict(), args.surrogate_checkpoint)\n"
        assert needle in after
        after = after.replace(needle, replacement, 1)
        needle = "    test_ap, test_auc, test_hit = link_pred_evaluation(node_feats, edge_feats, g, df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=1, mode='test', evaluation=False)"
        assert needle in after
        after = after.replace(needle, '    test_ap, test_auc, test_hit = 0., 0., 0.', 1)
    if name == 'attacker.py':
        after = after.replace('verbose=false', 'verbose=False').replace(' is not none', ' is not None').replace('drop=true', 'drop=True')
        after = after.replace('df = df.append(ptb_df)', 'df = pd.concat([df, ptb_df])')
        after = after.replace('if i_rows > 0:', 'if i_rows > 0 and int(len(rows) * ptb_rate) > 0:')
        after = after.replace('np.trunc(kde.sample(num_ptb_batch).squeeze())', 'np.atleast_1d(np.trunc(kde.sample(num_ptb_batch).squeeze()))')
        needle = "            self.args.model_path = f'./MODEL/AAAI/NON_ROBUST/{args.data}/none/{args.surrogate}_seed_{seed}_best_from51.pt'"
        assert needle in after
        after = after.replace(needle, '            self.args.model_path = args.surrogate_checkpoint')
        start = after.index('                        src_dst_pair = np.array(list(itertools.product(row_src_array, row_dst_array)))')
        end = after.index('                        num_hungarian = 0', start)
        after = after[:start] + "                        src_dst_score = score_pairs(model, mailbox, sampler, sample_param, gnn_param, node_feats, edge_feats, row_src_array, row_dst_array, float(ptb_ts_batch.min()))\n\n" + after[end:]
        after = after.replace('            model.eval()\n', '            model.eval()\n            model, mailbox = use_double_precision(model, mailbox)\n', 1)
        after = 'from .adapter_helpers import score_pairs, use_double_precision\n' + after
    after = after.replace('if args.data == "UCI" or args.data == "BITCOIN":', 'if args.data in {"UCI", "BITCOIN", "EPINIONS"}:')
    after = after.replace('if args.data != "UCI" and args.data != "BITCOIN":', 'if args.data not in {"UCI", "BITCOIN", "EPINIONS"}:')
    path.write_text(after)
    changes.extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile='upstream/tatk/' + name, tofile='runtime/tatk/' + name))
(ROOT / 'tspear_compatibility.patch').write_text(''.join(changes))
shutil.copyfile(Path(__file__).with_name('tspear_score.py'), runtime / 'tatk/adapter_helpers.py')
layer_path = runtime / 'tatk/layers.py'
before = layer_path.read_text()
after = before.replace('dgl.function.copy_edge(', 'dgl.function.copy_e(').replace('dgl.function.copy_src(', 'dgl.function.copy_u(')
after = after.replace('self.w(t.reshape((-1, 1)))', 'self.w(t.reshape((-1, 1)).to(dtype=self.w.weight.dtype))')
after = after.replace("return torch.zeros((b.num_dst_nodes(), self.dim_out), device=torch.device('cuda:0'))", "return torch.zeros((b.num_dst_nodes(), self.dim_out), dtype=self.w_out.weight.dtype, device=torch.device('cuda:0'))")
layer_path.write_text(after)
with (ROOT / 'tspear_compatibility.patch').open('a') as f:
    f.writelines(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile='upstream/tatk/layers.py', tofile='runtime/tatk/layers.py'))
(ROOT / 'tspear_compatibility.json').write_text(json.dumps({
    'external_source': 'https://github.com/wooner49/T-spear-shield',
    'changes': ['package-relative imports', 'remove unused imports for SVD defense and profiler',
                'correct Python literal capitalization', 'pandas concat instead of removed DataFrame.append',
                'include Epinions in the unipartite node-sampling branch',
                'skip zero-budget partial batches and retain one-element timestamp arrays',
                'validation-only surrogate checkpoint selection among epochs 51-100, matching the upstream expected best_from51 filename',
                'skip absent surrogate test set; victims alone receive original test candidates',
                'accept explicit surrogate checkpoint path'],
    'memory_optimization': 'cache node/time embeddings for all candidate pairs at the same scoring timestamp; original predictor and Hungarian assignment retained; require equivalence check before attack generation',
    'attack_scoring_precision': 'float64 CUDA inference with unchanged learned parameter values; avoid batch-shape-dependent rounding in large timestamp encodings; surrogate training remains float32',
    'attack_selection_algorithm_modified': False,
}, indent=2))
print(runtime)
