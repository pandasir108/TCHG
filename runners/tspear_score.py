import numpy as np
import torch

from tatk.utils import to_dgl_blocks, node_to_dgl_blocks, prepare_input


def use_double_precision(model, mailbox):
    model.double()
    if mailbox is not None:
        for name in ['node_memory', 'mailbox']:
            setattr(mailbox, name, getattr(mailbox, name).double())
    return model, mailbox


@torch.no_grad()
def score_pairs(model, mailbox, sampler, sample_param, gnn_param,
                node_feats, edge_feats, sources, destinations, timestamp):
    roots = np.unique(np.r_[sources, destinations]).astype(np.int32)
    times = np.full(len(roots), timestamp, dtype=np.float32)
    if sampler is not None:
        sampler.sample(roots, times)
        ret = sampler.get_ret()
    blocks = to_dgl_blocks(ret, sample_param['history']) if gnn_param['arch'] != 'identity' else node_to_dgl_blocks(roots, times)
    blocks = prepare_input(blocks, node_feats, edge_feats)
    if mailbox is not None:
        mailbox.prep_input_mails(blocks[0])
    embeddings = model.get_embeddings(blocks)
    src_indices = torch.as_tensor(np.searchsorted(roots, sources), device=embeddings.device)
    dst_indices = torch.as_tensor(np.searchsorted(roots, destinations), device=embeddings.device)
    left = model.edge_predictor.src_fc(embeddings[src_indices])
    right = model.edge_predictor.dst_fc(embeddings[dst_indices])
    outputs = []
    for start in range(0, len(left), 32):
        pair = torch.relu(left[start:start + 32, None, :] + right[None, :, :])
        outputs.append(model.edge_predictor.out_fc(pair).reshape(-1, 1))
    return torch.cat(outputs)
