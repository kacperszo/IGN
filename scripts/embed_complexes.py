"""Extract complex-level embeddings from IGN, with the prediction head removed.

IGN is the first model here with a clean split: `IGN.forward` ends with

    readouts, weights = self.readout(bg3, bond_feats3)
    return self.FC(readouts), weights

so `readouts` is a genuine graph-level vector and `self.FC` is the head. Nothing has to be
pooled by us — this embedding is **native**, unlike SS-GNN's and PLANET's.

Reads a directory of staged complexes (`<id>/<id>.pdb` + `<id>/<id>.sdf`, what
prepare_input.py builds) rather than the zip, so IGN's mode detection is bypassed entirely.

Only one checkpoint is used, not the five that predictions average. Averaging representations
from independently trained networks is meaningless — they do not share a basis — whereas
averaging their scalar outputs is fine. The choice of checkpoint is recorded in the manifest.

Written for Python 3.6, which is what environment.yml pins.

usage:
    python embed_complexes.py --complexes /outputs/staged --out /outputs/embeddings.npz
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dgl.data.utils import Subset  # noqa: F401  (imported for parity with their pipeline)
from graph_constructor import GraphDatasetIGN, collate_fn_ign
from model_v2 import IGN
from torch.utils.data import DataLoader
from utils import pocket_truncate

# defaults from model_ign_prediction.py — they must match the checkpoint
NODE_FEAT_SIZE = 54 + 40
EDGE_FEAT_SIZE_3D = 21
NUM_LAYERS = 3
GRAPH_FEAT_SIZE = 256
OUTDIM_G3 = 200
D_FC_LAYER = 200
N_FC_LAYER = 2
DROPOUT = 0.25
N_TASKS = 1


def main():
    parser = argparse.ArgumentParser(description='Embed complexes with the head removed')
    parser.add_argument('--complexes', required=True, help='directory of <id>/<id>.pdb + <id>.sdf')
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', default='../model_save/2021-07-10_21_59_17_51270.pth',
                        help='one checkpoint; predictions average five, embeddings use one')
    parser.add_argument('--workdir', required=True, help='writable scratch for pockets and graphs')
    parser.add_argument('--num_process', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    targets = sorted(d for d in os.listdir(args.complexes)
                     if os.path.isdir(os.path.join(args.complexes, d)))
    for sub in ('pockets', 'complexes', 'tmpfiles'):
        os.makedirs(os.path.join(args.workdir, sub), exist_ok=True)

    protein_dirs, ligand_dirs, pocket_out, complex_out = [], [], [], []
    for cid in targets:
        d = os.path.join(args.complexes, cid)
        protein_dirs.append(os.path.join(d, cid + '.pdb'))
        ligand_dirs.append(os.path.join(d, cid + '.sdf'))
        pocket_out.append(os.path.join(args.workdir, 'pockets', cid + '_pkt.pdb'))
        complex_out.append(os.path.join(args.workdir, 'complexes', cid))

    # serial rather than their multiprocessing Pool: a failure here should name the complex
    # that caused it instead of vanishing into a worker
    staged = []
    for cid, p, l, po, co in zip(targets, protein_dirs, ligand_dirs, pocket_out, complex_out):
        try:
            pocket_truncate(p, l, po, co)
            staged.append(cid)
        except Exception as e:
            print('skipping %s: %s: %s' % (cid, type(e).__name__, e))

    keys = [k for k in os.listdir(os.path.join(args.workdir, 'complexes'))]
    dirs = [os.path.join(args.workdir, 'complexes', k) for k in keys]
    dataset = GraphDatasetIGN(keys=keys, labels=[0 for _ in keys], data_dirs=dirs,
                              graph_ls_file=os.path.join(args.workdir, 'graphs.bin'),
                              graph_dic_path=os.path.join(args.workdir, 'tmpfiles'),
                              num_process=args.num_process, dis_threshold=8.00, path_marker='/')
    print('complexes with graphs: %d' % len(dataset))

    model = IGN(node_feat_size=NODE_FEAT_SIZE, edge_feat_size=EDGE_FEAT_SIZE_3D,
                num_layers=NUM_LAYERS, graph_feat_size=GRAPH_FEAT_SIZE, outdim_g3=OUTDIM_G3,
                d_FC_layer=D_FC_LAYER, n_FC_layer=N_FC_LAYER, dropout=DROPOUT, n_tasks=N_TASKS)
    model.load_state_dict(torch.load(args.model, map_location='cpu')['model_state_dict'])
    model.eval()

    # a pre-hook on the head captures its input, leaving their forward() untouched so the
    # same checkpoint keeps producing the same predictions
    captured = []
    handle = model.FC.register_forward_pre_hook(lambda m, inp: captured.append(inp[0]))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn_ign)
    ids, vectors = [], []
    with torch.no_grad():
        for batch in loader:
            bg, bg3, _labels, batch_keys = batch
            captured[:] = []
            model(bg, bg3)
            vectors.append(captured[0].cpu().numpy())
            ids.extend(list(batch_keys))
    handle.remove()

    matrix = np.concatenate(vectors, axis=0)
    if len(ids) != len(matrix):
        raise RuntimeError('%d ids but %d vectors' % (len(ids), len(matrix)))
    np.savez(args.out, ids=np.array(ids), vectors=matrix)
    print('\n%d embeddings of dimension %d -> %s' % (len(ids), matrix.shape[1], args.out))


if __name__ == '__main__':
    main()
