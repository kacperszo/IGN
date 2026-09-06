"""Export IGN's DGL graphs and the reference model's output, so the PyG port can be checked.

Runs inside the `ign-ref` image — the only configuration of IGN that reproduces its published
behaviour (torch 1.3.1, dgl 0.4.3, R=0.802 over 279 core-set complexes). The `modern` tier is
not usable as a reference: `gnnb verify` puts it at R=-0.155 against the same golden file.

What comes out is the input the model actually sees, as plain arrays, plus what it returned:

    g   the covalent graph   node features h, edge features e, and the edge list
    g3  the interaction graph  edge features e and its edge list
    y   the model's prediction for the same input

Dumping the graphs rather than rebuilding them is deliberate. It isolates the port's
arithmetic from its featurisation: if the numbers disagree here, the cause is in the layers,
not in how the graph was built.

The ligand is staged through `pocket.read_ligand`, so both sides of any later comparison get
the same molecule. Reading it any other way is not equivalent — `MolFromMolFile` on 1eby's
original sdf keeps its hydrogens where `SDMolSupplier` drops them, and the graph goes from
288 nodes to 382 with a prediction 4.4 log units adrift.

usage (inside ign-ref):
    python export_reference.py --complexes /data --model /ckpt/... --out /outputs
"""

# No `from __future__ import annotations`: this runs in the reference image, which is
# python 3.6 — the authors' own environment, and older than that import.
import argparse
import os
import sys

import numpy as np
import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--complexes", required=True)
    p.add_argument("--model", required=True, help="one of the five checkpoints")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--scripts", default="/work/scripts")
    args = p.parse_args()

    sys.path.insert(0, args.scripts)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rdkit import Chem  # noqa: E402
    from graph_constructor import GraphDatasetIGN, collate_fn_ign  # noqa: E402
    from utils import pocket_truncate  # noqa: E402
    from model_v2 import IGN  # noqa: E402
    from pocket import PocketError, read_ligand  # noqa: E402

    ids = sorted(d for d in os.listdir(args.complexes)
                 if os.path.isdir(os.path.join(args.complexes, d)))[: args.limit]
    print("complexes:", ids)

    # The graph builder does not read structures: it reads a pickle holding the (ligand,
    # pocket) pair that `pocket_truncate` produces. Skipping that step leaves the dataset
    # silently empty, because the graph generation runs in a multiprocessing Pool that
    # swallows the exception and writes a cache file containing nothing.
    work = os.path.join(args.out, "work")
    for sub in ("pockets", "complexes"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)

    staged, dirs = [], []
    for cid in ids:
        d = os.path.join(args.complexes, cid)
        # IGN's own naming, the same layout prepare_input.py stages
        protein = os.path.join(d, cid + ".pdb")
        if not os.path.exists(protein):
            protein = os.path.join(d, cid + "_protein.pdb")

        # The ligand goes through the port's reader, and both sides of the comparison then
        # see the same molecule. That is the point: this file exists to test the graph
        # builder, so the input to it must be held fixed. How the ligand is read is checked
        # separately, end to end, against the numbers model_ign_prediction.py produces.
        try:
            mol = read_ligand(d, cid, work=work)
        except PocketError as e:
            print("  %s: %s" % (cid, e))
            continue
        ligand = os.path.join(work, cid + "_staged.sdf")
        Chem.MolToMolFile(mol, ligand)

        out_pocket = os.path.join(work, "pockets", cid + "_pkt.pdb")
        out_complex = os.path.join(work, "complexes", cid)
        try:
            pocket_truncate(protein, ligand, out_pocket, out_complex)
        except Exception as e:
            print("  %s: pocket_truncate failed: %s: %s" % (cid, type(e).__name__, e))
            continue
        if not os.path.exists(out_complex):
            print("  %s: pocket_truncate wrote nothing" % cid)
            continue
        staged.append(cid)
        dirs.append(out_complex)
    if not staged:
        raise SystemExit("nothing staged")

    cache = os.path.join(args.out, "graphs")
    os.makedirs(cache, exist_ok=True)
    dataset = GraphDatasetIGN(keys=staged, labels=[0.0] * len(staged), data_dirs=dirs,
                              graph_ls_file=os.path.join(args.out, "graphs.bin"),
                              graph_dic_path=cache, num_process=1)
    print("dataset holds %d complexes" % len(dataset))

    # The sizes the published checkpoints were trained with, from `scripts/embed_complexes.py`.
    # Guessing them is not an option: a wrong graph_feat_size loads as a shape mismatch, but a
    # wrong dropout or n_layers would load cleanly and change the numbers.
    model = IGN(node_feat_size=54 + 40, edge_feat_size=21, num_layers=3, graph_feat_size=256,
                outdim_g3=200, d_FC_layer=200, n_FC_layer=2, dropout=0.25, n_tasks=1)
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    # Predictions come through a DataLoader with the authors' own collate function, which is
    # the path model_ign_prediction.py uses and the only one that produces a properly batched
    # graph — dgl.sum_edges reads `batch_size` off it and fails on a bare graph.
    from torch.utils.data import DataLoader

    def edges_by_eid(graph):
        """Edges in edge-id order, which is the order `edata` is indexed by.

        `edges()` with no argument makes no such promise: a CSR-backed graph returns them
        grouped by destination, and pairing that with `edata['e']` silently compares two
        different conventions.
        """
        try:
            return graph.edges(order="eid")
        except TypeError:  # dgl 0.4 signature without the keyword
            return graph.edges()

    # Read the tensors off the batched graph rather than going back to `dataset[i]`:
    # dgl 0.4's `batch` moves node and edge data into the batched graph and leaves the
    # constituents empty, so the second read raises KeyError on 'h'. A batch of one is the
    # same graph, so nothing is lost by taking it from here.
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_ign)
    arrays, order = {}, []
    with torch.no_grad():
        for bg, bg3, _l, keys in loader:
            key = keys[0]
            src, dst = edges_by_eid(bg)
            s3, d3 = edges_by_eid(bg3)
            # Read the inputs before the model runs: IGN.forward starts with
            # `bg.ndata.pop('h')` and `bg.edata.pop('e')`, so afterwards the graph no longer
            # carries what it was given.
            arrays["%s/h" % key] = bg.ndata["h"].numpy()
            arrays["%s/e" % key] = bg.edata["e"].numpy()
            arrays["%s/edge_index" % key] = np.stack([src.numpy(), dst.numpy()])
            arrays["%s/e3" % key] = bg3.edata["e"].numpy()
            arrays["%s/edge_index3" % key] = np.stack([s3.numpy(), d3.numpy()])
            out, _weights = model(bg, bg3)
            arrays["%s/y" % key] = out.numpy()
            order.append(key)
            print("  %s: %d atoms, %d bonds, %d interactions -> %.5f" %
                  (key, bg.number_of_nodes(), bg.number_of_edges(),
                   bg3.number_of_edges(), float(out)))

    np.savez(os.path.join(args.out, "reference.npz"), ids=np.array(order), **arrays)
    print(f"\n-> {os.path.join(args.out, 'reference.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
