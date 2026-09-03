"""Can IGN's encoder be moved to a different head? This decides whether transfer is possible.

`embed` showed the representation can be read out. Pretraining needs the reverse: take an
encoder trained on one task, put it in a fresh model, attach a head of a different shape, and
train that. If this does not work the whole programme is limited to whatever the authors' head
already predicts.

Four things have to hold, and each is a way it could fail quietly:

  1. the encoder weights arrive intact — not silently skipped by strict=False
  2. the head really is fresh, not carrying the old task's weights along
  3. a head of a *different output shape* runs, so the new task is not constrained to theirs
  4. gradients reach the encoder, so it can actually be fine-tuned rather than merely frozen

Written for Python 3.6, which is what this image pins.

usage: python test_encoder_transfer.py
"""

import os
import sys
import warnings

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
warnings.filterwarnings("ignore")

from model_v2 import IGN  # noqa: E402

CHECKPOINT = "/work/model_save/2021-07-10_21_59_17_51270.pth"
HEAD_PREFIXES = ["FC"]

ARCH = dict(node_feat_size=54 + 40, edge_feat_size=21, num_layers=3,
            graph_feat_size=256, outdim_g3=200, d_FC_layer=200, n_FC_layer=2,
            dropout=0.25, n_tasks=1)


def build():
    return IGN(**ARCH)


def split(state_dict):
    encoder, head = {}, {}
    for k, v in state_dict.items():
        (head if k.split(".")[0] in HEAD_PREFIXES else encoder)[k] = v
    return encoder, head


def main():
    trained = torch.load(CHECKPOINT, map_location="cpu")["model_state_dict"]
    encoder, head = split(trained)
    enc_params = sum(v.numel() for v in encoder.values())
    head_params = sum(v.numel() for v in head.values())
    print("split: encoder {} tensors / {:,} params ({:.0f}%) | head {} / {:,}".format(
        len(encoder), enc_params, 100.0 * enc_params / (enc_params + head_params),
        len(head), head_params))

    # ---- 1. the encoder arrives intact -------------------------------------------------
    model = build()
    before = {k: model.state_dict()[k].clone() for k in encoder}
    missing, unexpected = model.load_state_dict(encoder, strict=False)
    assert not unexpected, "unexpected keys: {}".format(unexpected[:5])
    assert all(k.split(".")[0] in HEAD_PREFIXES for k in missing), \
        "encoder tensors went missing: {}".format([k for k in missing
                                                   if k.split(".")[0] not in HEAD_PREFIXES][:5])
    after = model.state_dict()
    identical = sum(1 for k in encoder if torch.equal(after[k], encoder[k]))
    changed = sum(1 for k in encoder if not torch.equal(after[k], before[k]))
    print("1. encoder loaded : {}/{} tensors match the checkpoint exactly, "
          "{} differ from init".format(identical, len(encoder), changed))
    assert identical == len(encoder)

    # ---- 2. the head is fresh, not the old task's ---------------------------------------
    same_as_trained = sum(1 for k in head if torch.equal(after[k], head[k]))
    print("2. head is fresh  : {}/{} tensors still equal the trained head".format(
        same_as_trained, len(head)))
    assert same_as_trained == 0, "the old head survived the transfer"

    # ---- 3. a differently shaped head runs ----------------------------------------------
    # eight outputs rather than one: a new task, not a relabelling of theirs
    model.FC = nn.Sequential(nn.Linear(ARCH["outdim_g3"], 128), nn.ReLU(), nn.Linear(128, 8))
    graph, graph3 = make_batch()
    output, _weights = model(graph, graph3)
    print("3. new head runs  : output {} (was 1 task, now 8)".format(tuple(output.shape)))
    assert output.shape[1] == 8

    # ---- 4. gradients reach the encoder --------------------------------------------------
    model.zero_grad()
    output.sum().backward()
    encoder_modules = [m for name, m in model.named_parameters()
                       if name.split(".")[0] not in HEAD_PREFIXES]
    with_grad = [p for p in encoder_modules if p.grad is not None and p.grad.abs().sum() > 0]
    print("4. gradients flow : {}/{} encoder tensors received a non-zero gradient".format(
        len(with_grad), len(encoder_modules)))
    assert len(with_grad) > 0, "no gradient reached the encoder; it cannot be fine-tuned"

    print("\nPASS - the encoder transfers, and a new head can be trained through it.")


def make_batch():
    """One small synthetic complex in IGN's two-graph form."""
    import dgl
    n_atoms, n_bonds = 24, 60
    src = torch.randint(0, n_atoms, (n_bonds,)).tolist()
    dst = torch.randint(0, n_atoms, (n_bonds,)).tolist()

    # Built with the mutable 0.4 API their own graph_constructor uses. dgl.graph() produces a
    # heterograph here, and dgl.sum_edges in the readout then has no batch_size to work with.
    def build(edge_dim, node_dim=None):
        g = dgl.DGLGraph()
        g.add_nodes(n_atoms)
        g.add_edges(src, dst)
        if node_dim:
            g.ndata["h"] = torch.randn(n_atoms, node_dim)
        g.edata["e"] = torch.randn(n_bonds, edge_dim) if edge_dim > 1 \
            else torch.rand(n_bonds, 1) * 8
        return g

    g = build(ARCH["edge_feat_size"], ARCH["node_feat_size"])
    # the interaction graph carries one scalar per edge — the distance — which is why
    # DTIConvGraph3Layer is built with graph_feat_size + 1
    g3 = build(1)
    return dgl.batch([g]), dgl.batch([g3])


if __name__ == "__main__":
    main()
