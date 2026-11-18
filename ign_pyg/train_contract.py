"""Train the IGN port with an explicit split, resumable from a transferred encoder.

`scripts/model_train_ign.py` is the authors' trainer and it cannot be driven from here: its
paths are absolute to the authors' cluster (`/apdcephfs/private_dejunjiang/...`), its split
arrives as a pickle this project does not open, and it runs five independent repetitions in one
process. What it *does* give, and what is copied exactly below, is the hyperparameters and the
shape of the loop.

The model, the featuriser and the pocket selection are the port's, unchanged — the same code
`predict.py` scores with.

Outputs are `model.pt`, `history.csv`, `summary.json`.

## Batching, which is the only real work here

`predict.py` scores one complex at a time, so nothing in the port batches. The authors train at
`batch_size=128`, and a batch of one is a different optimisation problem, so the graphs have to
be merged properly:

* node features and both edge-feature tables concatenate;
* **both** edge index tables shift by the running node count — the covalent graph and the
  interaction graph address the same node space, so they take the same offset;
* `edge_batch` labels each interaction edge with its complex, which is what the readout scatters
  over, and `num_graphs` is the batch size.

Get the offset wrong on either table and nothing raises: the graphs silently connect to each
other and every number is plausible.

usage:
    python train_contract.py --train-graphs /cache/train.npz --val-graphs /cache/val.npz \
        --out /outputs --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import IGN  # noqa: E402
from transfer import set_training_mode, transfer_encoder  # noqa: E402

#: The authors' values, from the argparse defaults in scripts/model_train_ign.py.
AUTHOR_LR = 10 ** -3.5
AUTHOR_EPOCHS = 5000
AUTHOR_BATCH_SIZE = 128
AUTHOR_L2 = 0.00
AUTHOR_PATIENCE = 50
AUTHOR_TOLERANCE = 0.0

#: The sizes the published checkpoints were trained with — copied from predict.py, which copied
#: them from the authors' own scripts. A wrong graph_feat_size refuses to load; a wrong dropout
#: or n_FC_layer would load cleanly and change every number.
MODEL_ARGS = dict(node_feat_size=54 + 40, edge_feat_size=21, num_layers=3, graph_feat_size=256,
                  outdim_g3=200, d_FC_layer=200, n_FC_layer=2, dropout=0.25, n_tasks=1)

PARTS = ("h", "e", "edge_index", "e3", "edge_index3")


def pearson(a, b) -> float:
    """Pearson R in numpy; nan where it is undefined."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


class GraphStore:
    """The graphs for one split, held as arrays and merged on demand."""

    def __init__(self, path: str):
        store = np.load(path)
        self.ids = [str(x) for x in store["ids"]]
        self.labels = np.asarray(store["labels"], dtype=np.float32)
        self.graphs = [{name: store[f"{cid}/{name}"] for name in PARTS} for cid in self.ids]
        if len(self.labels) != len(self.ids):
            raise SystemExit(f"{path}: {len(self.ids)} ids but {len(self.labels)} labels")

    def __len__(self) -> int:
        return len(self.ids)


def collate(graphs: list[dict], labels: np.ndarray, device) -> tuple:
    """Merge complexes into one batched graph. See the module docstring for why this is the
    part worth reading twice."""
    hs, es, e3s, covs, inters, edge_batch = [], [], [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        n = g["h"].shape[0]
        hs.append(torch.from_numpy(g["h"]).float())
        es.append(torch.from_numpy(g["e"]).float())
        e3s.append(torch.from_numpy(g["e3"]).float())
        # Both tables index the same nodes, so both take the same offset.
        covs.append(torch.from_numpy(g["edge_index"]).long() + offset)
        inters.append(torch.from_numpy(g["edge_index3"]).long() + offset)
        edge_batch.append(torch.full((g["edge_index3"].shape[1],), i, dtype=torch.long))
        offset += n

    return (torch.cat(covs, dim=1).to(device),
            offset,
            torch.cat(hs).to(device),
            torch.cat(es).to(device),
            torch.cat(inters, dim=1).to(device),
            torch.cat(e3s).to(device),
            torch.cat(edge_batch).to(device),
            len(graphs),
            torch.from_numpy(labels).float().reshape(-1, 1).to(device))


def batches(store: GraphStore, batch_size: int, shuffle: bool, device, rng=None):
    order = list(range(len(store)))
    if shuffle:
        (rng or random).shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        yield collate([store.graphs[i] for i in idx], store.labels[idx], device)


def evaluate(model, store, batch_size, device, loss_fn) -> tuple[float, float]:
    """Mean loss and Pearson R over a split, model left as it was found."""
    was_training = model.training
    model.eval()
    losses, preds, trues = [], [], []
    with torch.no_grad():
        for *inputs, ys in batches(store, batch_size, False, device):
            outputs, _weights = model(*inputs)
            losses.append(float(loss_fn(outputs, ys)) * len(ys))
            preds.extend(outputs.reshape(-1).cpu().numpy().tolist())
            trues.extend(ys.reshape(-1).cpu().numpy().tolist())
    model.train(was_training)
    return sum(losses) / max(len(trues), 1), pearson(trues, preds)


def main() -> int:
    p = argparse.ArgumentParser(description="Train the IGN port with an explicit split, resumable from a transferred encoder")
    p.add_argument("--train-graphs", required=True, help="npz from prepare_graphs.py")
    p.add_argument("--val-graphs", default=None)
    p.add_argument("--out", required=True,
                   help="output directory; model.pt, history.csv and summary.json go here")
    p.add_argument("--epochs", type=int, default=None,
                   help=f"default: the authors' {AUTHOR_EPOCHS}")
    p.add_argument("--batch-size", type=int, default=None,
                   help=f"default: the authors' {AUTHOR_BATCH_SIZE}")
    p.add_argument("--lr", type=float, default=None,
                   help=f"default: the authors' {AUTHOR_LR:.3e}")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=AUTHOR_PATIENCE,
                   help="epochs without a better validation loss before stopping")
    p.add_argument("--init-encoder", default=None,
                   help="encoder-only .pt (no head tensors); makes this a fine-tune")
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--init-whole", action="store_true",
                   help="--init-encoder is a whole model.pt from an earlier run, head included: "
                        "the second stage of LP-FT, after a --freeze-encoder run trained the head")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    epochs = args.epochs if args.epochs is not None else AUTHOR_EPOCHS
    batch_size = args.batch_size if args.batch_size is not None else AUTHOR_BATCH_SIZE
    lr = args.lr if args.lr is not None else AUTHOR_LR

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device(args.device if (torch.cuda.is_available()
                                          and args.device.startswith("cuda")) else "cpu")
    train_store = GraphStore(args.train_graphs)
    print(f"device {device} | epochs {epochs} | batch {batch_size} | lr {lr:.3e}")
    print(f"{len(train_store)} training complexes")

    val_store = None
    if args.val_graphs:
        val_store = GraphStore(args.val_graphs)
        print(f"{len(val_store)} validation complexes")

    model = IGN(**MODEL_ARGS).to(device)
    provenance: dict[str, object] = {}
    if args.init_whole:
        # LP-FT (Kumar et al., 2022): start the full fine-tune from a head already fitted to the
        # pretrained features, so the first updates do not distort them through a random head.
        # transfer_encoder refuses head tensors on purpose, so a whole model loads here, strictly.
        if not args.init_encoder or args.freeze_encoder:
            raise SystemExit("--init-whole needs --init-encoder (the model.pt) and no --freeze-encoder")
        state = torch.load(args.init_encoder, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        provenance = {"init_whole": args.init_encoder, "transferred_tensors": len(state)}
        print(f"initialised the whole model, head included, from {args.init_encoder}")
    elif args.init_encoder:
        provenance = transfer_encoder(model, args.init_encoder, freeze=args.freeze_encoder)

    loss_fn = nn.MSELoss()
    if args.init_whole and val_store is not None:
        # the loaded model must score what its own run selected it on, or the load went wrong
        val_loss, val_r = evaluate(model, val_store, batch_size, device, loss_fn)
        print(f"before training: val {val_loss:.4f} R {val_r:.4f}")
        provenance.update(initial_val_loss=val_loss, initial_val_r=val_r)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=AUTHOR_L2)

    os.makedirs(args.out, exist_ok=True)
    history: list[dict] = []
    best = {"epoch": -1, "val_loss": float("inf"), "val_r": float("nan")}
    best_state = None
    since_improved = 0

    for epoch in range(epochs):
        set_training_mode(model, True, frozen_encoder=args.freeze_encoder)
        losses, preds, trues = [], [], []
        for *inputs, ys in batches(train_store, batch_size, True, device, rng):
            model.zero_grad()
            outputs, _weights = model(*inputs)
            loss = loss_fn(outputs, ys)
            loss.backward()
            optimizer.step()
            losses.append(float(loss) * len(ys))
            preds.extend(outputs.detach().reshape(-1).cpu().numpy().tolist())
            trues.extend(ys.reshape(-1).cpu().numpy().tolist())

        train_loss = sum(losses) / max(len(trues), 1)
        train_r = pearson(trues, preds)
        val_loss = val_r = ""
        if val_store is not None:
            val_loss, val_r = evaluate(model, val_store, batch_size, device, loss_fn)
            # The authors' early stopping takes a tolerance; at their default of 0.0 this is
            # plain "strictly better".
            if val_loss < best["val_loss"] - AUTHOR_TOLERANCE:
                best = {"epoch": epoch, "val_loss": val_loss, "val_r": val_r}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                since_improved = 0
            else:
                since_improved += 1

        history.append({"epoch": epoch, "train_loss": train_loss, "train_r": train_r,
                        "val_loss": val_loss, "val_r": val_r})
        print(f"epoch {epoch} | train {train_loss:.4f} R {train_r:.4f}"
              + (f" | val {val_loss:.4f} R {val_r:.4f}" if val_store is not None else ""))

        if val_store is not None and since_improved >= args.patience:
            print(f"no improvement for {args.patience} epochs; stopping")
            break

    # Without a validation split there is nothing to select on, so the last epoch is the model —
    # stated rather than implied.
    state = best_state if best_state is not None else model.state_dict()
    torch.save({k: v.detach().cpu() for k, v in state.items()},
               os.path.join(args.out, "model.pt"))

    with open(os.path.join(args.out, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "model": "ign",
        "epochs": len(history),
        "selected_epoch": best["epoch"] if best_state is not None else len(history) - 1,
        "selected_by": "val_loss" if best_state is not None else "last epoch (no val split)",
        "best_val_loss": best["val_loss"] if best_state is not None else None,
        "best_val_r": best["val_r"] if best_state is not None else None,
        "final_train_loss": history[-1]["train_loss"],
        "seed": args.seed, "epochs_requested": epochs, "batch_size": batch_size, "lr": lr,
        # One model, not the five the published numbers average. Said out loud because IGN's
        # 0.802 is an ensemble of five checkpoints and a single run is not comparable with it.
        "ensemble": 1,
        "freeze_encoder": bool(args.freeze_encoder),
        **provenance,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
