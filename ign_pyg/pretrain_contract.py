"""Pretrain IGN's encoder on MD: predict descriptors of a later frame from the current one.

The encoder is IGN's own, unchanged: `IGN.encode()` up to the pooled 200-d vector. The task head
is new and exists only here — `pretrain_head`, an MLP from that vector to the K descriptors — so
nothing of the affinity head `FC` is trained or kept. What this produces for the rest of the
programme is `encoder.pt`: encoder tensors only, in the `{"encoder": ...}` form
`transfer.read_encoder` accepts, loadable strictly into a fresh IGN by
`gnnb train --init-encoder`.

Targets are z-scored with statistics fitted on the training pairs only, and the loss is MSE over
the targets that are present — a NaN (a pocket with no residue in range, say) is masked out, not
zeroed, since a zero after z-scoring is a claim about the value.

**Two targets, `--target absolute|delta`.** `absolute` is Kuba's formulation: the descriptors of
frame n+gap. An early run on 137 complexes showed that at a 1 ns gap those are ~90% the current
frame's own descriptors — copying frame n scores a loss of 0.12 where predicting the mean scores
1.29 — so the absolute task is mostly *describing the structure in view*. `delta` asks for the
change, descriptors(n+gap) - descriptors(n): no structure-description shortcut exists there, and
whatever the encoder learns has to be about where the complex is going. Running both separates
"knows structure" from "knows motion".

Two numbers make the curve readable:
  * **persistence**: the validation loss of assuming nothing changes — copying frame n's
    descriptors (absolute), or predicting a zero change (delta). Beating it is the evidence that
    the model learned something about where the complex is going.
  * **mean**: the loss of predicting the training mean, which is 1.0 by construction of the
    z-scoring and is there to calibrate the eye.

Outputs follow `harness/training.py` — `model.pt`, `history.csv`, `summary.json` — plus
`encoder.pt`. Resumable with `--state` and `--max-minutes`, as SIGN's trainer is.

usage:
    python pretrain_contract.py --train /cache/ign_md_train --val /cache/ign_md_val \
        --out /outputs --epochs 20 --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import IGN  # noqa: E402
from train_contract import AUTHOR_BATCH_SIZE, AUTHOR_LR, MODEL_ARGS, PARTS, collate, pearson  # noqa: E402
from transfer import is_head, transfer_encoder  # noqa: E402

#: not an author value — IGN has no pretraining. A modest default; the curve decides.
DEFAULT_EPOCHS = 20
HEAD_HIDDEN = 256


def shards_of(prefix: str) -> list[str]:
    paths = sorted(glob.glob(f"{prefix}_[0-9][0-9][0-9].npz"))
    if not paths:
        raise SystemExit(f"no shards match {prefix}_NNN.npz")
    return paths


class Shard:
    """One npz shard held in memory: graphs, targets and current-frame descriptors."""

    def __init__(self, path: str):
        with np.load(path) as z:
            self.ids = [str(x) for x in z["ids"]]
            self.targets = z["targets"].astype(np.float32)
            self.current = z["current"].astype(np.float32)
            self.graphs = [{p: z[f"{i}/{p}"] for p in PARTS} for i in range(len(self.ids))]


#: shards held for the whole run when --in-memory is given; empty otherwise
SHARDS: dict[str, Shard] = {}


def raw_targets(targets: np.ndarray, current: np.ndarray, mode: str) -> np.ndarray:
    """What is predicted, before z-scoring: the later frame's descriptors, or their change."""
    return targets - current if mode == "delta" else targets


def no_change(targets: np.ndarray, current: np.ndarray, mode: str) -> np.ndarray:
    """The persistence prediction in raw units: frame n's descriptors, or a change of zero."""
    return np.zeros_like(targets) if mode == "delta" else current


def fit_normaliser(train_paths: list[str], mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-target mean and std over the training pairs, NaN-aware."""
    parts = []
    for p in train_paths:
        with np.load(p) as z:
            parts.append(raw_targets(z["targets"], z["current"], mode))
    stacked = np.concatenate(parts).astype(np.float64)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(target)
    diff = torch.where(mask, pred - torch.nan_to_num(target), torch.zeros_like(pred))
    return (diff ** 2).sum() / mask.sum().clamp(min=1)


def attach_head(model: IGN, k: int, style: str = "plain") -> None:
    """The pretraining head, as an attribute of IGN itself so encoder tensors keep IGN's names.

    `plain` is Linear-ReLU-Linear. `ign` is built like the authors' own affinity head `FC`:
    Linear, Dropout, LeakyReLU, BatchNorm1d, then the output layer. The difference matters
    because `IGN.encode` returns a *sum* over edges, so its scale grows with the complex: the
    plain head starts at losses of 8-120 on z-scored targets and, on 125 or 250 complexes or
    the 25-column target set, fell to predicting the mean from epoch 3 on. FC's BatchNorm is
    what absorbs that scale in the authors' own model."""
    d = MODEL_ARGS["outdim_g3"]
    if style == "ign":
        model.pretrain_head = nn.Sequential(
            nn.Linear(d, HEAD_HIDDEN), nn.Dropout(MODEL_ARGS["dropout"]), nn.LeakyReLU(),
            nn.BatchNorm1d(HEAD_HIDDEN), nn.Linear(HEAD_HIDDEN, k))
    else:
        model.pretrain_head = nn.Sequential(nn.Linear(d, HEAD_HIDDEN), nn.ReLU(), nn.Linear(HEAD_HIDDEN, k))


def forward(model: IGN, inputs) -> torch.Tensor:
    cov, n_nodes, h, e, inter, e3, edge_batch, n_graphs = inputs
    pooled, _ = model.encode(cov, n_nodes, h, e, inter, e3, edge_batch, n_graphs)
    return model.pretrain_head(pooled)


def encoder_state(model: IGN) -> dict[str, torch.Tensor]:
    """Encoder tensors only: neither the affinity head FC nor the pretraining head."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if not is_head(k) and not k.startswith("pretrain_head.")}


def run_epoch(model, paths, mean, std, batch_size, device, rng, optimizer=None, mode="absolute", mix=1):
    """One pass over a list of shards; trains if an optimizer is given. Returns loss, persistence
    loss, and mean Pearson across targets.

    Shards hold consecutive pairs of a sorted frame list, ~67 complexes each, so shuffling inside
    one shard leaves every batch with near-duplicate frames of the same few complexes. With
    `mix` > 1, training pools that many shards and shuffles across the pool."""
    training = optimizer is not None
    model.train(training)
    order = list(paths)
    if training:
        rng.shuffle(order)
    mean_t, std_t = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    tot, tot_persist, n_batches = 0.0, 0.0, 0
    preds, trues = [], []
    group = mix if training else 1
    for g in range(0, len(order), group):
        shards = [SHARDS.get(path) or Shard(path) for path in order[g:g + group]]
        idx = [(s, i) for s in range(len(shards)) for i in range(len(shards[s].ids))]
        if training:
            rng.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            b = idx[start:start + batch_size]
            *inputs, _ = collate([shards[s].graphs[i] for s, i in b], np.zeros(len(b), np.float32), device)
            targets = np.stack([shards[s].targets[i] for s, i in b])
            currents = np.stack([shards[s].current[i] for s, i in b])
            raw = raw_targets(targets, currents, mode)
            target = (torch.from_numpy(raw).to(device) - mean_t) / std_t
            with torch.set_grad_enabled(training):
                out = forward(model, inputs)
                loss = masked_mse(out, target)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            persist = no_change(targets, currents, mode)
            current = (torch.from_numpy(persist).to(device) - mean_t) / std_t
            tot += float(loss)
            tot_persist += float(masked_mse(torch.nan_to_num(current), target))
            n_batches += 1
            if not training:
                preds.append(out.detach().cpu().numpy())
                trues.append(target.cpu().numpy())
    r = float("nan")
    if preds:
        p, t = np.concatenate(preds), np.concatenate(trues)
        rs = [pearson(t[~np.isnan(t[:, j]), j], p[~np.isnan(t[:, j]), j]) for j in range(t.shape[1])]
        r = float(np.nanmean(rs))
    return tot / max(n_batches, 1), tot_persist / max(n_batches, 1), r


def main() -> int:
    ap = argparse.ArgumentParser(description="Pretrain IGN's encoder on MD next-frame descriptors")
    ap.add_argument("--train", required=True, help="shard prefix from prepare_md_pairs.py")
    ap.add_argument("--val", required=True, help="shard prefix for validation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=AUTHOR_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=AUTHOR_LR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--state", default=None, help="resumable state file, rewritten every epoch")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--init-encoder", default=None,
                    help="start the encoder from these weights instead of at random — published "
                         "weights, so that MD pretraining is a second stage on top of what the "
                         "model already knows; loaded strictly, as `gnnb train --init-encoder` does")
    ap.add_argument("--head", choices=["plain", "ign"], default="plain",
                    help="pretraining head; `ign` mirrors the authors' FC (see attach_head)")
    ap.add_argument("--in-memory", action="store_true",
                    help="read every shard once and keep it (~1 GB RAM per 7k pairs); by default each "
                         "epoch re-reads them, which leaves the GPU idle most of the time. Same "
                         "batches, same order, same numbers either way")
    ap.add_argument("--target", choices=["absolute", "delta"], default="absolute",
                    help="descriptors of frame n+gap, or their change from frame n — see above")
    ap.add_argument("--mix", type=int, default=1,
                    help="training shards pooled and shuffled together (see run_epoch); 1 = the "
                         "behaviour before 2026-10-20. ~1 GB RAM per 7k pairs held")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available()
                                          and args.device.startswith("cuda")) else "cpu")

    train_paths, val_paths = shards_of(args.train), shards_of(args.val)
    if args.in_memory:
        t0 = time.time()
        SHARDS.update({p: Shard(p) for p in train_paths + val_paths})
        print(f"{len(SHARDS)} shards in memory ({time.time() - t0:.0f} s)", flush=True)
    with open(f"{args.train}_meta.json") as f:
        meta = json.load(f)
    targets = meta["targets"]
    mean, std = fit_normaliser(train_paths, args.target)
    print(f"device {device} | {len(train_paths)} train shards, {len(val_paths)} val shards | "
          f"{len(targets)} targets ({args.target}) | epochs {args.epochs} | lr {args.lr:.3e}",
          flush=True)

    model = IGN(**MODEL_ARGS)
    # before the pretraining head exists: transfer_encoder insists that every tensor of the model
    # is either in the encoder file or in IGN's own head, and the new head is neither
    if args.init_encoder:
        transfer_encoder(model, args.init_encoder)
    attach_head(model, len(targets), args.head)
    model.to(device)
    # FC is never used here; excluding it from the optimiser keeps it at its initialisation
    params = [p for n, p in model.named_parameters() if not is_head(n)]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    history, best, best_epoch, best_encoder = [], float("inf"), -1, None
    first_epoch, started = 0, time.time()
    if args.state and os.path.exists(args.state):
        state = torch.load(args.state, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first_epoch = int(state["epoch"]) + 1
        best, best_epoch = float(state["best"]), int(state["best_epoch"])
        best_encoder, history = state["best_encoder"], list(state["history"])
        rng.setstate(tuple(state["rng"][0:1]) + (tuple(state["rng"][1]),) + tuple(state["rng"][2:]))
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed after epoch {first_epoch - 1}", flush=True)

    for epoch in range(first_epoch, args.epochs):
        tr_loss, tr_persist, _ = run_epoch(model, train_paths, mean, std, args.batch_size,
                                           device, rng, optimizer, mode=args.target, mix=args.mix)
        va_loss, va_persist, va_r = run_epoch(model, val_paths, mean, std, args.batch_size,
                                              device, rng, mode=args.target)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_r": "",
                        "val_loss": va_loss, "val_r": va_r,
                        "train_persistence": tr_persist, "val_persistence": va_persist})
        print(f"epoch {epoch} | train {tr_loss:.4f} | val {va_loss:.4f} R {va_r:.3f} | "
              f"persistence {va_persist:.4f}", flush=True)
        if va_loss < best:
            best, best_epoch, best_encoder = va_loss, epoch, encoder_state(model)
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                       os.path.join(args.out, "model.pt"))
        if args.state:
            tmp = args.state + ".tmp"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "best": best, "best_epoch": best_epoch,
                        "best_encoder": best_encoder, "history": history,
                        "rng": list(rng.getstate()), "torch_rng": torch.get_rng_state()}, tmp)
            os.replace(tmp, args.state)
        if args.max_minutes is not None and time.time() - started >= args.max_minutes * 60:
            print(f"time budget reached after epoch {epoch}; resume with the same --state")
            break

    torch.save({"encoder": best_encoder, "source": "ign_pyg/pretrain_contract.py",
                "targets": targets, "target_mode": args.target, "gap": meta["gap"],
                "best_epoch": best_epoch},
               os.path.join(args.out, "encoder.pt"))
    with open(os.path.join(args.out, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    last = history[-1]
    summary = {
        "model": "ign", "capability": "pretrain", "task": "MD next-frame descriptors",
        "target_mode": args.target,
        "targets": targets, "gap_frames": meta["gap"], "pairs_train": meta["pairs"],
        "epochs": len(history), "epochs_requested": args.epochs,
        "complete": len(history) >= args.epochs,
        "best_epoch": best_epoch, "best_val_loss": best,
        "val_persistence_loss": last["val_persistence"],
        "beats_persistence": best < last["val_persistence"],
        "normaliser": {"mean": mean.tolist(), "std": std.tolist()},
        "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size, "head": args.head, "init_encoder": args.init_encoder, "mix": args.mix,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("best_epoch", "best_val_loss",
                                               "val_persistence_loss", "beats_persistence")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
