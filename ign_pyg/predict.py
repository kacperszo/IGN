"""Score complexes with the ported IGN — pocket, features, model, csv.

This is the whole pipeline on a modern stack: nothing here imports dgl, dgllife, or any
compiled scatter extension. It replaces `prepare_input.py` plus
`scripts/model_ign_prediction.py`, which between them stage a zip, unzip it in place, and
pick their input mode from what sits at the archive's top level.

Predictions average the five published checkpoints, as upstream does. Embeddings, when
asked for, come from one — averaging representations across independently trained models
would mix bases that were never aligned, while averaging their scalar outputs is fine.

usage:
    python predict.py --complexes /data --checkpoints /ckpt --out /outputs/predictions.csv
    python predict.py --complexes /data --checkpoints /ckpt --embeddings /outputs/e.npz
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_graph  # noqa: E402
from model import IGN  # noqa: E402
from pocket import PocketError, read_ligand, truncate  # noqa: E402

# The sizes the published checkpoints were trained with, from `scripts/embed_complexes.py`.
# A wrong graph_feat_size refuses to load; a wrong dropout or n_FC_layer would load cleanly
# and change every number, so these are copied rather than inferred.
MODEL_ARGS = dict(node_feat_size=54 + 40, edge_feat_size=21, num_layers=3, graph_feat_size=256,
                  outdim_g3=200, d_FC_layer=200, n_FC_layer=2, dropout=0.25, n_tasks=1)


def load_models(checkpoint_dir: Path) -> list[IGN]:
    """Every .pth in the directory, in sorted order so a run is reproducible."""
    paths = sorted(checkpoint_dir.glob("*.pth"))
    if not paths:
        raise SystemExit(f"no .pth checkpoints in {checkpoint_dir}")
    models = []
    for path in paths:
        model = IGN(**MODEL_ARGS)
        # weights_only: these load on modern torch, so there is no reason to open a pickle
        # with the interpreter's full powers available to it
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state), strict=True)
        model.eval()
        models.append(model)
    return models


def score(models: list[IGN], graph: dict) -> tuple[float, np.ndarray]:
    """Mean prediction over the checkpoints, plus the first one's pooled representation."""
    h = torch.from_numpy(graph["h"]).float()
    e = torch.from_numpy(graph["e"]).float()
    ei = torch.from_numpy(graph["edge_index"]).long()
    e3 = torch.from_numpy(graph["e3"]).float()
    ei3 = torch.from_numpy(graph["edge_index3"]).long()
    edge_batch = torch.zeros(ei3.shape[1], dtype=torch.long)   # one complex per call

    preds, vector = [], None
    with torch.no_grad():
        for i, model in enumerate(models):
            pooled, _weights = model.encode(ei, h.shape[0], h, e, ei3, e3, edge_batch, 1)
            preds.append(float(model.FC(pooled).reshape(-1)[0]))
            if i == 0:
                vector = pooled.reshape(-1).numpy()
    return float(np.mean(preds)), vector


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--complexes", type=Path,
                        help="directory of <id>/ holding <id>_protein.pdb and <id>_ligand.sdf")
    source.add_argument("--graphs", type=Path,
                        help="an npz of prebuilt graphs from export_port_graphs.py, scored "
                             "without touching rdkit — the way to tell a featurisation "
                             "difference apart from a modelling one")
    p.add_argument("--checkpoints", required=True, type=Path,
                   help="directory of .pth files; all are averaged")
    p.add_argument("--out", type=Path, help="where to write predictions.csv")
    p.add_argument("--embeddings", type=Path, help="where to write embeddings.npz")
    p.add_argument("--work", type=Path, default=Path("/tmp/ign_work"),
                   help="scratch for the pocket pdb files rdkit has to read back")
    p.add_argument("--distance", type=float, default=5.0,
                   help="pocket radius in angstroms, selecting whole residues")
    args = p.parse_args()
    if not args.out and not args.embeddings:
        raise SystemExit("nothing to do: pass --out, --embeddings, or both")

    models = load_models(args.checkpoints)
    print(f"loaded {len(models)} checkpoint(s)")
    args.work.mkdir(parents=True, exist_ok=True)

    rows, ids, vectors, skipped = [], [], [], []

    if args.graphs:
        store = np.load(args.graphs)
        targets = [str(x) for x in store["ids"]]
        for cid in targets:
            graph = {name: store[f"{cid}/{name}"]
                     for name in ("h", "e", "edge_index", "e3", "edge_index3")}
            pred, vector = score(models, graph)
            rows.append((cid, pred))
            ids.append(cid)
            vectors.append(vector)
        return write(args, rows, ids, vectors, targets, skipped)

    targets = sorted(d for d in args.complexes.iterdir() if d.is_dir())
    for d in targets:
        cid = d.name
        try:
            ligand, pkt = truncate(str(d / f"{cid}_protein.pdb"), read_ligand(str(d), cid),
                                   str(args.work / f"{cid}_pocket.pdb"), distance=args.distance)
            graph = build_graph(ligand, pkt)
            pred, vector = score(models, graph)
        except PocketError as e:
            skipped.append((cid, str(e)))
            continue
        except Exception as e:  # noqa: BLE001 — one bad complex must not end the run
            skipped.append((cid, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
            continue
        rows.append((cid, pred))
        ids.append(cid)
        vectors.append(vector)

    return write(args, rows, ids, vectors, targets, skipped)


def write(args, rows, ids, vectors, targets, skipped) -> int:
    """Report what was scored and put it where it was asked for."""
    print(f"scored {len(rows)} of {len(targets)} complexes")
    for cid, why in skipped:
        print(f"  skipped {cid}: {why}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["complex_id", "y_pred"])
            writer.writerows(rows)
        print(f"-> {args.out}")
    if args.embeddings:
        args.embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.embeddings, ids=np.array(ids), vectors=np.stack(vectors))
        print(f"-> {args.embeddings}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
