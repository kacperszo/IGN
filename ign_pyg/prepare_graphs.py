"""Featurise the complexes named in a split into one cacheable npz.

`export_port_graphs.py` next to this does the same for a whole directory, and exists for a
different reason — it runs inside the *reference* image so that featurisation and scoring can be
told apart. This one takes a split and carries the labels, which is what training needs.

The featurisation itself is shared with the scoring path: `pocket.truncate` cuts whole residues
within 5 A, and `features.build_graph` produces the same five arrays `predict.py` scores. A
training set built any other way would not be graphs of the same kind the model is scored on.

usage:
    python prepare_graphs.py --complexes /data --labels /splits/train.csv \
        --out /cache/ign_train.npz
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_graph  # noqa: E402
from pocket import PocketError, read_ligand, truncate  # noqa: E402

#: the pocket radius predict.py defaults to, and what the checkpoints were trained with
DISTANCE = 5.0

#: the five arrays that make one complex's graph
PARTS = ("h", "e", "edge_index", "e3", "edge_index3")


def read_labels(path: str) -> dict[str, float]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    missing = {"complex_id", "y_true"} - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{path}: split file is missing {sorted(missing)}")
    return {r["complex_id"]: float(r["y_true"]) for r in rows}


def main() -> int:
    p = argparse.ArgumentParser(description="Featurise one split into an npz of graphs")
    p.add_argument("--complexes", required=True)
    p.add_argument("--labels", required=True, help="csv with complex_id,y_true")
    p.add_argument("--out", required=True, help="output .npz")
    p.add_argument("--work", default="/tmp/ign_prepare",
                   help="scratch for the pocket pdb files rdkit has to read back")
    p.add_argument("--distance", type=float, default=DISTANCE)
    args = p.parse_args()

    labels = read_labels(args.labels)
    present = [c for c in labels if os.path.isdir(os.path.join(args.complexes, c))]
    absent = len(labels) - len(present)
    if absent:
        print(f"{absent} of {len(labels)} complexes in the split are not in {args.complexes}")

    os.makedirs(args.work, exist_ok=True)
    print(f"featurising {len(present)} complexes, pocket radius {args.distance} A")

    arrays: dict[str, np.ndarray] = {}
    ids, ys, skipped = [], [], []
    for cid in present:
        d = os.path.join(args.complexes, cid)
        try:
            ligand, pkt = truncate(os.path.join(d, cid + "_protein.pdb"),
                                   read_ligand(d, cid),
                                   os.path.join(args.work, cid + "_pocket.pdb"),
                                   distance=args.distance)
            graph = build_graph(ligand, pkt)
        except PocketError as e:
            skipped.append((cid, str(e)))
            continue
        except Exception as e:  # noqa: BLE001 — one bad complex must not end the run
            skipped.append((cid, f"{type(e).__name__}: {e}"))
            continue
        for name in PARTS:
            arrays[f"{cid}/{name}"] = graph[name]
        ids.append(cid)
        ys.append(labels[cid])

    print(f"featurised {len(ids)} of {len(present)}")
    for cid, why in skipped[:10]:
        print(f"  skipped {cid}: {why}")
    if not ids:
        raise SystemExit("nothing was featurised")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, ids=np.array(ids), labels=np.array(ys, dtype=np.float32), **arrays)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
