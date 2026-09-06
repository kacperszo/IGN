"""Check the ported IGN against the reference tier, on the reference tier's own input.

`export_reference.py` dumps, from inside `ign-ref`, the graphs the DGL model actually saw and
the numbers it returned. This feeds those same arrays to the port and compares. Nothing here
builds a graph, so a disagreement can only come from the layers — featurisation is checked
separately, and mixing the two makes a mismatch impossible to localise.

The checkpoint loads by name into the port, without a key mapping. That is deliberate: a
mapping is a place to hide a mistake, and identical names mean `load_state_dict(strict=True)`
is itself a check that no parameter was renamed, dropped, or invented.

usage:
    python compare_with_reference.py --reference reference.npz --model <checkpoint.pth>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import IGN  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True, help="reference.npz from export_reference.py")
    p.add_argument("--model", required=True, help="the same checkpoint the export used")
    p.add_argument("--tolerance", type=float, default=1e-4,
                   help="largest absolute difference in the prediction still called a match")
    args = p.parse_args()

    ref = np.load(args.reference)
    ids = [str(x) for x in ref["ids"]]

    # The sizes the published checkpoints were trained with, from `scripts/embed_complexes.py`.
    # A wrong graph_feat_size fails to load; a wrong dropout or n_layers would load cleanly
    # and quietly change the numbers, which is why these are copied rather than guessed.
    model = IGN(node_feat_size=54 + 40, edge_feat_size=21, num_layers=3, graph_feat_size=256,
                outdim_g3=200, d_FC_layer=200, n_FC_layer=2, dropout=0.25, n_tasks=1)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    state = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(state, strict=True)
    model.eval()

    print("%-8s %12s %12s %12s" % ("complex", "reference", "port", "difference"))
    worst = 0.0
    with torch.no_grad():
        for cid in ids:
            h = torch.from_numpy(ref[cid + "/h"]).float()
            e = torch.from_numpy(ref[cid + "/e"]).float()
            ei = torch.from_numpy(ref[cid + "/edge_index"]).long()
            e3 = torch.from_numpy(ref[cid + "/e3"]).float()
            ei3 = torch.from_numpy(ref[cid + "/edge_index3"]).long()
            expected = float(ref[cid + "/y"].reshape(-1)[0])

            # one complex per call, so every interaction edge pools into the same graph
            edge_batch = torch.zeros(ei3.shape[1], dtype=torch.long)
            out, _weights = model(ei, h.shape[0], h, e, ei3, e3, edge_batch, 1)
            got = float(out.reshape(-1)[0])
            diff = abs(got - expected)
            worst = max(worst, diff)
            print("%-8s %12.6f %12.6f %12.3e" % (cid, expected, got, diff))

    print("\nlargest difference: %.3e over %d complexes" % (worst, len(ids)))
    if worst <= args.tolerance:
        print("MATCH — the port reproduces the reference tier's arithmetic")
        return 0
    print("MISMATCH — tolerance is %.1e" % args.tolerance)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
