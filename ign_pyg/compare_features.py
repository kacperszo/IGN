"""Check the DGL-free featuriser against the graphs the DGL constructor actually built.

Runs inside `ign-ref`, where both are available, so the only difference under test is the
code — same rdkit, same torchani, same pocket truncation. `export_reference.py` has already
written what the reference tier produced; this rebuilds the same complexes with
`features.py` and compares array by array.

Exact equality is the bar for the integer arrays and the one-hot blocks. The distance and
three-body columns go through float arithmetic in a different order, so those are compared
with a tolerance, reported per column so a systematic shift cannot hide behind an average.

usage (inside ign-ref):
    python compare_features.py --reference reference.npz --complexes /data --out /tmp/work
"""

# python 3.6 in the reference image, so no `from __future__ import annotations`.
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_graph  # noqa: E402
from pocket import PocketError, read_ligand, truncate  # noqa: E402


def stage(cid, complexes, work):
    """The port's own pocket truncation, so this compares that too.

    The reference arrays were built through upstream's `pocket_truncate`. Using the port's
    version here means a difference in how the pocket is cut shows up as a difference in the
    node features, which is the whole point — checking the featuriser against a pocket the
    featuriser's own pipeline did not produce would leave that step untested.
    """
    d = os.path.join(complexes, cid)
    try:
        return truncate(os.path.join(d, cid + "_protein.pdb"), read_ligand(d, cid),
                        os.path.join(work, cid + "_pkt.pdb"))
    except PocketError as e:
        print("  %s: %s" % (cid, e))
        return None


def report(name, expected, got, tol):
    if expected.shape != got.shape:
        print("    %-12s SHAPE %s vs %s" % (name, expected.shape, got.shape))
        return False
    diff = np.abs(expected.astype(np.float64) - got.astype(np.float64))
    worst = float(diff.max()) if diff.size else 0.0
    ok = worst <= tol
    print("    %-12s %-9s max |diff| = %.3e" % (name, "ok" if ok else "MISMATCH", worst))
    if not ok and expected.ndim == 2:
        per_col = diff.max(axis=0)
        bad = np.where(per_col > tol)[0]
        print("        columns off: %s" % bad[:20].tolist())
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--complexes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tolerance", type=float, default=1e-5)
    args = p.parse_args()

    ref = np.load(args.reference)
    ids = [str(x) for x in ref["ids"]]
    os.makedirs(args.out, exist_ok=True)

    failures = []
    for cid in ids:
        pair = stage(cid, args.complexes, args.out)
        if pair is None:
            print("  %s: could not be staged" % cid)
            failures.append(cid)
            continue
        g = build_graph(pair[0], pair[1])
        print("  %s" % cid)
        ok = True
        # the edge lists must agree exactly: a permutation here would put every feature on
        # the wrong edge while every shape still matched
        ok &= report("edge_index", ref[cid + "/edge_index"], g["edge_index"], 0)
        ok &= report("edge_index3", ref[cid + "/edge_index3"], g["edge_index3"], 0)
        ok &= report("h", ref[cid + "/h"], g["h"], args.tolerance)
        ok &= report("e", ref[cid + "/e"], g["e"], args.tolerance)
        ok &= report("e3", ref[cid + "/e3"], g["e3"], args.tolerance)
        if not ok:
            failures.append(cid)

    print()
    if failures:
        print("MISMATCH on %d of %d: %s" % (len(failures), len(ids), ", ".join(failures)))
        return 1
    print("MATCH — the featuriser reproduces the DGL constructor on %d complexes" % len(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
