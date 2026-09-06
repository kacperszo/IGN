"""Build the port's graphs with the reference image's rdkit and save them as arrays.

The point is to separate two things that a single end-to-end number cannot. The port scores
R=-0.04 against the golden file when it runs entirely on a current stack, and there are two
candidate causes: the featuriser is wrong, or rdkit changed how it reads these files between
2021 and 2026 and the molecules themselves are different.

So: featurise here, inside `ign-ref`, where rdkit is the version the checkpoints were trained
against. Score the result on a modern torch. If the correlation comes back, the port is right
and rdkit is the variable; if it does not, the featuriser has a real defect.

usage (inside ign-ref):
    python export_port_graphs.py --complexes /data --out /out/port_graphs.npz
"""

# python 3.6 in the reference image, so no `from __future__ import annotations`.
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_graph  # noqa: E402
from pocket import PocketError, read_ligand, truncate  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--complexes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", default="/tmp/port_graphs")
    args = p.parse_args()

    os.makedirs(args.work, exist_ok=True)
    targets = sorted(d for d in os.listdir(args.complexes)
                     if os.path.isdir(os.path.join(args.complexes, d)))

    arrays, ids, skipped = {}, [], []
    for cid in targets:
        d = os.path.join(args.complexes, cid)
        try:
            ligand, pkt = truncate(os.path.join(d, cid + "_protein.pdb"),
                                   read_ligand(d, cid),
                                   os.path.join(args.work, cid + "_pocket.pdb"))
            g = build_graph(ligand, pkt)
        except PocketError as e:
            skipped.append((cid, str(e)))
            continue
        except Exception as e:            # one bad complex must not end the run
            skipped.append((cid, "%s: %s" % (type(e).__name__, e)))
            continue
        for name, value in g.items():
            arrays["%s/%s" % (cid, name)] = value
        ids.append(cid)

    print("featurised %d of %d" % (len(ids), len(targets)))
    for cid, why in skipped:
        print("  skipped %s: %s" % (cid, why))
    np.savez(args.out, ids=np.array(ids), **arrays)
    print("-> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
