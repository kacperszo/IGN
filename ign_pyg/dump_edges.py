"""Dump one complex's graph from whichever `graph_constructor` is first on sys.path.

Run it twice — once against the copy baked into `ign-ref`, once against the repo's rewrite —
and compare. The rewrite replaced `dgl.DGLGraph()` plus mutation with `dgl.graph(...)`, and
its own comment warns that edge ORDER is load-bearing because features are assigned
positionally afterwards. If the two constructors emit edges in different orders, features land
on the wrong edges: the model still runs, still returns plausible numbers, and correlates at
R=-0.155 where the reference tier reaches 0.802. That is the hypothesis this script tests.

The graph is captured in memory rather than read back from the cache file: `graphs_from_mol_ign`
persists via `pickle.dump`, and nothing here needs to unpickle anything to compare edges.

usage (inside ign-ref):
    python dump_edges.py <scripts-dir> <complex-id> <data-dir> <out-dir> <tag>
"""
# No `from __future__ import annotations`: the reference image is python 3.6.
import os
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
import graph_constructor  # noqa: E402
from graph_constructor import graphs_from_mol_ign  # noqa: E402
from utils import pocket_truncate  # noqa: E402

cid, data, out, tag = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
work = os.path.join(out, "w_" + tag)
for sub in ("pockets", "complexes", "dic"):
    os.makedirs(os.path.join(work, sub), exist_ok=True)

d = os.path.join(data, cid)


def _pick_ligand(d, cid):
    """sdf when RDKit can read it, the mol2 beside it otherwise — IGN's own fallback.

    `pocket_truncate` reports a failure by printing and returning, leaving no complex file
    behind; the next step then dies on a missing path rather than on the real cause.
    """
    from rdkit import Chem

    sdf = os.path.join(d, cid + "_ligand.sdf")
    if os.path.exists(sdf) and any(m is not None for m in Chem.SDMolSupplier(sdf)):
        return sdf
    return os.path.join(d, cid + "_ligand.mol2")


pocket_truncate(os.path.join(d, cid + "_protein.pdb"),
                _pick_ligand(d, cid),
                os.path.join(work, "pockets", cid + "_pkt.pdb"),
                os.path.join(work, "complexes", cid))
if not os.path.exists(os.path.join(work, "complexes", cid)):
    raise SystemExit("%s: pocket_truncate wrote no complex for %s" % (tag, cid))


class _Capture(object):
    """Stands in for the `pickle` module inside graph_constructor, keeping the graph in RAM.

    Only `dump` is intercepted. `load` is delegated, because the constructor reads back the
    (ligand, pocket) pair that `pocket_truncate` wrote moments earlier in this same container
    — a file this run produced from the PDB and SDF text, not foreign data.
    """

    def __init__(self, real):
        self._real = real
        self.obj = None

    def dump(self, obj, f):
        self.obj = obj

    def __getattr__(self, name):
        return getattr(self._real, name)


cap = _Capture(graph_constructor.pickle)
graph_constructor.pickle = cap

# called directly, not through GraphDatasetIGN: the dataset runs this inside a Pool that
# swallows exceptions and silently leaves an empty cache behind
graphs_from_mol_ign(os.path.join(work, "complexes", cid), cid, 0.0, os.path.join(work, "dic"))
if cap.obj is None:
    raise SystemExit("%s: graph construction failed — see the traceback above" % tag)

g, g3 = cap.obj["g"], cap.obj["g3"]


def _edges_by_eid(graph):
    """Edges in edge-id order, which is the order `edata` is indexed by.

    `edges()` with no argument does not promise that order: a CSR-backed graph returns them
    grouped by destination instead, so pairing that output with `edata['e']` compares two
    different conventions and invents a misalignment that is not there.
    """
    try:
        return graph.edges(order="eid")
    except TypeError:  # older signature without the keyword
        return graph.edges()


src, dst = _edges_by_eid(g)
s3, d3 = _edges_by_eid(g3)
np.savez(os.path.join(out, "edges_%s.npz" % tag),
         edge_index=np.stack([src.numpy(), dst.numpy()]),
         h=g.ndata["h"].numpy(), e=g.edata["e"].numpy(),
         edge_index3=np.stack([s3.numpy(), d3.numpy()]), e3=g3.edata["e"].numpy())
print("%s: %d nodes, %d bonds, %d interactions" %
      (tag, g.number_of_nodes(), g.number_of_edges(), g3.number_of_edges()))
