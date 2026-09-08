# CLAUDE.md — IGN

## What this is

Affinity prediction from an interaction graph network, averaged over five checkpoints. The
authors' code is the most foreign in the benchmark and the one the container isolation exists
for.

Three tiers off one source tree. **Use `ign.torch`** unless you are reproducing a published
number: it is the model reimplemented on plain PyTorch, with no DGL, dgllife, PyG or compiled
scatter extension, so it runs on any current torch.

| variant | stack | `gnnb verify` |
|---|---|---|
| `ign.reference` | torch 1.3.1, dgl 0.4.3, CPU only | 279/279, max abs diff 1e-06 |
| `ign.modern` | torch 2.4.1+cu124, dgl 2.4.0 | 279/279, 0.598 (tolerance 0.75) |
| `ign.torch` | torch 2.5.1+cu124, no DGL | 279/279, 0.591 (tolerance 0.75) |

## Current state

Done for `predict` and `embed` on all three tiers.

| | |
|---|---|
| CASF-2016 scoring | **R 0.802** — highest in the benchmark — RMSE 1.468, c-index 0.827 |
| ranking | rho 0.626, top-1 0.526 |
| embedding | 200d, **native**, probe R 0.765, retains **95%** |
| n | 279 of 285; six ligands will not sanitise under the reference tier's 2021 RDKit |

Two things travel with that 0.802. It **averages five checkpoints**, so it is an ensemble
against single models — and its training list is not published, while 266 of the 285 core-set
complexes sit inside PDBbind refined.

The embedding is native because `IGN.forward` ends `self.FC(readouts)`: `readouts` is already a
graph vector and `FC` is the head. Nothing had to be invented, and it retains more than any
pooling we chose ourselves.

## Hard-won facts (do NOT regress these)

- **DGL 0.4.3 asks for its backend interactively on first import.** There is no stdin in a
  build, so the image dies on `EOFError` from `input()` unless `DGLBACKEND=pytorch` is set.
- **Pillow, pulled in by matplotlib, needs `libtiff.so.5`.** The base image has no libtiff and
  current Debian ships `.so.6`, so it has to come from conda-forge pinned at 4.2.
  `graph_constructor.py` does `from pylab import *`, so matplotlib is on the import path
  whether or not anything plots.
- **The input protocol is a directory holding exactly one zip**, whose top level must contain
  only directories. A stray `.pdb` or `.sdf` there silently switches IGN to mode1, which
  assumes one rigid protein and many ligand poses.
- **A half-staged complex is worse than a missing one.** IGN's mode2 loop assigns `sdf_file`
  only when it sees one and never resets it between targets, so a directory with a protein
  and no ligand either crashes on an undefined name or silently reuses the previous complex's
  ligand. `prepare_input.py` therefore reads the ligand before creating anything.
- **`--test_file_path` must be writable**: IGN unzips into it and writes `prediction.csv`
  there, so it is `/outputs` and never `/data`.
- **There is no single checkpoint.** Five weight files are loaded from a path relative to the
  script and averaged, so they are baked into the image and the registry declares none.
- **`weights_only` does not exist in torch 1.3.1**, three years before the parameter. What
  compensates: all five checkpoints scan clean — `collections.OrderedDict`, the two torch
  storage classes and `torch._utils._rebuild_tensor_v2`, exactly what a plain state dict
  holds — plus the container isolation, which `tools/verify_isolation.py` measures rather
  than assumes. A clean scan means "only allowlisted references", never "the file is empty".
- **`FC` builds three Linear layers when asked for two.** Two consecutive `if`s where an
  `if/elif` was meant, so `j == 0` takes both branches. Writing the obvious `elif` changes the
  architecture; `load_state_dict(strict=True)` is what catches it, which is why the port keeps
  upstream's parameter names rather than tidier ones.
- **The three chirality features are constant zero, and always were.** `pocket_truncate`
  pickles `[ligand, pocket]` and the graph builder unpickles them; rdkit's pickler drops atom
  properties, so `_CIPCode` and `_ChiralityPossible` are gone before the featuriser looks.
  Computing them properly makes the model disagree with its own checkpoints. Restoring them is
  a retraining decision, not a porting one.
- **The ligand's reader decides the molecule.** The chain is `SDMolSupplier` → `MolToMolFile`
  → `MolFromMolFile`. Reading the original file with `MolFromMolFile` instead is not
  equivalent: on 1eby it keeps 40 hydrogens the supplier drops, the complex goes from 288
  nodes to 382, and the prediction moves 4.4 log units.
- **`MolFromMolFile` cannot open a mol2 at all**, so a mol2 has to be rewritten as an sdf
  first. Pointing the reader at it looks like a fallback and never fires — that is how 89 of
  285 core-set complexes were being dropped.
- **Ask for `edges(order="eid")` before pairing edges with `edata`.** A CSR-backed DGL graph
  returns edges grouped by destination otherwise, and comparing the two conventions invents a
  misalignment that is not there.

## rdkit is part of the model, as much as torch is

Three changes between rdkit 2021.03 and 2026.03 each destroy the reproduction on their own,
and none of them raises anything — every array keeps its shape:

| change | effect if ignored |
|---|---|
| hydrogens survive the sdf round trip | 288 nodes become 382; predictions go negative |
| MOL bond type 4 no longer flags **atoms** aromatic | `atom_is_aromatic` all zero on ligands |
| `RemoveHs` moved a pyrrole N's H to `numExplicitHs` | one total-H feature column shifts |

Fixed one at a time, the port went R = -0.07 → 0.77 → 0.978 → 0.9993. The same two corrections
in `pocket_truncate` took `ign.modern` from max abs diff 14.6 to 0.598 — **two independent
codebases, the same three fixes, the same recovery.** That is what cleared DGL, which had been
blamed for the regression for months; the `DGLGraph()` → `dgl.graph()` rewrite produces
bit-identical graphs in edge-id order, tested with `ign_pyg/dump_edges.py`.

**Handle the behaviour in code rather than pinning a version**, so the model stops depending
on which rdkit happens to be installed. `ign_pyg/pocket.py` does exactly that.

DGL is still a dead end for this model — no commits in about a year, CUDA wheels published
only up to torch 2.1, and `data.dgl.ai` serving a templated `repo.html` for paths that were
never populated. That is why `ign.torch` exists and why it is the tier to build on.

## The port

`ign_pyg/` reimplements the model and its featurisation. Parameter names match
`scripts/model_v2.py`, so the published checkpoints load with `strict=True` and a divergence
can only be arithmetic. The two graph operations DGL provided are twelve lines of torch, so
there is no extension wheel to match against a torch version.

Checked in three places, deliberately separate, because one end-to-end number cannot say which
half is wrong:

| check | what it holds fixed | result |
|---|---|---|
| `ign_pyg/compare_with_reference.py` | the graphs, replayed from the DGL model's own input | 279/279, max 2.4e-06 |
| `export_port_graphs.py` + `predict.py --graphs` | rdkit — featurise in `ign-ref`, score on torch 2.5.1 | **279/279, R = 1.000000** |
| `gnnb verify ign.torch` | nothing | R = 0.999272, 264/279 within 1e-4 |

The middle row is the one that says the port is right. The last is the price of a five-year
newer rdkit, charged to fifteen molecules — 4jia worst at 0.591, and 1o0h where the two
versions perceive a different formal charge and one different bond. For an exact reproduction,
use the middle path.

`ign_pyg/test_invariance.py` checks the prediction does not move under rotation, translation or
a relabelling of the ligand's atoms — every geometric feature here is a distance, an angle, an
area or an AEV, so it must not.

## Build & run

```bash
podman build --format=docker -f Containerfile.torch -t ign-torch:latest .   # the one to use
podman build --format=docker -t ign-ref:latest .                            # reference tier
podman build --format=docker -f Containerfile.modern -t ign:latest .        # dgl on cuda

gnnb verify --variant ign.torch --dataset data/CASF-2016/coreset
gnnb run --variant ign.torch --capability predict --dataset <complexes> --gpu
gnnb run --variant ign.torch --capability embed   --dataset <complexes>
```

Never rebuild a working image under a tag something depends on: build to a candidate tag, run
`verify`, then retag.
