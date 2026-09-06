# CLAUDE.md — IGN

## What this is

Affinity prediction from an interaction graph network, averaged over five checkpoints. The
authors' code, from a Chinese lab, run unmodified inside a container — the most foreign code
in the benchmark and the one the isolation exists for.

Runs as `ign.reference`. CPU-only: the authors pinned torch 1.3.1, which predates our GPUs.

## Current state

Done for `predict` and `embed`, on both tiers.

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

Predictions come back as `keys,test_pred` and the adapter normalises them to
`complex_id,y_pred`.

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
  than assumes. (This said *no code references at all* until `scan_pickle.py` was fixed to
  read past the first of a pre-1.6 checkpoint's five pickle streams.)

## DGL looks abandoned, and that caps this model

Kacper's observation, not verified here: DGL has had no commits for about a year. What we saw
building the modern tier is consistent with it — CUDA wheels published only up to torch 2.1,
and `data.dgl.ai` serving a templated `repo.html` for paths like `torch-2.4/cu121` that were
never populated, so pip resolves a wheel URL that answers 403 (S3's reply for an object that
is not there when listing is denied).

Consequences worth carrying:

- **This model is fine for now.** The container pins a working DGL, and a frozen library
  inside an image is exactly what the image is for.
- **But the torch ceiling is closed.** Anything on DGL stays at torch 2.1, which covers sm_86
  and sm_89 today. The next GPU generation puts IGN back in the position torch 1.3.1 put it
  in, a few years later.
- **It is a selection criterion for the remaining roster.** If any of the uncloned repos build
  on DGL, that is worth knowing before investing in them.

The contrast is instructive: EquiBind is in this benchmark as a PyG rewrite Kacper did rather
than a DGL fork, so it runs on any current torch. That looked like tidying at the time; it
turns out to have been an exit from a dead end.

## The GPU tier works, but does not verify bit-for-bit

`ign.modern` (torch 2.1.2+cu121, dgl 2.1.0) runs and reaches the GPU: torch reports sm_89 and
DGL graphs move to `cuda:0`. Training is unblocked.

Six incompatibilities had to be closed, all handled by fallbacks so **one source tree serves
both tiers** — which is what makes `verify` a comparison of environments rather than of two
diverging codebases:

| what | reference (dgl 0.4.3) | modern (dgl 2.1.0) |
|---|---|---|
| featurizer import | `dgl.data.chem` | `dgllife.utils` |
| edge copy | `fn.copy_edge` | `fn.copy_e` |
| edge multiply | `fn.src_mul_edge` | `fn.u_mul_e` |
| graph construction | `DGLGraph()` + `add_edges` | `dgl.graph((src, dst))` |
| `torch.compiler.is_compiling` | not called | called; shimmed to `False` |
| torchani | 2.2 | **pinned to 2.2** — newer releases reject hydrogen |

**But the two tiers are not bit-comparable, and the reason is RDKit, not DGL.** The reference
image carries rdkit 2021.03.4 (conda) and the modern one 2026.03.5 (pip) — five years of
molecule-parsing changes:

- `1a30`: 7.0858803 vs 7.0858793 — identical to float32 noise, so the port itself is right
- `1bcu`: 4.8139734 vs 2.6389904 — a real difference
- `1bzc`: unreadable under the old RDKit, parsed fine by the new one

So `verify` cannot be strict here until rdkit is pinned to a matching version in the modern
tier, which may not be installable on Python 3.11. Until then the modern tier is **usable for
training but not a bit-exact reproduction**, and that has to travel with any number from it.

The general lesson is worth more than this model: **pinning the framework is not enough.** The
chemistry toolkit that turns files into features is as much a part of the environment as torch,
and it changes the inputs rather than the arithmetic.

## The port: IGN without DGL, and what it took to make it agree

`ign_pyg/` reimplements the model and its featurisation on plain torch — no dgl, no dgllife,
no PyG, no `torch_scatter`. The two graph operations IGN needs are twelve lines of torch, so
there is no compiled extension pinned to a torch ABI and nothing to reinstall when torch
moves. `ign.torch` runs it.

Two checks, deliberately separate, because a single end-to-end number cannot say which half
is wrong:

| check | what it holds fixed | result |
|---|---|---|
| `compare_with_reference.py` | the graphs — replayed from the DGL model's own input | 279/279, max 2.4e-06 |
| `compare_features.py` | the molecules — both featurisers, one rdkit | see below |
| end to end vs `model_ign_prediction.py` | nothing | R=0.9995, mean 0.007 |

### Four things had to be discovered, not read off the code

**`FC` builds three Linear layers when asked for two.** Two consecutive `if`s where an
`if/elif` was meant, so `j == 0` takes both branches. Writing the obvious `elif` changes the
architecture; `load_state_dict(strict=True)` is what caught it, which is the reason the port
keeps upstream's parameter names rather than tidier ones.

**The chirality features are constant zero, and always were.** `pocket_truncate` pickles
`[ligand, pocket]` and the graph builder unpickles them; rdkit's pickler drops atom
properties, so `_CIPCode` and `_ChiralityPossible` are gone by the time the featuriser looks
for them. Three of the 40 atom features never take a value. Computing them properly in the
port made it *disagree* with the checkpoints — restoring them is a retraining decision.

**The ligand's reader decides the molecule.** Upstream's chain is `SDMolSupplier` →
`MolToMolFile` → `MolFromMolFile`. Reading the original file with `MolFromMolFile` instead
looks equivalent and is not: on 1eby it keeps 40 hydrogens the supplier drops, the complex
goes from 288 nodes to 382, and the prediction moves 4.4 log units.

**89 of 285 complexes were being dropped by a fallback that cannot work.** `pocket_truncate`
calls `MolFromMolFile`, which reads sdf and mol only. Pointing it at a mol2 always fails, so
the mol2 has to be rewritten as an sdf first — which is what `prepare_input.py` does.

### rdkit is the environment, as much as torch is

Three changes between rdkit 2021.03 and 2026.03 each broke reproduction on their own, and the
port now handles all three explicitly rather than inheriting whatever the installed version
does:

| change | effect if ignored |
|---|---|
| hydrogens survive the sdf round trip | 88 atoms instead of 48; predictions go negative |
| MOL bond type 4 no longer flags **atoms** aromatic | `atom_is_aromatic` all zero on ligands |
| `RemoveHs` moved a pyrrole N's H to `numExplicitHs`; manual deletion does not | one total-H column shifts |

Measured on the full core set, the three cost R=-0.07, R=0.77 and R=0.978 respectively as
they were fixed, ending at 0.9995. Note what that means: **a chemistry-toolkit upgrade is a
model change.** Neither `weights_only` nor a pinned torch protects against it, and none of it
raises an error — every array keeps its shape.

This is also the most likely explanation for `ign.modern`'s R=-0.155, which was blamed on DGL
for months. That tier runs the same `graph_constructor.py` against a 2026 rdkit and would hit
all three. The graph constructor itself is exonerated: its rewrite from `DGLGraph()` to
`dgl.graph(...)` produces bit-identical graphs in edge-id order, tested directly with
`ign_pyg/dump_edges.py`.

### What remains

One complex in the core set, 1o0h, still disagrees: the two rdkits perceive a different
formal charge, hybridisation and one different bond. That is a genuine difference in
chemistry perception, not something the port can paper over, and it is the reason `verify`
for `ign.torch` should carry a tolerance rather than demand equality.

## Planned: a GPU tier for training

Wanted, and a genuine port rather than a version bump. What it requires, in order:

1. **torch ≥1.8** for sm_86, realistically 2.x.
2. **dgl ≥0.6**, which is past the 0.5 break that removed `dgl.data.chem`. Every
   `BaseBondFeaturizer` import moves to `dgllife`, and `fn.copy_edge` is gone in dgl 1.0.
3. **torchani** and **ProDy** rebuilt against the new stack; the pinned 2.2 and 2.0 are from
   the same era as torch 1.3.
4. A `verify` run against this tier's golden file, because none of the above is behaviourally
   neutral.

Do this as a second variant (`ign.modern`) beside the reference tier, never in place of it.
The reference tier is what proves the port did not change the model.

## Build & run

```bash
podman build --format=docker -t ign-ref:latest .
gnnb run --variant ign.reference --capability predict --dataset <complexes>
```
