# IGN — interaction graph network for protein-ligand affinity

> A maintained fork. The authors' own README is kept as
> [README.upstream.md](README.upstream.md) for attribution and for their description of
> the method — **its build and run instructions are not current for this fork.**

## Upstream

InteractionGraphNet (IGN) by the authors of <https://github.com/zjujdj/IGN>; licence Apache-2.0.
The authors' description of the method is in [README.upstream.md](README.upstream.md), which
carries no paper citation, so none is given here. Their scripts are partly based on dgl-lifesci
(<https://github.com/awslabs/dgl-lifesci>). Edits to upstream files are commented in place with
the reason (for example `scripts/model_v2.py` and `scripts/model_ign_prediction.py`); `ign_pyg/`
is a port written here.

## What it is

A graph network over an explicit protein-ligand interaction graph: ligand and pocket
atoms are nodes, contacts are edges, and every feature is a distance, an angle, a triangle area or an
atom-environment vector — so the prediction is invariant to where the complex sits in space. Affinity
averages five checkpoints; `readout` pools to a 200-dim graph vector and `FC` is the head, which makes
the encoder/head boundary the authors' own.

Three tiers exist, each with its own Containerfile, because the original pins torch 1.3.1 and
dgl 0.4.3:

- `ign.reference` (`Containerfile`): that original environment, CPU only, kept as the fidelity
  baseline.
- `ign.modern` (`Containerfile.modern`): the same source on torch 2.1.2 (cu121) and a current
  DGL, for training on GPUs the reference environment does not support.
- `ign.torch` (`Containerfile.torch`): `ign_pyg/`, the model and its featurisation reimplemented
  on plain PyTorch 2.5.1 without dgl. It is the one to use.

## State

| what | measured |
|---|---|
| CASF-2016 scoring | **R 0.802**, RMSE 1.468, n=279 of 285 (six ligands will not sanitise under the reference tier's 2021 RDKit) |
| embedding | 200d, native (the head's own input), probe R 0.765 |
| agreement with reference predictions | 279/279 — reference tier max abs diff 1.0e-06, port 5.9e-01 (a current RDKit; see below) |
| invariance | rotation, translation and ligand relabelling, to 8.6e-07 (`ign_pyg/test_invariance.py`) |

The 0.802 averages five checkpoints, so it is an ensemble figure, and their training list is not
published while 266 of the 285 core-set complexes sit inside PDBbind refined.

## Build

```bash
podman build --format=docker -t ign-ref:latest .                           # reference tier, torch 1.3.1
podman build --format=docker -f Containerfile.torch -t ign-torch:latest .   # the one to use
podman build --format=docker -f Containerfile.modern -t ign:latest .        # dgl on a current torch
```

## Run it

Every command below runs with `--network=none` and a read-only root filesystem. No run
commands are given for the `modern` tier.

Input is one directory per complex:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2

Also reads `<id>_pocket.pdb` when present; otherwise it truncates the pocket itself from the protein.

```bash
# ign.reference — localhost/ign-ref:latest

# predict
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    localhost/ign-ref:latest \
    sh -c 'set -e; python prepare_input.py --complexes /data --out /outputs; cd scripts && python model_ign_prediction.py --test_file_path /outputs'

# embed
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    localhost/ign-ref:latest \
    sh -c 'set -e; python prepare_input.py --complexes /data --out /outputs --staged_only; cd scripts && python embed_complexes.py --complexes /outputs/ign_input --out /outputs/embeddings.npz --workdir /outputs/work'
```

```bash
# ign.torch — localhost/ign-torch:latest

# predict
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    localhost/ign-torch:latest \
    sh -c 'set -e; cd /work/ign_pyg && python predict.py --complexes /data --checkpoints /work/model_save --out /outputs/predictions.csv --work /outputs/work'

# embed
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    localhost/ign-torch:latest \
    sh -c 'set -e; cd /work/ign_pyg && python predict.py --complexes /data --checkpoints /work/model_save --embeddings /outputs/embeddings.npz --work /outputs/work'

# train
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/ign-torch:latest \
    sh -c 'cd /work/ign_pyg && if [ -f /cache/ign_train_r5.npz ]; then echo reusing /cache/ign_train_r5.npz; else python prepare_graphs.py --complexes /data --labels /splits/train.csv --out /cache/ign_train_r5.npz --distance 5 --work /tmp/ign_prepare; fi && if [ -f /cache/ign_val_r5.npz ]; then echo reusing /cache/ign_val_r5.npz; else python prepare_graphs.py --complexes /data --labels /splits/val.csv --out /cache/ign_val_r5.npz --distance 5 --work /tmp/ign_prepare; fi && python train_contract.py --train-graphs /cache/ign_train_r5.npz --out /outputs --seed 0 --val-graphs /cache/ign_val_r5.npz --epochs 30 --device cpu'

# finetune  (encoder frozen; drop --freeze-encoder to tune all of it)
# needs an encoder-only file you supply, see "Weights" below
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    -v /path/to/encoders:/ckpt:ro \
    --shm-size 4g \
    localhost/ign-torch:latest \
    sh -c 'cd /work/ign_pyg && if [ -f /cache/ign_train_r5.npz ]; then echo reusing /cache/ign_train_r5.npz; else python prepare_graphs.py --complexes /data --labels /splits/train.csv --out /cache/ign_train_r5.npz --distance 5 --work /tmp/ign_prepare; fi && if [ -f /cache/ign_val_r5.npz ]; then echo reusing /cache/ign_val_r5.npz; else python prepare_graphs.py --complexes /data --labels /splits/val.csv --out /cache/ign_val_r5.npz --distance 5 --work /tmp/ign_prepare; fi && python train_contract.py --train-graphs /cache/ign_train_r5.npz --out /outputs --seed 0 --val-graphs /cache/ign_val_r5.npz --epochs 30 --init-encoder /ckpt/encoder.pt --freeze-encoder --device cpu'
```

## Weights

The five authors' checkpoints are in `model_save/` and are baked into every image; `predict` and
`embed` need nothing else. The `finetune` command needs `encoder.pt`, which is not in the
repository: supply a file holding encoder tensors only (every tensor of a trained model except
those under `FC`), in the form `ign_pyg/transfer.py:read_encoder` accepts. `ign_pyg/pretrain_contract.py`
writes one. `ign_pyg/transfer.py` refuses a file that still carries head tensors.

## What comes out

| file | holds |
|---|---|
| `predictions.csv` | `complex_id,y_pred` — the reference tier writes `keys,test_pred` and the adapter renames; the port writes the standard header |
| `embeddings.npz` | `ids` and `vectors`, 200-dim, the pooled graph vector its head reads |

## Before you trust the numbers

**RDKit's version is part of this model.** Predictions average five checkpoints trained against RDKit 2021.03, and three behaviour changes since then are each enough to destroy the reproduction silently — hydrogens surviving an sdf round trip, MOL bond type 4 no longer flagging atoms aromatic, and `RemoveHs` moving a pyrrole nitrogen's hydrogen into `numExplicitHs`. The corrections are pinned in `scripts/utils.py` and `ign_pyg/pocket.py` rather than pinned by version, so both tiers run from one source tree. Fifteen core-set complexes still differ because a current RDKit genuinely perceives them differently; that residual is why the registry declares `tolerance = 0.75`.

## Porting notes

What someone reusing this model needs to know, and what breaks if it is changed back.

**RDKit is part of the model.** Three changes between RDKit 2021.03 and 2026.03 each destroy the
reproduction on their own, and none raises anything: every array keeps its shape.

| change | effect if ignored |
|---|---|
| hydrogens survive the sdf round trip | 288 nodes become 382; predictions go negative |
| MOL bond type 4 no longer flags **atoms** aromatic | `atom_is_aromatic` all zero on ligands |
| `RemoveHs` moved a pyrrole N's H to `numExplicitHs` | one total-H feature column shifts |

Fixed one at a time, the port went R = -0.07, 0.77, 0.978, 0.9993. The same two corrections in
`pocket_truncate` took the DGL-based modern tier from max abs diff 14.6 to 0.598: two independent
codebases, the same three fixes, the same recovery. Handle the behaviour in code rather than
pinning a version (`ign_pyg/pocket.py` does), so the model does not depend on which RDKit is
installed.

**DGL is a dead end for this model.** Releases that support torch 1.8+ start at dgl 0.6, which is
already past the 0.5 break that removed `BaseBondFeaturizer`, so there is no small version bump.
Wheels for CUDA were published only up to torch 2.1 and the project was inactive for about a
year, which is why `ign_pyg/` exists. The `DGLGraph()` to `dgl.graph()` rewrite produces
bit-identical graphs in edge-id order (`ign_pyg/dump_edges.py`), so DGL itself was not the source
of the regression it had been blamed for.

**`FC` builds three Linear layers when asked for two.** Two consecutive `if`s where an
`if/elif` was meant, so `j == 0` takes both branches. Writing the obvious `elif` changes the
architecture; `load_state_dict(strict=True)` is what catches it, which is why the port keeps
upstream's parameter names rather than tidier ones.

**The port** reimplements the model and its featurisation in `ign_pyg/`. Parameter names match
`scripts/model_v2.py`, so the published checkpoints load with `strict=True` and a divergence can
only be arithmetic. The two graph operations DGL provided are twelve lines of torch, so no
extension wheel has to match a torch version. It is checked in three places, kept separate
because one end-to-end number cannot say which half is wrong:

| check | what it holds fixed | result |
|---|---|---|
| `ign_pyg/compare_with_reference.py` | the graphs, replayed from the DGL model's own input | 279/279, max 2.4e-06 |
| `export_port_graphs.py` then `predict.py --graphs` | RDKit: featurise in the reference image, score on torch 2.5.1 | 279/279, R = 1.000000 |
| end to end | nothing | R = 0.999272, 264/279 within 1e-4 |

The middle row says the port is right. The last is the price of a five-year-newer RDKit, charged
to fifteen molecules (4jia worst at 0.591; on 1o0h the two versions perceive a different formal
charge and one different bond). For an exact reproduction use the middle path.

**Input handling.**

- `--test_file_path` must be a directory holding exactly one zip whose top level contains only
  directories. A stray `.pdb` or `.sdf` there silently switches IGN to mode1, which assumes one
  rigid protein and many ligand poses. It must also be writable: IGN unzips into it and writes
  `prediction.csv` there.
- A half-staged complex is worse than a missing one. IGN's mode2 loop assigns `sdf_file` only
  when it sees one and never resets it between targets, so a directory with a protein and no
  ligand either crashes on an undefined name or silently reuses the previous complex's ligand;
  `prepare_input.py` reads the ligand before creating anything.
- The ligand's reader decides the molecule. The chain is `SDMolSupplier`, `MolToMolFile`,
  `MolFromMolFile`; reading the original file with `MolFromMolFile` instead is not equivalent
  (on 1eby it keeps 40 hydrogens the supplier drops, 288 nodes become 382 and the prediction
  moves 4.4 log units). `MolFromMolFile` cannot open a mol2 at all, so a mol2 must be rewritten
  as an sdf first; pointing the reader at it looks like a fallback and never fires.
- The three chirality features are constant zero, and always were: `pocket_truncate` pickles
  `[ligand, pocket]` and RDKit's pickler drops atom properties, so `_CIPCode` and
  `_ChiralityPossible` are gone before the featuriser looks. Computing them properly makes the
  model disagree with its own checkpoints; restoring them is a retraining decision.
- Ask for `edges(order="eid")` before pairing DGL edges with `edata`; a CSR-backed graph returns
  edges grouped by destination otherwise.

**Reference-tier environment.** DGL 0.4.3 asks for its backend interactively on first import; with
no stdin in a build the image dies on `EOFError` unless `DGLBACKEND=pytorch` is set. Pillow,
pulled in by matplotlib, needs `libtiff.so.5`, which current Debian no longer ships, so it comes
from conda-forge pinned at 4.2 (`graph_constructor.py` does `from pylab import *`, so matplotlib
is on the import path whether or not anything plots).

**Loading weights.** There is no single checkpoint: five weight files are loaded from a path
relative to the script and averaged. The reference tier's torch 1.3.1 predates `weights_only`,
so its checkpoints cannot be opened with it; all five contain only `collections.OrderedDict`,
the two torch storage classes and `torch._utils._rebuild_tensor_v2`, which is what a plain state
dict is made of, but scan any other pickle before loading it.
