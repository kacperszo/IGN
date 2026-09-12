# IGN — interaction graph network for protein-ligand affinity

> A fork maintained for [gnn-benchmark](../../README.md). The authors' own README is
> kept as [README.upstream.md](README.upstream.md) for attribution and for their
> description of the method — **its build and run instructions are not current for
> this fork.**

## What it is

A graph network over an explicit protein-ligand interaction graph: ligand and pocket
atoms are nodes, contacts are edges, and every feature is a distance, an angle, a triangle area or an
atom-environment vector — so the prediction is invariant to where the complex sits in space. Affinity
averages five checkpoints; `readout` pools to a 200-dim graph vector and `FC` is the head, which makes
the encoder/head boundary the authors' own.

Three tiers exist because the original pins torch 1.3.1 and dgl 0.4.3. `ign.reference` is that
environment, kept as the fidelity baseline; `ign.torch` reimplements the model and its featurisation
without dgl on torch 2.5.1, and is the one to use.

## State

| | |
|---|---|
| CASF-2016 scoring | **R 0.802**, RMSE 1.468, n=279 |
| embedding | 200d, **native**, probe R 0.765 |
| `gnnb verify` | 279/279 — reference 1.0e-06, port 5.9e-01 (tolerance 0.75, a current RDKit) |
| invariance | rotation, translation and ligand relabelling, to 8.6e-07 |

## Build

```bash
podman build --format=docker -t ign-ref:latest .                           # reference tier, torch 1.3.1
podman build --format=docker -f Containerfile.torch -t ign-torch:latest .   # the one to use
```

## Run it, without the harness

Generated from this model's adapter by `gnnb howto`, so these are the exact commands
the benchmark issues — regenerate with `python tools/sync_model_readmes.py`. Every one
runs with `--network=none` and a read-only root filesystem.

Input is one directory per complex:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2; several models try both

Also reads `<id>_pocket.pdb` when present; otherwise it truncates the pocket itself from the protein.

```bash
# ign.reference — localhost/ign-ref:latest
# source: models/ign

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
# source: models/ign

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
```

## What comes out

| file | holds |
|---|---|
| `predictions.csv` | `complex_id,y_pred` — the reference tier writes `keys,test_pred` and the adapter renames; the port writes the standard header |
| `embeddings.npz` | `ids` and `vectors`, 200-dim, the pooled graph vector its head reads |

## Before you trust the numbers

**RDKit's version is part of this model.** Predictions average five checkpoints trained against RDKit 2021.03, and three behaviour changes since then are each enough to destroy the reproduction silently — hydrogens surviving an sdf round trip, MOL bond type 4 no longer flagging atoms aromatic, and `RemoveHs` moving a pyrrole nitrogen's hydrogen into `numExplicitHs`. The corrections are pinned in `scripts/utils.py` and `ign_pyg/pocket.py` rather than pinned by version, so both tiers run from one source tree. Fifteen core-set complexes still differ because a current RDKit genuinely perceives them differently; that residual is why the registry declares `tolerance = 0.75`.

## Maintainer notes

`CLAUDE.md` in this directory holds what breaks if it is changed back.
