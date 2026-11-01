"""Build IGN pretraining pairs from MD frames: the graph of frame n, the descriptors of frame n+gap.

The pretext task — ported from Kuba's MD_pretraining (`pretraining/dataset.py:build_frame_dataset`)
to IGN — is to predict, from one MD frame alone, structural descriptors of the frame `gap` steps
later in the same trajectory. At AI3's 10 frames per ns the default gap of 10 is 1 ns.

Inputs are files, not another repository's code:
  * `--frames`: PDBbind-layout frame directories `<pdb>_run_<r>_frame_<NNN>/`, as written by
    MD_pretraining's `sources/ai3/process_archives.py`;
  * `--descriptors`: that script's merged `descriptors.csv`, keyed (file=<pdb>_run_<r>, frame);
  * `--ids`: CSV with a `complex_id` column naming the complexes of this split.

A pair is kept only when both frames pass Kuba's own filter — `ligand_rmsd_first < 8` A, i.e. the
ligand has not left the pocket — as `modify_md_descrs_df` requires of both frames of a pair. The
graph is IGN's own, built by the same `pocket.truncate` + `features.build_graph` that scoring
uses, so pretraining and fine-tuning see graphs of one kind — after one correction: the frames
come from AMBER and name residues by protonation state (HIE, HID, HIP, CYX, ASH, GLH, LYN), and
RDKit's PDB reader assigns bond orders only to the standard names. Left as they are, a histidine
ring comes out saturated and the backbone C=O of those residues as C-OH, where every crystal
pocket has them aromatic and double-bonded. They are renamed to the standard residue before the
pocket is cut (`--keep-amber-names` turns this off).

Stored beside each target vector is the same descriptor vector for frame n. The trainer uses it
for a *persistence* baseline — predict frame n+gap by copying frame n — which is the bar the model
has to clear before the pretext task can be said to teach anything about dynamics.

Output is sharded npz, because the full set does not fit in memory: ~175 KB per graph, ~30 pairs
per run. The build is resumable: shards already on disk are kept, and the metadata file that
marks the build complete is written last.

usage:
    python prepare_md_pairs.py --frames /data --descriptors /data/descriptors.csv \
        --ids /splits/train.csv --out /cache/ign_md_train --workers 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_graph  # noqa: E402
from pocket import PocketError, read_ligand, truncate  # noqa: E402

#: frame-intrinsic descriptors of frame n+gap. Excluded on purpose: anything measured against the
#: first or last frame of the trajectory (rmsd_first/last, contacts_q1/q2) — the model never sees
#: those frames — `ligand_rmsd_from_previous`, SASA (freesasa is not installed where these were
#: computed, so the columns are empty), and every binding-free-energy term, which is an affinity
#: proxy rather than dynamics.
TARGETS = (
    "heavy_atom_contacts_3_5", "heavy_atom_contacts_4_5",
    "hydrogen_bonds_prot_lig", "hydrophobic_contacts_4",
    "ligand_PMI1", "ligand_PMI2", "ligand_PMI3", "ligand_NPR1", "ligand_NPR2",
    "ligand_RadiusOfGyration", "ligand_InertialShapeFactor", "ligand_Eccentricity",
    "ligand_Asphericity", "ligand_SpherocityIndex", "ligand_PBF",
    "pocket_RadiusOfGyration",
)
#: Kuba's escape filter: frames where the ligand drifted this far from its start are dropped
RMSD_CUTOFF = 8.0
PARTS = ("h", "e", "edge_index", "e3", "edge_index3")
#: AMBER protonation-state residue names -> the standard name RDKit has a bond-order template for
AMBER_NAMES = {"HIE": "HIS", "HID": "HIS", "HIP": "HIS", "CYX": "CYS", "CYM": "CYS",
               "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}
FRAME_DIR = re.compile(r"^(?P<file>(?P<pdb>[0-9a-z]{4})_run_\d)_frame_(?P<n>\d{3})$")


def read_descriptors(path: str) -> dict[tuple[str, int], dict[str, float]]:
    """(file, frame) -> row, keeping only frames that pass the escape filter."""
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if float(r["ligand_rmsd_first"]) >= RMSD_CUTOFF:
                continue
            rows[(r["file"], int(r["frame"]))] = r
    return rows


def vector(row: dict[str, str], targets) -> np.ndarray:
    return np.array([float(row[c]) if row[c] not in ("", "nan") else np.nan for c in targets],
                    dtype=np.float32)


def targets_of(descriptors: str) -> tuple[str, ...]:
    """The dataset's own target list if it declares one, else TARGETS.

    A `targets.txt` beside the descriptor table (one column name per line) names what this
    dataset is to be pretrained on — a table computed elsewhere need not carry all sixteen
    default columns."""
    path = os.path.join(os.path.dirname(os.path.abspath(descriptors)), "targets.txt")
    if not os.path.exists(path):
        return TARGETS
    with open(path) as f:
        return tuple(line.strip() for line in f if line.strip())


def torch_one_thread() -> None:
    """Worker initialiser: one intra-op thread, so N workers use N cores and not N x cores."""
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def standard_names(protein_pdb: str, out_pdb: str) -> str:
    """A copy of `protein_pdb` with AMBER residue names replaced by the standard ones."""
    with open(protein_pdb) as src, open(out_pdb, "w") as dst:
        for line in src:
            if line.startswith(("ATOM", "HETATM")) and line[17:20] in AMBER_NAMES:
                line = line[:17] + AMBER_NAMES[line[17:20]] + line[20:]
            dst.write(line)
    return out_pdb


def featurise(task):
    """One frame's IGN graph, or None."""
    frames_root, frame_id, distance, work, rename = task
    d = os.path.join(frames_root, frame_id)
    protein = os.path.join(d, f"{frame_id}_protein.pdb")
    try:
        if rename:
            protein = standard_names(protein, os.path.join(work, f"{frame_id}_protein.pdb"))
        lig, pkt = truncate(protein, read_ligand(d, frame_id),
                            os.path.join(work, f"{frame_id}_pocket.pdb"), distance=distance)
        graph = build_graph(lig, pkt)
        os.remove(os.path.join(work, f"{frame_id}_pocket.pdb"))
        if rename:
            os.remove(protein)
        return frame_id, graph
    except (PocketError, Exception):  # noqa: BLE001 — one bad frame must not end the run
        return frame_id, None


def main() -> int:
    p = argparse.ArgumentParser(description="IGN pretraining pairs: frame n -> descriptors of n+gap")
    p.add_argument("--frames", required=True)
    p.add_argument("--descriptors", required=True)
    p.add_argument("--ids", required=True, help="CSV with a complex_id column")
    p.add_argument("--out", required=True, help="output prefix; shards are <out>_<k>.npz")
    p.add_argument("--gap", type=int, default=10, help="frames ahead; 10 = 1 ns in AI3")
    p.add_argument("--stride", type=int, default=1,
                   help="start a pair only at every stride-th frame. MISATO has 100 frames per run "
                        "against AI3's ~40, so --stride 3 keeps ~30 pairs per run as in AI3: the "
                        "sources then differ in what they show, not in how many near-duplicate "
                        "neighbouring frames they repeat")
    p.add_argument("--shard-size", type=int, default=2000)
    p.add_argument("--distance", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--work", default="/tmp/ign_md_prepare")
    p.add_argument("--keep-amber-names", action="store_true",
                   help="do not rename HIE/HID/HIP/CYX/ASH/GLH/LYN (the behaviour before 2026-10-20)")
    args = p.parse_args()

    with open(args.ids, newline="") as f:
        wanted = {r["complex_id"].lower() for r in csv.DictReader(f)}
    desc = read_descriptors(args.descriptors)
    targets = targets_of(args.descriptors)
    missing = [c for c in targets if c not in next(iter(desc.values()))]
    if missing:
        raise SystemExit(f"{args.descriptors} lacks target columns {missing}")
    print(f"{len(targets)} targets: {', '.join(targets)}")

    # Every (frame n, frame n+gap) pair of the same run where both frames exist on disk and both
    # pass the escape filter. The graph is needed only for n; n+gap contributes its descriptors.
    present = {}
    for name in os.listdir(args.frames):
        m = FRAME_DIR.match(name)
        if m and m["pdb"] in wanted:
            present[(m["file"], int(m["n"]))] = name
    pairs = sorted((present[(file, n)], file, n) for (file, n) in present
                   if n % args.stride == 0 and (file, n) in desc and (file, n + args.gap) in desc)
    complexes = {f.split("_run_")[0] for _, f, _ in pairs}
    print(f"{len(wanted)} complexes in the split, {len(complexes)} with frames; "
          f"{len(pairs)} pairs at gap {args.gap}, stride {args.stride}")
    if not pairs:
        raise SystemExit("no pairs — check that --frames and --descriptors come from the same run")

    os.makedirs(args.work, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    rename = not args.keep_amber_names
    tasks = [(args.frames, frame_id, args.distance, args.work, rename) for frame_id, _, _ in pairs]
    lookup = {frame_id: (file, n) for frame_id, file, n in pairs}

    # One shard per `--shard-size` consecutive pairs, written under a temporary name and renamed
    # when complete. A rerun skips every shard already on disk, so an interrupted build loses at
    # most the shard in progress. Shard k always holds the same pairs, whatever failed before it.
    chunks = [tasks[i:i + args.shard_size] for i in range(0, len(tasks), args.shard_size)]
    # torch keeps its own intra-op pool regardless of OMP_NUM_THREADS once imported; pin it too
    pool = Pool(args.workers, initializer=torch_one_thread) if args.workers else None
    written, failed, reused = 0, 0, 0
    for k, chunk in enumerate(chunks):
        path, empty = f"{args.out}_{k:03d}.npz", f"{args.out}_{k:03d}.empty"
        if os.path.exists(path):
            with np.load(path) as z:
                written += len(z["ids"])
                failed += len(chunk) - len(z["ids"])
            reused += 1
            continue
        if os.path.exists(empty):
            failed += len(chunk)
            reused += 1
            continue
        shard, ids, y_later, current = {}, [], [], []
        results = pool.imap(featurise, chunk, chunksize=8) if pool else map(featurise, chunk)
        for frame_id, graph in results:
            if graph is None:
                failed += 1
                continue
            file, n = lookup[frame_id]
            key = str(len(ids))
            for part in PARTS:
                shard[f"{key}/{part}"] = graph[part]
            ids.append(frame_id)
            y_later.append(vector(desc[(file, n + args.gap)], targets))
            current.append(vector(desc[(file, n)], targets))
        if ids:
            tmp = f"{args.out}_{k:03d}.tmp.npz"
            np.savez(tmp, ids=np.array(ids), targets=np.stack(y_later), current=np.stack(current),
                     **shard)
            os.replace(tmp, path)
            written += len(ids)
        else:
            open(empty, "w").close()
        print(f"  shard {k + 1}/{len(chunks)}: {written} pairs so far, {failed} frames failed",
              flush=True)
    if pool:
        pool.close()
    shard_k = len(chunks)
    if reused:
        print(f"reused {reused} shards already on disk")

    with open(f"{args.out}_meta.json", "w") as f:
        json.dump({"targets": list(targets), "gap": args.gap, "stride": args.stride,
                   "pairs": written,
                   "failed": failed, "shards": shard_k, "complexes": sorted(complexes),
                   "rmsd_cutoff": RMSD_CUTOFF, "pocket_distance": args.distance,
                   "amber_names_renamed": rename}, f, indent=2)
    print(f"{written} pairs in {shard_k} shards, {failed} frames failed -> {args.out}_*.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
