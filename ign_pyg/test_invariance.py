"""IGN predicts a scalar from a rigid complex, so moving the complex must change nothing.

Every geometric quantity the featuriser computes is a distance, an angle, a triangle area or
an atom environment vector, and all four are invariant under rotation and translation. That
is a property of the model, not of this port — which is what makes it a useful check on the
port: if a coordinate leaked into a feature directly, or an axis got transposed while the
shapes stayed right, the prediction would move here and nowhere else.

Run inside the ign-torch image:
    python test_invariance.py --complexes /data --checkpoints /work/model_save

A second check goes with it. Permuting the ligand's atom numbering is relabelling, not
motion: the graph is the same graph, so the prediction must also be unchanged. That one
catches the opposite mistake — an aggregation that depends on the order rows arrive in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolTransforms  # noqa: F401  (imported for its side effects on Mol)
from rdkit.Geometry import Point3D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_graph  # noqa: E402
from pocket import read_ligand, truncate  # noqa: E402
from predict import load_models, score  # noqa: E402


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random rotation, via the QR decomposition of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))            # fix the signs so q is not a reflection
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def move(mol: Chem.Mol, rotation: np.ndarray, shift: np.ndarray) -> Chem.Mol:
    """A copy of the molecule with every atom rotated and translated."""
    moved = Chem.Mol(mol)
    conf = moved.GetConformer()
    coords = np.array(conf.GetPositions())
    new = coords @ rotation.T + shift
    for i in range(moved.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*[float(x) for x in new[i]]))
    return moved


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--complexes", required=True, type=Path)
    p.add_argument("--checkpoints", required=True, type=Path)
    p.add_argument("--n", type=int, default=3, help="how many complexes to test")
    p.add_argument("--trials", type=int, default=3, help="random placements per complex")
    p.add_argument("--tolerance", type=float, default=1e-3,
                   help="largest prediction shift still called invariant")
    p.add_argument("--work", type=Path, default=Path("/tmp/ign_invariance"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    models = load_models(args.checkpoints)
    args.work.mkdir(parents=True, exist_ok=True)
    targets = sorted(d for d in args.complexes.iterdir() if d.is_dir())[: args.n]

    failures = []
    for d in targets:
        cid = d.name
        try:
            ligand, pkt = truncate(str(d / f"{cid}_protein.pdb"), read_ligand(str(d), cid),
                                   str(args.work / f"{cid}_pocket.pdb"))
        except Exception as e:  # noqa: BLE001
            print(f"  {cid}: could not be staged ({e})")
            continue

        base, _ = score(models, build_graph(ligand, pkt))
        print(f"  {cid}: base {base:.6f}")

        for trial in range(args.trials):
            rotation = random_rotation(rng)
            shift = rng.normal(scale=25.0, size=3)
            moved, _ = score(models, build_graph(move(ligand, rotation, shift),
                                                 move(pkt, rotation, shift)))
            diff = abs(moved - base)
            flag = "ok" if diff <= args.tolerance else "MOVED"
            print(f"      rotation+translation {trial}: {moved:.6f}  diff {diff:.2e}  {flag}")
            if diff > args.tolerance:
                failures.append(f"{cid} placement {trial} ({diff:.2e})")

        # relabelling the ligand's atoms: same graph, different row order
        order = list(rng.permutation(ligand.GetNumAtoms()))
        relabelled = Chem.RenumberAtoms(ligand, [int(i) for i in order])
        permuted, _ = score(models, build_graph(relabelled, pkt))
        diff = abs(permuted - base)
        flag = "ok" if diff <= args.tolerance else "MOVED"
        print(f"      ligand atom permutation: {permuted:.6f}  diff {diff:.2e}  {flag}")
        if diff > args.tolerance:
            failures.append(f"{cid} permutation ({diff:.2e})")

    print()
    if failures:
        print("NOT INVARIANT: " + "; ".join(failures))
        return 1
    print("invariant under rotation, translation and ligand relabelling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
