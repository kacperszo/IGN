"""Turn a canonical complex directory into the archive IGN expects.

IGN's input protocol is unusual and unforgiving (`scripts/model_ign_prediction.py:89`):

  * `--test_file_path` must be a directory holding exactly one `.zip`
  * the directory inside that archive must be named like the archive itself, because the
    script lists `<zip name without .zip>/` after unzipping
  * only directories may sit at that top level — a stray `.pdb` or `.sdf` there silently
    switches it to mode1, which assumes one rigid protein and many ligand poses
  * files are `<id>/<id>.pdb` and `<id>/<id>.sdf`, not the `_protein`/`_ligand` names the
    rest of the benchmark uses

Output keys come back as `<pdb stem>_<sdf _Name>`, so the ligand is written with `_Name`
set to a known tag and the suffix is stripped when collecting.

usage:
    python prepare_input.py --complexes /data --out /outputs


Written for Python 3.6, which is what `environment.yml` pins: no `from __future__ import
annotations`, no built-in generics in annotations, nothing newer than f-strings.
"""

import argparse
import os
import shutil
import tempfile
import zipfile

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

LIGAND_TAG = "lig"
ARCHIVE_STEM = "ign_input"


def stage_one(complex_dir, stage_root):
    """Stage one complex, leaving nothing behind if it cannot be staged whole.

    A directory holding a protein but no ligand is worse than no directory at all: IGN's
    mode2 loop assigns `sdf_file` only when it sees one, and never resets it between
    targets, so a half-staged complex either crashes on an undefined name or silently reuses
    the previous complex's ligand.
    """
    cid = os.path.basename(complex_dir.rstrip("/"))
    protein = os.path.join(complex_dir, f"{cid}_protein.pdb")
    ligand = os.path.join(complex_dir, f"{cid}_ligand.sdf")
    if not os.path.exists(ligand):
        ligand = os.path.join(complex_dir, f"{cid}_ligand.mol2")

    # Read the ligand before creating anything. Some PDBbind SDFs do not sanitise — 1a30 is
    # one — and the mol2 beside them usually does, so both are tried before giving up.
    mols = []
    for path in (ligand, os.path.join(complex_dir, f"{cid}_ligand.mol2")):
        if not os.path.exists(path):
            continue
        if path.endswith(".sdf"):
            mols = [m for m in Chem.SDMolSupplier(path) if m is not None]
        else:
            mols = [m for m in [Chem.MolFromMol2File(path)] if m is not None]
        if mols:
            break
    if not mols:
        raise ValueError("no readable ligand")

    # everything that can fail has failed by now, so the directory is safe to create
    target = os.path.join(stage_root, cid)
    os.makedirs(target, exist_ok=True)
    shutil.copyfile(protein, os.path.join(target, f"{cid}.pdb"))

    mol = mols[0]
    # an empty _Name leaves a trailing underscore in IGN's output keys
    mol.SetProp("_Name", LIGAND_TAG)
    with Chem.SDWriter(os.path.join(target, f"{cid}.sdf")) as w:
        w.write(mol)


def main():
    parser = argparse.ArgumentParser(description="Build IGN's input archive")
    parser.add_argument("--complexes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--staged_only", action="store_true",
                        help="leave the staged directory in place and skip the archive; the "
                             "embedding path reads it directly instead of going through "
                             "IGN's zip-and-mode-detect protocol")
    args = parser.parse_args()

    dirs = sorted(os.path.join(args.complexes, d) for d in os.listdir(args.complexes)
                  if os.path.isdir(os.path.join(args.complexes, d)))
    os.makedirs(args.out, exist_ok=True)
    archive = os.path.join(args.out, f"{ARCHIVE_STEM}.zip")

    staged = 0
    failed = []

    def stage_all(root):
        nonlocal staged
        os.makedirs(root)
        for d in dirs:
            try:
                stage_one(d, root)
                staged += 1
            except Exception as e:
                failed.append(f"{os.path.basename(d)}: {type(e).__name__}: {e}")

    if args.staged_only:
        # the embedding path reads these directories directly, so no archive is built and
        # the staging survives the call
        destination = os.path.join(args.out, ARCHIVE_STEM)
        stage_all(destination)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, ARCHIVE_STEM)
            stage_all(root)
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
                for base, _dirs, files in os.walk(root):
                    for name in files:
                        full = os.path.join(base, name)
                        zf.write(full, os.path.relpath(full, tmp))
        destination = archive

    if failed:
        print(f"{len(failed)} complexes could not be staged:")
        for f in failed[:10]:
            print("  ", f)
    print(f"{staged} complexes -> {destination}")


if __name__ == "__main__":
    main()
