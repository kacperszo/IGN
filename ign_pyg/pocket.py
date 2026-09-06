"""Cut the binding pocket out of a protein, the way IGN's own `pocket_truncate` does.

Same steps and the same tools — prody selects, rdkit reads back — with two differences that
matter and one that does not.

What matters: the pair is returned rather than pickled. Upstream writes `[ligand, pocket]`
to disk and the graph builder reads it back, so a run's intermediate state is a pickle file;
here the mols stay in memory and nothing has to be unpickled later. And a failure says which
step failed instead of printing and returning `None`, which upstream turns into an empty
dataset several stages downstream.

What does not matter: the pocket PDB is written to a caller-chosen path, because rdkit needs
a file to read. That file is a byproduct, not a result.
"""

# Kept importable on python 3.6 so it can be compared inside the reference image.
import os
import tempfile

from rdkit import Chem


class PocketError(Exception):
    """Raised when a complex cannot be prepared, naming the step that failed."""


def strip_hydrogens(mol):
    """Heavy atoms only, decided here rather than by whichever rdkit is installed.

    The reference pipeline's molecules contain no hydrogens — `pocket_truncate` even says so
    in a comment — but it gets there by accident, relying on `removeHs` in rdkit 2021.03.
    That is not stable: reading 1eby's ligand and writing it back out drops its 40 hydrogens
    on rdkit 2021 and keeps every one of them on 2026, and neither `RemoveHs` nor
    `RemoveAllHs` removes them on the newer version. The molecule then has 88 atoms instead
    of 48, the complex graph has 382 nodes instead of 288, and the prediction lands 12 log
    units away — negative, well outside the range an affinity can take.

    So the hydrogens are deleted outright and the molecule sanitised afterwards, which
    recomputes the implicit hydrogen counts the featuriser reads through `GetTotalNumHs`.
    This is the same policy on every rdkit, which is the point.
    """
    hydrogens = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
    if not hydrogens:
        return mol
    editable = Chem.RWMol(mol)
    for idx in sorted(hydrogens, reverse=True):
        # An aromatic neighbour keeps the hydrogen as an explicit count. This is rdkit's own
        # rule, not an embellishment: a pyrrole-type nitrogen's hydrogen cannot be inferred
        # back from its valence, so dropping it outright turns `GetTotalNumHs` from 1 into 0
        # and moves one of the five total-H columns.
        #
        # Aromaticity is read off the bonds, not the atom flag. The flag is exactly what the
        # newer rdkit fails to set on these files, and the sanitisation below recomputes it
        # anyway — so `flag_aromatic_atoms` has to run after this function, and this test
        # cannot depend on it having run before.
        for neighbour in editable.GetAtomWithIdx(idx).GetNeighbors():
            if any(b.GetBondType() == Chem.BondType.AROMATIC for b in neighbour.GetBonds()):
                neighbour.SetNumExplicitHs(neighbour.GetNumExplicitHs() + 1)
        editable.RemoveAtom(idx)
    stripped = editable.GetMol()
    try:
        Chem.SanitizeMol(stripped)
    except Exception:
        # Kekulisation is the step that fails on these files, and it is also the one the
        # featuriser does not depend on: every feature it reads is an element, a degree, a
        # charge, a hybridisation or an aromatic flag.
        Chem.SanitizeMol(stripped, Chem.SanitizeFlags.SANITIZE_ALL
                         ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    return stripped


def flag_aromatic_atoms(mol):
    """Mark an atom aromatic when one of its bonds is, which is what rdkit 2021 did.

    PDBbind's ligand sdf files use MOL bond type 4 — "aromatic" — which is not part of the
    MDL standard. rdkit 2021.03 read that as aromaticity and set the flag on the atoms;
    rdkit 2026 keeps the bond type and leaves the atoms unflagged, so the same benzene ring
    writes as `c1ccccc1` on one and `C1:C:C:C:C:C:1` on the other. Sanitisation does not
    settle it either way: the molecule cannot be kekulised from bonds alone.

    `atom_is_aromatic` is one of the 40 atom features, and the published checkpoints were
    trained with it set. Recomputing aromaticity by some model of our own would be a
    different feature; copying the old reader's rule reproduces the one they saw.

    Only ligands need this. The pocket comes from a pdb, which has no bond block at all, and
    the two rdkit versions already agree on it.
    """
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            bond.GetBeginAtom().SetIsAromatic(True)
            bond.GetEndAtom().SetIsAromatic(True)
    return mol


def read_ligand(complex_dir, cid, work=None, sanitize=True):
    """The ligand exactly as the reference pipeline delivers it to the featuriser.

    The chain matters more than it looks. Upstream reads the file with `SDMolSupplier`,
    writes the molecule back out with `MolToMolFile`, and only then does `pocket_truncate`
    open it with `MolFromMolFile`. Reading the original file with `MolFromMolFile` directly
    is not a shortcut to the same molecule: on 1eby it returns 88 atoms where the supplier
    returns 48, because the hydrogens survive one route and not the other. Featurising the
    88-atom version gives a complex with 382 nodes instead of 288 and a prediction 4.4 log
    units off — a difference no shape check can see.

    The sdf comes first and the mol2 beside it is the fallback. Both are needed: some
    PDBbind sdf files do not sanitise — 1owh and 1a30 among them — and their mol2 usually
    does, while other complexes ship no mol2 at all. The mol2 has to be rewritten as an sdf
    regardless, since `MolFromMolFile` cannot open one.
    """
    sdf = os.path.join(complex_dir, "%s_ligand.sdf" % cid)
    mol2 = os.path.join(complex_dir, "%s_ligand.mol2" % cid)

    mol = None
    if os.path.exists(sdf):
        found = [m for m in Chem.SDMolSupplier(sdf, sanitize=sanitize) if m is not None]
        mol = found[0] if found else None
    if mol is None and os.path.exists(mol2):
        mol = Chem.MolFromMol2File(mol2, sanitize=sanitize)
    if mol is None:
        raise PocketError("no readable ligand in %s" % complex_dir)

    # The write-and-reread is the pipeline, not tidying: what the featuriser sees is always
    # a molecule that has been through MolToMolFile and MolFromMolFile.
    if work is None:
        work = tempfile.mkdtemp(prefix="ign_ligand_")
    os.makedirs(work, exist_ok=True)
    staged = os.path.join(work, "%s_lig.sdf" % cid)
    Chem.MolToMolFile(mol, staged)
    mol = Chem.MolFromMolFile(staged, sanitize=sanitize)
    if mol is None:
        raise PocketError("the staged ligand could not be read back: %s" % staged)
    return flag_aromatic_atoms(strip_hydrogens(mol))


def truncate(protein_file, ligand, pocket_out_file, distance=5, sanitize=True):
    """(protein pdb, ligand mol) -> (ligand mol, pocket mol), both with a conformer.

    `distance` is in angstroms and selects whole residues, not atoms: a residue is kept if
    any of its atoms is within range, so side chains are never cut in half.
    """
    from prody import parsePDB, writePDB

    if isinstance(ligand, str):     # a path, read with the same fallback as read_ligand
        ligand_file, ligand = ligand, Chem.MolFromMolFile(ligand, sanitize=sanitize)
        if ligand is None:
            raise PocketError("rdkit could not read the ligand: %s" % ligand_file)

    structure = parsePDB(protein_file)
    if structure is None:
        raise PocketError("prody could not read the protein: %s" % protein_file)

    protein = structure.select("protein")   # drops water, ions and other heteroatoms
    if protein is None:
        raise PocketError("no protein atoms in %s" % protein_file)

    selected = protein.select("same residue as within %s of ligand" % distance,
                              ligand=ligand.GetConformer().GetPositions())
    if selected is None:
        raise PocketError("no residue within %s A of the ligand in %s" % (distance, protein_file))

    directory = os.path.dirname(pocket_out_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    writePDB(pocket_out_file, selected)

    # rdkit reads the pocket back from the file prody just wrote; the hydrogens are removed
    # explicitly below, and the featuriser counts them through GetTotalNumHs instead.
    pocket = Chem.MolFromPDBFile(pocket_out_file, sanitize=sanitize)
    if pocket is None:
        raise PocketError("rdkit could not read the pocket back: %s" % pocket_out_file)
    # The pocket needs the same treatment: prody writes hydrogens into the pdb, and whether
    # rdkit drops them on the way back in is another thing that changed between versions.
    return ligand, strip_hydrogens(pocket)
