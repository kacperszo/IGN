"""IGN's graph construction without DGL or dgllife — arrays in, arrays out.

`scripts/graph_constructor.py` builds two DGL graphs and hangs features off them. This
produces the same tensors as plain arrays: an `edge_index` pair per graph plus the node and
edge features, which is all `model.py` consumes.

Written to run unchanged in the reference image (python 3.6, torch 1.3.1) as well as on a
current stack. That is what makes it checkable: `compare_features.py` runs this inside
`ign-ref` against the very arrays the DGL constructor produced there, so a disagreement is
the port's fault and not the environment's.

The dgllife featurisers are reimplemented here rather than imported. dgllife pins dgl, so
importing it would defeat the point. The vocabularies below are copied from that package,
not recalled — a wrong entry shifts a one-hot column and changes nothing else that any
shape check could catch.

Sizes, which the checkpoints fix and nothing may drift from:

    atom features   17 type + 6 degree + 1 charge + 1 radicals + 6 hybridisation
                    + 1 aromatic + 5 total-H + 3 chirality            =  40
    AEV             9 radial + 45 angular (num_species=9)             =  54  -> h is 94
    bond features   4 type + 1 conjugated + 1 in-ring + 5 stereo      =  11
                    + 1 distance + 9 three-body                       ->  e is 21
"""

# No `from __future__ import annotations`: this has to import on python 3.6 as well, which
# is what the reference image runs and where the comparison happens.
import numpy as np
import torch
from rdkit import Chem
from scipy.spatial import distance_matrix

# ---------------------------------------------------------------- dgllife, reimplemented

ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si',
              'Fe', 'Zn', 'Cu', 'Mn', 'Mo']
HYBRIDIZATIONS = [Chem.rdchem.HybridizationType.SP,
                  Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3,
                  Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2]
BOND_TYPES = [Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]
BOND_STEREO = [Chem.rdchem.BondStereo.STEREONONE,
               Chem.rdchem.BondStereo.STEREOANY,
               Chem.rdchem.BondStereo.STEREOZ,
               Chem.rdchem.BondStereo.STEREOE]


def one_hot(x, allowable_set, encode_unknown=False):
    """dgllife's `one_hot_encoding`, minus its habit of mutating the caller's list.

    Upstream appends `None` to the list it was handed, so a `partial` sharing one list is
    changed by its first call. The result is the same either way — the append happens once —
    but doing it on a copy means the vocabularies above stay what they say they are.
    """
    if encode_unknown and allowable_set[-1] is not None:
        allowable_set = list(allowable_set) + [None]
    if encode_unknown and x not in allowable_set:
        x = None
    return [x == s for s in allowable_set]


def chirality(atom):
    """AttentiveFP's chirality triple — three zeros, and that is not a stub.

    Upstream reads `_CIPCode` and `_ChiralityPossible` off the atom. Neither is ever set by
    the time it looks: `pocket_truncate` pickles `[ligand, pocket]` and `graphs_from_mol_ign`
    unpickles them, and rdkit's pickler drops atom properties unless
    `SetDefaultPickleProperties` says otherwise. Measured on the reference image: a ligand
    with two `_ChiralityPossible` flags comes back from a pickle round trip with none.

    So these three columns are constant zero in every graph the published checkpoints were
    trained on, and constant zero in the golden predictions. Computing them properly here
    would feed the network three features it has never seen take a value — the opposite of
    reproducing it. Restoring them is a retraining decision, not a porting one.
    """
    return [False, False, False]


def atom_features(mol):
    """(n_atoms, 40), in the order MyAtomFeaturizer concatenates them."""
    rows = []
    for atom in mol.GetAtoms():
        rows.append(
            one_hot(atom.GetSymbol(), ATOM_TYPES, encode_unknown=True)
            + one_hot(atom.GetDegree(), list(range(6)))
            + [atom.GetFormalCharge()]
            + [atom.GetNumRadicalElectrons()]
            + one_hot(atom.GetHybridization(), HYBRIDIZATIONS, encode_unknown=True)
            + [atom.GetIsAromatic()]
            + one_hot(atom.GetTotalNumHs(), list(range(5)))
            + chirality(atom)
        )
    return np.array(rows, dtype=np.float32)


N_BOND_FEATURES = 11


def bond_features(mol):
    """(n_bonds, 11), one row per bond — not per directed edge."""
    rows = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        rows.append(
            one_hot(bond.GetBondType(), BOND_TYPES)
            + [bond.GetIsConjugated()]
            + [bond.IsInRing()]
            + one_hot(bond.GetStereo(), BOND_STEREO, encode_unknown=True)
        )
    if not rows:
        # a molecule with no bonds is rare but real — a single metal ion, say. Reshaping an
        # empty array to (0, -1) cannot infer the width and raises, so state it.
        return np.zeros((0, N_BOND_FEATURES), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------- three-body edge features

def _d3_info(a, b, c):
    ab, ac = b - a, c - a
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_


def _d3_features(edge_index, pos):
    """Nine numbers per directed edge, from the angles it makes with the source's neighbours.

    For edge (u, v): every other in-neighbour w of u contributes the angle at u between v
    and w, the triangle's area, and the distance u-w; the nine values are the max, sum and
    mean of each. An edge whose source has no other neighbour gets zeros, which is what
    makes the edge count and the feature count agree.
    """
    src, dst = edge_index[0], edge_index[1]
    # in-neighbours of every node, with multiplicity — DGL's `predecessors` reads them
    # straight out of the adjacency, so a repeated edge is counted twice and the sums and
    # means below would shift if this deduplicated.
    preds = [[] for _ in range(pos.shape[0])]
    for u, v in zip(src, dst):
        preds[v].append(u)

    out = np.zeros((len(src), 9), dtype=np.float64)
    for i in range(len(src)):
        u, v = int(src[i]), int(dst[i])
        others = list(preds[u])
        if v in others:
            others.remove(v)          # remove one occurrence, as list.remove does upstream
        if not others:
            continue
        angles, areas, distances = [], [], []
        for w in others:
            angle, area, distance = _d3_info(pos[u], pos[v], pos[w])
            angles.append(angle)
            areas.append(area)
            distances.append(distance)
        out[i] = [np.max(angles) * 0.01, np.sum(angles) * 0.01, np.mean(angles) * 0.01,
                  np.max(areas), np.sum(areas), np.mean(areas),
                  np.max(distances) * 0.1, np.sum(distances) * 0.1, np.mean(distances) * 0.1]
    return out


# ---------------------------------------------------------------- the AEV block

def atom_environment_vectors(mol1, mol2, EtaR=4.00, ShfR=3.17, Zeta=8.00, ShtZ=3.14):
    """torchani's Behler-Parrinello vectors, computed over the complex as one system.

    Ligand and pocket go in together on purpose: an atom's environment is what the other
    molecule puts around it, and splitting them would describe two isolated fragments.
    """
    from torchani import AEVComputer, SpeciesConverter

    converter = SpeciesConverter(['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'])
    numbers = ([a.GetAtomicNum() for a in mol1.GetAtoms()]
               + [a.GetAtomicNum() for a in mol2.GetAtoms()])
    coords = np.concatenate([mol1.GetConformer().GetPositions(),
                             mol2.GetConformer().GetPositions()], axis=0)
    species = torch.tensor(numbers, dtype=torch.long).unsqueeze(0)
    positions = torch.tensor(coords, dtype=torch.float64).unsqueeze(0)
    res = converter((species, positions))
    computer = AEVComputer(Rcr=12.0, Rca=12.0,
                           EtaR=torch.tensor([EtaR]), ShfR=torch.tensor([ShfR]),
                           EtaA=torch.tensor([3.5]), Zeta=torch.tensor([Zeta]),
                           ShfA=torch.tensor([0]), ShfZ=torch.tensor([ShtZ]),
                           num_species=9)
    return computer((res.species, res.coordinates)).aevs[0].float().numpy()


# ---------------------------------------------------------------- the graph

def build_graph(mol1, mol2, dis_threshold=8.0, EtaR=4.00, ShfR=3.17, Zeta=8.00, ShtZ=3.14):
    """(ligand, pocket) -> the arrays `model.py` takes.

    Returns h, e, edge_index for the covalent graph and e3, edge_index3 for the interaction
    graph. Node ids are the ligand's atoms first, then the pocket's, offset by the ligand's
    count — the same numbering the checkpoints were trained under.
    """
    n1, n2 = mol1.GetNumAtoms(), mol2.GetNumAtoms()
    n = n1 + n2

    # Edges go in as the original appended them: ligand bonds, then pocket bonds, each in
    # both directions with all the forward directions before all the reverse ones. Order is
    # load-bearing, because features are attached by position further down.
    src1 = [mol1.GetBondWithIdx(i).GetBeginAtomIdx() for i in range(mol1.GetNumBonds())]
    dst1 = [mol1.GetBondWithIdx(i).GetEndAtomIdx() for i in range(mol1.GetNumBonds())]
    src2 = [mol2.GetBondWithIdx(i).GetBeginAtomIdx() + n1 for i in range(mol2.GetNumBonds())]
    dst2 = [mol2.GetBondWithIdx(i).GetEndAtomIdx() + n1 for i in range(mol2.GetNumBonds())]

    src = np.concatenate([src1, dst1, src2, dst2]).astype(np.int64)
    dst = np.concatenate([dst1, src1, dst2, src2]).astype(np.int64)
    edge_index = np.stack([src, dst])

    dis_matrix = distance_matrix(mol1.GetConformer().GetPositions(),
                                 mol2.GetConformer().GetPositions())
    inter = np.where(dis_matrix < dis_threshold)
    edge_index3 = np.stack([inter[0].astype(np.int64), (inter[1] + n1).astype(np.int64)])
    e3 = (dis_matrix[inter[0], inter[1]] * 0.1).astype(np.float32).reshape(-1, 1)

    h = np.zeros((n, 94), dtype=np.float32)
    h[:n1, :40] = atom_features(mol1)
    h[n1:, :40] = atom_features(mol2)
    h[:, 40:] = atom_environment_vectors(mol1, mol2, EtaR, ShfR, Zeta, ShtZ)

    # One bond feature row per bond, repeated for the forward block and again for the
    # reverse block, so an edge and its opposite carry the same chemistry.
    b1, b2 = bond_features(mol1), bond_features(mol2)
    e = np.zeros((len(src), N_BOND_FEATURES), dtype=np.float32)
    e[:2 * len(src1)] = np.concatenate([b1, b1]) if len(src1) else e[:0]
    e[2 * len(src1):] = np.concatenate([b2, b2]) if len(src2) else e[2 * len(src1):]

    pos = np.zeros((n, 3), dtype=np.float32)
    pos[:n1] = mol1.GetConformer().GetPositions()
    pos[n1:] = mol2.GetConformer().GetPositions()

    d1 = distance_matrix(mol1.GetConformer().GetPositions(), mol1.GetConformer().GetPositions())
    d2 = distance_matrix(mol2.GetConformer().GetPositions(), mol2.GetConformer().GetPositions())
    lig = np.concatenate([src1, dst1]).astype(int)
    lig_dst = np.concatenate([dst1, src1]).astype(int)
    pkt = np.concatenate([src2, dst2]).astype(int) - n1
    pkt_dst = np.concatenate([dst2, src2]).astype(int) - n1
    dist = np.concatenate([d1[lig, lig_dst], d2[pkt, pkt_dst]]).astype(np.float32)

    e = np.concatenate([e, (dist * 0.1).reshape(-1, 1)], axis=1)
    e = np.concatenate([e, _d3_features(edge_index, pos).astype(np.float32)], axis=1)

    return {"h": h, "e": e, "edge_index": edge_index,
            "e3": e3, "edge_index3": edge_index3}
