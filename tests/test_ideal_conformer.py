"""generate_ideal_conformer: stereo-preserving ETKDG ideal conformer for SMILES ligands.

The SMILES-ligand restraint target must keep the SMILES stereo (cis/trans, chirality);
a model's predicted conformer often does not (e.g. maleate predicted trans). These guard
that the helper embeds the correct isomer so bond/angle/chiral/cistrans targets are ideal.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from rgi_utils._mol_build import generate_ideal_conformer


def _cc_torsion(mol, coords):
    """|carboxyl-C - C=C - carboxyl-C| torsion of the (single) acyclic C=C, or None."""
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i in range(len(coords)):
        conf.SetAtomPosition(i, (float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])))
    m.AddConformer(conf, assignId=True)
    for b in m.GetBonds():
        a2, a3 = b.GetBeginAtom(), b.GetEndAtom()
        if b.GetBondType() == Chem.BondType.DOUBLE and a2.GetSymbol() == "C" and a3.GetSymbol() == "C":
            def cn(a, o):
                return [
                    n for n in a.GetNeighbors()
                    if n.GetIdx() != o.GetIdx() and n.GetSymbol() == "C"
                    and any(x.GetSymbol() == "O" for x in n.GetNeighbors())
                ]
            c2, c4 = cn(a2, a3), cn(a3, a2)
            if c2 and c4:
                return abs(rdMolTransforms.GetDihedralDeg(
                    m.GetConformer(), c2[0].GetIdx(), a2.GetIdx(), a3.GetIdx(), c4[0].GetIdx()))
    return None


def test_maleate_is_cis():
    mol = Chem.MolFromSmiles(r"OC(=O)/C=C\C(=O)O")
    coords = generate_ideal_conformer(mol)
    assert coords is not None and len(coords) == mol.GetNumAtoms()
    t = _cc_torsion(mol, coords)
    assert t is not None and t < 30.0, f"maleate must embed cis (~0deg), got {t}"


def test_fumarate_is_trans():
    mol = Chem.MolFromSmiles("OC(=O)/C=C/C(=O)O")
    coords = generate_ideal_conformer(mol)
    assert coords is not None and len(coords) == mol.GetNumAtoms()
    t = _cc_torsion(mol, coords)
    assert t is not None and t > 150.0, f"fumarate must embed trans (~180deg), got {t}"


def test_atom_count_matches():
    mol = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")  # one stereocentre
    coords = generate_ideal_conformer(mol)
    assert coords is not None and len(coords) == mol.GetNumAtoms()


def test_reorder_to_target_keeps_stereo():
    """generate_ideal_conformer(smol, target_mol) reorders the ideal coords to target_mol's
    atom order via a substructure match (the protenix path: target_mol carries real bond
    orders, atom order == atom_array order). The reordered coords must keep the correct E/Z."""
    import numpy as np

    from rgi_utils._mol_build import build_ligand_mol

    smol = Chem.MolFromSmiles("OC(=O)/C=C/C(=O)O")  # fumarate (trans)
    canon = generate_ideal_conformer(smol)
    assert canon is not None
    elems = np.array([a.GetAtomicNum() for a in smol.GetAtoms()])
    bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble()))
        for b in smol.GetBonds()
    ]
    target = build_ligand_mol(elems, canon, bonds)  # real bond orders, atom_array order
    out = generate_ideal_conformer(smol, target_mol=target)
    assert out is not None and len(out) == smol.GetNumAtoms()
    t = _cc_torsion(smol, out)
    assert t is not None and t > 150.0, f"reordered fumarate must stay trans, got {t}"
