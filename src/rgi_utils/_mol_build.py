"""Shared RDKit mol builder for ligand conformer restraints.

Tools that expose a per-atom structure object (elements + bonds + ideal
reference coordinates) rebuild the ligand mol from that subset instead of
looking it up by CCD name. protenix, openfold-3 and chai all share this builder
so the bond/angle/chiral featurization sees the same mol regardless of tool
(building from the present atoms also makes it leaving-atom-correct).
"""

from __future__ import annotations


def atomic_number(symbol: str) -> int:
    """Atomic number for an element symbol; 0 if unknown (treated as padding)."""
    from rdkit.Chem import GetPeriodicTable

    try:
        return int(GetPeriodicTable().GetAtomicNumber(str(symbol).capitalize()))
    except Exception:
        return 0


def build_ligand_mol(elements, coords, bonds_local, perceive_bonds=False):
    """Build an RDKit mol from a ligand's atoms (element symbols), 3D coords and
    local bond tuples ``(i, j, order)``. Chirality is assigned from the 3D
    geometry, so bond order is immaterial to the geometric (bond/angle/chiral)
    restraints — a tool that lacks bond orders may pass order=1 for every bond.

    ``perceive_bonds``: when True AND no explicit ``bonds_local`` are supplied,
    derive connectivity from the 3D reference conformer via RDKit
    ``DetermineConnectivity``. This is for tools (chai) whose structure context
    exposes a reference conformer (atom_ref_pos) but NOT intra-ligand bonds — the
    perceived topology is self-consistent with that conformer, so the resulting
    bond/angle/chiral restraints keep the ligand at its reference geometry. Tools
    that do expose real bonds (protenix/openfold) pass them and leave this False.
    """
    from rdkit import Chem

    rw = Chem.RWMol()
    conf = Chem.Conformer(len(elements))
    for i, sym in enumerate(elements):
        # accept an element symbol ("C") or an atomic number (6); the latter is
        # what tools storing ref_element as Z provide (e.g. chai). RDKit's
        # Chem.Atom() takes either, so int() decides which constructor to use.
        try:
            atom = Chem.Atom(int(sym))
        except (ValueError, TypeError):
            atom = Chem.Atom(str(sym).capitalize())
        rw.AddAtom(atom)
        conf.SetAtomPosition(
            i, (float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2]))
        )
    order_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    for li, lj, order in bonds_local:
        rw.AddBond(int(li), int(lj), order_map.get(int(order), Chem.BondType.SINGLE))
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)
    if perceive_bonds and not bonds_local and len(elements) > 1:
        # Derive connectivity from the reference-conformer geometry (chai path).
        # NOTE: only CONNECTIVITY is perceived, not bond ORDERS — chai exposes
        # heavy atoms only (no H) and no bonds, so RDKit DetermineBonds cannot
        # solve valences (it reads the H-less skeleton as highly charged and
        # throws). Consequently every perceived bond is SINGLE, so the dihedral
        # (cis/trans) restraint — which keys on BondType.DOUBLE — finds nothing on
        # chai ligands (dihedrals=0, graceful). The proper fix is to thread chai's
        # source SMILES/ConformerData mol (which carries real bond orders + E/Z)
        # into the adapter; until then bond/angle/chiral (order-agnostic) work but
        # cis/trans does not. The other four tools supply real bond orders.
        try:
            from rdkit.Chem import rdDetermineBonds

            rdDetermineBonds.DetermineConnectivity(mol)
            # DetermineConnectivity leaves every atom noImplicit=True with 0 implicit
            # H, so heavy-atom stereocenters look 3-coordinate and the
            # AssignStereochemistryFrom3D below would find ZERO chiral tags (silently
            # dropping all chiral restraints). Re-enable implicit-H accounting so
            # stereo perception sees the stereocenters. Scoped to the perceive branch
            # only — the explicit-bond path (protenix/openfold) is untouched.
            for a in mol.GetAtoms():
                a.SetNoImplicit(False)
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # geometry-only restraints don't need a clean valence model
        pass
    try:
        Chem.AssignStereochemistryFrom3D(mol)  # chiral tags for chiral restraints
    except Exception:
        pass
    return mol
