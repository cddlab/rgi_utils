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

    Bond ORDER does matter for one downstream consumer: the conformer restraint's
    UFF-relax (``featurizer._extract_conformer`` -> :func:`uff_relax`) needs aromatic
    rings to be perceivable as aromatic, or it puckers a flat single-bond ring to
    sp3. ``order`` therefore accepts BOTH conventions seen across tools: the
    RDKit-ish 1/2/3 (single/double/triple) and biotite ``BondType`` codes, whose
    aromatic rings come Kekule-encoded as 5 (AROMATIC_SINGLE) / 6 (AROMATIC_DOUBLE)
    plus 9 (AROMATIC). Mapping 5/6 back to single/double hands ``SanitizeMol`` a
    valid Kekule structure, so it re-perceives the ring as aromatic.

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
        4: Chem.BondType.AROMATIC,  # legacy explicit-aromatic convention
        5: Chem.BondType.SINGLE,  # biotite AROMATIC_SINGLE -> Kekule single
        6: Chem.BondType.DOUBLE,  # biotite AROMATIC_DOUBLE -> Kekule double
        7: Chem.BondType.TRIPLE,  # biotite AROMATIC_TRIPLE (rare)
        9: Chem.BondType.AROMATIC,  # biotite AROMATIC (generic)
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


def generate_ideal_conformer(mol, target_mol=None):
    """Stereo-preserving ETKDGv3 + UFF ideal conformer for ``mol`` (whose chiral tags +
    bond E/Z are ALREADY correct, e.g. from SMILES). Returns an (n_heavy, 3) float64
    array, or None on failure.

    This is the SMILES-ligand restraint TARGET source. A model's own reference conformer
    (protenix ``atom_array.coord`` / af3,openfold ``ref_pos``) is often not the ideal
    isomer (e.g. a maleate predicted trans), so its bond/angle/dihedral targets are wrong
    even though the restraint converges to them. Embedding from the correct-stereo mol
    gives the ideal geometry (cis maleate ~0deg) exactly like boltz's ligand_mol
    conformer. It does NOT call AssignStereochemistryFrom3D (that would re-perceive stereo
    from geometry); the caller's mol must already carry the intended stereo.

    ``target_mol``: when given (the adapter's mol, in the coordinate-tensor / global_indices
    atom order), the returned coords are REORDERED to ``target_mol``'s atom order via a
    substructure match — ``mol`` supplies the SMILES stereo, ``target_mol`` fixes the atom
    order (a SMILES mol's RDKit-canonical order often differs from the tool's internal atom
    order, e.g. af3 token order). Returns None if the match fails: a wrong mapping would
    corrupt the restraint, so the caller falls back to the model's own coords.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        mh = Chem.AddHs(mol)  # explicit H gives a sensible 3D embedding
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D  # deterministic target geometry across runs
        cid = AllChem.EmbedMolecule(mh, params)
        if cid == -1:  # retry with random coords for hard cases
            params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(mh, params)
        if cid == -1:
            return None
        try:
            AllChem.UFFOptimizeMolecule(mh, confId=cid, maxIters=1000)
        except Exception:  # a geometry-only target doesn't need a clean FF result
            pass
        # AddHs appends Hs AFTER the heavy atoms; RemoveHs restores the original
        # heavy-atom order, so the rows line up 1:1 with the input ``mol``'s atoms.
        mh = Chem.RemoveHs(mh)
        if mh.GetNumAtoms() != mol.GetNumAtoms():
            return None
        coords = np.asarray(mh.GetConformer(0).GetPositions(), dtype=np.float64)
        if target_mol is None:
            return coords
        # reorder mol's atom order -> target_mol's: match[i] is the target_mol atom that
        # corresponds to mol atom i, so out[match[i]] = coords[i].
        match = target_mol.GetSubstructMatch(mol)
        if len(match) != mol.GetNumAtoms():
            match = target_mol.GetSubstructMatch(mol, useChirality=False)
        if len(match) != mol.GetNumAtoms():
            return None  # atom-order mapping failed -> caller keeps the model coords
        out = np.zeros_like(coords)
        for i, j in enumerate(match):
            out[j] = coords[i]
        return out
    except Exception:
        return None


def uff_relax(mol, coords):
    """UFF-relax ``coords`` (a conformer of ``mol``) to ideal bond/angle geometry while
    KEEPING the input fold.

    Unlike :func:`generate_ideal_conformer` (a from-scratch ETKDG embed that mis-folds
    big/flexible/phosphate ligands), this starts from the tool's existing conformer and
    runs a LOCAL force-field minimisation, so the global fold is preserved while
    Kekule-localized aromatic rings, stretched bonds and bent angles relax to their
    force-field-ideal values. Used to derive bond/angle restraint TARGETS that are
    consistent across tools (every tool's cached conformer otherwise carries its own
    bond/angle idiosyncrasies). Stereo is preserved (local minimisation from a fixed
    start). Returns heavy-atom coords in ``mol`` atom order, or None on failure.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        coords = np.asarray(coords, dtype=np.float64)
        m = Chem.Mol(mol)
        conf = Chem.Conformer(m.GetNumAtoms())
        for i in range(m.GetNumAtoms()):
            conf.SetAtomPosition(i, (float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])))
        m.RemoveAllConformers()
        m.AddConformer(conf, assignId=True)
        mh = Chem.AddHs(m, addCoords=True)  # Hs placed from heavy-atom geometry
        if AllChem.UFFOptimizeMolecule(mh, maxIters=200) not in (0, 1):
            return None  # not converged / no force field -> keep the tool's conformer
        mh = Chem.RemoveHs(mh)
        if mh.GetNumAtoms() != mol.GetNumAtoms():
            return None
        return np.asarray(mh.GetConformer(0).GetPositions(), dtype=np.float64)
    except Exception:
        return None
