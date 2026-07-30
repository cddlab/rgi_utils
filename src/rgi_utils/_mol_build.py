"""Shared RDKit mol builder for ligand conformer restraints.

Tools that expose a per-atom structure object (elements + bonds + ideal
reference coordinates) rebuild the ligand mol from that subset instead of
looking it up by CCD name. protenix, openfold-3 and chai all share this builder
so the bond/angle/chiral featurization sees the same mol regardless of tool
(building from the present atoms also makes it leaving-atom-correct).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Force fields available to `ff_relax` /
# `conformer_restraints_config.relax_force_field.ligand`. "none" is not a force field:
# it disables the relax so the targets come straight off the tool's cached conformer.
_RELAX_FORCE_FIELDS = ("uff", "mmff94", "mmff94s", "none")


class RelaxError(ValueError):
    """A force-field relax the user EXPLICITLY asked for could not be performed.

    Raised only for `mmff94`/`mmff94s`. UFF is the default (nobody asked for it), so a UFF
    failure stays a soft fall-back to the cached conformer -- see :func:`ff_relax`.
    """


def parse_relax_force_field(conformer_config: dict | None) -> str:
    """Validated ligand ``relax_force_field``; ``"uff"`` when omitted.

    Mirrors ``monlib_geom.parse_config``: the module that consumes the vocabulary owns its
    validation, and ``config.py`` calls this while PARSING so a typo raises there rather
    than several minutes later. Validating only the config KEY would not be enough -- an
    unknown value would slip through unnoticed on any run where no ligand opts in, leaving
    a `mmf94` typo silently running UFF.

    The public shape is ``relax_force_field: {ligand: uff}``. A missing ``ligand`` or a
    YAML ``null`` value counts as omitted (matching ``_conf_slack``'s null handling); the
    STRING ``"none"`` is the explicit "do not relax at all".
    """
    config = conformer_config or {}
    if "relax_force_field" not in config:
        return "uff"
    spec = config["relax_force_field"]
    if not isinstance(spec, dict):
        raise ValueError(
            "conformer_restraints_config.relax_force_field must be a mapping; "
            "the scalar form is no longer supported. Use "
            f"relax_force_field: {{ligand: {spec!r}}}."
        )
    unknown = {str(key) for key in spec if key != "ligand"}
    if unknown:
        raise ValueError(
            "conformer_restraints_config.relax_force_field: unknown key(s) "
            f"{sorted(unknown)}. Known keys: ['ligand']"
        )
    value = spec.get("ligand")
    if value is None:
        return "uff"
    ff = str(value).lower()
    if ff not in _RELAX_FORCE_FIELDS:
        raise ValueError(
            "conformer_restraints_config.relax_force_field.ligand: unknown value "
            f"{value!r}, "
            f"expected one of {list(_RELAX_FORCE_FIELDS)}"
        )
    return ff


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
    force-field relax (``featurizer._extract_conformer`` -> :func:`ff_relax`) needs aromatic
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
        # throws). Consequently every perceived bond is SINGLE, so the cistrans
        # (cis/trans) restraint — which keys on BondType.DOUBLE — finds nothing
        # via THIS branch (cistrans=0, graceful). chai's adapter therefore PREFERS
        # its source-SMILES path (chai/adapter.py _mol_from_smiles), which carries
        # real Kekulized bond orders + E/Z, and only falls back to this perceive
        # branch when no SMILES is available; bond/angle/chiral (order-agnostic) work
        # either way. The other tools supply real bond orders directly (boltz/protenix/
        # openfold/AF3 via CCD/biotite; esmfold2 via ligand_bond_orders).
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
        except Exception as exc:
            # Don't abort (geometry-only restraints don't need connectivity), but a
            # bond-less mol silently drops chiral/cistrans restraints, so surface it.
            logger.warning(
                "connectivity perception failed for ligand; chiral/cistrans "
                "restraints may be dropped: %s",
                exc,
            )
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
    isomer (e.g. a maleate predicted trans), so its bond/angle/cistrans targets are wrong
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


def ff_relax(mol, coords, force_field="uff"):
    """Force-field-relax ``coords`` (a conformer of ``mol``) to ideal bond/angle geometry
    while KEEPING the input fold.

    Unlike :func:`generate_ideal_conformer` (a from-scratch ETKDG embed that mis-folds
    big/flexible/phosphate ligands), this starts from the tool's existing conformer and
    runs a LOCAL force-field minimisation, so the global fold is preserved while
    Kekule-localized aromatic rings, stretched bonds and bent angles relax to their
    force-field-ideal values. Used to derive bond/angle restraint TARGETS that are
    consistent across tools (every tool's cached conformer otherwise carries its own
    bond/angle idiosyncrasies). Stereo is preserved (local minimisation from a fixed
    start). Returns heavy-atom coords in ``mol`` atom order, or None on failure.

    ``force_field`` (``conformer_restraints_config.relax_force_field.ligand``):
    ``"uff"`` (default) / ``"mmff94"`` / ``"mmff94s"``. Neither is uniformly better --
    on adenosine monophosphate MMFF lands the conjugated exocyclic C6-N6 far closer to
    the monomer library (1.376-1.389 vs UFF 1.428, library 1.330) while UFF wins on the
    glycosidic C1'-N9 -- hence a user-selectable option rather than a new default.
    ``"none"`` is handled by the CALLER (it means "do not call this at all").

    **The failure policies differ deliberately.** UFF is the default that nobody asked
    for, so a failure returns None and the caller keeps its cached conformer (unchanged
    long-standing behaviour). MMFF is only ever reached because the user explicitly set
    ``relax_force_field.ligand``, so any failure raises :class:`RelaxError` rather
    than silently producing un-relaxed targets that look like MMFF ones. This matters in
    practice: MMFF has NO metal parameters (``MMFFHasAllMoleculeParams`` is False for Fe
    where UFF is
    True, and ``MMFFOptimizeMolecule`` then returns -1 without raising).

    **MMFF mutates the mol it is handed.** ``MMFFHasAllMoleculeParams`` /
    ``MMFFGetMoleculeProperties`` / ``MMFFOptimizeMolecule`` KEKULIZE their argument --
    aromatic flags cleared, bonds rewritten to SINGLE/DOUBLE (reproducible on caffeine).
    ``featurizer._extract_conformer`` perceives ``cistrans`` from ``BondType.DOUBLE`` and
    ``plane`` from ring info, so leaking that into the caller's mol would silently change
    which restraints get built. Every RDKit call below therefore runs on the local
    ``Chem.Mol(mol)`` copy (deep enough to protect the caller) -- never on ``mol``.

    ``maxIters=200`` is shared by both force fields. RDKit reports rc=1 ("iteration limit")
    there for all of UFF/MMFF94/MMFF94s on ATP, but the geometry is converged in practice:
    re-running at 2000 and 20000 iterations gives bond lengths identical to 3 decimals and
    only flips rc to 0. So rc=1 is accepted, and raising the budget would buy nothing while
    moving every existing UFF target.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    ff = str(force_field).lower()
    try:
        coords = np.asarray(coords, dtype=np.float64)
        m = Chem.Mol(mol)  # deep copy: shields the caller from MMFF's kekulization
        conf = Chem.Conformer(m.GetNumAtoms())
        for i in range(m.GetNumAtoms()):
            conf.SetAtomPosition(
                i, (float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2]))
            )
        m.RemoveAllConformers()
        m.AddConformer(conf, assignId=True)
        mh = Chem.AddHs(m, addCoords=True)  # Hs placed from heavy-atom geometry
        if ff == "uff":
            if AllChem.UFFOptimizeMolecule(mh, maxIters=200) not in (0, 1):
                return (
                    None  # not converged / no force field -> keep the tool's conformer
                )
        else:
            # Pre-check on the COPY: a clearer message than the bare rc == -1 that a
            # metal-containing (or otherwise un-typeable) ligand would otherwise produce.
            variant = "MMFF94s" if ff == "mmff94s" else "MMFF94"
            if not AllChem.MMFFHasAllMoleculeParams(mh):
                raise RelaxError(
                    f"relax_force_field.ligand={force_field!r}: RDKit has no {variant} "
                    "parameters for this ligand (MMFF covers no metals, unlike UFF). Use "
                    "relax_force_field: {ligand: uff}, or {ligand: none} to skip the "
                    "relax entirely."
                )
            rc = AllChem.MMFFOptimizeMolecule(mh, maxIters=200, mmffVariant=variant)
            if rc == -1:
                raise RelaxError(
                    f"relax_force_field.ligand={force_field!r}: {variant} force-field "
                    "setup failed for this ligand. Use relax_force_field: "
                    "{ligand: uff}, or {ligand: none} to skip the relax entirely."
                )
        mh = Chem.RemoveHs(mh)
        if mh.GetNumAtoms() != mol.GetNumAtoms():
            return None
        return np.asarray(mh.GetConformer(0).GetPositions(), dtype=np.float64)
    except RelaxError:
        raise
    except Exception as exc:
        if ff.startswith("mmff"):
            # Explicitly requested -> never degrade silently. This also covers the mol
            # whose SanitizeMol failed in build_ligand_mol (a bare except: pass there), on
            # which RDKit raises a Pre-condition Violation RuntimeError.
            raise RelaxError(
                f"relax_force_field.ligand={force_field!r}: relax failed for this "
                f"ligand: {exc}"
            ) from exc
        return None
