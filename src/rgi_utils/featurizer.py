"""Build a backend-agnostic ``RestraintSpec`` from ligand conformers + distances.

This is the single place where conformer restraints (bond/angle/chiral/cistrans/
improper) are derived from RDKit mols. Each ligand supplies its own
``global_indices``, so multiple ligands produce non-colliding restraints — there is
no per-batch state and no hard-coded ligand index (the multi-ligand bug in the old
code).

Flow:
  1. extract bond/angle/chiral/cistrans/improper restraints per ligand in GLOBAL atom
     indices,
  2. collect distance restraint atom groups (already resolved to global indices),
  3. active_sites = union of all referenced global atoms (sorted, unique),
  4. remap every index to a LOCAL index into active_sites and pack padded arrays.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
from rdkit import Chem

from rgi_utils._mol_build import uff_relax
from rgi_utils.atom_context import LigandConf
from rgi_utils.spec import (
    DIST_TYPE_CODES,
    AngleArrays,
    BondArrays,
    ChiralArrays,
    CisTransArrays,
    DistanceArrays,
    GroupAngleArrays,
    GroupDihedralArrays,
    ImproperArrays,
    RestraintSpec,
    RmsdArrays,
    VdwArrays,
    VdwConfig,
)

logger = logging.getLogger(__name__)

_ANGLE_PATT = Chem.MolFromSmarts("*~*~*")
_CHIRAL_TAGS = (
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
)


def _bond_length(crds: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(crds[i] - crds[j]))


def _angle_rad(crds: np.ndarray, i: int, j: int, k: int) -> float:
    rij = crds[i] - crds[j]
    rkj = crds[k] - crds[j]
    cos = np.dot(rij, rkj) / (np.linalg.norm(rij) * np.linalg.norm(rkj) + 1e-12)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _chiral_vol(crds: np.ndarray, c: int, n1: int, n2: int, n3: int) -> float:
    v1 = crds[n1] - crds[c]
    v2 = crds[n2] - crds[c]
    v3 = crds[n3] - crds[c]
    return float(np.dot(v1, np.cross(v2, v3)))


def _cistrans_rad(crds: np.ndarray, i: int, j: int, k: int, ll: int) -> float:
    """Signed torsion angle (radians) for atoms i-j-k-ll about the j-k axis.

    Identical formula to the energy backends' ``cistrans_energy`` so the target
    computed here equals the value the energy sees at the reference conformer
    (residual starts at zero before any perturbation).
    """
    b1 = crds[j] - crds[i]
    b2 = crds[k] - crds[j]
    b3 = crds[ll] - crds[k]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2n = b2 / np.sqrt(np.dot(b2, b2) + 1e-12)
    m1 = np.cross(n1, b2n)
    # Mirror the energy backends' dihedral exactly (incl. the atan2(0,0) guard) so
    # phi0 == the value the energy sees at the reference geometry.
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    if x == 0.0 and y == 0.0:
        x = 1e-12
    return float(np.arctan2(y, x))


def _extract_conformer(ligand_confs: list[LigandConf]):
    """Return bond/angle/chiral/cistrans/improper restraint tuples in GLOBAL atom
    indices."""
    bonds = []  # (g0, g1, r0)
    angles = []  # (g0, g1, g2, th0)
    chirals = []  # (g0, g1, g2, g3, vol0)
    cistrans = []  # (g0, g1, g2, g3, phi0)
    impropers = []  # (g0, g1, g2, g3, vol0) — sp2 planarity, center is g0

    for lc in ligand_confs:
        mol = lc.mol
        crds = np.asarray(lc.conf_coords, dtype=np.float64)
        gidx = np.asarray(lc.global_indices, dtype=np.int64)
        # Derive the bond/angle/chiral/cistrans TARGETS from a UFF-relaxed copy of the
        # tool's conformer. Each tool's cached conformer carries its own idiosyncrasies
        # (the boltz v2 ~/.boltz/mols cache Kekule-localizes aromatic rings ~1.34/1.48;
        # other tools' ref_pos has non-ideal bond/angle lengths), so without this the
        # restraint just reproduces them and is a no-op vs the force-field-ideal the emb
        # metric measures against. uff_relax KEEPS the fold (local minimisation from the
        # existing conformer) -- unlike a from-scratch ETKDG embed, which mis-folds
        # big/flexible/phosphate ligands -- and brings bonds/angles onto the shared
        # force-field ideal. Falls back to the cached coords if UFF fails.
        # Guard: only when the mol has REAL bond orders (an aromatic or double bond).
        # chai/esmfold2 expose no bond orders -> their mol is all-single, so UFF would
        # localize aromatic rings to single-bond lengths (~1.5), corrupting the target;
        # for those tools the cached conformer holds the real reference geometry.
        if any(
            b.GetIsAromatic() or b.GetBondType() == Chem.BondType.DOUBLE
            for b in mol.GetBonds()
        ):
            _relaxed = uff_relax(mol, crds)
            if _relaxed is not None and len(_relaxed) == len(crds):
                crds = _relaxed

        for b in mol.GetBonds():
            ai, aj = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bonds.append((int(gidx[ai]), int(gidx[aj]), _bond_length(crds, ai, aj)))

        for ai, aj, ak in mol.GetSubstructMatches(_ANGLE_PATT):
            angles.append(
                (
                    int(gidx[ai]),
                    int(gidx[aj]),
                    int(gidx[ak]),
                    _angle_rad(crds, ai, aj, ak),
                )
            )

        for atom in mol.GetAtoms():
            if atom.GetChiralTag() not in _CHIRAL_TAGS:
                continue
            ci = atom.GetIdx()
            nei = [b.GetOtherAtom(atom).GetIdx() for b in atom.GetBonds()]
            for cand in itertools.combinations(nei, 3):
                vol = _chiral_vol(crds, ci, cand[0], cand[1], cand[2])
                if lc.invert_chirality:
                    vol = -vol
                chirals.append(
                    (
                        int(gidx[ci]),
                        int(gidx[cand[0]]),
                        int(gidx[cand[1]]),
                        int(gidx[cand[2]]),
                        vol,
                    )
                )

        # cis/trans (E/Z): hold each acyclic, non-aromatic double bond at its
        # reference-conformer dihedral. Detection uses only bond ORDER (DOUBLE)
        # + connectivity, both consistent across tools; `not IsInRing()` excludes
        # aromatic/ring double bonds (incl. Kekule rings) which cannot isomerise.
        try:
            Chem.FastFindRings(mol)  # ensure IsInRing() has ring info
        except Exception:
            pass
        for b in mol.GetBonds():
            if b.GetBondType() != Chem.BondType.DOUBLE:
                continue
            if b.GetIsAromatic() or b.IsInRing():
                continue
            aj, ak = b.GetBeginAtom(), b.GetEndAtom()
            j, k = aj.GetIdx(), ak.GetIdx()
            nbr_j = [n.GetIdx() for n in aj.GetNeighbors() if n.GetIdx() != k]
            nbr_k = [n.GetIdx() for n in ak.GetNeighbors() if n.GetIdx() != j]
            if not nbr_j or not nbr_k:
                continue  # terminal double bond (e.g. C=O) — no dihedral
            # Enumerate every (i, l) substituent pair across the bond, like the
            # chiral combination enumeration: an order-independent restraint set.
            for i in nbr_j:
                for ll in nbr_k:
                    phi0 = _cistrans_rad(crds, i, j, k, ll)
                    cistrans.append(
                        (
                            int(gidx[i]),
                            int(gidx[j]),
                            int(gidx[k]),
                            int(gidx[ll]),
                            phi0,
                        )
                    )

        # improper (planarity): hold each sp2 double-bond centre in its substituents'
        # plane. Same bond filter as cistrans (acyclic, non-aromatic DOUBLE — the
        # topological `not IsInRing()` is parity-safe even if SanitizeMol failed to
        # perceive aromaticity, unlike GetIsAromatic() alone). A carbonyl C=O is an
        # EXOCYCLIC bond (`IsInRing()` is False), so carbonyl / amide / ester / carboxyl
        # centres are kept; only in-ring non-aromatic C=C are dropped. Each endpoint
        # with exactly 3 heavy neighbours (mol is H-removed) gets one improper on
        # (centre; its 3 neighbours); target vol0 = reference signed volume (~0).
        seen_improper: set[int] = set()
        for b in mol.GetBonds():
            if b.GetBondType() != Chem.BondType.DOUBLE:
                continue
            if b.GetIsAromatic() or b.IsInRing():
                continue
            for atom in (b.GetBeginAtom(), b.GetEndAtom()):
                ci = atom.GetIdx()
                if ci in seen_improper:
                    continue
                nei = [n.GetIdx() for n in atom.GetNeighbors()]
                if len(nei) != 3:
                    continue  # not a 3-substituted sp2 centre (no signed volume)
                seen_improper.add(ci)
                vol = _chiral_vol(crds, ci, nei[0], nei[1], nei[2])
                impropers.append(
                    (
                        int(gidx[ci]),
                        int(gidx[nei[0]]),
                        int(gidx[nei[1]]),
                        int(gidx[nei[2]]),
                        vol,
                    )
                )

    return bonds, angles, chirals, cistrans, impropers


def _pad_groups(restraints, n_groups, g2l):
    """Pad each restraint's ``n_groups`` atom groups into (n, max_grp) local-index and
    {0,1} mask arrays. Reads ``restraints[k].target_sites{1..n_groups}`` (global
    indices). ``max_grp`` spans every group of every restraint so one padded width
    covers them all. Returns (idx_arrays, mask_arrays): each a list of ``n_groups``
    arrays in group order. Used for group-centroid angle (3) and dihedral (4)."""
    n = len(restraints)
    max_grp = max(
        len(getattr(r, f"target_sites{g + 1}"))
        for r in restraints
        for g in range(n_groups)
    )
    idx_arrays = [np.zeros((n, max_grp), dtype=np.int64) for _ in range(n_groups)]
    mask_arrays = [np.zeros((n, max_grp)) for _ in range(n_groups)]
    for ri, r in enumerate(restraints):
        for g in range(n_groups):
            local = [g2l[int(s)] for s in getattr(r, f"target_sites{g + 1}")]
            idx_arrays[g][ri, : len(local)] = local
            mask_arrays[g][ri, : len(local)] = 1.0
    return idx_arrays, mask_arrays


def _group_geom_params(restraints):
    """(geom_type codes, target1, target2, move_free) arrays for group angle/dihedral
    restraints. The string type maps to the SAME int code as the distance restraint
    (``DIST_TYPE_CODES``: harmonic=0 / flat-bottomed=1 / flat-bottomed1=2 (lower) /
    flat-bottomed2=3 (upper)); target1/target2 are radians (0.0 where unused). move_free
    is an (n, n_groups) {0,1} mask (1 = that group is free to move)."""
    codes = np.array([DIST_TYPE_CODES[r.geom_type] for r in restraints], dtype=np.int64)
    t1 = np.array([float(r.target1) for r in restraints])
    t2 = np.array([float(r.target2) for r in restraints])
    move_free = np.array([[1.0 if f else 0.0 for f in r.move_free] for r in restraints])
    return codes, t1, t2, move_free


def _group_sigmas(restraints, conf_start_sigma):
    """Per-restraint (start_sigma, stop_sigma) arrays for group angle/dihedral. Mirrors
    the distance/rmsd convention: an omitted start_sigma (None) falls back to
    ``conf_start_sigma`` (from_dict normally pre-fills +inf), stop_sigma defaults -1."""
    start = np.array(
        [
            float(r.start_sigma)
            if getattr(r, "start_sigma", None) is not None
            else conf_start_sigma
            for r in restraints
        ]
    )
    stop = np.array([float(getattr(r, "stop_sigma", -1.0)) for r in restraints])
    return start, stop


def _group_steps(restraints):
    """Per-restraint (start_step, stop_step) arrays for group angle/dihedral. The step
    window is the alternative gate axis to the sigma window (mutually exclusive at config
    time) and is independent of the conformer window — omitted -> the data object's
    -inf/+inf default (always active). Mirrors the distance/rmsd step plumbing."""
    start = np.array(
        [float(getattr(r, "start_step", float("-inf"))) for r in restraints]
    )
    stop = np.array([float(getattr(r, "stop_step", float("inf"))) for r in restraints])
    return start, stop


def _dist_params(dr) -> tuple[int, float, float]:
    """Map a DistanceData (string type + targets) to (code, target1, target2)."""
    t = dr.distance_restraint_type
    code = DIST_TYPE_CODES[t]
    if t == "harmonic":
        return code, float(dr.target_distance), 0.0
    if t == "flat-bottomed":
        return code, float(dr.target_distance1), float(dr.target_distance2)
    if t == "flat-bottomed1":
        return code, float(dr.target_distance1), 0.0
    if t == "flat-bottomed2":
        return code, 0.0, float(dr.target_distance2)
    raise ValueError(f"unknown distance type {t!r}")


def _vdw_radius(z: int) -> float:
    """VdW radius (A) for atomic number ``z`` from RDKit's periodic table."""
    if z is None or z < 1 or z > 118:
        return 0.0
    return float(Chem.GetPeriodicTable().GetRvdw(int(z)))


def _conf_weight(conformer_config: dict | None, key: str) -> float:
    """Weight of a conformer sub-term (bond/angle/chiral/cistrans/vdw/improper).

    Uniform "default 1.0, off if not configured" rule, shared with every other
    restraint type (distance/rmsd/angle/dihedral/custom all default weight 1.0): a
    sub-block PRESENT in ``conformer_restraints_config`` defaults its weight to 1.0
    (override with an explicit ``weight``); an ABSENT sub-block is OFF (weight 0). An
    explicit ``weight: 0`` / null also disables the term. (A bare ``key:`` with no
    body — None in YAML — counts as present, so it activates the term at 1.0.)
    """
    cfg = conformer_config or {}
    if key not in cfg:
        return 0.0
    w = (cfg.get(key) or {}).get("weight", 1.0)
    return float(w) if w is not None else 0.0


def _conf_slack(conformer_config: dict | None, key: str, default: float) -> float:
    """Slack (flat-bottom half-width) of a conformer sub-term, with uniform null handling.

    Shared by all five conformer terms (bond/angle/chiral/cistrans/improper) so the
    omitted / explicit-0 / null cases can't drift between them: an OMITTED ``slack`` ->
    the per-term ``default``; an explicit ``slack: 0`` -> 0.0 (a hard, zero-width
    restraint, NOT the default); a ``slack: null`` -> the ``default`` (null is treated as
    omitted, matching ``apply_window_params``). The truthiness trap (``slack or default``
    would silently turn an explicit 0 into the default) is what this avoids.
    """
    v = (conformer_config or {}).get(key)
    v = (v or {}).get("slack", default)
    return float(default if v is None else v)


def _build_vdw_config(
    ligand_confs: list[LigandConf],
    conformer_config: dict,
    active_sites: np.ndarray,
    g2l: dict,
    elements: np.ndarray | None,
) -> VdwConfig | None:
    """Build the dynamic ligand-protein VdW config (torch + jax optimizers).

    Ligand atoms (the moving set) come from the RDKit mols; the protein
    background is every heavy atom NOT in ``active_sites`` (i.e. not optimised),
    so the VdW term pushes the ligand out of the fixed pocket. Returns ``None``
    when VdW is disabled (weight<=0), there are no ligands, or element info is
    unavailable.
    """
    vcfg = (conformer_config or {}).get("vdw", {}) or {}
    weight = _conf_weight(conformer_config, "vdw")
    if weight <= 0.0 or not ligand_confs or elements is None:
        return None

    # ligand atoms + per-atom radii from the mols (global index -> radius)
    lig_radius: dict[int, float] = {}
    for lc in ligand_confs:
        gidx = np.asarray(lc.global_indices, dtype=np.int64)
        for i, atom in enumerate(lc.mol.GetAtoms()):
            lig_radius[int(gidx[i])] = _vdw_radius(atom.GetAtomicNum())
    ligand_global = np.array(sorted(lig_radius), dtype=np.int64)
    # every ligand atom is in active_sites (added in build_spec when VdW is on)
    ligand_local = np.array([g2l[int(g)] for g in ligand_global], dtype=np.int64)
    ligand_radii = np.array(
        [lig_radius[int(g)] for g in ligand_global], dtype=np.float64
    )

    # protein background = heavy atoms that are NOT optimised (not in active_sites)
    elements = np.asarray(elements)
    active_set = {int(a) for a in active_sites}
    protein_global = np.array(
        [
            a
            for a in range(len(elements))
            if int(elements[a]) > 0 and a not in active_set
        ],
        dtype=np.int64,
    )
    if len(protein_global) == 0:
        return None
    protein_radii = np.array(
        [_vdw_radius(int(elements[a])) for a in protein_global], dtype=np.float64
    )

    return VdwConfig(
        weight=weight,
        ligand_local=ligand_local,
        ligand_radii=ligand_radii,
        protein_global=protein_global,
        protein_radii=protein_radii,
        scale=float(vcfg.get("scale", 0.75)),
        dmax=float(vcfg.get("dmax", 5.0)),
    )


def _build_intramolecular_vdw(
    ligand_confs: list[LigandConf],
    conformer_config: dict,
    g2l: dict,
) -> VdwArrays | None:
    """Static intramolecular VdW repulsion within each ligand (all backends).

    Penalizes non-bonded atom pairs within one ligand — topological distance > 2
    (so 1-2 bonds and 1-3 angles are skipped) and reference-conformer distance
    < ``dmax`` — with a lower bound ``scale * (r_i + r_j)``. Unlike the dynamic
    ligand-protein ``VdwConfig``, the pair list is fixed, so this term also works in
    the jax/numpy backends via ``VdwArrays``. Enabled when
    ``conformer_config['vdw']['mode']`` is ``'intramolecular'`` or ``'both'`` (the
    DEFAULT); ``'ligand_protein'`` leaves it off.
    """
    vcfg = (conformer_config or {}).get("vdw", {}) or {}
    weight = _conf_weight(conformer_config, "vdw")
    if weight <= 0.0 or not ligand_confs:
        return None
    from rdkit.Chem import rdmolops

    scale = float(vcfg.get("scale", 0.75))
    dmax = float(vcfg.get("dmax", 5.0))
    idx_pairs: list[list[int]] = []
    r_min_list: list[float] = []
    for lc in ligand_confs:
        mol = lc.mol
        crds = np.asarray(lc.conf_coords, dtype=np.float64)
        gidx = np.asarray(lc.global_indices, dtype=np.int64)
        n = mol.GetNumAtoms()
        if n < 2:
            continue
        topo = rdmolops.GetDistanceMatrix(mol)
        radii = [_vdw_radius(a.GetAtomicNum()) for a in mol.GetAtoms()]
        for i in range(n):
            for j in range(i + 1, n):
                if topo[i, j] <= 2:  # skip 1-2 (bond) and 1-3 (angle) pairs
                    continue
                if _bond_length(crds, i, j) >= dmax:
                    continue
                idx_pairs.append([g2l[int(gidx[i])], g2l[int(gidx[j])]])
                r_min_list.append(scale * (radii[i] + radii[j]))
    if not idx_pairs:
        return None
    n_pair = len(idx_pairs)
    return VdwArrays(
        idx=np.array(idx_pairs, dtype=np.int64),
        r_min=np.array(r_min_list, dtype=np.float64),
        weight=np.full(n_pair, weight),
        mask=np.ones(n_pair),
    )


def _build_interligand_vdw(
    ligand_confs: list[LigandConf],
    conformer_config: dict,
    g2l: dict,
) -> VdwArrays | None:
    """Static inter-ligand VdW repulsion BETWEEN distinct ligands (all backends).

    The third VdW flavour, alongside intramolecular (within one ligand) and the
    dynamic ligand-protein term. Both endpoints of every pair live in
    ``active_sites`` (each ligand moves under its own conformer restraint), so
    autodiff drives BOTH ligands apart — exactly like ``vdw_energy``'s
    intramolecular pairs, only the pair list crosses molecules. Unlike
    ``_build_intramolecular_vdw`` there is NO topological skip (different mols have
    no shared bonds) and NO reference-distance ``dmax`` cutoff (two ligands'
    ``conf_coords`` live in independent frames, so a build-time distance is
    meaningless); every cross pair is listed and the ``clamp(d - r_min, max=0)``
    penalty contributes zero beyond contact. Ligands are H-removed and small, so
    the all-pairs list stays cheap. Built only when VdW is on, ``mode`` is the
    default ``'both'``, and at least two ligands opted in.
    """
    vcfg = (conformer_config or {}).get("vdw", {}) or {}
    weight = _conf_weight(conformer_config, "vdw")
    if weight <= 0.0 or len(ligand_confs) < 2:
        return None

    scale = float(vcfg.get("scale", 0.75))
    radii = [
        [_vdw_radius(a.GetAtomicNum()) for a in lc.mol.GetAtoms()]
        for lc in ligand_confs
    ]
    idx_pairs: list[list[int]] = []
    r_min_list: list[float] = []
    for a in range(len(ligand_confs)):
        gA = np.asarray(ligand_confs[a].global_indices, dtype=np.int64)
        for b in range(a + 1, len(ligand_confs)):
            gB = np.asarray(ligand_confs[b].global_indices, dtype=np.int64)
            for i in range(len(gA)):
                li = g2l[int(gA[i])]
                ri = radii[a][i]
                for j in range(len(gB)):
                    idx_pairs.append([li, g2l[int(gB[j])]])
                    r_min_list.append(scale * (ri + radii[b][j]))
    if not idx_pairs:
        return None
    n_pair = len(idx_pairs)
    return VdwArrays(
        idx=np.array(idx_pairs, dtype=np.int64),
        r_min=np.array(r_min_list, dtype=np.float64),
        weight=np.full(n_pair, weight),
        mask=np.ones(n_pair),
    )


def _concat_vdw_arrays(*parts: VdwArrays | None) -> VdwArrays | None:
    """Stack several ``VdwArrays`` into one (drops ``None``s). They are scored by the
    same ``vdw_energy`` term under the same conformer gate, so concatenating their
    ``idx``/``r_min``/``weight``/``mask`` rows composes the flavours with no extra
    wiring. Returns ``None`` when every part is empty."""
    present = [p for p in parts if p is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return VdwArrays(
        idx=np.concatenate([p.idx for p in present], axis=0),
        r_min=np.concatenate([p.r_min for p in present]),
        weight=np.concatenate([p.weight for p in present]),
        mask=np.concatenate([p.mask for p in present]),
    )


def build_spec(
    ligand_confs: list[LigandConf] | None = None,
    distance_restraints: list | None = None,
    conformer_config: dict | None = None,
    elements: np.ndarray | None = None,
    # Omitted start_sigma -> +inf = active at EVERY diffusion step (the documented
    # contract; the gate is `sigma <= start_sigma` and sigma >= 0, so the old -1.0
    # default silently disabled the conformer AND any distance/rmsd/group entry whose
    # start_sigma falls back to this value). config.py mirrors this default; the only
    # production caller (combined.setup) passes it explicitly, so this just makes direct
    # build_spec() callers agree with the config path.
    conf_start_sigma: float = float("inf"),
    conf_stop_sigma: float = -1.0,
    conf_start_step: float = float("-inf"),
    conf_stop_step: float = float("inf"),
    rmsd_restraints: list | None = None,
    angle_restraints: list | None = None,
    dihedral_restraints: list | None = None,
    custom_restraints: list | None = None,
) -> RestraintSpec:
    """Build a RestraintSpec. ``distance_restraints`` are DistanceData with
    ``target_sites1``/``target_sites2`` already resolved to global indices;
    ``rmsd_restraints`` are RmsdData with ``target_sites``/``ref_coords`` resolved;
    ``angle_restraints``/``dihedral_restraints`` are AngleRestraintData /
    DihedralRestraintData with their group ``target_sites{1..N}`` resolved to global
    indices (N=3 for angle, N=4 for dihedral)."""
    ligand_confs = ligand_confs or []
    cfg = conformer_config or {}
    # Conformer restraints are OPT-IN -- this is the single enforcement point for every
    # tool: (1) with no conformer_restraints_config (e.g. a distance-only run) build no
    # conformer at all; (2) otherwise restrain only ligands flagged
    # conformer_restraints=True. EVERY tool defaults the per-ligand flag to False, so a
    # ligand must explicitly opt in (boltz/protenix/af3/openfold/esm via a per-ligand
    # input field; chai via the sidecar conformer_restraints map).
    # A config holding only documentation keys ("_comment") counts as absent.
    cfg_present = any(not str(k).startswith("_") for k in cfg)
    if not cfg_present:
        ligand_confs = []
    _n_before = len(ligand_confs)
    ligand_confs = [
        lc for lc in ligand_confs if getattr(lc, "conformer_restraints", False)
    ]
    # Loud signal for the silent no-op: conformer_restraints_config is present and ligands
    # exist, but none opted in -> zero conformer restraints built (NOT "satisfied"). A
    # finalize term reading 0.00000 because the spec has 0 of that restraint is a no-op.
    if cfg_present and _n_before and not ligand_confs:
        # print() (not just logger.warning, which the package NullHandler mutes) so this
        # misconfiguration alert survives host logging configs -- the same reasoning as
        # the "NO ACTIVE RESTRAINTS" print in combined.setup. Fires only on the genuine
        # footgun (conformer config + ligands present, but none opted in), so it is rare.
        msg = (
            "conformer_restraints_config present but no ligand opted in "
            "(set conformer_restraints: true on the ligand) -- no conformer restraints built"
        )
        logger.warning(msg)
        print(f"[rgi_utils] WARNING: {msg}", flush=True)
    distance_restraints = [
        dr for dr in (distance_restraints or []) if getattr(dr, "run_restr", False)
    ]
    rmsd_restraints = [
        rr for rr in (rmsd_restraints or []) if getattr(rr, "run_restr", False)
    ]
    angle_restraints = [
        ar for ar in (angle_restraints or []) if getattr(ar, "run_restr", False)
    ]
    dihedral_restraints = [
        dr for dr in (dihedral_restraints or []) if getattr(dr, "run_restr", False)
    ]
    custom_restraints = [
        c for c in (custom_restraints or []) if getattr(c, "run_restr", False)
    ]
    # Every conformer sub-term follows the uniform "default 1.0, off if not configured"
    # rule (see _conf_weight): a sub-block PRESENT in the conformer config is active at
    # weight 1.0 (override with an explicit weight); an ABSENT sub-block is OFF. So a
    # ligand that opts in but lists e.g. only `bond:` gets ONLY bond — angle/chiral/
    # cistrans/vdw/improper stay off until their own sub-block is added. slack defaults
    # stay per-term (chiral/improper flat-bottom ~0.05 signed-volume; bond/angle/cistrans
    # 0.0 = pure harmonic toward the reference; cistrans slack is in radians).
    bw = _conf_weight(cfg, "bond")
    bsl = _conf_slack(cfg, "bond", 0.0)
    aw = _conf_weight(cfg, "angle")
    asl = _conf_slack(cfg, "angle", 0.0)
    cw = _conf_weight(cfg, "chiral")
    csl = _conf_slack(cfg, "chiral", 0.05)
    dw = _conf_weight(cfg, "cistrans")
    dsl = _conf_slack(cfg, "cistrans", 0.0)
    vdw_weight = _conf_weight(cfg, "vdw")
    iw = _conf_weight(cfg, "improper")
    isl = _conf_slack(cfg, "improper", 0.05)

    bonds, angles, chirals, cistrans, impropers = _extract_conformer(ligand_confs)
    # weight<=0 means "disable": drop the term BEFORE the active_sites union so its
    # atoms do not become optimisable and it is never iterated — uniform across all
    # conformer terms. weight<=0 now also covers an ABSENT sub-block (_conf_weight -> 0),
    # so an unlisted term is simply dropped here.
    if bw <= 0:
        bonds = []
    if aw <= 0:
        angles = []
    if cw <= 0:
        chirals = []
    if dw <= 0:
        cistrans = []
    if iw <= 0:
        impropers = []  # OFF by default (iw defaults to 0): opt-in planarity term

    # ---- collect every referenced global atom -> active_sites -----------------
    active: set[int] = set()
    for g0, g1, _ in bonds:
        active.update((g0, g1))
    for g0, g1, g2, _ in angles:
        active.update((g0, g1, g2))
    for g0, g1, g2, g3, _ in chirals:
        active.update((g0, g1, g2, g3))
    for g0, g1, g2, g3, _ in cistrans:
        active.update((g0, g1, g2, g3))
    for g0, g1, g2, g3, _ in impropers:
        active.update((g0, g1, g2, g3))
    for dr in distance_restraints:
        active.update(int(s) for s in dr.target_sites1)
        active.update(int(s) for s in dr.target_sites2)
    for rr in rmsd_restraints:
        active.update(int(s) for s in rr.fit_target_sites)
        active.update(int(s) for s in rr.calc_target_sites)
    for gar in angle_restraints:
        for g in range(3):
            active.update(int(s) for s in getattr(gar, f"target_sites{g + 1}"))
    for gdr in dihedral_restraints:
        for g in range(4):
            active.update(int(s) for s in getattr(gdr, f"target_sites{g + 1}"))
    # custom restraints contribute every atom their selections resolved to.
    for cr in custom_restraints:
        active.update(int(s) for s in cr.iter_global_sites())
    # VdW pushes the whole ligand, so every ligand atom must be optimisable even
    # if it carries no bond/angle/chiral term (e.g. a monatomic ion).
    if vdw_weight > 0:
        for lc in ligand_confs:
            active.update(int(g) for g in lc.global_indices)

    active_sites = np.array(sorted(active), dtype=np.int64)
    g2l = {int(g): i for i, g in enumerate(active_sites)}
    # custom restraints: remap each resolved selection to LOCAL indices (CustomSpec).
    custom_specs = [cr.build_spec(g2l) for cr in custom_restraints]
    # Three VdW flavours share the conformer_config['vdw'] block: static intramolecular
    # (within one ligand) and inter-ligand (between distinct restrained ligands), both
    # VdwArrays in the spec -> energy layer / all backends; and the dynamic ligand-protein
    # term (VdwConfig -> torch/jax optimizer). `mode` picks the intra/background split, or
    # the DEFAULT "both" enables intramolecular + ligand-protein together. The inter-ligand
    # flavour rides the DEFAULT "both" too (it is neither purely intra nor background): two
    # restrained ligands each sit in active_sites, so neither is in the other's fixed
    # background and ligand-protein alone never repels A<->B. The intra + inter pairs are
    # CONCATENATED into one VdwArrays (same energy term, same conformer gate); ligand-protein
    # stays a separate spec field. The explicit "intramolecular"/"ligand_protein" modes keep
    # their precise meaning (no inter). All halves run on torch AND jax; on numpy (energy
    # reference only) the ligand-protein half is inert.
    vdw_mode = (cfg.get("vdw", {}) or {}).get("mode", "both")
    if vdw_mode not in ("ligand_protein", "intramolecular", "both"):
        raise ValueError(
            "conformer vdw mode must be 'ligand_protein', 'intramolecular', or "
            f"'both', got {vdw_mode!r}"
        )
    vdw_intra = (
        _build_intramolecular_vdw(ligand_confs, cfg, g2l)
        if vdw_mode in ("intramolecular", "both")
        else None
    )
    vdw_inter = (
        _build_interligand_vdw(ligand_confs, cfg, g2l) if vdw_mode == "both" else None
    )
    vdw_arrays = _concat_vdw_arrays(vdw_intra, vdw_inter)
    vdw_config = (
        _build_vdw_config(ligand_confs, cfg, active_sites, g2l, elements)
        if vdw_mode in ("ligand_protein", "both")
        else None
    )

    # ---- conformer arrays (local indices) -------------------------------------
    bond = None
    if bonds:
        idx = np.array([[g2l[g0], g2l[g1]] for g0, g1, _ in bonds], dtype=np.int64)
        bond = BondArrays(
            idx=idx,
            r0=np.array([r for _, _, r in bonds]),
            slack=np.full(len(bonds), bsl),
            weight=np.full(len(bonds), bw),
            half=np.zeros(len(bonds)),
            mask=np.ones(len(bonds)),
        )
    angle = None
    if angles:
        idx = np.array(
            [[g2l[g0], g2l[g1], g2l[g2]] for g0, g1, g2, _ in angles], dtype=np.int64
        )
        angle = AngleArrays(
            idx=idx,
            th0=np.array([t for _, _, _, t in angles]),
            slack=np.full(len(angles), asl),
            weight=np.full(len(angles), aw),
            mask=np.ones(len(angles)),
        )
    chiral = None
    if chirals:
        idx = np.array(
            [[g2l[g0], g2l[g1], g2l[g2], g2l[g3]] for g0, g1, g2, g3, _ in chirals],
            dtype=np.int64,
        )
        chiral = ChiralArrays(
            idx=idx,
            vol0=np.array([v for _, _, _, _, v in chirals]),
            slack=np.full(len(chirals), csl),
            weight=np.full(len(chirals), cw),
            mask=np.ones(len(chirals)),
        )
    cistrans_arr = None
    if cistrans:
        idx = np.array(
            [[g2l[g0], g2l[g1], g2l[g2], g2l[g3]] for g0, g1, g2, g3, _ in cistrans],
            dtype=np.int64,
        )
        cistrans_arr = CisTransArrays(
            idx=idx,
            phi0=np.array([p for _, _, _, _, p in cistrans]),
            slack=np.full(len(cistrans), dsl),
            weight=np.full(len(cistrans), dw),
            mask=np.ones(len(cistrans)),
        )
    improper = None
    if impropers:
        idx = np.array(
            [[g2l[g0], g2l[g1], g2l[g2], g2l[g3]] for g0, g1, g2, g3, _ in impropers],
            dtype=np.int64,
        )
        improper = ImproperArrays(
            idx=idx,
            vol0=np.array([v for _, _, _, _, v in impropers]),
            slack=np.full(len(impropers), isl),
            weight=np.full(len(impropers), iw),
            mask=np.ones(len(impropers)),
        )

    # ---- distance arrays (padded, local indices) ------------------------------
    distance = None
    if distance_restraints:
        n = len(distance_restraints)
        max_grp = max(
            max(len(dr.target_sites1), len(dr.target_sites2))
            for dr in distance_restraints
        )
        grp1_idx = np.zeros((n, max_grp), dtype=np.int64)
        grp2_idx = np.zeros((n, max_grp), dtype=np.int64)
        grp1_mask = np.zeros((n, max_grp))
        grp2_mask = np.zeros((n, max_grp))
        target1 = np.zeros(n)
        target2 = np.zeros(n)
        dist_type = np.zeros(n, dtype=np.int64)
        move_mode = np.zeros(n, dtype=np.int64)  # 0=both / 1=grp1 only / 2=grp2 only
        dist_weight = np.ones(n)  # relative strength (no-op unless over-constrained)
        dist_start_sigma = np.full(n, -1.0)
        dist_stop_sigma = np.full(n, -1.0)  # -1 = never released (off)
        dist_start_step = np.full(n, float("-inf"))  # step-window; -inf/+inf = always
        dist_stop_step = np.full(n, float("inf"))
        for di, dr in enumerate(distance_restraints):
            s1 = [g2l[int(s)] for s in dr.target_sites1]
            s2 = [g2l[int(s)] for s in dr.target_sites2]
            grp1_idx[di, : len(s1)] = s1
            grp2_idx[di, : len(s2)] = s2
            grp1_mask[di, : len(s1)] = 1.0
            grp2_mask[di, : len(s2)] = 1.0
            code, t1, t2 = _dist_params(dr)
            dist_type[di] = code
            move_mode[di] = int(getattr(dr, "move_mode", 0))
            dist_weight[di] = float(getattr(dr, "weight", 1.0))
            target1[di] = t1
            target2[di] = t2
            ss = getattr(dr, "start_sigma", None)
            dist_start_sigma[di] = float(ss) if ss is not None else conf_start_sigma
            dist_stop_sigma[di] = float(getattr(dr, "stop_sigma", -1.0))
            dist_start_step[di] = float(getattr(dr, "start_step", float("-inf")))
            dist_stop_step[di] = float(getattr(dr, "stop_step", float("inf")))
        distance = DistanceArrays(
            grp1_idx=grp1_idx,
            grp2_idx=grp2_idx,
            grp1_mask=grp1_mask,
            grp2_mask=grp2_mask,
            target1=target1,
            target2=target2,
            dist_type=dist_type,
            move_mode=move_mode,
            weight=dist_weight,
            mask=np.ones(n),
            start_sigma=dist_start_sigma,
            stop_sigma=dist_stop_sigma,
            start_step=dist_start_step,
            stop_step=dist_stop_step,
        )

    # ---- RMSD arrays (padded; fit = superposition atoms, calc = measured atoms) --
    rmsd = None
    if rmsd_restraints:
        n = len(rmsd_restraints)
        max_fit = max(len(rr.fit_target_sites) for rr in rmsd_restraints)
        max_calc = max(len(rr.calc_target_sites) for rr in rmsd_restraints)
        fit_idx = np.zeros((n, max_fit), dtype=np.int64)
        fit_mask = np.zeros((n, max_fit))
        fit_ref = np.zeros((n, max_fit, 3))
        calc_idx = np.zeros((n, max_calc), dtype=np.int64)
        calc_mask = np.zeros((n, max_calc))
        calc_ref = np.zeros((n, max_calc, 3))
        target1 = np.zeros(n)
        target2 = np.zeros(n)
        geom_type = np.zeros(n, dtype=np.int64)
        rmsd_weight = np.zeros(n)
        rmsd_start_sigma = np.full(n, -1.0)
        rmsd_stop_sigma = np.full(n, -1.0)  # -1 = never released (active to sigma=0)
        rmsd_start_step = np.full(n, float("-inf"))  # step-window; -inf/+inf = always
        rmsd_stop_step = np.full(n, float("inf"))
        for ri, rr in enumerate(rmsd_restraints):
            f_local = [g2l[int(s)] for s in rr.fit_target_sites]
            kf = len(f_local)
            fit_idx[ri, :kf] = f_local
            fit_mask[ri, :kf] = 1.0
            fit_ref[ri, :kf] = np.asarray(rr.fit_ref_coords, dtype=np.float64)
            c_local = [g2l[int(s)] for s in rr.calc_target_sites]
            kc = len(c_local)
            calc_idx[ri, :kc] = c_local
            calc_mask[ri, :kc] = 1.0
            calc_ref[ri, :kc] = np.asarray(rr.calc_ref_coords, dtype=np.float64)
            target1[ri] = float(rr.target1)
            target2[ri] = float(rr.target2)
            geom_type[ri] = DIST_TYPE_CODES[rr.rmsd_type]
            # rr.weight is already normalized in set_config (None -> 1.0); pass it
            # through verbatim so an explicit weight: 0 yields a zero-energy term
            # (do NOT coerce a falsy 0 to 1.0 like the conformer terms, which instead
            # drop weight<=0 entries before the active_sites union).
            rmsd_weight[ri] = float(rr.weight)
            ss = getattr(rr, "start_sigma", None)
            rmsd_start_sigma[ri] = float(ss) if ss is not None else conf_start_sigma
            rmsd_stop_sigma[ri] = float(getattr(rr, "stop_sigma", -1.0))
            rmsd_start_step[ri] = float(getattr(rr, "start_step", float("-inf")))
            rmsd_stop_step[ri] = float(getattr(rr, "stop_step", float("inf")))
        rmsd = RmsdArrays(
            fit_idx=fit_idx,
            fit_mask=fit_mask,
            fit_ref=fit_ref,
            calc_idx=calc_idx,
            calc_mask=calc_mask,
            calc_ref=calc_ref,
            target1=target1,
            target2=target2,
            geom_type=geom_type,
            weight=rmsd_weight,
            start_sigma=rmsd_start_sigma,
            stop_sigma=rmsd_stop_sigma,
            start_step=rmsd_start_step,
            stop_step=rmsd_stop_step,
            mask=np.ones(n),
        )

    # ---- group-centroid angle / dihedral arrays (padded, local indices) ------------
    group_angle = None
    if angle_restraints:
        n = len(angle_restraints)
        (g1, g2, g3), (m1, m2, m3) = _pad_groups(angle_restraints, 3, g2l)
        start_sigma, stop_sigma = _group_sigmas(angle_restraints, conf_start_sigma)
        start_step, stop_step = _group_steps(angle_restraints)
        codes, t1, t2, move_free = _group_geom_params(angle_restraints)
        group_angle = GroupAngleArrays(
            grp1_idx=g1,
            grp2_idx=g2,
            grp3_idx=g3,
            grp1_mask=m1,
            grp2_mask=m2,
            grp3_mask=m3,
            target1=t1,
            target2=t2,
            geom_type=codes,
            move_free=move_free,
            weight=np.array([float(r.weight) for r in angle_restraints]),
            mask=np.ones(n),
            start_sigma=start_sigma,
            stop_sigma=stop_sigma,
            start_step=start_step,
            stop_step=stop_step,
        )
    group_dihedral = None
    if dihedral_restraints:
        n = len(dihedral_restraints)
        (g1, g2, g3, g4), (m1, m2, m3, m4) = _pad_groups(dihedral_restraints, 4, g2l)
        start_sigma, stop_sigma = _group_sigmas(dihedral_restraints, conf_start_sigma)
        start_step, stop_step = _group_steps(dihedral_restraints)
        codes, t1, t2, move_free = _group_geom_params(dihedral_restraints)
        group_dihedral = GroupDihedralArrays(
            grp1_idx=g1,
            grp2_idx=g2,
            grp3_idx=g3,
            grp4_idx=g4,
            grp1_mask=m1,
            grp2_mask=m2,
            grp3_mask=m3,
            grp4_mask=m4,
            target1=t1,
            target2=t2,
            geom_type=codes,
            move_free=move_free,
            weight=np.array([float(r.weight) for r in dihedral_restraints]),
            mask=np.ones(n),
            start_sigma=start_sigma,
            stop_sigma=stop_sigma,
            start_step=start_step,
            stop_step=stop_step,
        )

    spec = RestraintSpec(
        n_active=len(active_sites),
        active_sites=active_sites,
        bond=bond,
        angle=angle,
        chiral=chiral,
        improper=improper,
        cistrans=cistrans_arr,
        distance=distance,
        rmsd=rmsd,
        group_angle=group_angle,
        group_dihedral=group_dihedral,
        vdw=vdw_arrays,
        vdw_config=vdw_config,
        conf_start_sigma=conf_start_sigma,
        conf_stop_sigma=conf_stop_sigma,
        conf_start_step=conf_start_step,
        conf_stop_step=conf_stop_step,
        custom=custom_specs,
    )
    vdw_parts = []
    if vdw_intra is not None:
        vdw_parts.append(f"{len(vdw_intra.idx)}intra")
    if vdw_inter is not None:
        vdw_parts.append(f"{len(vdw_inter.idx)}inter")
    if vdw_config is not None:
        vdw_parts.append(
            f"{len(vdw_config.ligand_local)}lig/{len(vdw_config.protein_global)}prot"
        )
    vdw_desc = "+".join(vdw_parts) if vdw_parts else "off"
    logger.info(
        "built spec: n_active=%d bonds=%d angles=%d chirals=%d impropers=%d cistrans=%d "
        "distances=%d rmsd=%d group_angle=%d group_dihedral=%d vdw=%s custom=%d",
        spec.n_active,
        len(bonds),
        len(angles),
        len(chirals),
        len(impropers),
        len(cistrans),
        len(distance_restraints),
        len(rmsd_restraints),
        len(angle_restraints),
        len(dihedral_restraints),
        vdw_desc,
        len(custom_specs),
    )
    return spec
