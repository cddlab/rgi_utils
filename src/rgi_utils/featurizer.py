"""Build a backend-agnostic ``RestraintSpec`` from ligand conformers + distances.

This is the single place where conformer restraints (bond/angle/chiral/cistrans/
plane) are derived from RDKit mols. Each ligand supplies its own
``global_indices``, so multiple ligands produce non-colliding restraints — there is
no per-batch state and no hard-coded ligand index (the multi-ligand bug in the old
code).

Flow:
  1. extract bond/angle/chiral/cistrans/plane restraints per ligand in GLOBAL atom
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

from rgi_utils._config_util import (
    VDW_MAX_ATOM_STEP_DEFAULT,
    VDW_NEIGHBOR_REBUILD_INTERVAL_DEFAULT,
    VDW_NEIGHBOR_SKIN_DEFAULT,
    VDW_SCALE_DEFAULT,
    validate_vdw_config,
)
from rgi_utils._mol_build import ff_relax, parse_relax_force_field
from rgi_utils.atom_context import LigandConf
from rgi_utils.spec import (
    DIST_TYPE_CODES,
    ActiveVdwConfig,
    AngleArrays,
    BondArrays,
    ChiralArrays,
    CisTransArrays,
    DistanceArrays,
    GroupAngleArrays,
    GroupDihedralArrays,
    GroupImproperArrays,
    GroupPlaneArrays,
    PlaneArrays,
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
# A candidate plane group (ring / sp2 functional group) is kept only if its reference
# conformer is coplanar to within this max out-of-plane deviation (Angstrom). Aromatic /
# conjugated rings are flat (<0.05 A); a puckered saturated ring (cyclohexane chair)
# deviates ~0.25 A, so 0.1 A cleanly selects planar groups WITHOUT trusting the
# (SanitizeMol-dependent, unreliable) RDKit aromaticity flag.
_PLANE_TOL = 0.1


def _max_plane_dev(crds, idxs):
    """Max out-of-plane distance (Angstrom) of atoms ``idxs`` from their own best-fit
    plane, in the reference conformer ``crds``. The plane normal is the smallest-
    eigenvalue eigenvector of the centred covariance (build-time numpy, no autodiff)."""
    pts = crds[list(idxs)]
    x0 = pts - pts.mean(axis=0)
    _w, vecs = np.linalg.eigh(
        x0.T @ x0
    )  # ascending eigenvalues; columns = eigenvectors
    return float(np.max(np.abs(x0 @ vecs[:, 0])))


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


def _extract_conformer(
    ligand_confs: list[LigandConf],
    *,
    relax: bool = True,
    force_field: str = "uff",
):
    """Return bond/angle/chiral/cistrans restraint tuples and plane groups in GLOBAL atom
    indices.

    ``relax`` is the STRUCTURAL switch (the polymer call site passes False: monomer-library
    residues are never force-field relaxed); ``force_field`` is the user's
    ``conformer_restraints_config.relax_force_field.ligand`` choice, applied to LIGANDS
    only.
    """
    bonds = []  # (g0, g1, r0)
    angles = []  # (g0, g1, g2, th0)
    chirals = []  # (g0, g1, g2, g3, vol0)
    cistrans = []  # (g0, g1, g2, g3, phi0)
    planes = []  # tuple(global idx, ...) — a planar atom group (ring or sp2 group)
    ff = str(force_field).lower()
    do_relax = relax and ff != "none"

    for li, lc in enumerate(ligand_confs):
        mol = lc.mol
        crds = np.asarray(lc.conf_coords, dtype=np.float64)
        gidx = np.asarray(lc.global_indices, dtype=np.int64)
        # Derive the bond/angle/chiral/cistrans TARGETS from a force-field-relaxed copy of
        # the tool's conformer. Each tool's cached conformer carries its own idiosyncrasies
        # (the boltz v2 ~/.boltz/mols cache Kekule-localizes aromatic rings ~1.34/1.48;
        # other tools' ref_pos has non-ideal bond/angle lengths), so without this the
        # restraint just reproduces them and is a no-op vs the force-field-ideal the emb
        # metric measures against. ff_relax KEEPS the fold (local minimisation from the
        # existing conformer) -- unlike a from-scratch ETKDG embed, which mis-folds
        # big/flexible/phosphate ligands -- and brings bonds/angles onto the shared
        # force-field ideal. Falls back to the cached coords if UFF fails (an explicitly
        # requested MMFF raises instead -- see ff_relax).
        # Guard: only when the mol has REAL bond orders (an aromatic or double bond).
        # chai/esmfold2 expose no bond orders -> their mol is all-single, so the relax would
        # localize aromatic rings to single-bond lengths (~1.5), corrupting the target;
        # for those tools the cached conformer holds the real reference geometry.
        has_orders = any(
            b.GetIsAromatic() or b.GetBondType() == Chem.BondType.DOUBLE
            for b in mol.GetBonds()
        )
        if do_relax and not has_orders and ff != "uff":
            # An explicitly requested MMFF that would silently not run at all. Raise rather
            # than skip -- "I set relax_force_field.ligand: mmff94s and got
            # un-relaxed targets" is exactly the invisible outcome the explicit setting
            # is meant to rule out.
            _at = f"global atom index {int(gidx[0])}, " if len(gidx) else ""
            raise ValueError(
                "conformer_restraints_config.relax_force_field."
                f"ligand={force_field!r}: ligand #{li} "
                f"({_at}{mol.GetNumAtoms()} atoms) has "
                "no aromatic or double bond, so the relax is skipped and the force field "
                "would never run. Either the tool supplied no real bond orders (chai / "
                "esmfold2 without SMILES -- relaxing an all-single mol would collapse "
                "aromatic rings to ~1.5 A), or the ligand is genuinely saturated. Supply "
                "the ligand as SMILES/CCD, or set relax_force_field: {ligand: uff} "
                "(same skip, no error) or {ligand: none}."
            )
        if do_relax and has_orders:
            _relaxed = ff_relax(mol, crds, ff)
            if _relaxed is not None and len(_relaxed) == len(crds):
                crds = _relaxed

        for b in mol.GetBonds():
            ai, aj = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bonds.append(
                (int(gidx[ai]), int(gidx[aj]), _bond_length(crds, ai, aj), None)
            )

        for ai, aj, ak in mol.GetSubstructMatches(_ANGLE_PATT):
            angles.append(
                (
                    int(gidx[ai]),
                    int(gidx[aj]),
                    int(gidx[ak]),
                    _angle_rad(crds, ai, aj, ak),
                    None,
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

        # plane: hold each planar atom GROUP coplanar (servalcat-style best-fit plane).
        # Two group sources, each CONFIRMED planar in the reference conformer
        # (_max_plane_dev < _PLANE_TOL) rather than trusting GetIsAromatic():
        #   (a) rings — whole SSSR ring as one group (GetRingInfo). Aromatic/conjugated
        #       rings are flat and kept; saturated (puckered) rings deviate and are
        #       dropped. Ring topology is bond-order-independent, so this fires even for
        #       tools whose mol lost bond orders (chai/esmfold2 geometry-perceived path).
        #   (b) non-ring sp2 groups — each acyclic (`not IsInRing()`) non-aromatic DOUBLE
        #       bond centre + its heavy neighbours (mol is H-removed): carbonyl / carboxyl
        #       / amide / trisubstituted-alkene centres form a >=4-atom coplanar group. A
        #       2-heavy-neighbour alkene centre gives only 3 atoms (trivially planar, no
        #       restraint force) and is skipped by the len>=4 filter.
        candidates = [tuple(r) for r in mol.GetRingInfo().AtomRings()]
        for b in mol.GetBonds():
            if (
                b.GetBondType() != Chem.BondType.DOUBLE
                or b.GetIsAromatic()
                or b.IsInRing()
            ):
                continue
            for atom in (b.GetBeginAtom(), b.GetEndAtom()):
                candidates.append(
                    tuple([atom.GetIdx()] + [n.GetIdx() for n in atom.GetNeighbors()])
                )
        seen_planes: set[frozenset] = set()
        for cand in candidates:
            key = frozenset(cand)
            if len(key) < 4 or key in seen_planes:
                continue  # a plane needs >=4 atoms (3 are trivially coplanar)
            local = sorted(key)
            if _max_plane_dev(crds, local) >= _PLANE_TOL:
                continue  # not coplanar in the reference (e.g. saturated ring)
            seen_planes.add(key)
            planes.append(tuple(int(gidx[i]) for i in local))

    return bonds, angles, chirals, cistrans, planes


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


def _window_arrays(restraints, default_start_sigma):
    """Return the four common per-entry sigma/step window arrays."""
    start_sigma = np.array(
        [
            float(restraint.start_sigma)
            if getattr(restraint, "start_sigma", None) is not None
            else default_start_sigma
            for restraint in restraints
        ]
    )
    stop_sigma = np.array(
        [float(getattr(restraint, "stop_sigma", -1.0)) for restraint in restraints]
    )
    start_step = np.array(
        [
            float(getattr(restraint, "start_step", float("-inf")))
            for restraint in restraints
        ]
    )
    stop_step = np.array(
        [
            float(getattr(restraint, "stop_step", float("inf")))
            for restraint in restraints
        ]
    )
    return start_sigma, stop_sigma, start_step, stop_step


def _build_group_geom_arrays(
    restraints, n_groups, array_type, g2l, default_start_sigma
):
    """Build one padded angle/dihedral/improper array family."""
    indices, masks = _pad_groups(restraints, n_groups, g2l)
    codes, target1, target2, move_free = _group_geom_params(restraints)
    start_sigma, stop_sigma, start_step, stop_step = _window_arrays(
        restraints, default_start_sigma
    )
    groups = {}
    for group, (idx, group_mask) in enumerate(zip(indices, masks), start=1):
        groups[f"grp{group}_idx"] = idx
        groups[f"grp{group}_mask"] = group_mask
    return array_type(
        **groups,
        target1=target1,
        target2=target2,
        geom_type=codes,
        move_free=move_free,
        weight=np.array([float(restraint.weight) for restraint in restraints]),
        mask=np.ones(len(restraints)),
        start_sigma=start_sigma,
        stop_sigma=stop_sigma,
        start_step=start_step,
        stop_step=stop_step,
    )


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


def _build_distance_arrays(restraints, g2l, default_start_sigma):
    """Build padded distance arrays using the shared group/window encoders."""
    indices, masks = _pad_groups(restraints, 2, g2l)
    params = [_dist_params(restraint) for restraint in restraints]
    start_sigma, stop_sigma, start_step, stop_step = _window_arrays(
        restraints, default_start_sigma
    )
    return DistanceArrays(
        grp1_idx=indices[0],
        grp2_idx=indices[1],
        grp1_mask=masks[0],
        grp2_mask=masks[1],
        target1=np.array([param[1] for param in params]),
        target2=np.array([param[2] for param in params]),
        dist_type=np.array([param[0] for param in params], dtype=np.int64),
        move_mode=np.array(
            [int(getattr(restraint, "move_mode", 0)) for restraint in restraints],
            dtype=np.int64,
        ),
        weight=np.array(
            [float(getattr(restraint, "weight", 1.0)) for restraint in restraints]
        ),
        mask=np.ones(len(restraints)),
        start_sigma=start_sigma,
        stop_sigma=stop_sigma,
        start_step=start_step,
        stop_step=stop_step,
    )


def _vdw_radius(z: int) -> float:
    """VdW radius (A) for atomic number ``z`` from RDKit's periodic table."""
    if z is None or z < 1 or z > 118:
        return 0.0
    return float(Chem.GetPeriodicTable().GetRvdw(int(z)))


def _conf_weight(conformer_config: dict | None, key: str) -> float:
    """Weight of a conformer sub-term (bond/angle/chiral/cistrans/vdw/plane).

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

    Shared by all five conformer terms (bond/angle/chiral/cistrans/plane) so the
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
    polymer_atoms: np.ndarray,
    conformer_config: dict,
    active_sites: np.ndarray,
    g2l: dict,
    elements: np.ndarray | None,
) -> VdwConfig | None:
    """Build the dynamic fixed-background VdW config (torch + jax optimizers).

    The fixed-background half of the ``intermolecular`` category. Ligand atoms (the
    moving set) come from the RDKit mols; the background is every non-padding atom NOT
    in ``active_sites`` (i.e. not optimised) — protein, DNA/RNA, and any non-restrained
    ligand — so the VdW term pushes the ligand out of the fixed pocket. Returns
    ``None`` when VdW is disabled (weight<=0), there are no ligands, or element info
    is unavailable.
    """
    vcfg = (conformer_config or {}).get("vdw", {}) or {}
    weight = _conf_weight(conformer_config, "vdw")
    if (
        weight <= 0.0
        or (not ligand_confs and len(polymer_atoms) == 0)
        or elements is None
    ):
        return None

    # ligand atoms + per-atom radii from the mols (global index -> radius)
    lig_radius: dict[int, float] = {}
    for lc in ligand_confs:
        gidx = np.asarray(lc.global_indices, dtype=np.int64)
        for i, atom in enumerate(lc.mol.GetAtoms()):
            lig_radius[int(gidx[i])] = _vdw_radius(atom.GetAtomicNum())
    elements = np.asarray(elements)
    for g in polymer_atoms:
        if 0 <= int(g) < len(elements) and int(elements[int(g)]) > 0:
            lig_radius[int(g)] = _vdw_radius(int(elements[int(g)]))
    ligand_global = np.array(sorted(lig_radius), dtype=np.int64)
    # every ligand atom is in active_sites (added in build_spec when VdW is on)
    ligand_local = np.array([g2l[int(g)] for g in ligand_global], dtype=np.int64)
    ligand_radii = np.array(
        [lig_radius[int(g)] for g in ligand_global], dtype=np.float64
    )

    # fixed background = all non-padding atoms NOT optimised (not in active_sites):
    # protein / DNA/RNA / non-restrained ligand. element code 0 is the padding sentinel.
    active_set = {int(a) for a in active_sites}
    background_global = np.array(
        [
            a
            for a in range(len(elements))
            if int(elements[a]) > 0 and a not in active_set
        ],
        dtype=np.int64,
    )
    if len(background_global) == 0:
        return None
    background_radii = np.array(
        [_vdw_radius(int(elements[a])) for a in background_global], dtype=np.float64
    )
    max_neighbors = int(vcfg.get("max_neighbors", 32))
    if max_neighbors < 1:
        raise ValueError("conformer vdw max_neighbors must be >= 1")

    return VdwConfig(
        weight=weight,
        ligand_local=ligand_local,
        ligand_radii=ligand_radii,
        background_global=background_global,
        background_radii=background_radii,
        scale=float(vcfg.get("scale", VDW_SCALE_DEFAULT)),
        dmax=float(vcfg.get("dmax", 5.0)),
        max_neighbors=max_neighbors,
    )


def _build_active_vdw_config(
    polymer_atoms: np.ndarray,
    conformer_config: dict,
    active_sites: np.ndarray,
    elements: np.ndarray | None,
    bonds,
    angles,
) -> ActiveVdwConfig | None:
    """Build dynamic polymer-involving VdW metadata in local active-site space."""

    weight = _conf_weight(conformer_config, "vdw")
    if weight <= 0.0 or len(polymer_atoms) == 0 or elements is None:
        return None
    elements = np.asarray(elements)
    g2l = {int(g): i for i, g in enumerate(active_sites)}
    radii = np.array(
        [
            _vdw_radius(int(elements[g])) if int(elements[g]) > 0 else 0.0
            for g in active_sites
        ],
        dtype=np.float64,
    )
    polymer_set = {int(g) for g in polymer_atoms}
    polymer_mask = np.array([int(g) in polymer_set for g in active_sites], dtype=bool)
    n_active = len(active_sites)
    excluded = set()

    def exclude(global_i, global_j):
        if global_i not in g2l or global_j not in g2l:
            return
        i, j = sorted((g2l[int(global_i)], g2l[int(global_j)]))
        if i != j:
            excluded.add(i * n_active + j)

    adjacency: dict[int, set[int]] = {}
    for g0, g1, *_ in bonds:
        g0, g1 = int(g0), int(g1)
        adjacency.setdefault(g0, set()).add(g1)
        adjacency.setdefault(g1, set()).add(g0)

    # Exclude every pair separated by at most three covalent bonds (1-2/1-3/1-4),
    # including paths that cross peptide or phosphodiester links.
    for start in adjacency:
        seen = {start}
        frontier = {start}
        for _distance in range(3):
            frontier = {
                neighbour
                for atom in frontier
                for neighbour in adjacency.get(atom, ())
                if neighbour not in seen
            }
            for end in frontier:
                exclude(start, end)
            seen.update(frontier)

    # Keep explicit angle exclusions even if an incomplete external geometry source
    # supplied an angle without both constituent bonds.
    for g0, g1, g2, *_ in angles:
        exclude(g0, g1)
        exclude(g1, g2)
        exclude(g0, g2)

    vcfg = (conformer_config or {}).get("vdw", {}) or {}
    max_neighbors = int(vcfg.get("max_neighbors", 32))
    if max_neighbors < 1:
        raise ValueError("conformer vdw max_neighbors must be >= 1")
    return ActiveVdwConfig(
        weight=weight,
        radii=radii,
        polymer_mask=polymer_mask,
        excluded_codes=np.asarray(sorted(excluded), dtype=np.int64),
        scale=float(vcfg.get("scale", VDW_SCALE_DEFAULT)),
        dmax=float(vcfg.get("dmax", 5.0)),
        max_neighbors=max_neighbors,
    )


def _build_intramolecular_vdw(
    ligand_confs: list[LigandConf],
    conformer_config: dict,
    g2l: dict,
) -> VdwArrays | None:
    """Static intramolecular VdW repulsion within each ligand (all backends).

    Penalizes every atom pair within one ligand whose topological distance is > 3
    (so 1-2 bonds, 1-3 angles, and 1-4 dihedrals are skipped), with a lower bound
    ``scale * (r_i + r_j)``. Reference distance is deliberately not a build filter. Unlike the
    dynamic fixed-background ``VdwConfig``, the pair list is fixed, so this term also
    works in the jax/numpy backends via ``VdwArrays``. Enabled when
    ``conformer_config['vdw']['mode']`` is ``'intramolecular'`` or ``'both'`` (the
    DEFAULT); ``'intermolecular'`` leaves it off.
    """
    vcfg = (conformer_config or {}).get("vdw", {}) or {}
    weight = _conf_weight(conformer_config, "vdw")
    if weight <= 0.0 or not ligand_confs:
        return None
    from rdkit.Chem import rdmolops

    scale = float(vcfg.get("scale", VDW_SCALE_DEFAULT))
    idx_pairs: list[list[int]] = []
    r_min_list: list[float] = []
    for lc in ligand_confs:
        mol = lc.mol
        gidx = np.asarray(lc.global_indices, dtype=np.int64)
        n = mol.GetNumAtoms()
        if n < 2:
            continue
        topo = rdmolops.GetDistanceMatrix(mol)
        radii = [_vdw_radius(a.GetAtomicNum()) for a in mol.GetAtoms()]
        for i in range(n):
            for j in range(i + 1, n):
                if topo[i, j] <= 3:  # skip 1-2, 1-3, and 1-4 pairs
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

    The restrained-ligand half of the ``intermolecular`` category (the other half is
    the dynamic fixed-background ``VdwConfig``). Both endpoints of every pair live in
    ``active_sites`` (each ligand moves under its own conformer restraint), so
    autodiff drives BOTH ligands apart — exactly like ``vdw_energy``'s
    intramolecular pairs, only the pair list crosses molecules. Unlike
    ``_build_intramolecular_vdw`` there is NO topological skip (different mols have
    no shared bonds) and NO reference-distance ``dmax`` cutoff (two ligands'
    ``conf_coords`` live in independent frames, so a build-time distance is
    meaningless); every cross pair is listed and the ``clamp(d - r_min, max=0)``
    penalty contributes zero beyond contact. Ligands are H-removed and small, so
    the all-pairs list stays cheap. Built only when VdW is on, ``mode`` is
    ``'intermolecular'`` or ``'both'``, and at least two ligands opted in.
    """
    vcfg = (conformer_config or {}).get("vdw", {}) or {}
    weight = _conf_weight(conformer_config, "vdw")
    if weight <= 0.0 or len(ligand_confs) < 2:
        return None

    scale = float(vcfg.get("scale", VDW_SCALE_DEFAULT))
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
    polymer_geometry=None,
    plane_restraints: list | None = None,
    improper_restraints: list | None = None,
) -> RestraintSpec:
    """Build a RestraintSpec. ``distance_restraints`` are DistanceData with
    ``target_sites1``/``target_sites2`` already resolved to global indices;
    ``rmsd_restraints`` are RmsdData with ``target_sites``/``ref_coords`` resolved;
    ``angle_restraints``/``dihedral_restraints``/``improper_restraints`` carry
    resolved group ``target_sites{1..N}`` global indices (N=3 for angle, N=4 for
    dihedral/improper). ``plane_restraints`` are
    PlaneRestraintData with ``target_sites`` (a LIST of per-group global-index lists)
    resolved — the standalone ``plane_restraints_config`` term, which is independent of
    the conformer ``plane`` sub-block (its own weight/type/gate per entry). The base-pair
    coplanarity macro also arrives here (combined.setup hands over pre-resolved
    PlaneRestraintData), which is why there is no longer an ``extra_plane_groups``
    back-door into the conformer plane arrays."""
    ligand_confs = ligand_confs or []
    cfg = conformer_config or {}
    validate_vdw_config(cfg)
    # Conformer restraints are OPT-IN -- this is the single enforcement point for every
    # tool: (1) with no conformer_restraints_config (e.g. a distance-only run) build no
    # conformer at all; (2) otherwise restrain only ligand conformers whose chain opted
    # in. Every tool defaults the per-chain flag to False.
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
    if cfg_present and _n_before and not ligand_confs and polymer_geometry is None:
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
    improper_restraints = [
        ir for ir in (improper_restraints or []) if getattr(ir, "run_restr", False)
    ]
    plane_restraints = [
        pr for pr in (plane_restraints or []) if getattr(pr, "run_restr", False)
    ]
    custom_restraints = [
        c for c in (custom_restraints or []) if getattr(c, "run_restr", False)
    ]
    # Every conformer sub-term follows the uniform "default 1.0, off if not configured"
    # rule (see _conf_weight): a sub-block PRESENT in the conformer config is active at
    # weight 1.0 (override with an explicit weight); an ABSENT sub-block is OFF. So a
    # ligand that opts in but lists e.g. only `bond:` gets ONLY bond — angle/chiral/
    # cistrans/vdw/plane stay off until their own sub-block is added. slack defaults
    # stay per-term (chiral flat-bottom ~0.05 signed-volume; bond/angle/cistrans/plane
    # 0.0 = pure harmonic toward the reference; cistrans slack is in radians, plane in
    # Angstrom out-of-plane deviation).
    bw = _conf_weight(cfg, "bond")
    bsl = _conf_slack(cfg, "bond", 0.0)
    aw = _conf_weight(cfg, "angle")
    asl = _conf_slack(cfg, "angle", 0.0)
    cw = _conf_weight(cfg, "chiral")
    csl = _conf_slack(cfg, "chiral", 0.05)
    dw = _conf_weight(cfg, "cistrans")
    dsl = _conf_slack(cfg, "cistrans", 0.0)
    vdw_weight = _conf_weight(cfg, "vdw")
    pw = _conf_weight(cfg, "plane")
    psl = _conf_slack(cfg, "plane", 0.0)

    # Which force field idealises the reference conformer before the targets are measured
    # off it. LIGANDS only -- the polymer call below stays relax=False (monomer-library
    # residues are never relaxed), so this can never fire there.
    relax_ff = parse_relax_force_field(cfg)
    bonds, angles, chirals, cistrans, planes = _extract_conformer(
        ligand_confs, force_field=relax_ff
    )
    polymer_atoms = np.empty(0, dtype=np.int64)
    if polymer_geometry is not None:
        pb, pa, pc, _pd, pp = _extract_conformer(
            polymer_geometry.residue_confs, relax=False
        )
        # Monomer-library targets REPLACE the reference-conformer ones residue by
        # residue (`monomer_library` config; see monlib_geom). Every conformer-derived
        # tuple is intra-residue, so "all its atoms are in a covered residue" identifies
        # exactly the tuples the library re-states -- drop those and keep the rest, so a
        # partially covered structure mixes sources per residue and never doubles up.
        lib_atoms = polymer_geometry.library_atoms
        if lib_atoms:
            pb = [t for t in pb if not lib_atoms.issuperset(t[:2])]
            pa = [t for t in pa if not lib_atoms.issuperset(t[:3])]
            pp = [t for t in pp if not lib_atoms.issuperset(t)]
        bonds.extend(pb)
        bonds.extend(polymer_geometry.library_bonds)
        bonds.extend(polymer_geometry.link_bonds)
        angles.extend(pa)
        angles.extend(polymer_geometry.library_angles)
        angles.extend(polymer_geometry.link_angles)
        # Chirality stays reference-conformer-derived even under a library: only the
        # SIGN protects stereochemistry, and the library's ChiralityType convention
        # would have to be reconciled with _chiral_vol's atom ordering first.
        chirals.extend(pc)  # residue-local stereocentres (Calpha) only
        # Polymer planarity: residue-local aromatic rings (His/Phe/Tyr/Trp side chains,
        # nucleic-acid bases) from _extract_conformer -- or the library's named plane
        # groups, which put a whole nucleobase (ring + exocyclic atoms + C1') in ONE
        # group where SSSR perception splits a purine into two fused rings -- plus the
        # canonical peptide plane (a 5-atom group in global indices; appended directly,
        # bypassing the residue-local coplanarity check like link_bonds/link_angles).
        planes.extend(pp)
        planes.extend(polymer_geometry.library_planes)
        planes.extend(polymer_geometry.link_planes)
        polymer_atoms = np.asarray(polymer_geometry.atom_indices, dtype=np.int64)
    # VdW covalent exclusions must survive even when bond/angle energy blocks are off.
    exclusion_bonds = list(bonds)
    exclusion_angles = list(angles)
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
    if pw <= 0:
        planes = []  # OFF by default (pw defaults to 0): opt-in plane term

    # ---- collect every referenced global atom -> active_sites -----------------
    active: set[int] = set()
    for g0, g1, *_ in bonds:
        active.update((g0, g1))
    for g0, g1, g2, *_ in angles:
        active.update((g0, g1, g2))
    for g0, g1, g2, g3, _ in chirals:
        active.update((g0, g1, g2, g3))
    for g0, g1, g2, g3, _ in cistrans:
        active.update((g0, g1, g2, g3))
    for grp in planes:
        active.update(grp)
    resolved_restraints = itertools.chain(
        distance_restraints,
        rmsd_restraints,
        angle_restraints,
        dihedral_restraints,
        improper_restraints,
        plane_restraints,
        custom_restraints,
    )
    for restraint in resolved_restraints:
        active.update(int(site) for site in restraint.iter_global_sites())
    # VdW pushes the whole ligand, so every ligand atom must be optimisable even
    # if it carries no bond/angle/chiral term (e.g. a monatomic ion).
    if vdw_weight > 0:
        for lc in ligand_confs:
            active.update(int(g) for g in lc.global_indices)
        active.update(int(g) for g in polymer_atoms)

    active_sites = np.array(sorted(active), dtype=np.int64)
    g2l = {int(g): i for i, g in enumerate(active_sites)}
    # custom restraints: remap each resolved selection to LOCAL indices (CustomSpec).
    custom_specs = [cr.build_spec(g2l) for cr in custom_restraints]
    # VdW has two CATEGORIES, set by conformer_config['vdw']['mode']:
    #   - "intramolecular": clashes WITHIN one ligand (static VdwArrays -> energy layer).
    #   - "intermolecular": clashes between that ligand and EVERY OTHER molecule. This is
    #     itself two pieces sharing one category: (a) vs the FIXED background — every heavy
    #     atom not in active_sites, i.e. protein/DNA/RNA/non-restrained ligand — built as
    #     VdwConfig and run in the torch/jax optimizer (the background needs no gradient);
    #     (b) vs OTHER RESTRAINED ligands — both move, so it is a static VdwArrays scored in
    #     the energy layer with the inter-ligand pairs CONCATENATED onto the intramolecular
    #     rows (same energy term, same conformer gate, all backends).
    #   - "both" (DEFAULT): intramolecular + intermolecular.
    # The old "ligand_protein" value (the fixed-background piece only) is REMOVED -> raise a
    # migration hint, mirroring the rejected `backend:` key. Both halves run on torch AND
    # jax; on numpy (energy reference only) the optimizer fixed-background half is inert.
    vdw_mode = (cfg.get("vdw", {}) or {}).get("mode", "both")
    if vdw_mode == "ligand_protein":
        raise ValueError(
            "conformer vdw mode 'ligand_protein' was renamed to 'intermolecular', which "
            "now repels the ligand off EVERY other molecule (protein/DNA/RNA/non-restrained "
            "ligand background AND other restrained ligands), not just the fixed background "
            "-- update your config to mode: intermolecular"
        )
    if vdw_mode not in ("intramolecular", "intermolecular", "both"):
        raise ValueError(
            "conformer vdw mode must be 'intramolecular', 'intermolecular', or "
            f"'both', got {vdw_mode!r}"
        )
    vdw_intra = (
        _build_intramolecular_vdw(ligand_confs, cfg, g2l)
        if vdw_mode in ("intramolecular", "both")
        else None
    )
    vdw_inter = (
        _build_interligand_vdw(ligand_confs, cfg, g2l)
        if vdw_mode in ("intermolecular", "both")
        else None
    )
    vdw_arrays = _concat_vdw_arrays(vdw_intra, vdw_inter)
    vdw_config = (
        _build_vdw_config(ligand_confs, polymer_atoms, cfg, active_sites, g2l, elements)
        if vdw_mode in ("intermolecular", "both")
        else None
    )
    active_vdw_config = _build_active_vdw_config(
        polymer_atoms, cfg, active_sites, elements, exclusion_bonds, exclusion_angles
    )

    # ---- conformer arrays (local indices) -------------------------------------
    bond = None
    if bonds:
        idx = np.array([[g2l[g0], g2l[g1]] for g0, g1, *_ in bonds], dtype=np.int64)
        bond = BondArrays(
            idx=idx,
            r0=np.array([r for _, _, r, _ in bonds]),
            # A library-derived restraint carries its own sigma; use it as the
            # flat-bottom half-width. Refmac/servalcat weight by 1/sigma^2 and let
            # the experimental data decide where inside that sigma the atom sits;
            # with no data term a weighted harmonic just converges to the exact
            # target no matter the weight, so the only way sigma can mean anything
            # here is as a tolerance. Without it the CG removes the scatter real
            # structures carry (measured on QBP: N-CA-C spread 1.83 -> 0.88 deg
            # against 2.90 in the 1GGG crystal) and the strain it can no longer
            # absorb locally reappears as clashes.
            slack=np.array([bsl if e is None else float(e) for *_, e in bonds]),
            weight=np.full(len(bonds), bw),
            half=np.zeros(len(bonds)),
            mask=np.ones(len(bonds)),
        )
    angle = None
    if angles:
        idx = np.array(
            [[g2l[g0], g2l[g1], g2l[g2]] for g0, g1, g2, *_ in angles],
            dtype=np.int64,
        )
        angle = AngleArrays(
            idx=idx,
            th0=np.array([t for _, _, _, t, _ in angles]),
            slack=np.array([asl if e is None else float(e) for *_, e in angles]),
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
    plane = None
    # conformer/polymer planes all carry the shared conformer weight/slack (pw/psl) and the
    # shared conformer gate. Selection-driven planes are NOT here — they are the standalone
    # `group_plane` term below, with their own per-entry type/weight/gate.
    plane_groups = list(planes)
    if plane_groups:
        # variable group size -> pad to the widest group; padding columns hold local
        # index 0 (a valid atom) and are zeroed in grp_mask (same layout as distance).
        n_plane = len(plane_groups)
        max_atoms = max(len(grp) for grp in plane_groups)
        idx = np.zeros((n_plane, max_atoms), dtype=np.int64)
        grp_mask = np.zeros((n_plane, max_atoms), dtype=np.float64)
        for r, grp in enumerate(plane_groups):
            for c, g in enumerate(grp):
                idx[r, c] = g2l[g]
                grp_mask[r, c] = 1.0
        plane = PlaneArrays(
            idx=idx,
            grp_mask=grp_mask,
            slack=np.full(n_plane, psl),
            weight=np.full(n_plane, pw),
            mask=np.ones(n_plane),
        )

    # ---- distance arrays (padded, local indices) ------------------------------
    distance = (
        _build_distance_arrays(distance_restraints, g2l, conf_start_sigma)
        if distance_restraints
        else None
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
        (
            rmsd_start_sigma,
            rmsd_stop_sigma,
            rmsd_start_step,
            rmsd_stop_step,
        ) = _window_arrays(rmsd_restraints, conf_start_sigma)
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

    # ---- group-centroid angle / torsion arrays -------------------------------
    group_angle = (
        _build_group_geom_arrays(
            angle_restraints, 3, GroupAngleArrays, g2l, conf_start_sigma
        )
        if angle_restraints
        else None
    )
    group_dihedral = (
        _build_group_geom_arrays(
            dihedral_restraints, 4, GroupDihedralArrays, g2l, conf_start_sigma
        )
        if dihedral_restraints
        else None
    )
    group_improper = (
        _build_group_geom_arrays(
            improper_restraints, 4, GroupImproperArrays, g2l, conf_start_sigma
        )
        if improper_restraints
        else None
    )

    # ---- standalone best-fit-plane arrays (padded, local indices) -------------------
    group_plane = None
    if plane_restraints:
        n = len(plane_restraints)
        # Each entry POOLS all of its groups into one plane, so a row is the concatenated
        # atom list (padded to the widest entry). `free` is therefore per-ATOM: it repeats
        # each group's `move_free` flag across that group's atoms.
        pooled = [
            [int(s) for grp in pr.target_sites for s in grp] for pr in plane_restraints
        ]
        free_flags = [
            [
                1.0 if pr.move_free[gi] else 0.0
                for gi, grp in enumerate(pr.target_sites)
                for _ in grp
            ]
            for pr in plane_restraints
        ]
        max_atoms = max(len(row) for row in pooled)
        idx = np.zeros((n, max_atoms), dtype=np.int64)
        grp_mask = np.zeros((n, max_atoms))
        free = np.zeros((n, max_atoms))
        for r, (row, flags) in enumerate(zip(pooled, free_flags)):
            local = [g2l[g] for g in row]
            idx[r, : len(local)] = local
            grp_mask[r, : len(local)] = 1.0
            free[r, : len(flags)] = flags
        start_sigma, stop_sigma, start_step, stop_step = _window_arrays(
            plane_restraints, conf_start_sigma
        )
        group_plane = GroupPlaneArrays(
            idx=idx,
            grp_mask=grp_mask,
            free=free,
            target1=np.array([float(r.target1) for r in plane_restraints]),
            target2=np.array([float(r.target2) for r in plane_restraints]),
            geom_type=np.array(
                [DIST_TYPE_CODES[r.geom_type] for r in plane_restraints], dtype=np.int64
            ),
            weight=np.array([float(r.weight) for r in plane_restraints]),
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
        plane=plane,
        cistrans=cistrans_arr,
        distance=distance,
        rmsd=rmsd,
        group_angle=group_angle,
        group_dihedral=group_dihedral,
        group_plane=group_plane,
        vdw=vdw_arrays,
        vdw_config=vdw_config,
        group_improper=group_improper,
        active_vdw_config=active_vdw_config,
        vdw_max_atom_step=float(
            (cfg.get("vdw", {}) or {}).get("max_atom_step", VDW_MAX_ATOM_STEP_DEFAULT)
        ),
        vdw_neighbor_rebuild_interval=int(
            (cfg.get("vdw", {}) or {}).get(
                "neighbor_rebuild_interval", VDW_NEIGHBOR_REBUILD_INTERVAL_DEFAULT
            )
        ),
        vdw_neighbor_skin=float(
            (cfg.get("vdw", {}) or {}).get("neighbor_skin", VDW_NEIGHBOR_SKIN_DEFAULT)
        ),
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
            f"{len(vdw_config.ligand_local)}lig/"
            f"{len(vdw_config.background_global)}bg/{vdw_config.max_neighbors}nn"
        )
    if active_vdw_config is not None:
        vdw_parts.append(
            f"{int(active_vdw_config.polymer_mask.sum())}poly/"
            f"{active_vdw_config.max_neighbors}nn"
        )
    vdw_desc = "+".join(vdw_parts) if vdw_parts else "off"
    logger.info(
        "built spec: n_active=%d bonds=%d angles=%d chirals=%d plane=%d cistrans=%d "
        "distances=%d rmsd=%d group_angle=%d group_dihedral=%d "
        "group_improper=%d group_plane=%d "
        "vdw=%s custom=%d relax_ff=%s",
        spec.n_active,
        len(bonds),
        len(angles),
        len(chirals),
        len(plane_groups),
        len(cistrans),
        len(distance_restraints),
        len(rmsd_restraints),
        len(angle_restraints),
        len(dihedral_restraints),
        len(improper_restraints),
        len(plane_restraints),
        vdw_desc,
        len(custom_specs),
        relax_ff,
    )
    return spec
