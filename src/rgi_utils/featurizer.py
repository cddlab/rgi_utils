"""Build a backend-agnostic ``RestraintSpec`` from ligand conformers + distances.

This is the single place where conformer restraints (bond/angle/chiral) are
derived from RDKit mols. Each ligand supplies its own ``global_indices``, so
multiple ligands produce non-colliding restraints — there is no per-batch state
and no hard-coded ligand index (the multi-ligand bug in the old code).

Flow:
  1. extract bond/angle/chiral restraints per ligand in GLOBAL atom indices,
  2. collect distance restraint atom groups (already resolved to global indices),
  3. active_sites = union of all referenced global atoms (sorted, unique),
  4. remap every index to a LOCAL index into active_sites and pack padded arrays.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
from rdkit import Chem

from rgi_utils.atom_context import LigandConf
from rgi_utils.spec import (
    DIST_TYPE_CODES,
    AngleArrays,
    BondArrays,
    ChiralArrays,
    DistanceArrays,
    RestraintSpec,
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


def _extract_conformer(ligand_confs: list[LigandConf]):
    """Return bond/angle/chiral restraint tuples in GLOBAL atom indices."""
    bonds = []  # (g0, g1, r0)
    angles = []  # (g0, g1, g2, th0)
    chirals = []  # (g0, g1, g2, g3, vol0)

    for lc in ligand_confs:
        mol = lc.mol
        crds = np.asarray(lc.conf_coords, dtype=np.float64)
        gidx = np.asarray(lc.global_indices, dtype=np.int64)

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

    return bonds, angles, chirals


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


def _build_vdw_config(
    ligand_confs: list[LigandConf],
    conformer_config: dict,
    active_sites: np.ndarray,
    g2l: dict,
    elements: np.ndarray | None,
) -> VdwConfig | None:
    """Build the dynamic ligand-protein VdW config (torch optimizer only).

    Ligand atoms (the moving set) come from the RDKit mols; the protein
    background is every heavy atom NOT in ``active_sites`` (i.e. not optimised),
    so the VdW term pushes the ligand out of the fixed pocket. Returns ``None``
    when VdW is disabled (weight<=0), there are no ligands, or element info is
    unavailable.
    """
    vcfg = (conformer_config or {}).get("vdw", {}) or {}
    weight = float(vcfg.get("weight", 0.0) or 0.0)
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


def build_spec(
    ligand_confs: list[LigandConf] | None = None,
    distance_restraints: list | None = None,
    conformer_config: dict | None = None,
    elements: np.ndarray | None = None,
    conf_start_sigma: float = -1.0,
) -> RestraintSpec:
    """Build a RestraintSpec. ``distance_restraints`` are DistanceData with
    ``target_sites1``/``target_sites2`` already resolved to global indices."""
    ligand_confs = ligand_confs or []
    distance_restraints = [
        dr for dr in (distance_restraints or []) if getattr(dr, "run_restr", False)
    ]
    cfg = conformer_config or {}
    bw = cfg.get("bond", {}).get("weight", 0.05)
    bsl = cfg.get("bond", {}).get("slack", 0.0)
    aw = cfg.get("angle", {}).get("weight", 0.05)
    asl = cfg.get("angle", {}).get("slack", 0.0)
    cw = cfg.get("chiral", {}).get("weight", 0.1)
    csl = cfg.get("chiral", {}).get("slack", 0.05)
    vdw_weight = float((cfg.get("vdw", {}) or {}).get("weight", 0.0) or 0.0)

    bonds, angles, chirals = _extract_conformer(ligand_confs)

    # ---- collect every referenced global atom -> active_sites -----------------
    active: set[int] = set()
    for g0, g1, _ in bonds:
        active.update((g0, g1))
    for g0, g1, g2, _ in angles:
        active.update((g0, g1, g2))
    for g0, g1, g2, g3, _ in chirals:
        active.update((g0, g1, g2, g3))
    for dr in distance_restraints:
        active.update(int(s) for s in dr.target_sites1)
        active.update(int(s) for s in dr.target_sites2)
    # VdW pushes the whole ligand, so every ligand atom must be optimisable even
    # if it carries no bond/angle/chiral term (e.g. a monatomic ion).
    if vdw_weight > 0:
        for lc in ligand_confs:
            active.update(int(g) for g in lc.global_indices)

    active_sites = np.array(sorted(active), dtype=np.int64)
    g2l = {int(g): i for i, g in enumerate(active_sites)}
    vdw_config = _build_vdw_config(ligand_confs, cfg, active_sites, g2l, elements)

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
        dist_start_sigma = np.full(n, -1.0)
        for di, dr in enumerate(distance_restraints):
            s1 = [g2l[int(s)] for s in dr.target_sites1]
            s2 = [g2l[int(s)] for s in dr.target_sites2]
            grp1_idx[di, : len(s1)] = s1
            grp2_idx[di, : len(s2)] = s2
            grp1_mask[di, : len(s1)] = 1.0
            grp2_mask[di, : len(s2)] = 1.0
            code, t1, t2 = _dist_params(dr)
            dist_type[di] = code
            target1[di] = t1
            target2[di] = t2
            ss = getattr(dr, "start_sigma", None)
            dist_start_sigma[di] = float(ss) if ss is not None else conf_start_sigma
        distance = DistanceArrays(
            grp1_idx=grp1_idx,
            grp2_idx=grp2_idx,
            grp1_mask=grp1_mask,
            grp2_mask=grp2_mask,
            target1=target1,
            target2=target2,
            dist_type=dist_type,
            mask=np.ones(n),
            start_sigma=dist_start_sigma,
        )

    spec = RestraintSpec(
        n_active=len(active_sites),
        active_sites=active_sites,
        bond=bond,
        angle=angle,
        chiral=chiral,
        distance=distance,
        vdw_config=vdw_config,
        conf_start_sigma=conf_start_sigma,
    )
    logger.info(
        "built spec: n_active=%d bonds=%d angles=%d chirals=%d distances=%d vdw=%s",
        spec.n_active,
        len(bonds),
        len(angles),
        len(chirals),
        len(distance_restraints),
        "off"
        if vdw_config is None
        else f"{len(vdw_config.ligand_local)}lig/{len(vdw_config.protein_global)}prot",
    )
    return spec
