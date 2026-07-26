"""Featurizer tests: multi-ligand has no index collision; distance+conformer union."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rgi_utils.atom_context import LigandConf
from rgi_utils.distance_restr_data import DistanceData
from rgi_utils.featurizer import build_spec
from rgi_utils.group_geom_restr_data import AngleRestraintData, DihedralRestraintData


def _ethane():
    m = Chem.MolFromSmiles("CC")
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=1)
    conf = np.asarray(m.GetConformer().GetPositions())
    return m, conf


def test_multiligand_no_collision():
    m1, c1 = _ethane()
    m2, c2 = _ethane()
    n = m1.GetNumAtoms()
    lc1 = LigandConf(
        mol=m1, conf_coords=c1, global_indices=np.arange(n), conformer_restraints=True
    )
    lc2 = LigandConf(
        mol=m2,
        conf_coords=c2,
        global_indices=np.arange(n) + 100,
        conformer_restraints=True,
    )

    spec = build_spec([lc1, lc2], [], {"bond": {"weight": 0.05}})

    assert spec.bond is not None
    # both ligands contribute their bonds, with no overlap
    assert spec.bond.idx.shape[0] == 2 * m1.GetNumBonds()
    assert spec.n_active == 2 * n
    assert {int(x) for x in spec.active_sites} == set(range(n)) | set(
        range(100, 100 + n)
    )
    # all local indices are valid (the collision bug would push these out of range)
    assert int(spec.bond.idx.max()) < spec.n_active


def test_distance_and_conformer_active_union():
    m, c = _ethane()
    n = m.GetNumAtoms()
    lc = LigandConf(
        mol=m, conf_coords=c, global_indices=np.arange(n), conformer_restraints=True
    )

    # a distance restraint referencing atoms outside the ligand
    dd = DistanceData()
    dd.set_config(
        {
            "atom_selection1": "index 50",
            "atom_selection2": "index 60",
            "harmonic": {"target_distance": 5.0},
        }
    )
    # pre-resolve sites (normally done by an adapter)
    dd.target_sites1 = [50]
    dd.target_sites2 = [60]

    spec = build_spec([lc], [dd], {"bond": {"weight": 0.05}})

    # active sites = ligand atoms (conformer) UNION distance atoms
    assert spec.distance is not None
    assert {int(x) for x in spec.active_sites} == set(range(n)) | {50, 60}
    # distance group local indices point back at 50 and 60
    g2l = {int(g): i for i, g in enumerate(spec.active_sites)}
    assert int(spec.distance.grp1_idx[0, 0]) == g2l[50]
    assert int(spec.distance.grp2_idx[0, 0]) == g2l[60]


def test_group_angle_dihedral_active_union_and_padding():
    """Group angle/dihedral atoms join active_sites; group indices are remapped to local
    and padded per group with a {0,1} mask (max_grp spans every group)."""
    ad = AngleRestraintData()
    ad.geom_type, ad.target1, ad.target2 = "harmonic", 1.0, 0.0
    ad.move_free, ad.weight, ad.run_restr = (True, True, True), 1.0, True
    ad.start_sigma, ad.stop_sigma = 1e30, -1.0
    ad.target_sites1 = [10, 11, 12]  # group of 3 -> sets max_grp
    ad.target_sites2 = [20]  # group of 1 (padded)
    ad.target_sites3 = [30, 31]

    dd = DihedralRestraintData()
    dd.geom_type, dd.target1, dd.target2 = "harmonic", 2.0, 0.0
    dd.move_free, dd.weight, dd.run_restr = (True, True, True, True), 1.0, True
    dd.start_sigma, dd.stop_sigma = 1e30, -1.0
    dd.target_sites1, dd.target_sites2 = [40], [50]
    dd.target_sites3, dd.target_sites4 = [60], [70]

    spec = build_spec(angle_restraints=[ad], dihedral_restraints=[dd])
    assert {int(x) for x in spec.active_sites} == {
        10,
        11,
        12,
        20,
        30,
        31,
        40,
        50,
        60,
        70,
    }

    g2l = {int(g): i for i, g in enumerate(spec.active_sites)}
    ga = spec.group_angle
    assert ga is not None
    assert ga.grp1_idx.shape[1] == 3  # max_grp = largest group (group1 has 3 atoms)
    assert list(ga.grp1_idx[0]) == [g2l[10], g2l[11], g2l[12]]  # remapped to local
    assert list(ga.grp1_mask[0]) == [1.0, 1.0, 1.0]
    assert int(ga.grp2_idx[0, 0]) == g2l[20]  # group of 1: first slot valid
    assert list(ga.grp2_mask[0]) == [1.0, 0.0, 0.0]  # rest masked padding
    assert float(ga.target1[0]) == 1.0
    assert int(ga.geom_type[0]) == 0  # harmonic
    assert list(ga.move_free[0]) == [1.0, 1.0, 1.0]  # all groups free (default)

    gd = spec.group_dihedral
    assert gd is not None
    assert int(gd.grp4_idx[0, 0]) == g2l[70]
    assert float(gd.target1[0]) == 2.0
    assert spec.has_group_angle() and spec.has_group_dihedral()


def test_vdw_config_fixed_background():
    m, c = _ethane()
    n = m.GetNumAtoms()
    lc = LigandConf(
        mol=m, conf_coords=c, global_indices=np.arange(n), conformer_restraints=True
    )

    n_atom = n + 3
    elements = np.zeros(n_atom, dtype=np.int64)
    for i, atom in enumerate(m.GetAtoms()):
        elements[i] = atom.GetAtomicNum()
    elements[n] = 6  # protein heavy atom
    elements[n + 1] = 7  # protein heavy atom
    # elements[n + 2] stays 0 -> padding, excluded from the VdW background

    spec = build_spec([lc], [], {"vdw": {"weight": 1.0}}, elements=elements)
    assert spec.vdw_config is not None
    # every ligand atom is optimisable and addressed by a local index
    assert len(spec.vdw_config.ligand_local) == n
    assert int(spec.vdw_config.ligand_local.max()) < spec.n_active
    # fixed background = non-padding atoms NOT in active_sites; padding (n+2) excluded
    assert {int(x) for x in spec.vdw_config.background_global} == {n, n + 1}
    assert spec.vdw_config.background_radii.shape == (2,)

    # disabled when weight <= 0
    spec0 = build_spec([lc], [], {"vdw": {"weight": 0.0}}, elements=elements)
    assert spec0.vdw_config is None


def test_intramolecular_vdw_static_arrays():
    """vdw mode=intramolecular builds a static spec.vdw (works in jax/numpy too),
    not the dynamic fixed-background vdw_config."""
    m = Chem.MolFromSmiles("CCCC")  # butane: only C1-C4 has topological dist > 2
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=1)
    m = Chem.RemoveHs(m)  # 4 heavy atoms
    n = m.GetNumAtoms()
    c = np.asarray(m.GetConformer().GetPositions())
    lc = LigandConf(
        mol=m, conf_coords=c, global_indices=np.arange(n), conformer_restraints=True
    )

    spec = build_spec(
        [lc], [], {"vdw": {"weight": 1.0, "mode": "intramolecular", "dmax": 10.0}}
    )
    # static VdwArrays, not the dynamic fixed-background config
    assert spec.vdw is not None
    assert spec.vdw_config is None
    # butane: the only non-bonded heavy pair is C1-C4 (topological distance 3)
    assert spec.vdw.idx.shape == (1, 2)
    assert float(spec.vdw.weight[0]) == 1.0
    assert int(spec.vdw.idx.max()) < spec.n_active  # valid local indices

    # explicit mode=intermolecular keeps ONLY the dynamic/inter paths (no static intra);
    # a single ligand has no inter pairs, so spec.vdw stays None
    spec_dyn = build_spec([lc], [], {"vdw": {"weight": 1.0, "mode": "intermolecular"}})
    assert spec_dyn.vdw is None


def test_vdw_both_modes_compose():
    """vdw mode='both' builds the static intramolecular spec.vdw AND the dynamic
    fixed-background vdw_config together (separate spec fields, scored independently)."""
    m = Chem.MolFromSmiles("CCCC")  # butane: one non-bonded heavy pair (C1-C4)
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=1)
    m = Chem.RemoveHs(m)
    n = m.GetNumAtoms()
    c = np.asarray(m.GetConformer().GetPositions())
    lc = LigandConf(
        mol=m, conf_coords=c, global_indices=np.arange(n), conformer_restraints=True
    )
    elements = np.zeros(n + 2, dtype=np.int64)
    for i, atom in enumerate(m.GetAtoms()):
        elements[i] = atom.GetAtomicNum()
    elements[n] = 6  # protein heavy background atoms (not in active_sites)
    elements[n + 1] = 7

    spec = build_spec(
        [lc],
        [],
        {"vdw": {"weight": 1.0, "mode": "both", "dmax": 10.0}},
        elements=elements,
    )
    # BOTH flavours present and independent
    assert spec.vdw is not None  # static intramolecular (all backends)
    assert spec.vdw.idx.shape == (1, 2)  # butane C1-C4
    assert spec.vdw_config is not None  # dynamic fixed-background (torch)
    assert {int(x) for x in spec.vdw_config.background_global} == {n, n + 1}


def test_vdw_unknown_mode_raises():
    """An unknown vdw mode raises, not a silent fallback to ligand_protein."""
    m, c = _ethane()
    lc = LigandConf(
        mol=m,
        conf_coords=c,
        global_indices=np.arange(m.GetNumAtoms()),
        conformer_restraints=True,
    )
    with pytest.raises(ValueError, match="vdw mode must be"):
        build_spec([lc], [], {"vdw": {"weight": 1.0, "mode": "typo"}})


def _lig_heavy(smi: str) -> LigandConf:
    """A conformer-restrained LigandConf from SMILES, H-removed (heavy atoms only) like
    the real adapters produce."""
    m = Chem.MolFromSmiles(smi)
    mh = Chem.AddHs(m)
    AllChem.EmbedMolecule(mh, randomSeed=1)
    AllChem.UFFOptimizeMolecule(mh)
    m = Chem.RemoveHs(mh)
    crds = np.asarray(m.GetConformer(0).GetPositions(), dtype=np.float64)
    return LigandConf(
        mol=m,
        conf_coords=crds,
        global_indices=np.arange(m.GetNumAtoms()),
        conformer_restraints=True,
    )


def test_plane_perception():
    """plane fires on whole planar atom GROUPS (servalcat-style best-fit plane):
    aromatic/conjugated rings + non-ring sp2 groups, each CONFIRMED coplanar in the
    reference conformer (so aromaticity flags are never trusted). Variable group size is
    a padded (n_plane, max_atoms) idx + grp_mask. OFF unless explicitly opted in."""
    cfg = {"plane": {"weight": 1.0}, "cistrans": {"weight": 1.0}}
    nrow = lambda a: 0 if a is None else len(a.idx)  # noqa: E731

    # fumarate: 2 carboxyl groups (each carboxyl C + O + O + alkene C = 4 coplanar atoms)
    # -> plane=2; no ring. The C=C alkene centre has only 2 heavy neighbours (a 3-atom,
    # trivially-planar group, skipped) so the bond's E/Z is held by cistrans (=1).
    spec = build_spec([_lig_heavy(r"OC(=O)/C=C/C(=O)O")], [], cfg)
    assert nrow(spec.plane) == 2
    assert nrow(spec.cistrans) == 1
    assert spec.plane.idx.shape == (2, 4)  # two 4-atom groups
    assert (spec.plane.grp_mask.sum(axis=1) == 4).all()  # no padding for 4-atom groups
    assert int(spec.plane.idx.max()) < spec.n_active  # valid local indices

    # acetamide amide group (carbonyl C + methyl C + O + N = 4 coplanar atoms) -> plane=1
    assert nrow(build_spec([_lig_heavy("CC(=O)N")], [], cfg).plane) == 1

    # aromatic ring: NOW restrained (the whole 6-membered ring is one plane group). This
    # is the servalcat behaviour the old per-centre signed-volume term could not do (a
    # ring CH has only 2 heavy neighbours). Detected via ring topology + reference
    # coplanarity, NOT GetIsAromatic (benzene here even fails to kekulize, yet is caught).
    bz = build_spec([_lig_heavy("c1ccccc1")], [], cfg).plane
    assert nrow(bz) == 1 and int(bz.grp_mask.sum()) == 6  # one 6-atom ring group

    # fused rings: ATP's adenine is a 6+5 fused bicyclic -> two ring plane groups; the
    # puckered ribose + tetrahedral phosphate are NOT coplanar so contribute nothing.
    atp = build_spec(
        [_lig_heavy("Nc1ncnc2n(cnc12)C3OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C3O")],
        [],
        cfg,
    ).plane
    assert nrow(atp) == 2
    assert sorted(int(m.sum()) for m in atp.grp_mask) == [5, 6]

    # saturated ring dropped: cyclohexane is puckered (chair), so its reference is NOT
    # coplanar (max out-of-plane deviation > _PLANE_TOL) and no plane fires.
    assert build_spec([_lig_heavy("C1CCCCC1")], [], cfg).plane is None

    # OFF by default: no plane key -> no plane term even alongside other conformer terms
    # (preserves every existing conformer run).
    off = build_spec([_lig_heavy("CC(=O)N")], [], {"bond": {"weight": 0.05}})
    assert off.plane is None and off.bond is not None


def _lig_heavy_at(smi: str, base: int, seed: int = 1) -> tuple[LigandConf, int]:
    """Like ``_lig_heavy`` but with a global-index offset ``base`` so several ligands get
    disjoint global indices (what the adapters guarantee per structure)."""
    m = Chem.MolFromSmiles(smi)
    mh = Chem.AddHs(m)
    AllChem.EmbedMolecule(mh, randomSeed=seed)
    AllChem.UFFOptimizeMolecule(mh)
    m = Chem.RemoveHs(mh)
    n = m.GetNumAtoms()
    crds = np.asarray(m.GetConformer(0).GetPositions(), dtype=np.float64)
    lc = LigandConf(
        mol=m,
        conf_coords=crds,
        global_indices=np.arange(n) + base,
        conformer_restraints=True,
    )
    return lc, n


def test_interligand_vdw_default_both():
    """Two restrained ligands + the DEFAULT mode='both' add inter-ligand VdW pairs to
    spec.vdw: the cross product of the two ligands' atoms (ethane has no intramolecular
    pair — its two carbons are bonded — so spec.vdw is purely inter). Every pair crosses
    the two ligands' atom sets, and no protein background means vdw_config is None."""
    lcA, n = _lig_heavy_at("CC", base=0)
    lcB, _ = _lig_heavy_at("CC", base=100)
    spec = build_spec([lcA, lcB], [], {"vdw": {"weight": 1.0}})
    assert spec.vdw is not None
    assert spec.vdw_config is None  # no elements -> no protein background
    assert spec.vdw.idx.shape == (n * n, 2)  # inter cross product only
    assert int(spec.vdw.idx.max()) < spec.n_active  # valid local indices
    # every pair has one local index from each ligand's block (crosses molecules)
    g2l = {int(g): i for i, g in enumerate(spec.active_sites)}
    locA = {g2l[g] for g in range(n)}
    locB = {g2l[g] for g in range(100, 100 + n)}
    for li, lj in spec.vdw.idx:
        a, b = int(li), int(lj)
        assert (a in locA and b in locB) or (a in locB and b in locA)


def test_interligand_vdw_composes_with_intra():
    """mode='both' CONCATENATES intra (per ligand) + inter (cross) pairs into one
    spec.vdw: two butanes give 2 intra (one C1-C4 each) + n*n inter."""
    lcA, n = _lig_heavy_at("CCCC", base=0, seed=1)  # butane: 1 intra pair (C1-C4)
    lcB, _ = _lig_heavy_at("CCCC", base=100, seed=2)
    spec = build_spec([lcA, lcB], [], {"vdw": {"weight": 1.0, "dmax": 10.0}})
    assert spec.vdw.idx.shape == (2 + n * n, 2)


def test_interligand_vdw_only_in_both_mode():
    """Inter-ligand pairs ride the default 'both' only: explicit 'intramolecular' and the
    single-ligand case never produce them (ethane intra=0 -> spec.vdw stays None)."""
    lcA, _ = _lig_heavy_at("CC", base=0)
    lcB, _ = _lig_heavy_at("CC", base=100)
    # explicit intramolecular: no inter, ethane has no intra pair -> spec.vdw None
    spec_intra = build_spec(
        [lcA, lcB], [], {"vdw": {"weight": 1.0, "mode": "intramolecular"}}
    )
    assert spec_intra.vdw is None
    # single ligand under 'both': inter needs >=2 ligands -> none built
    assert build_spec([lcA], [], {"vdw": {"weight": 1.0}}).vdw is None


def test_vdw_mode_intermolecular_excludes_intra():
    """mode='intermolecular' = fixed background + inter-ligand, but NO intramolecular: two
    butanes (each would add 1 intra C1-C4 pair under 'both') give ONLY the n*n inter cross
    pairs in spec.vdw, plus a fixed-background vdw_config from a non-active heavy atom."""
    lcA, n = _lig_heavy_at("CCCC", base=0, seed=1)
    lcB, _ = _lig_heavy_at("CCCC", base=4, seed=2)
    elements = np.zeros(9, dtype=np.int64)
    elements[:8] = 6  # the two butanes (in active_sites)
    elements[8] = 7  # a background heavy atom (NOT in active_sites)
    spec = build_spec(
        [lcA, lcB],
        [],
        {"vdw": {"weight": 1.0, "mode": "intermolecular"}},
        elements=elements,
    )
    assert spec.vdw is not None
    assert spec.vdw.idx.shape == (n * n, 2)  # inter only, NO intra (would be 2 + n*n)
    assert spec.vdw_config is not None  # fixed-background half present
    assert {int(x) for x in spec.vdw_config.background_global} == {8}


def test_vdw_mode_ligand_protein_removed():
    """The old 'ligand_protein' mode value is removed -> raises a migration hint pointing
    to 'intermolecular' (mirrors the rejected `backend:` key)."""
    lcA, _ = _lig_heavy_at("CC", base=0)
    with pytest.raises(ValueError, match="ligand_protein.*renamed to 'intermolecular'"):
        build_spec([lcA], [], {"vdw": {"weight": 1.0, "mode": "ligand_protein"}})


@pytest.mark.parametrize("term,default", [("bond", 0.0), ("chiral", 0.05)])
def test_conf_slack_null_handling_uniform(term, default):
    """slack: omitted/null -> per-term default; explicit 0 -> 0.0 (the truthiness trap).

    Guards the consistency fix that routed all conformer terms through _conf_slack so the
    null/zero handling can't drift (it previously diverged: only some terms had an `or 0.0`
    guard, so `slack: null` crashed bond/angle and silently zeroed a non-zero default). The
    chiral case (default 0.05) exercises the trap that a zero default cannot."""
    from rgi_utils.featurizer import _conf_slack

    assert _conf_slack({}, term, default) == default  # key absent
    assert _conf_slack({term: {}}, term, default) == default  # slack omitted
    assert _conf_slack({term: None}, term, default) == default  # bare key, None body
    assert _conf_slack({term: {"slack": None}}, term, default) == default  # null
    assert _conf_slack({term: {"slack": 0}}, term, default) == 0.0  # explicit 0 kept
    assert _conf_slack({term: {"slack": 0.3}}, term, default) == pytest.approx(0.3)
