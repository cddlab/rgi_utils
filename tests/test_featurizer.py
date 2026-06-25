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


def test_vdw_config_protein_background():
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
    # protein background = heavy atoms NOT in active_sites; padding (n+2) excluded
    assert {int(x) for x in spec.vdw_config.protein_global} == {n, n + 1}
    assert spec.vdw_config.protein_radii.shape == (2,)

    # disabled when weight <= 0
    spec0 = build_spec([lc], [], {"vdw": {"weight": 0.0}}, elements=elements)
    assert spec0.vdw_config is None


def test_intramolecular_vdw_static_arrays():
    """vdw mode=intramolecular builds a static spec.vdw (works in jax/numpy too),
    not the dynamic ligand-protein vdw_config."""
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
    # static VdwArrays, not the dynamic ligand-protein config
    assert spec.vdw is not None
    assert spec.vdw_config is None
    # butane: the only non-bonded heavy pair is C1-C4 (topological distance 3)
    assert spec.vdw.idx.shape == (1, 2)
    assert float(spec.vdw.weight[0]) == 1.0
    assert int(spec.vdw.idx.max()) < spec.n_active  # valid local indices

    # explicit mode=ligand_protein keeps ONLY the dynamic path (no static vdw); the
    # default is now "both", so request ligand_protein to isolate it
    spec_dyn = build_spec([lc], [], {"vdw": {"weight": 1.0, "mode": "ligand_protein"}})
    assert spec_dyn.vdw is None


def test_vdw_both_modes_compose():
    """vdw mode='both' builds the static intramolecular spec.vdw AND the dynamic
    ligand-protein vdw_config together (separate spec fields, scored independently)."""
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
    assert spec.vdw_config is not None  # dynamic ligand-protein (torch)
    assert {int(x) for x in spec.vdw_config.protein_global} == {n, n + 1}


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


def test_improper_perception():
    """improper (sp2 planarity) fires on acyclic, non-aromatic double-bond endpoints
    with exactly 3 heavy neighbours, reusing chiral's signed volume. Parity-safe scope:
    aromatic + in-ring double bonds are excluded (mirrors cistrans's bond filter), and
    the term is OFF unless explicitly opted in."""
    cfg = {"improper": {"weight": 1.0}, "cistrans": {"weight": 1.0}}
    nrow = lambda a: 0 if a is None else len(a.idx)  # noqa: E731

    # fumarate: 2 carboxyl carbons (exocyclic C=O, 3 heavy neighbours) -> improper=2;
    # the C=C alkene carbons have only 2 heavy neighbours each, so the bond's E/Z is held
    # by cistrans (=1), not improper.
    spec = build_spec([_lig_heavy(r"OC(=O)/C=C/C(=O)O")], [], cfg)
    assert nrow(spec.improper) == 2
    assert nrow(spec.cistrans) == 1
    assert int(spec.improper.idx.max()) < spec.n_active  # valid local indices

    # acetamide carbonyl carbon (CH3-C, O, N = 3 heavy neighbours) -> improper=1
    assert nrow(build_spec([_lig_heavy("CC(=O)N")], [], cfg).improper) == 1

    # aromatic centres excluded (SanitizeMol is best-effort -> don't trust GetIsAromatic
    # alone; here it correctly perceives the ring as aromatic and the term stays empty)
    assert build_spec([_lig_heavy("c1ccccc1")], [], cfg).improper is None
    # in-ring non-aromatic C=C excluded by the topological IsInRing guard (parity-safe)
    assert build_spec([_lig_heavy("C1CCC=CC1")], [], cfg).improper is None

    # OFF by default: no improper key -> no improper term even alongside other conformer
    # terms (preserves every existing conformer run).
    off = build_spec([_lig_heavy("CC(=O)N")], [], {"bond": {"weight": 0.05}})
    assert off.improper is None and off.bond is not None


@pytest.mark.parametrize("term,default", [("bond", 0.0), ("improper", 0.05)])
def test_conf_slack_null_handling_uniform(term, default):
    """slack: omitted/null -> per-term default; explicit 0 -> 0.0 (the truthiness trap).

    Guards the consistency fix that routed all five conformer terms through _conf_slack so
    the null/zero handling can't drift (it previously diverged: only cistrans/improper had
    an `or 0.0` guard, so `slack: null` crashed bond/angle/chiral and silently zeroed
    improper's 0.05 default)."""
    from rgi_utils.featurizer import _conf_slack

    assert _conf_slack({}, term, default) == default  # key absent
    assert _conf_slack({term: {}}, term, default) == default  # slack omitted
    assert _conf_slack({term: None}, term, default) == default  # bare key, None body
    assert _conf_slack({term: {"slack": None}}, term, default) == default  # null
    assert _conf_slack({term: {"slack": 0}}, term, default) == 0.0  # explicit 0 kept
    assert _conf_slack({term: {"slack": 0.3}}, term, default) == pytest.approx(0.3)
