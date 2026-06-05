"""Featurizer tests: multi-ligand has no index collision; distance+conformer union."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from rgi_utils.atom_context import LigandConf
from rgi_utils.distance_restr_data import DistanceData
from rgi_utils.featurizer import build_spec


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

    # default mode keeps the dynamic path (no static vdw; needs elements)
    spec_dyn = build_spec([lc], [], {"vdw": {"weight": 1.0}})
    assert spec_dyn.vdw is None
