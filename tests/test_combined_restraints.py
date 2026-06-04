"""Integration tests for the new CombinedRestraints (set_config / setup / minimize).

Uses the numpy backend (CPU, no GPU) with a mock adapter, exercising the full
flow: config parse -> distance resolution -> conformer spec build -> minimize.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rgi_utils import CombinedRestraints
from rgi_utils.atom_context import AtomRecord, LigandConf


class MockAdapter:
    """Mock FrameworkAdapter + ConformerAdapter for testing without a framework."""

    def __init__(self, atoms, ligand_confs=None):
        self._atoms = atoms
        self._ligand_confs = ligand_confs or []

    def iter_atoms(self):
        yield from self._atoms

    def iter_ligand_confs(self):
        yield from self._ligand_confs

    def num_atoms(self):
        return max((a.index for a in self._atoms), default=-1) + 1


@pytest.fixture(autouse=True)
def _reset_singleton():
    CombinedRestraints.reset()
    yield
    CombinedRestraints.reset()


def test_singleton():
    assert CombinedRestraints.get_instance() is CombinedRestraints.get_instance()


def test_config_defaults():
    cr = CombinedRestraints.get_instance()
    cr.set_config({})
    assert cr.config.verbose is False
    assert cr.config.gpu is False
    assert cr.config.method == "CG"
    assert cr.config.resolve_backend() == "numpy"


def test_start_sigma_validation():
    """Top-level start_sigma is rejected; omitting it per restraint defaults to +inf
    (active at every step); explicit per-restraint values are honored."""
    import math

    cr = CombinedRestraints.get_instance()
    dist = {
        "atom_selection1": "chain A",
        "atom_selection2": "chain B",
        "start_sigma": 1e30,
        "harmonic": {"target_distance": 5.0},
    }
    # (a) top-level start_sigma is rejected
    with pytest.raises(ValueError, match="top-level"):
        cr.set_config({"start_sigma": 1e30, "distance_restraints_config": [dist]})
    # (b) a distance entry without start_sigma defaults to +inf (every step)
    no_ss = {k: v for k, v in dist.items() if k != "start_sigma"}
    cr.set_config({"distance_restraints_config": [no_ss]})
    assert math.isinf(cr.config.distance_data[0].start_sigma)
    # (c) conformer terms without start_sigma default to +inf (every step)
    cr.set_config({"conformer_restraints_config": {"bond": {"weight": 1.0}}})
    assert math.isinf(cr.config.conf_start_sigma)
    # (d) explicit per-restraint values are honored
    cr.set_config(
        {
            "distance_restraints_config": [dist],
            "conformer_restraints_config": {"start_sigma": 1.0, "bond": {"weight": 1.0}},
        }
    )
    assert cr.config.conf_start_sigma == 1.0


def test_setup_empty_is_inactive():
    cr = CombinedRestraints.get_instance()
    cr.set_config({})
    cr.setup(MockAdapter([AtomRecord("A", 1, 0), AtomRecord("A", 2, 1)]))
    assert not cr.is_active()
    coords = np.zeros((1, 2, 3))
    # minimize is a no-op and must not raise
    cr.minimize(coords, 0, sigma=0.0)


def test_distance_resolve_and_minimize():
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "backend": "numpy",
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "start_sigma": 1e30,
                    "harmonic": {"target_distance": 5.0},
                }
            ],
        }
    )
    atoms = [
        AtomRecord("A", 1, 0),
        AtomRecord("A", 2, 1),
        AtomRecord("B", 1, 2),
        AtomRecord("B", 2, 3),
    ]
    cr.setup(MockAdapter(atoms))
    assert cr.is_active()
    dd = cr.config.distance_data[0]
    assert set(dd.target_sites1) == {0, 1}
    assert set(dd.target_sites2) == {2, 3}

    coords = np.zeros((1, 4, 3))
    coords[0, 2:, 0] = 20.0  # group B far away on x
    d0 = np.linalg.norm(coords[0, 2:].mean(0) - coords[0, :2].mean(0))
    cr.minimize(coords, 0, sigma=0.0)
    d1 = np.linalg.norm(coords[0, 2:].mean(0) - coords[0, :2].mean(0))
    assert abs(d1 - 5.0) < abs(d0 - 5.0)


def test_minimize_skipped_above_start_sigma():
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "backend": "numpy",
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "start_sigma": 1.0,
                    "harmonic": {"target_distance": 5.0},
                }
            ],
        }
    )
    cr.setup(MockAdapter([AtomRecord("A", 1, 0), AtomRecord("B", 1, 1)]))
    coords = np.zeros((1, 2, 3))
    coords[0, 1, 0] = 20.0
    original = coords.copy()
    cr.minimize(coords, 0, sigma=2.0)  # sigma > start_sigma -> skip
    assert np.allclose(coords, original)


def test_multiligand_conformer_setup():
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "backend": "numpy",
            "conformer_restraints_config": {"start_sigma": 1e30, "bond": {"weight": 0.1}},
        }
    )
    m = Chem.AddHs(Chem.MolFromSmiles("CC"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    c = np.asarray(m.GetConformer().GetPositions())
    n = m.GetNumAtoms()
    lcs = [LigandConf(m, c, np.arange(n)), LigandConf(m, c, np.arange(n) + n)]
    atoms = [AtomRecord("A", i + 1, i) for i in range(2 * n)]
    cr.setup(MockAdapter(atoms, ligand_confs=lcs))
    assert cr.is_active()
    assert cr.spec.n_active == 2 * n
    assert cr.spec.bond.idx.shape[0] == 2 * m.GetNumBonds()
