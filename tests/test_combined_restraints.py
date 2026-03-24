import math
import numpy as np
import torch
import pytest
from rgi_utils import CombinedRestraints
from rgi_utils.atom_context import AtomRecord


class MockAdapter:
    """Mock FrameworkAdapter for testing without boltz."""

    def __init__(self, atoms: list[AtomRecord]):
        self._atoms = atoms

    def iter_atoms(self):
        yield from self._atoms


def _reset_singleton():
    """Reset CombinedRestraints singleton between tests."""
    CombinedRestraints._instance = None


def _make_trivial_tensors(natoms: int):
    """Make minimal tensors for setup_site (CPU, no GPU mode)."""
    conformer_restraint = torch.zeros(1, natoms, dtype=torch.long)
    atom_pad_mask = torch.ones(1, natoms, dtype=torch.bool)
    ref_element = torch.zeros(1, natoms, 128)
    return conformer_restraint, atom_pad_mask, ref_element


class TestCombinedRestraintsSetup:
    def setup_method(self):
        _reset_singleton()

    def test_singleton(self):
        a = CombinedRestraints.get_instance()
        b = CombinedRestraints.get_instance()
        assert a is b

    def test_set_config_defaults(self):
        cr = CombinedRestraints.get_instance()
        cr.set_config({})
        assert cr.verbose is False
        assert cr.gpu is False
        assert cr.method == "CG"

    def test_setup_empty_no_error(self):
        """setup with no conformer or distance restraints should not raise."""
        cr = CombinedRestraints.get_instance()
        cr.set_config({})

        atoms = [
            AtomRecord(chain="A", resid=1, index=0),
            AtomRecord(chain="A", resid=2, index=1),
        ]
        adapter = MockAdapter(atoms)
        conformer_restraint, atom_pad_mask, ref_element = _make_trivial_tensors(2)
        cr.setup(adapter, conformer_restraint, atom_pad_mask, ref_element, nbatch=1)
        # active_sites is empty (no conformer restraints) → early return
        assert cr.active_sites == [] or len(cr.active_sites) == 0


class TestCombinedRestraintsMinimize:
    def setup_method(self):
        _reset_singleton()

    def test_minimize_skipped_above_start_sigma(self):
        """minimize should be no-op when sigma_t > start_sigma."""
        cr = CombinedRestraints.get_instance()
        cr.set_config({"start_sigma": 1.0})

        cr.active_sites = [0, 1]
        cr.nbatch = 1
        cr.natoms = 2

        crds = torch.rand(1, 5, 3)
        original = crds.clone()
        cr.minimize(crds, istep=0, sigma_t=2.0)  # sigma_t > start_sigma
        assert torch.allclose(crds, original)


class TestDistanceRestraintIntegration:
    def setup_method(self):
        _reset_singleton()

    def test_resolve_sites_selects_correct_atoms(self):
        """Distance restraint should select atoms matching atom_selection."""
        cr = CombinedRestraints.get_instance()
        config = {
            "start_sigma": 999999,
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "harmonic": {"target_distance": 10.0},
                }
            ],
        }
        cr.set_config(config)

        atoms = [
            AtomRecord(chain="A", resid=1, index=0),
            AtomRecord(chain="A", resid=2, index=1),
            AtomRecord(chain="B", resid=1, index=2),
            AtomRecord(chain="B", resid=2, index=3),
        ]
        adapter = MockAdapter(atoms)

        dist_restr = cr.distance_data[0]
        dist_restr.resolve_sites(adapter)

        assert set(dist_restr.target_sites1) == {0, 1}
        assert set(dist_restr.target_sites2) == {2, 3}
