import numpy as np
import pytest
import torch

from rgi_utils import CombinedRestraints
from rgi_utils.atom_context import AtomRecord
from rgi_utils.bond_restr_data import BondData


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


class TestCombinedRestraintsCalcGrad:
    """Integration tests for CombinedRestraints.calc() and grad()."""

    def setup_method(self):
        _reset_singleton()

    def _cr_with_bond(self, r0: float, w: float = 1.0) -> CombinedRestraints:
        cr = CombinedRestraints.get_instance()
        cr.set_config({})
        cr.bond_data.append(BondData(0, 1, r0=r0, w=w))
        cr.nbatch = 1
        cr.natoms = 2
        return cr

    def test_calc_zero_at_ideal(self):
        cr = self._cr_with_bond(r0=2.0)
        crds = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]).reshape(-1)
        assert cr.calc(crds) == pytest.approx(0.0)

    def test_calc_nonzero_when_stretched(self):
        cr = self._cr_with_bond(r0=1.0)
        crds = np.array([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]).reshape(-1)
        # w=1, delta=2 → energy = 4
        assert cr.calc(crds) == pytest.approx(4.0)

    def test_grad_nonzero_when_stretched(self):
        cr = self._cr_with_bond(r0=1.0)
        crds = np.array([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]).reshape(-1)
        g = cr.grad(crds)
        assert np.any(np.abs(g) > 0)

    def test_reset_indices_clears_bond_atom_refs(self):
        cr = CombinedRestraints.get_instance()
        cr.set_config({})
        b = BondData(5, 7, r0=1.0)
        cr.bond_data.append(b)
        cr.reset_indices()
        assert b.aid0 == -1
        assert b.aid1 == -1
