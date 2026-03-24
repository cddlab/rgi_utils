import numpy as np
import pytest

from rgi_utils.atom_context import AtomRecord
from rgi_utils.distance_restr_data import DistanceData


class MockAdapter:
    def __init__(self, atoms: list[AtomRecord]):
        self._atoms = atoms

    def iter_atoms(self):
        yield from self._atoms


def _make_dd(config: dict) -> DistanceData:
    dd = DistanceData()
    dd.set_config(config)
    return dd


def _harmonic(target: float) -> DistanceData:
    return _make_dd(
        {
            "atom_selection1": "chain A",
            "atom_selection2": "chain B",
            "harmonic": {"target_distance": target},
        }
    )


def _flat_bottomed(d1: float, d2: float) -> DistanceData:
    return _make_dd(
        {
            "atom_selection1": "chain A",
            "atom_selection2": "chain B",
            "flat-bottomed": {"target_distance1": d1, "target_distance2": d2},
        }
    )


def _set_sites(dd: DistanceData, s1: list[int], s2: list[int]) -> None:
    dd.target_local_sites1 = list(s1)
    dd.target_local_sites2 = list(s2)


class TestDistanceDataSetConfig:
    def test_harmonic_config(self):
        dd = _harmonic(10.0)
        assert dd.distance_restraint_type == "harmonic"
        assert dd.target_distance == pytest.approx(10.0)
        assert dd.run_restr is True

    def test_flat_bottomed_config(self):
        dd = _flat_bottomed(3.0, 10.0)
        assert dd.distance_restraint_type == "flat-bottomed"
        assert dd.target_distance1 == pytest.approx(3.0)
        assert dd.target_distance2 == pytest.approx(10.0)

    def test_flat_bottomed1_config(self):
        dd = _make_dd(
            {
                "atom_selection1": "chain A",
                "atom_selection2": "chain B",
                "flat-bottomed1": {"target_distance1": 5.0},
            }
        )
        assert dd.distance_restraint_type == "flat-bottomed1"
        assert dd.target_distance1 == pytest.approx(5.0)

    def test_flat_bottomed2_config(self):
        dd = _make_dd(
            {
                "atom_selection1": "chain A",
                "atom_selection2": "chain B",
                "flat-bottomed2": {"target_distance2": 8.0},
            }
        )
        assert dd.distance_restraint_type == "flat-bottomed2"
        assert dd.target_distance2 == pytest.approx(8.0)

    def test_flat_bottomed_raises_when_d1_gt_d2(self):
        with pytest.raises(ValueError, match="target_distance1 must be smaller"):
            _flat_bottomed(10.0, 3.0)

    def test_invalid_calc_method_raises(self):
        with pytest.raises(ValueError, match="calc_method"):
            _make_dd(
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "calc_method": "invalid",
                    "harmonic": {"target_distance": 5.0},
                }
            )

    def test_is_valid_after_set_config(self):
        dd = _harmonic(5.0)
        assert dd.is_valid() is True


class TestDistanceDataCalc:
    def test_harmonic_zero_at_target(self):
        dd = _harmonic(5.0)
        _set_sites(dd, [0], [1])
        crds = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        assert dd.calc(crds) == pytest.approx(0.0)

    def test_harmonic_nonzero_away_from_target(self):
        dd = _harmonic(5.0)
        _set_sites(dd, [0], [1])
        crds = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
        # delta = 8 - 5 = 3 → energy = 9
        assert dd.calc(crds) == pytest.approx(9.0)

    def test_flat_bottomed_zero_inside(self):
        dd = _flat_bottomed(3.0, 10.0)
        _set_sites(dd, [0], [1])
        crds = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        assert dd.calc(crds) == pytest.approx(0.0)

    def test_flat_bottomed_energy_below_d1(self):
        dd = _flat_bottomed(3.0, 10.0)
        _set_sites(dd, [0], [1])
        # dist=1, delta = 1 - 3 = -2 → energy = 4
        crds = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert dd.calc(crds) == pytest.approx(4.0)

    def test_flat_bottomed_energy_above_d2(self):
        dd = _flat_bottomed(3.0, 10.0)
        _set_sites(dd, [0], [1])
        # dist=12, delta = 12 - 10 = 2 → energy = 4
        crds = np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
        assert dd.calc(crds) == pytest.approx(4.0)

    def test_flat_bottomed1_energy_below_d1(self):
        dd = _make_dd(
            {
                "atom_selection1": "chain A",
                "atom_selection2": "chain B",
                "flat-bottomed1": {"target_distance1": 5.0},
            }
        )
        _set_sites(dd, [0], [1])
        # dist=2, delta = 2 - 5 = -3 → energy = 9
        crds = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        assert dd.calc(crds) == pytest.approx(9.0)

    def test_flat_bottomed1_zero_above_d1(self):
        dd = _make_dd(
            {
                "atom_selection1": "chain A",
                "atom_selection2": "chain B",
                "flat-bottomed1": {"target_distance1": 5.0},
            }
        )
        _set_sites(dd, [0], [1])
        crds = np.array([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
        assert dd.calc(crds) == pytest.approx(0.0)

    def test_flat_bottomed2_energy_above_d2(self):
        dd = _make_dd(
            {
                "atom_selection1": "chain A",
                "atom_selection2": "chain B",
                "flat-bottomed2": {"target_distance2": 5.0},
            }
        )
        _set_sites(dd, [0], [1])
        # dist=8, delta = 8 - 5 = 3 → energy = 9
        crds = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
        assert dd.calc(crds) == pytest.approx(9.0)

    def test_flat_bottomed2_zero_below_d2(self):
        dd = _make_dd(
            {
                "atom_selection1": "chain A",
                "atom_selection2": "chain B",
                "flat-bottomed2": {"target_distance2": 5.0},
            }
        )
        _set_sites(dd, [0], [1])
        crds = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        assert dd.calc(crds) == pytest.approx(0.0)

    def test_com_of_multiple_atoms(self):
        # Two atoms per site: COM is midpoint
        dd = _harmonic(0.0)
        _set_sites(dd, [0, 1], [2, 3])
        # COM1 = (0+2)/2 = 1, COM2 = (4+6)/2 = 5, dist = 4
        crds = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
        )
        # delta = 4 - 0 = 4 → energy = 16
        assert dd.calc(crds) == pytest.approx(16.0)

    def test_calc_sd_harmonic(self):
        dd = _harmonic(5.0)
        _set_sites(dd, [0], [1])
        crds = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
        # dist=8, delta = 8-5=3, sd = 9
        assert dd.calc_sd(crds) == pytest.approx(9.0)


class TestDistanceDataGrad:
    def test_harmonic_grad_zero_at_target(self):
        dd = _harmonic(5.0)
        _set_sites(dd, [0], [1])
        crds = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        grad = np.zeros_like(crds)
        dd.grad(crds, grad)
        assert np.allclose(grad, 0.0, atol=1e-9)

    def test_harmonic_grad_direction_when_too_close(self):
        # dist < target: gradient at site1 is positive along connecting vector
        # (optimizer will move site1 away from site2)
        dd = _harmonic(10.0)
        _set_sites(dd, [0], [1])
        crds = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        grad = np.zeros_like(crds)
        dd.grad(crds, grad)
        # dE/d(site1_x) > 0: site1 should move in -x (away from site2)
        assert grad[0, 0] > 0
        # dE/d(site2_x) < 0: site2 should move in +x (away from site1)
        assert grad[1, 0] < 0

    def test_flat_bottomed_no_grad_inside(self):
        dd = _flat_bottomed(3.0, 10.0)
        _set_sites(dd, [0], [1])
        crds = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        grad = np.zeros_like(crds)
        dd.grad(crds, grad)
        assert np.allclose(grad, 0.0, atol=1e-9)


class TestDistanceDataResolveSites:
    def test_resolve_sites_maps_correct_atoms(self):
        dd = _make_dd(
            {
                "atom_selection1": "chain A and resid 1",
                "atom_selection2": "chain B",
                "harmonic": {"target_distance": 5.0},
            }
        )
        atoms = [
            AtomRecord(chain="A", resid=1, index=0),
            AtomRecord(chain="A", resid=2, index=1),
            AtomRecord(chain="B", resid=1, index=2),
        ]
        dd.resolve_sites(MockAdapter(atoms))
        assert dd.target_sites1 == [0]
        assert dd.target_sites2 == [2]

    def test_resolve_sites_empty_selection_raises(self):
        dd = _make_dd(
            {
                "atom_selection1": "chain Z",
                "atom_selection2": "chain B",
                "harmonic": {"target_distance": 5.0},
            }
        )
        atoms = [AtomRecord(chain="A", resid=1, index=0)]
        with pytest.raises(AssertionError):
            dd.resolve_sites(MockAdapter(atoms))
