import math

import numpy as np
import pytest

from rgi_utils.angle_restr_data import AngleData
from rgi_utils.bond_restr_data import BondData
from rgi_utils.chiral_data import ChiralData, calc_chiral_vol


class TestBondData:
    def _make_bond(self, r0=1.5, w=1.0, slack=0.0):
        b = BondData(0, 1, r0=r0, w=w, slack=slack)
        return b

    def test_calc_zero_at_ideal(self):
        b = self._make_bond(r0=1.5)
        crds = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        assert b.calc(crds) == 0.0

    def test_calc_nonzero_when_stretched(self):
        b = self._make_bond(r0=1.0, w=1.0)
        crds = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        # dist=2.0, delta=2.0-1.0=1.0, ene=1.0*1.0*1.0=1.0
        assert b.calc(crds) == pytest.approx(1.0)

    def test_grad_zero_at_ideal(self):
        b = self._make_bond(r0=1.5)
        crds = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        grad = np.zeros_like(crds)
        b.grad(crds, grad)
        assert np.allclose(grad, 0.0, atol=1e-10)

    def test_grad_nonzero_when_stretched(self):
        b = self._make_bond(r0=1.0, w=1.0)
        crds = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        grad = np.zeros_like(crds)
        b.grad(crds, grad)
        # gradient should be nonzero along x
        assert abs(grad[0, 0]) > 0 or abs(grad[1, 0]) > 0

    def test_slack_allows_range(self):
        b = self._make_bond(r0=1.5, w=1.0, slack=0.2)
        # within slack: no energy
        crds = np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]])
        assert b.calc(crds) == 0.0
        # outside slack: energy > 0
        crds2 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        assert b.calc(crds2) > 0.0


class TestAngleData:
    def _make_angle(self, th0, w=1.0, slack=0.0):
        a = AngleData(0, 1, 2, th0=th0, w=w, slack=slack)
        return a

    def test_calc_zero_at_ideal(self):
        th0 = math.radians(109.5)
        a = self._make_angle(th0=th0, slack=math.radians(1.0))
        # Place atoms at th0 angle
        crds = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [math.cos(th0), math.sin(th0), 0.0],
            ]
        )
        assert a.calc(crds) == 0.0

    def test_calc_nonzero_when_bent(self):
        th0 = math.radians(90.0)
        a = self._make_angle(th0=th0, w=1.0, slack=0.0)
        # 180 degree (straight) angle: deviation = 90 degrees = pi/2 rad
        crds = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert a.calc(crds) > 0.0

    def test_grad_nonzero_when_bent(self):
        th0 = math.radians(90.0)
        a = self._make_angle(th0=th0, w=1.0, slack=0.0)
        # 120-degree angle (non-degenerate): deviation = 30 degrees from ideal 90
        # atom2 at (0.5, sqrt(3)/2, 0) gives 120 degrees at atom1
        crds = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, math.sqrt(3) / 2, 0.0]]
        )
        grad = np.zeros_like(crds)
        a.grad(crds, grad)
        assert np.any(np.abs(grad) > 0)


class TestChiralData:
    def _make_chiral(self, vol):
        c = ChiralData(0, 1, 2, 3, chiral_vol=vol, w=1.0, slack=0.0)
        return c

    def _tetrahedral_crds(self):
        # ideal tetrahedral geometry centered at atom 0
        return np.array(
            [
                [0.0, 0.0, 0.0],  # center
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def test_calc_zero_at_ideal(self):
        crds = self._tetrahedral_crds()
        vol = calc_chiral_vol(crds, 0, [1, 2, 3])
        c = self._make_chiral(vol)
        # With slack=0, energy should be very small (vol matches vol)
        assert c.calc(crds) == pytest.approx(0.0, abs=1e-10)

    def test_calc_nonzero_when_inverted(self):
        crds = self._tetrahedral_crds()
        vol = calc_chiral_vol(crds, 0, [1, 2, 3])
        c = self._make_chiral(vol)
        # Invert z-axis to flip chirality
        crds_inv = crds.copy()
        crds_inv[3] = [0.0, 0.0, -1.0]
        assert c.calc(crds_inv) > 0.0

    def test_grad_nonzero_when_inverted(self):
        crds = self._tetrahedral_crds()
        vol = calc_chiral_vol(crds, 0, [1, 2, 3])
        c = self._make_chiral(vol)
        crds_inv = crds.copy()
        crds_inv[3] = [0.0, 0.0, -1.0]
        grad = np.zeros_like(crds_inv)
        c.grad(crds_inv, grad)
        assert np.any(np.abs(grad) > 0)
