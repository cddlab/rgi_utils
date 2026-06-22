"""Tests for ScanMinimizer (framework-free scan-time minimizer wrapper).

Uses fake rgi + fake minimizer (numpy arrays duck-type for positions), so no jax /
no GPU. Verifies the reshape contract (flatten to (-1, 3) for the minimizer, restore
the original shape) and the duck-typed is_active / n_active / finalize delegation.
"""

from __future__ import annotations

import numpy as np

from rgi_utils.optim.scan_runner import ScanMinimizer


class _FakeSpec:
    def __init__(self, n_active):
        self.n_active = n_active


class _FakeRgi:
    def __init__(self, active=True, n_active=5):
        self._active = active
        self.spec = _FakeSpec(n_active)
        self.finalized = []  # records (flat_shape, istep)

    def is_active(self):
        return self._active

    def finalize(self, flat, istep=0):
        self.finalized.append((np.asarray(flat).shape, istep))


def test_minimize_gpu_flattens_and_restores_shape():
    seen = {}

    def minimizer(flat, sigma, step=0):
        seen["shape"] = flat.shape
        seen["sigma"] = sigma
        return flat + 1.0  # any transform; shape round-trip is what we check

    sm = ScanMinimizer(_FakeRgi(), minimizer)
    pos = np.zeros((4, 3, 3))  # (num_tokens=4, max=3, 3) -> flat (12, 3)
    out = sm.minimize_gpu(pos, 2.5)
    assert seen["shape"] == (12, 3)  # minimizer saw the flat atom axis
    assert seen["sigma"] == 2.5
    assert out.shape == (4, 3, 3)  # restored to the original shape
    assert np.allclose(out, 1.0)  # transform applied through the round-trip


def test_minimize_gpu_shape_agnostic():
    # an already-flat (n, 3) tensor round-trips unchanged in shape
    seen = {}

    def minimizer(flat, sigma, step=0):
        seen["shape"] = flat.shape
        return flat

    sm = ScanMinimizer(_FakeRgi(), minimizer)
    out = sm.minimize_gpu(np.zeros((7, 3)), 1.0)
    assert seen["shape"] == (7, 3)
    assert out.shape == (7, 3)


def test_minimize_gpu_noop_when_no_minimizer():
    sm = ScanMinimizer(_FakeRgi(active=True), None)
    pos = np.arange(6.0).reshape(2, 3)
    out = sm.minimize_gpu(pos, 1.0)
    assert out is pos  # untouched (same object)
    assert sm.is_active() is False  # minimizer None -> inactive


def test_is_active_and_n_active():
    sm = ScanMinimizer(_FakeRgi(active=True, n_active=7), lambda f, s: f)
    assert sm.is_active() is True
    assert sm.n_active == 7
    # inactive rgi -> not active even with a minimizer
    assert ScanMinimizer(_FakeRgi(active=False), lambda f, s: f).is_active() is False


def test_n_active_zero_when_no_spec():
    rgi = _FakeRgi()
    rgi.spec = None
    assert ScanMinimizer(rgi, lambda f, s: f).n_active == 0


def test_finalize_flattens_and_delegates():
    rgi = _FakeRgi()
    ScanMinimizer(rgi, lambda f, s: f).finalize(np.zeros((4, 3, 3)), istep=9)
    assert rgi.finalized == [((12, 3), 9)]


def test_finalize_noop_when_no_minimizer():
    rgi = _FakeRgi()
    ScanMinimizer(rgi, None).finalize(np.zeros((4, 3, 3)))
    assert rgi.finalized == []  # no minimizer -> no finalize call
