"""SciPy CPU optimizer — fallback / debugging path (``gpu: false``).

Uses the numpy reference energy with scipy.optimize (numerical jacobian). Slower
than the GPU backends but dependency-free and handy for mdtraj-based distance
debugging.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import optimize

from rgi_utils.energy import numpy_energy

logger = logging.getLogger(__name__)


class NumpyRestraintOptimizer:
    def __init__(self, spec, max_iter: int = 100, method: str = "CG"):
        self.spec = spec
        self.prepared = numpy_energy.prepare_spec(spec)
        self.active_sites = np.asarray(spec.active_sites)
        self.max_iter = max_iter
        self.method = method

    def minimize(self, coords, sigma=None, start_sigma=None, max_iter=None):
        """Optimize ``coords`` (..., n_atom, 3) numpy array in-place."""
        if not self.spec.is_active():
            return coords
        if sigma is not None and start_sigma is not None and sigma > start_sigma:
            return coords
        active = coords[..., self.active_sites, :]
        shape = active.shape
        prepared = self.prepared

        def f(x):
            return float(numpy_energy.total_energy(x.reshape(shape), prepared))

        x0 = np.asarray(active, dtype=np.float64).reshape(-1)
        res = optimize.minimize(
            f, x0, method=self.method, options={"maxiter": max_iter or self.max_iter}
        )
        coords[..., self.active_sites, :] = res.x.reshape(shape)
        return coords

    def energy(self, coords) -> float:
        if not self.spec.is_active():
            return 0.0
        active = coords[..., self.active_sites, :]
        return float(numpy_energy.total_energy(active, self.prepared))
