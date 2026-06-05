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
from rgi_utils.optim.distance_shift import apply_distance_shift_numpy

logger = logging.getLogger(__name__)


class NumpyRestraintOptimizer:
    def __init__(self, spec, max_iter: int = 100, method: str = "CG"):
        self.spec = spec
        self.prepared = numpy_energy.prepare_spec(spec)
        self.active_sites = np.asarray(spec.active_sites)
        self.max_iter = max_iter
        self.method = method
        # Lazy (torch, torch_energy, cpu_prepared_spec) for the analytic jacobian;
        # None = not yet probed, False = torch unavailable (use scipy finite-diff).
        self._torch = None
        self._prepared_t = None

    def minimize(self, coords, sigma=None, start_sigma=None, max_iter=None):
        """Optimize ``coords`` (..., n_atom, 3) numpy array in-place. Distance
        restraints are applied in closed form (rigid COM shift); conformer terms are
        optimised with scipy. Each restraint is gated on ``sigma <= start_sigma``."""
        if not self.spec.is_active():
            return coords
        if sigma is not None and sigma > self.spec.max_start_sigma():
            return coords
        active = np.array(coords[..., self.active_sites, :], dtype=np.float64, copy=True)

        # 1) Distance restraints: closed-form rigid COM translation (no solver).
        if self.spec.has_distance():
            active = apply_distance_shift_numpy(active, self.prepared["distance"], sigma)

        # 2) Conformer restraints: scipy on the conformer-only energy.
        if self.spec.has_conformer():
            shape = active.shape
            prepared = self.prepared

            def f(x):
                return float(
                    numpy_energy.total_energy(
                        x.reshape(shape), prepared, sigma, include_distance=False
                    )
                )

            jac = self._make_jac(shape, sigma)
            res = optimize.minimize(
                f,
                active.reshape(-1),
                jac=jac,
                method=self._scipy_method(),
                options={"maxiter": max_iter or self.max_iter},
            )
            active = res.x.reshape(shape)

        coords[..., self.active_sites, :] = active
        return coords

    def _scipy_method(self) -> str:
        """Map the shared ``method`` aliases to a scipy solver name so one
        ``restraints_config`` behaves the same on every backend. torch/jax accept
        ``{cg,ncg,nonlinear-cg,nonlinearcg}`` as CG and anything else as LBFGS, both
        case-insensitively; scipy needs exact names. Without this, a config that runs
        on the GPU tools (e.g. ``method: "LBFGS"`` or ``"ncg"``) would crash the numpy
        (default CPU) backend with ValueError('Unknown solver ...')."""
        m = (self.method or "cg").lower()
        if m in ("cg", "ncg", "nonlinear-cg", "nonlinearcg"):
            return "CG"
        return "L-BFGS-B"

    def _make_jac(self, shape, sigma):
        """Analytic gradient for scipy via torch autodiff over the (parity-identical)
        torch energy. Without it scipy CG finite-differences the gradient — O(DOF)
        energy evals per gradient, each O(DOF) -> ~O(DOF^2), unusably slow at protein
        scale. Returns None (scipy falls back to finite differences) when torch is
        unavailable, preserving the pure-numpy fallback."""
        tg = self._torch_energy()
        if tg is None:
            return None
        torch, torch_energy, prepared_t = tg

        def jac(x):
            # Re-enable autograd: callers (boltz/openfold) run under inference_mode.
            with torch.inference_mode(False), torch.enable_grad():
                xt = torch.tensor(
                    x.reshape(shape), dtype=torch.float64, requires_grad=True
                )
                torch_energy.total_energy(
                    xt, prepared_t, sigma, include_distance=False
                ).backward()
                g = xt.grad.detach().cpu().numpy()
            return np.asarray(g, dtype=np.float64).reshape(-1)

        return jac

    def _torch_energy(self):
        """Cached ``(torch, torch_energy, cpu_prepared_spec)`` or None if torch is
        unavailable. Built once per optimizer (instance-scoped per structure)."""
        if self._torch is False:
            return None
        if self._torch is None:
            try:
                import torch

                from rgi_utils.energy import torch_energy

                with torch.inference_mode(False):
                    self._prepared_t = torch_energy.prepare_spec(
                        self.spec, dtype=torch.float64, device="cpu"
                    )
                self._torch = (torch, torch_energy)
            except Exception:
                self._torch = False
                return None
        torch, torch_energy = self._torch
        return torch, torch_energy, self._prepared_t

    def energy(self, coords) -> float:
        if not self.spec.is_active():
            return 0.0
        active = coords[..., self.active_sites, :]
        return float(numpy_energy.total_energy(active, self.prepared))
