"""GPU restraint optimizer for JAX tools (alphafold3).

Builds a JIT/scan/vmap-compatible minimizer using a ``jaxopt`` solver
(NonlinearCG or LBFGS) over an analytic ``jax.grad`` energy, gated on the noise
level with ``jax.lax.cond``. There is NO ``pure_callback`` and NO scipy backend
— jaxopt's solvers are pure JAX, so the whole optimization runs inside XLA on
the accelerator. This fixes the slow AF3 restraint path (which used
``jaxopt.ScipyMinimize`` outside JIT) while giving proper line-searched
convergence rather than fixed-step gradient descent.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp

from rgi_utils.energy import jax_energy

logger = logging.getLogger(__name__)


def make_minimizer(
    spec,
    max_iter: int = 200,
    learning_rate: float = 0.01,
    method: str = "lbfgs",
):
    """Return ``minimize(coords, sigma) -> coords``.

    ``coords`` has shape (..., n_atom, 3). The returned function is pure and
    JIT/vmap-able, so it runs inside the diffusion loop's ``hk.scan``/``hk.vmap``.
    ``method`` selects the jaxopt solver: ``"cg"`` -> NonlinearCG, else LBFGS.
    ``learning_rate`` is accepted for API compatibility but unused (the solver
    runs its own line search). Per-restraint gating uses ``spec.max_start_sigma()``
    and the per-term masks baked into the spec, so there is no ``start_sigma`` arg.
    """
    import jaxopt

    active_idx = jnp.asarray(spec.active_sites, dtype=jnp.int32)
    prepared = jax_energy.prepare_spec(spec)
    max_ss = spec.max_start_sigma()
    # ``backtracking`` enforces sufficient decrease (Armijo), like torch's
    # strong-Wolfe line search. The default ``zoom``/``hager-zhang`` searches can
    # accept a huge first step that collapses atoms onto each other, where the
    # eps-regularised distance makes the gradient vanish (a false stationary
    # point) — backtracking rejects that step.
    is_cg = (method or "lbfgs").lower() in ("cg", "ncg", "nonlinear-cg", "nonlinearcg")

    def _descend(coords, sigma):
        # Build the (sigma-gated) energy + solver per call so each restraint is
        # active only when sigma <= its start_sigma (folded into the energy mask).
        def energy_fn(active):
            return jax_energy.total_energy(active, prepared, sigma)

        solver_cls = jaxopt.NonlinearCG if is_cg else jaxopt.LBFGS
        solver = solver_cls(
            fun=energy_fn,
            maxiter=max_iter,
            linesearch="backtracking",
            implicit_diff=False,
        )
        active = coords[..., active_idx, :]
        active_opt = solver.run(active).params
        # Robustness: a degenerate geometry can still make a solver step diverge;
        # keep the input coordinates if the result is non-finite.
        active_opt = jnp.where(jnp.all(jnp.isfinite(active_opt)), active_opt, active)
        return coords.at[..., active_idx, :].set(active_opt)

    def minimize(coords, sigma):
        if not spec.is_active():
            return coords
        # skip the whole step only when sigma exceeds every restraint's start_sigma
        return jax.lax.cond(
            jnp.asarray(sigma) <= max_ss,
            lambda c: _descend(c, sigma),
            lambda c: c,
            coords,
        )

    return minimize


def energy_of(spec, coords) -> float:
    """Restraint energy at ``coords`` (for stats); host-side, not for the loop."""
    if not spec.is_active():
        return 0.0
    active_idx = jnp.asarray(spec.active_sites, dtype=jnp.int32)
    prepared = jax_energy.prepare_spec(spec)
    active = coords[..., active_idx, :]
    return float(jax_energy.total_energy(active, prepared))
