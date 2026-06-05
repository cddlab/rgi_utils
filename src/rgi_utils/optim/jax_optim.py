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
from rgi_utils.optim.distance_shift import apply_distance_shift_jax

logger = logging.getLogger(__name__)


def make_minimizer(
    spec,
    max_iter: int = 100,
    learning_rate: float = 0.01,
    method: str = "cg",
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
    has_dist = spec.has_distance()
    has_conf = spec.has_conformer()
    has_rmsd = spec.has_rmsd()
    dist_prepared = prepared.get("distance")

    def _descend(coords, sigma):
        active = coords[..., active_idx, :]
        # 1) Distance restraints: closed-form rigid COM translation (pure jnp, no
        #    solver) -- a COM-distance restraint is 1-DOF. Gated per-restraint inside.
        if has_dist:
            active = apply_distance_shift_jax(active, dist_prepared, sigma)
        # 2) Conformer + RMSD restraints: jaxopt on the non-distance energy (distance is
        #    applied above; total_energy(include_distance=False) covers conformer AND
        #    RMSD). Skipped entirely for a distance-only spec.
        if has_conf or has_rmsd:

            def energy_fn(a):
                return jax_energy.total_energy(a, prepared, sigma, include_distance=False)

            solver_cls = jaxopt.NonlinearCG if is_cg else jaxopt.LBFGS
            solver = solver_cls(
                fun=energy_fn,
                maxiter=max_iter,
                linesearch="backtracking",
                implicit_diff=False,
            )
            opt = solver.run(active).params
            # keep the input coordinates if the solver diverged to non-finite
            active = jnp.where(jnp.all(jnp.isfinite(opt)), opt, active)
        return coords.at[..., active_idx, :].set(active)

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
