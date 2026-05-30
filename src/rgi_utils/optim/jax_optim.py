"""GPU restraint optimizer for JAX tools (alphafold3).

Builds a JIT/scan/vmap-compatible minimizer: gradient descent via
``jax.lax.fori_loop`` over an analytic ``jax.grad`` energy, gated on the noise
level with ``jax.lax.cond``. There is NO ``pure_callback`` and NO scipy — the
whole optimization runs inside XLA on the accelerator, which is what fixes the
slow distance restraints in the old AF3 prototype.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp

from rgi_utils.energy import jax_energy

logger = logging.getLogger(__name__)


def make_minimizer(
    spec, max_iter: int = 200, learning_rate: float = 0.01, start_sigma: float = -1.0
):
    """Return ``minimize(coords, sigma) -> coords``.

    ``coords`` has shape (..., n_atom, 3). The returned function is pure and
    JIT-able, so it can be called inside ``jax.lax.scan`` in the diffusion loop.
    """
    active_idx = jnp.asarray(spec.active_sites, dtype=jnp.int32)
    prepared = jax_energy.prepare_spec(spec)

    def energy_fn(active):
        return jax_energy.total_energy(active, prepared)

    grad_fn = jax.grad(energy_fn)

    def _descend(coords):
        active = coords[..., active_idx, :]

        def step(_, a):
            return a - learning_rate * grad_fn(a)

        active_opt = jax.lax.fori_loop(0, max_iter, step, active)
        return coords.at[..., active_idx, :].set(active_opt)

    def minimize(coords, sigma):
        if not spec.is_active():
            return coords
        return jax.lax.cond(
            jnp.asarray(sigma) <= start_sigma, _descend, lambda c: c, coords
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
