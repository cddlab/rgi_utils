"""GPU restraint optimizer for JAX tools (alphafold3).

Builds a JIT/scan/vmap-compatible minimizer over an analytic ``jax.grad`` energy,
gated on the noise level with ``jax.lax.cond``. ``method='CG'`` (the default, shared
with torch) runs ``_cg_minimize`` — a pure-jax port of the torch nonlinear CG
(``lax.while_loop``); ``method='l-bfgs'`` uses ``jaxopt.LBFGS`` (lazily imported).
There is NO ``pure_callback`` and NO scipy, so the whole optimization runs inside XLA
on the accelerator. The custom CG matches torch exactly and converges the RMSD energy
(whose fixed-rotation gradient stalls jaxopt's NonlinearCG + backtracking line search).
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp

from rgi_utils.energy import jax_energy
from rgi_utils.optim.distance_shift import apply_distance_shift_jax

logger = logging.getLogger(__name__)


def _cg_minimize(energy_fn, x0, max_iter, max_ls=20, gtol=1e-7, ftol=1e-9):
    """Pure-jax nonlinear conjugate gradient (Polak-Ribiere+, backtracking Armijo line
    search, restart on non-descent) — a port of the torch ``TorchRestraintOptimizer.
    _minimize_cg`` built from ``jax.lax.while_loop`` so it stays JIT/scan/vmap-able.
    ``x0`` is the active-site coords; ``energy_fn(x) -> scalar`` is the restraint energy.
    Returns the optimized coords. Identical CG to the torch backend, so ``method='CG'``
    is the same algorithm everywhere and (unlike jaxopt NonlinearCG) it converges the
    RMSD energy."""
    vg = jax.value_and_grad(energy_fn)
    eps = 1e-12

    def line_search(x_base, d, f, g_proto, slope):
        def cond(s):
            _step, accepted, _x, _f, _g, i = s
            return jnp.logical_and(jnp.logical_not(accepted), i < max_ls)

        def body(s):
            step, _acc, _x, _f, _g, i = s
            xt = x_base + step * d
            ft, gt = vg(xt)
            ok = ft <= f + 1e-4 * step * slope  # Armijo sufficient decrease
            return (jnp.where(ok, step, step * 0.5), ok, xt, ft, gt, i + 1)

        init = (jnp.asarray(1.0), jnp.asarray(False), x_base, f, g_proto, jnp.asarray(0))
        _s, accepted, xt, ft, gt, _i = jax.lax.while_loop(cond, body, init)
        return xt, ft, gt, accepted

    f0, g0 = vg(x0)
    gg0 = jnp.sum(g0 * g0)
    stop0 = jnp.max(jnp.abs(g0)) < gtol

    def cond(st):
        _x, _f, _g, _d, _gg, it, stop = st
        return jnp.logical_and(jnp.logical_not(stop), it < max_iter)

    def body(st):
        x, f, g, d, gg, it, _stop = st
        bad = jnp.logical_or(jnp.logical_not(jnp.isfinite(gg)), gg <= 1e-20)
        d = jnp.where(jnp.sum(d * g) >= 0.0, -g, d)  # restart if not a descent dir
        slope = jnp.sum(d * g)
        xt, ft, gt, accepted = line_search(x, d, f, g, slope)
        conv = jnp.logical_or(
            jnp.max(jnp.abs(gt)) < gtol,
            jnp.abs(ft - f) < ftol * (1.0 + jnp.abs(f)),
        )
        beta = jnp.maximum(0.0, jnp.sum(gt * (gt - g)) / (gg + eps))  # PR+
        use = jnp.logical_and(accepted, jnp.logical_not(bad))
        nx = jnp.where(use, xt, x)
        nf = jnp.where(use, ft, f)
        ng = jnp.where(use, gt, g)
        nd = jnp.where(use, -gt + beta * d, d)
        ngg = jnp.where(use, jnp.sum(gt * gt), gg)
        stop_next = jnp.logical_or(bad, jnp.logical_or(jnp.logical_not(accepted), conv))
        return (nx, nf, ng, nd, ngg, it + 1, stop_next)

    init = (x0, f0, g0, -g0, gg0, jnp.asarray(0), stop0)
    x = jax.lax.while_loop(cond, body, init)[0]
    return x


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
    active_idx = jnp.asarray(spec.active_sites, dtype=jnp.int32)
    prepared = jax_energy.prepare_spec(spec)
    max_ss = spec.max_start_sigma()
    is_cg = (method or "cg").lower() in ("cg", "ncg", "nonlinear-cg", "nonlinearcg")
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

            if is_cg:
                # the custom CG matches torch and converges the RMSD energy (jaxopt's
                # NonlinearCG + backtracking stalls on its fixed-rotation gradient)
                opt = _cg_minimize(energy_fn, active, max_iter)
            else:
                import jaxopt  # only the non-default l-bfgs method needs jaxopt

                opt = (
                    jaxopt.LBFGS(
                        fun=energy_fn, maxiter=max_iter, linesearch="backtracking",
                        implicit_diff=False,
                    )
                    .run(active)
                    .params
                )
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
