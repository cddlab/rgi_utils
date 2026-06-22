"""GPU restraint optimizer for JAX tools (alphafold3).

Builds a JIT/scan/vmap-compatible minimizer over an analytic ``jax.grad`` energy,
gated on the noise level with ``jax.lax.cond``. ``method='CG'`` (the default, shared
with torch) runs ``_cg_minimize`` — a pure-jax port of the torch nonlinear CG
(``lax.while_loop``); ``method='l-bfgs'`` uses ``jaxopt.LBFGS`` (lazily imported).
There is NO ``pure_callback`` and NO scipy, so the whole optimization runs inside XLA
on the accelerator. The custom CG is the SAME algorithm + constants as the torch CG and
converges the RMSD energy (whose fixed-rotation gradient stalls jaxopt's NonlinearCG +
backtracking line search). Note: it is not bit-identical to torch — XLA float-op
reordering inside ``lax.while_loop`` can make the jax minimum differ from torch's by a
small, input-dependent amount, and on some inputs at high ``max_iter`` (>~300) the
backtracking line search can stall a little earlier than the eager torch loop; at the
default ``max_iter=100`` they agree on the standard fixtures.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp

from rgi_utils.energy import jax_energy
from rgi_utils.optim._cg_config import (
    ARMIJO_C1,
    BACKTRACK,
    EPS,
    FTOL,
    GG_FLOOR,
    GTOL,
    MAX_LS,
)
from rgi_utils.optim.distance_shift import apply_distance_shift_jax

logger = logging.getLogger(__name__)


def _cg_minimize(energy_fn, x0, max_iter, max_ls=MAX_LS, gtol=GTOL, ftol=FTOL):
    """Pure-jax nonlinear conjugate gradient (Polak-Ribiere+, backtracking Armijo line
    search, restart on non-descent) — a port of the torch ``TorchRestraintOptimizer.
    _minimize_cg`` built from ``jax.lax.while_loop`` so it stays JIT/scan/vmap-able.
    ``x0`` is the active-site coords; ``energy_fn(x) -> scalar`` is the restraint energy.
    Returns the optimized coords. Same algorithm + constants as the torch backend (so
    ``method='CG'`` means CG everywhere) and (unlike jaxopt NonlinearCG) it converges the
    RMSD energy — though XLA float reordering in ``lax.while_loop`` can make its minimum
    differ from torch's by a small, input-dependent amount (see the module docstring)."""
    vg = jax.value_and_grad(energy_fn)

    def line_search(x_base, d, f, g_proto, slope):
        def cond(s):
            _step, accepted, _x, _f, _g, i = s
            return jnp.logical_and(jnp.logical_not(accepted), i < max_ls)

        def body(s):
            step, _acc, _x, _f, _g, i = s
            xt = x_base + step * d
            ft, gt = vg(xt)
            ok = ft <= f + ARMIJO_C1 * step * slope  # Armijo sufficient decrease
            return (jnp.where(ok, step, step * BACKTRACK), ok, xt, ft, gt, i + 1)

        init = (
            jnp.asarray(1.0),
            jnp.asarray(False),
            x_base,
            f,
            g_proto,
            jnp.asarray(0),
        )
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
        bad = jnp.logical_or(jnp.logical_not(jnp.isfinite(gg)), gg <= GG_FLOOR)
        d = jnp.where(jnp.sum(d * g) >= 0.0, -g, d)  # restart if not a descent dir
        # On a degenerate iteration (non-finite / underflowed gg) zero the search
        # direction so the line search accepts step 0 in ONE eval (no wasted
        # backtracking) — matching torch's pre-line-search break; `use` discards it.
        d = jnp.where(bad, jnp.zeros_like(d), d)
        slope = jnp.sum(d * g)
        xt, ft, gt, accepted = line_search(x, d, f, g, slope)
        conv = jnp.logical_or(
            jnp.max(jnp.abs(gt)) < gtol,
            jnp.abs(ft - f) < ftol * (1.0 + jnp.abs(f)),
        )
        beta = jnp.maximum(0.0, jnp.sum(gt * (gt - g)) / (gg + EPS))  # PR+
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


def _vdw_pair_energy(active, prot_pos, lig_local, lig_r, prot_r, scale, weight):
    """jnp port of the torch dynamic ligand-protein VdW (``optim/_torch_cg_gpu.py``):
    the moving ligand atoms (``active[lig_local]``) vs the FIXED protein background
    ``prot_pos``, all-pairs ``weight * sum(clamp(d - scale*(r_i+r_j), max=0)^2)`` (zero
    gradient beyond contact). Pure jnp so it composes into the scan/vmap energy. Same
    formula + ``EPS`` as the torch impl, so the two agree on value and gradient."""
    lig = active[..., lig_local, :]
    diff = lig[..., :, None, :] - prot_pos[..., None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + EPS)
    r_min = scale * (lig_r[:, None] + prot_r[None, :])
    delta = jnp.minimum(dist - r_min, 0.0)  # torch clamp(max=0.0)
    return weight * jnp.sum(delta**2)


def make_minimizer(
    spec,
    max_iter: int = 100,
    method: str = "cg",
):
    """Return ``minimize(coords, sigma, step) -> coords``.

    ``coords`` has shape (..., n_atom, 3); ``step`` is the diffusion step index (for the
    step-window gate, alongside ``sigma`` for the sigma-window gate). The returned function
    is pure and JIT/vmap-able, so it runs inside the diffusion loop's ``hk.scan``/``hk.vmap``
    (``step`` is a traced scalar there). ``method='cg'`` (the default) runs the pure-jax
    ``_cg_minimize``; any other value uses ``jaxopt.LBFGS`` (lazily imported). Per-restraint
    gating uses ``spec.max_start_sigma()`` and the per-term masks baked into the spec, so
    there is no ``start_sigma`` arg.
    """
    active_idx = jnp.asarray(spec.active_sites, dtype=jnp.int32)
    prepared = jax_energy.prepare_spec(spec)
    max_ss = spec.max_start_sigma()
    is_cg = (method or "cg").lower() in ("cg", "ncg", "nonlinear-cg", "nonlinearcg")
    has_dist = spec.has_distance()
    has_conf = spec.has_conformer()
    has_rmsd = spec.has_rmsd()
    # group-centroid angle/dihedral are CG-solved (energy terms gated per-restraint inside
    # total_energy via sigma), so the solver branch must run when either is present.
    has_group = spec.has_group_angle() or spec.has_group_dihedral()
    # custom restraints -> jnp closures (active_coords) -> scalar (weight folded);
    # selections baked as static jnp index arrays, so they trace inside lax.scan. Added to
    # the CG objective with a per-entry sigma gate (jnp.where).
    has_custom = spec.has_custom()
    from rgi_utils.custom.closure import build_terms

    custom_terms = build_terms(spec.custom, "jax") if has_custom else []
    dist_prepared = prepared.get("distance")
    # dynamic ligand-protein VdW (formerly torch-only; now jax too). The protein
    # background is read from the FULL coords at minimize time (it moves per diffusion
    # step), so it is NOT baked into the spec -- only the indices/radii are. Gated on
    # conf_start_sigma like the other conformer terms.
    _vc = getattr(spec, "vdw_config", None)
    has_vdw = _vc is not None and _vc.weight > 0
    if has_vdw:
        vdw_lig_local = jnp.asarray(_vc.ligand_local, dtype=jnp.int32)
        vdw_lig_r = jnp.asarray(_vc.ligand_radii)
        vdw_prot_global = jnp.asarray(_vc.protein_global, dtype=jnp.int32)
        vdw_prot_r = jnp.asarray(_vc.protein_radii)
        vdw_scale = jnp.asarray(float(_vc.scale))
        vdw_weight = jnp.asarray(float(_vc.weight))
        conf_ss = jnp.asarray(float(spec.conf_start_sigma))
        conf_stop = jnp.asarray(float(getattr(spec, "conf_stop_sigma", -1.0)))
        # conformer STEP window (the alternative gate axis; ANDed with the sigma window).
        conf_sstep = jnp.asarray(float(getattr(spec, "conf_start_step", float("-inf"))))
        conf_estep = jnp.asarray(float(getattr(spec, "conf_stop_step", float("inf"))))

    def _descend(coords, sigma, step):
        active = coords[..., active_idx, :]
        # 1) Distance restraints: closed-form rigid centroid translation (pure jnp, no
        #    solver) -- a centroid-distance restraint is 1-DOF. Gated per-restraint inside.
        if has_dist:
            active = apply_distance_shift_jax(active, dist_prepared, sigma, step)
        # 2) Conformer + RMSD restraints: jaxopt on the non-distance energy (distance is
        #    applied above; total_energy(include_distance=False) covers conformer AND
        #    RMSD), plus the ligand-protein VdW term (gated on conf_start_sigma -- the
        #    `jnp.where` zeroes its weight AND gradient above the gate). Skipped for a
        #    distance-only spec. has_conf is already True when vdw_config is set.
        if has_conf or has_rmsd or has_vdw or has_group or has_custom:
            if has_vdw:
                prot_pos = coords[..., vdw_prot_global, :]
                # conformer window: active sigma window (conf_stop <= sigma <= conf_start)
                # AND active step window (conf_sstep <= step <= conf_estep). NOT a _TERMS
                # entry, so this gate is maintained by hand (mirrors torch_optim).
                _s = jnp.asarray(sigma)
                _st = jnp.asarray(step)
                in_win = (
                    (_s <= conf_ss)
                    & (_s >= conf_stop)
                    & (_st >= conf_sstep)
                    & (_st <= conf_estep)
                )
                vdw_w = jnp.where(in_win, vdw_weight, 0.0)

            def energy_fn(a):
                e = jax_energy.total_energy(
                    a, prepared, sigma, step, include_distance=False
                )
                if has_vdw:
                    e = e + _vdw_pair_energy(
                        a,
                        prot_pos,
                        vdw_lig_local,
                        vdw_lig_r,
                        vdw_prot_r,
                        vdw_scale,
                        vdw_w,
                    )
                # per-custom gate: active sigma window AND active step window.
                for _name, start, stop, start_step, stop_step, closure in custom_terms:
                    _sc = jnp.asarray(sigma)
                    _stc = jnp.asarray(step)
                    gate = jnp.where(
                        (_sc <= start)
                        & (_sc >= stop)
                        & (_stc >= start_step)
                        & (_stc <= stop_step),
                        1.0,
                        0.0,
                    )
                    e = e + gate * closure(a)
                return e

            if is_cg:
                # the custom CG matches torch and converges the RMSD energy (jaxopt's
                # NonlinearCG + backtracking stalls on its fixed-rotation gradient)
                opt = _cg_minimize(energy_fn, active, max_iter)
            else:
                import jaxopt  # only the non-default l-bfgs method needs jaxopt

                opt = (
                    jaxopt.LBFGS(
                        fun=energy_fn,
                        maxiter=max_iter,
                        linesearch="backtracking",
                        implicit_diff=False,
                    )
                    .run(active)
                    .params
                )
            # keep the input coordinates if the solver diverged to non-finite
            active = jnp.where(jnp.all(jnp.isfinite(opt)), opt, active)
        return coords.at[..., active_idx, :].set(active)

    def minimize(coords, sigma, step=0):
        if not spec.is_active():
            return coords
        # Skip the whole step only when sigma exceeds every restraint's start_sigma. A
        # step-windowed restraint keeps start_sigma=+inf -> max_ss=+inf -> never skipped
        # here; its step gate (inside _descend) handles activation. The per-restraint
        # sigma+step gates inside the energy still zero anything out of its window.
        return jax.lax.cond(
            jnp.asarray(sigma) <= max_ss,
            lambda c: _descend(c, sigma, step),
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
