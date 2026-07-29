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
from rgi_utils.spec import check_active_vdw_int32_safe

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


def _vdw_pair_energy(active, bg_pos, lig_local, lig_r, bg_r, scale, weight):
    """jnp port of the torch dynamic fixed-background VdW (``optim/_torch_cg_gpu.py``):
    the moving ligand atoms (``active[lig_local]``) vs the FIXED background
    ``bg_pos``, all-pairs ``weight * sum(clamp(d - scale*(r_i+r_j), max=0)^2)`` (zero
    gradient beyond contact). Pure jnp so it composes into the scan/vmap energy. Same
    formula + ``EPS`` as the torch impl, so the two agree on value and gradient."""
    lig = active[..., lig_local, :]
    diff = lig[..., :, None, :] - bg_pos[..., None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + EPS)
    r_min = scale * (lig_r[:, None] + bg_r[None, :])
    delta = jnp.minimum(dist - r_min, 0.0)  # torch clamp(max=0.0)
    return weight * jnp.sum(delta**2)


def _build_active_vdw_pairs(
    active,
    radii,
    polymer_mask,
    excluded_codes,
    dmax,
    max_neighbors,
):
    """Pure-jax fixed-width neighbour builder matching the torch implementation."""

    n_atom = active.shape[-2]
    batch = active.reshape((-1, n_atom, 3))
    if n_atom < 2:  # nothing to pair (mirrors the torch builder's guard)
        empty = jnp.zeros((batch.shape[0], n_atom, 0), dtype=jnp.int32)
        return empty, empty.astype(active.dtype)
    k = min(int(max_neighbors), n_atom - 1)
    diff = batch[:, :, None, :] - batch[:, None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + EPS)
    dist = jnp.where(jnp.eye(n_atom, dtype=bool)[None, :, :], jnp.inf, dist)
    neg_values, neighbours = jax.lax.top_k(-dist, k)
    values = -neg_values
    source = jnp.arange(n_atom, dtype=jnp.int32).reshape((1, n_atom, 1))
    lo = jnp.minimum(source, neighbours)
    hi = jnp.maximum(source, neighbours)
    codes = lo * n_atom + hi
    valid = values <= dmax
    valid = valid & (polymer_mask[source] | polymer_mask[neighbours])
    valid = valid & (radii[source] > 0) & (radii[neighbours] > 0)
    if excluded_codes.shape[0] > 0:
        positions = jnp.searchsorted(excluded_codes, codes)
        positions = jnp.minimum(positions, excluded_codes.shape[0] - 1)
        valid = valid & (excluded_codes[positions] != codes)

    batch_idx = jnp.arange(batch.shape[0], dtype=jnp.int32).reshape((-1, 1, 1))
    reverse_neighbours = neighbours[batch_idx, neighbours]
    reverse_valid = valid[batch_idx, neighbours]
    reverse = jnp.any(
        (reverse_neighbours == source[..., None]) & reverse_valid, axis=-1
    )
    pair_factor = valid.astype(active.dtype) / (1.0 + reverse.astype(active.dtype))
    return neighbours, pair_factor


def _active_vdw_pair_energy(active, neighbours, pair_factor, radii, scale, weight):
    """VdW energy over the per-step active-active neighbour list."""

    n_atom = active.shape[-2]
    batch = active.reshape((-1, n_atom, 3))
    batch_idx = jnp.arange(batch.shape[0], dtype=jnp.int32).reshape((-1, 1, 1))
    other = batch[batch_idx, neighbours]
    diff = batch[:, :, None, :] - other
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + EPS)
    r_min = scale * (radii[None, :, None] + radii[neighbours])
    delta = jnp.minimum(dist - r_min, 0.0)
    return weight * jnp.sum(pair_factor * delta**2)


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
    # group-centroid angle/dihedral and the standalone best-fit plane are CG-solved (energy
    # terms gated per-restraint inside total_energy via sigma), so the solver branch must
    # run when any is present.
    has_group = (
        spec.has_group_angle() or spec.has_group_dihedral() or spec.has_group_plane()
    )
    # custom restraints -> jnp closures (active_coords) -> scalar (weight folded);
    # selections baked as static jnp index arrays, so they trace inside lax.scan. Added to
    # the CG objective with a per-entry sigma gate (jnp.where).
    has_custom = spec.has_custom()
    from rgi_utils.custom.closure import build_terms

    custom_terms = build_terms(spec.custom, "jax") if has_custom else []
    # dynamic fixed-background VdW (formerly torch-only; now jax too). The fixed
    # background is read from the FULL coords at minimize time (it moves per diffusion
    # step), so it is NOT baked into the spec -- only the indices/radii are. Gated on
    # conf_start_sigma like the other conformer terms.
    _vc = getattr(spec, "vdw_config", None)
    has_vdw = _vc is not None and _vc.weight > 0
    if has_vdw:
        vdw_lig_local = jnp.asarray(_vc.ligand_local, dtype=jnp.int32)
        vdw_lig_r = jnp.asarray(_vc.ligand_radii)
        vdw_bg_global = jnp.asarray(_vc.background_global, dtype=jnp.int32)
        vdw_bg_r = jnp.asarray(_vc.background_radii)
        vdw_scale = jnp.asarray(float(_vc.scale))
        vdw_weight = jnp.asarray(float(_vc.weight))
    _ac = getattr(spec, "active_vdw_config", None)
    has_active_vdw = _ac is not None and _ac.weight > 0
    if has_active_vdw:
        # int32 pair-code encoding: fail loudly above the JAX-safe active-atom count
        # rather than silently corrupting the covalent-pair exclusion (see spec.py).
        check_active_vdw_int32_safe(int(_ac.radii.shape[0]))
        active_vdw_radii = jnp.asarray(_ac.radii)
        active_vdw_polymer = jnp.asarray(_ac.polymer_mask, dtype=bool)
        active_vdw_excluded = jnp.asarray(_ac.excluded_codes, dtype=jnp.int32)
        active_vdw_scale = jnp.asarray(float(_ac.scale))
        active_vdw_weight = jnp.asarray(float(_ac.weight))
        active_vdw_dmax = jnp.asarray(float(_ac.dmax))
        active_vdw_max_neighbors = int(_ac.max_neighbors)
    if has_vdw or has_active_vdw:
        conf_ss = jnp.asarray(float(spec.conf_start_sigma))
        conf_stop = jnp.asarray(float(getattr(spec, "conf_stop_sigma", -1.0)))
        # conformer STEP window (the alternative gate axis; ANDed with the sigma window).
        conf_sstep = jnp.asarray(float(getattr(spec, "conf_start_step", float("-inf"))))
        conf_estep = jnp.asarray(float(getattr(spec, "conf_stop_step", float("inf"))))

    def _descend(coords, sigma, step):
        active = coords[..., active_idx, :]
        # Distance + conformer + RMSD + group restraints all minimise ONE objective via the
        # CG (total_energy sums every active term; distance is now an autodiff CG term whose
        # reduced-mass-rescaled centroid gradient translates each group rigidly — no
        # closed-form shift), plus the fixed-background VdW term (gated on conf_start_sigma —
        # the `jnp.where` zeroes its weight AND gradient above the gate). has_conf is already
        # True when vdw_config is set.
        if (
            has_dist
            or has_conf
            or has_rmsd
            or has_vdw
            or has_active_vdw
            or has_group
            or has_custom
        ):
            if has_vdw or has_active_vdw:
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
            if has_vdw:
                bg_pos = coords[..., vdw_bg_global, :]
                vdw_w = jnp.where(in_win, vdw_weight, 0.0)
            if has_active_vdw:
                active_vdw_w = jnp.where(in_win, active_vdw_weight, 0.0)
                neighbours, pair_factor = _build_active_vdw_pairs(
                    jax.lax.stop_gradient(active),
                    active_vdw_radii,
                    active_vdw_polymer,
                    active_vdw_excluded,
                    active_vdw_dmax,
                    active_vdw_max_neighbors,
                )

            def energy_fn(a):
                e = jax_energy.total_energy(a, prepared, sigma, step)
                if has_vdw:
                    e = e + _vdw_pair_energy(
                        a,
                        bg_pos,
                        vdw_lig_local,
                        vdw_lig_r,
                        vdw_bg_r,
                        vdw_scale,
                        vdw_w,
                    )
                if has_active_vdw:
                    e = e + _active_vdw_pair_energy(
                        a,
                        neighbours,
                        pair_factor,
                        active_vdw_radii,
                        active_vdw_scale,
                        active_vdw_w,
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


def dynamic_vdw_energy(spec, coords) -> float:
    """The dynamic (optimizer-only) VdW terms alone (>= 0); for finalize stats.

    The jax twin of ``TorchRestraintOptimizer.dynamic_vdw_energy``. Both the fixed
    background (``spec.vdw_config``) and the active-active polymer neighbour list
    (``spec.active_vdw_config``) are applied ONLY inside the optimizer, so
    ``energy_breakdown`` -- which reads ``spec.vdw`` (the static intra + inter-ligand
    rows) -- cannot see them. Without this, AF3's finalize printed ``vdw=0.00000`` for
    terms that ran at every diffusion step, i.e. a number that could not distinguish
    "no clash" from "never measured".

    Host-side (not for the loop) and ungated, matching torch: this is a residual report
    at the final coordinates, not a step of the CG objective.
    """
    if not spec.is_active():
        return 0.0
    vc = getattr(spec, "vdw_config", None)
    ac = getattr(spec, "active_vdw_config", None)
    has_vdw = vc is not None and vc.weight > 0
    has_active_vdw = ac is not None and ac.weight > 0
    if not (has_vdw or has_active_vdw):
        return 0.0
    active = coords[..., jnp.asarray(spec.active_sites, dtype=jnp.int32), :]
    total = 0.0
    if has_vdw:
        total += float(
            _vdw_pair_energy(
                active,
                coords[..., jnp.asarray(vc.background_global, dtype=jnp.int32), :],
                jnp.asarray(vc.ligand_local, dtype=jnp.int32),
                jnp.asarray(vc.ligand_radii),
                jnp.asarray(vc.background_radii),
                jnp.asarray(float(vc.scale)),
                jnp.asarray(float(vc.weight)),
            )
        )
    if has_active_vdw:
        # same int32 pair-code guard as make_minimizer (fail loudly, never corrupt the
        # covalent-pair exclusion) -- stats must not diverge from what was minimised.
        check_active_vdw_int32_safe(int(ac.radii.shape[0]))
        radii = jnp.asarray(ac.radii)
        neighbours, pair_factor = _build_active_vdw_pairs(
            active,
            radii,
            jnp.asarray(ac.polymer_mask, dtype=bool),
            jnp.asarray(ac.excluded_codes, dtype=jnp.int32),
            jnp.asarray(float(ac.dmax)),
            int(ac.max_neighbors),
        )
        total += float(
            _active_vdw_pair_energy(
                active,
                neighbours,
                pair_factor,
                radii,
                jnp.asarray(float(ac.scale)),
                jnp.asarray(float(ac.weight)),
            )
        )
    return total
