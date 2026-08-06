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

from rgi_utils._array_ops import VDW_OVERLAP_EPS
from rgi_utils.energy import jax_energy
from rgi_utils.optim._cell_list import (
    CELL_CHUNK_SIZE,
    CELL_HASH_PRIMES,
    CELL_OFFSETS,
)
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


def _cg_minimize(
    energy_fn, x0, max_iter, max_ls=MAX_LS, gtol=GTOL, ftol=FTOL, max_atom_step=None
):
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
            delta = step * d
            if max_atom_step is not None:
                atom_norm = jnp.sqrt(
                    jnp.sum(delta * delta, axis=-1, keepdims=True) + EPS
                )
                delta = delta * jnp.minimum(1.0, max_atom_step / atom_norm)
            xt = x_base + delta
            ft, gt = vg(xt)
            predicted = (
                step * slope if max_atom_step is None else jnp.sum(g_proto * delta)
            )
            ok = ft <= f + ARMIJO_C1 * predicted
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


def _cell_hash_jax(cells):
    """Return an int32 spatial hash; callers verify cells after hash lookup."""
    return (
        cells[..., 0] * jnp.int32(CELL_HASH_PRIMES[0])
        ^ cells[..., 1] * jnp.int32(CELL_HASH_PRIMES[1])
        ^ cells[..., 2] * jnp.int32(CELL_HASH_PRIMES[2])
    )


def _batched_searchsorted_jax(sorted_values, queries, side):
    return jax.vmap(lambda values, query: jnp.searchsorted(values, query, side=side))(
        sorted_values, queries
    ).astype(jnp.int32)


def _merge_cell_candidates_jax(best_dist2, best_idx, dist2, candidate, k):
    """Keep the lexicographically smallest ``(distance squared, atom index)`` rows."""
    candidate = candidate.astype(best_idx.dtype)
    merged_dist2 = jnp.concatenate((best_dist2, dist2), axis=-1)
    merged_idx = jnp.concatenate((best_idx, candidate), axis=-1)
    merged_dist2, merged_idx = jax.lax.sort(
        (merged_dist2, merged_idx),
        dimension=-1,
        is_stable=True,
        num_keys=2,
    )
    return merged_dist2[..., :k], merged_idx[..., :k].astype(best_idx.dtype)


def _build_cell_pairs_jax(
    query,
    target,
    dmax,
    max_neighbors,
    exclude_self=False,
    query_radii=None,
    target_radii=None,
    pair_scale=None,
    query_polymer=None,
    target_polymer=None,
    excluded_codes=None,
    pair_code_size=None,
):
    """Return target indices and ranking scores from a sorted cell list."""

    n_query, n_target = query.shape[-2], target.shape[-2]
    query_batch = jax.lax.stop_gradient(query.reshape((-1, n_query, 3)))
    target_batch = jax.lax.stop_gradient(target.reshape((-1, n_target, 3)))
    if query_batch.shape[0] != target_batch.shape[0]:
        raise ValueError("query and target VdW batch dimensions must match")
    if query_radii is not None:
        query_radii = jnp.asarray(query_radii, dtype=query_batch.dtype)
        target_radii = jnp.asarray(target_radii, dtype=query_batch.dtype)
        pair_scale = jnp.asarray(pair_scale, dtype=query_batch.dtype)
    max_candidates = n_target - int(exclude_self)
    if n_query == 0 or max_candidates <= 0:
        empty_idx = jnp.zeros((query_batch.shape[0], n_query, 0), dtype=jnp.int32)
        return empty_idx, empty_idx.astype(query.dtype)
    if n_target == 1:
        dist2 = jnp.sum(
            (query_batch - target_batch[:, :1, :]) ** 2, axis=-1, keepdims=True
        )
        dmax_value = jnp.asarray(dmax, dtype=query_batch.dtype)
        valid = (
            jnp.isfinite(dist2) & (dmax_value > 0) & (dist2 <= dmax_value * dmax_value)
        )
        score = dist2
        source = jnp.arange(n_query, dtype=jnp.int32).reshape((1, n_query, 1))
        neighbours = jnp.zeros(dist2.shape, dtype=jnp.int32)
        if query_radii is not None:
            source_r = query_radii[source]
            target_r = target_radii[neighbours]
            valid = valid & (source_r > 0) & (target_r > 0)
            score = jnp.sqrt(dist2 + EPS) - pair_scale * (source_r + target_r)
        if query_polymer is not None:
            valid = valid & (query_polymer[source] | target_polymer[neighbours])
        if excluded_codes is not None and excluded_codes.shape[0] > 0:
            lo = jnp.minimum(source, neighbours)
            hi = jnp.maximum(source, neighbours)
            codes = lo * pair_code_size + hi
            positions = jnp.searchsorted(excluded_codes, codes)
            positions = jnp.minimum(positions, excluded_codes.shape[0] - 1)
            valid = valid & (excluded_codes[positions] != codes)
        return neighbours, jnp.where(valid, score, jnp.inf)
    k = min(int(max_neighbors), max_candidates)
    n_batch = query_batch.shape[0]
    dmax_value = jnp.asarray(dmax, dtype=query_batch.dtype)
    cell_width = jnp.where(
        dmax_value > 0, dmax_value, jnp.asarray(1.0, query_batch.dtype)
    )
    safe_query = jnp.nan_to_num(query_batch, nan=0.0, posinf=0.0, neginf=0.0)
    safe_target = jnp.nan_to_num(target_batch, nan=0.0, posinf=0.0, neginf=0.0)
    query_cells = jnp.floor(safe_query / cell_width).astype(jnp.int32)
    target_cells = jnp.floor(safe_target / cell_width).astype(jnp.int32)
    target_hashes = _cell_hash_jax(target_cells)
    order = jnp.argsort(target_hashes, axis=-1, stable=True).astype(jnp.int32)
    sorted_hashes = jnp.take_along_axis(target_hashes, order, axis=-1)
    own_start = _batched_searchsorted_jax(sorted_hashes, target_hashes, "left")
    own_end = _batched_searchsorted_jax(sorted_hashes, target_hashes, "right")
    max_bucket = jnp.max(own_end - own_start)

    best_dist2 = jnp.full((n_batch, n_query, k), jnp.inf, dtype=query_batch.dtype)
    best_idx = jnp.zeros((n_batch, n_query, k), dtype=jnp.int32)
    batch_idx = jnp.arange(n_batch, dtype=jnp.int32).reshape((-1, 1, 1))
    source = jnp.arange(n_query, dtype=jnp.int32).reshape((1, n_query, 1))
    offsets = jnp.asarray(CELL_OFFSETS, dtype=jnp.int32)
    chunk_offsets = jnp.arange(CELL_CHUNK_SIZE, dtype=jnp.int32).reshape((1, 1, -1))
    cutoff2 = dmax_value * dmax_value

    def offset_body(offset_index, state):
        best_dist2, best_idx = state
        adjacent_cells = query_cells + offsets[offset_index]
        query_hashes = _cell_hash_jax(adjacent_cells)
        starts = _batched_searchsorted_jax(sorted_hashes, query_hashes, "left")
        ends = _batched_searchsorted_jax(sorted_hashes, query_hashes, "right")

        def chunk_cond(chunk_state):
            base, _best_dist2, _best_idx = chunk_state
            return base < max_bucket

        def chunk_body(chunk_state):
            base, best_dist2, best_idx = chunk_state
            positions = starts[..., None] + base + chunk_offsets
            position_valid = positions < ends[..., None]
            safe_positions = jnp.minimum(positions, n_target - 1)
            candidate = order[batch_idx, safe_positions]
            candidate_cells = target_cells[batch_idx, candidate]
            same_cell = jnp.all(
                candidate_cells == adjacent_cells[..., None, :], axis=-1
            )
            candidate_coords = target_batch[batch_idx, candidate]
            delta = query_batch[:, :, None, :] - candidate_coords
            dist2 = jnp.sum(delta * delta, axis=-1)
            valid_candidate = (
                position_valid
                & same_cell
                & jnp.isfinite(dist2)
                & (dmax_value > 0)
                & (dist2 <= cutoff2)
            )
            if exclude_self:
                valid_candidate = valid_candidate & (candidate != source)
            score = dist2
            if query_radii is not None:
                source_r = query_radii[source]
                target_r = target_radii[candidate]
                valid_candidate = valid_candidate & (source_r > 0) & (target_r > 0)
                score = jnp.sqrt(dist2 + EPS) - pair_scale * (source_r + target_r)
            if query_polymer is not None:
                valid_candidate = valid_candidate & (
                    query_polymer[source] | target_polymer[candidate]
                )
            if excluded_codes is not None and excluded_codes.shape[0] > 0:
                lo = jnp.minimum(source, candidate)
                hi = jnp.maximum(source, candidate)
                codes = lo * pair_code_size + hi
                positions = jnp.searchsorted(excluded_codes, codes)
                positions = jnp.minimum(positions, excluded_codes.shape[0] - 1)
                valid_candidate = valid_candidate & (excluded_codes[positions] != codes)
            score = jnp.where(valid_candidate, score, jnp.inf)
            candidate = jnp.where(valid_candidate, candidate, 0)
            best_dist2, best_idx = _merge_cell_candidates_jax(
                best_dist2, best_idx, score, candidate, k
            )
            return base + CELL_CHUNK_SIZE, best_dist2, best_idx

        _base, best_dist2, best_idx = jax.lax.while_loop(
            chunk_cond,
            chunk_body,
            (jnp.int32(0), best_dist2, best_idx),
        )
        return best_dist2, best_idx

    best_dist2, best_idx = jax.lax.fori_loop(
        0, len(CELL_OFFSETS), offset_body, (best_dist2, best_idx)
    )
    return best_idx, best_dist2


def _build_active_vdw_pairs(
    active,
    radii,
    polymer_mask,
    excluded_codes,
    dmax,
    max_neighbors,
    scale=0.75,
):
    """Pure-jax sorted-cell neighbour builder matching the torch implementation.

    It is static-shape and JIT/scan compatible; exclusions are applied before K
    candidates are selected by smallest VdW clearance. Hash buckets are traversed completely in
    fixed-width chunks, so no dense or hash-colliding cell silently loses candidates.
    """

    n_atom = active.shape[-2]
    batch = jax.lax.stop_gradient(active.reshape((-1, n_atom, 3)))
    neighbours, best_dist2 = _build_cell_pairs_jax(
        batch,
        batch,
        dmax,
        max_neighbors,
        exclude_self=True,
        query_radii=radii,
        target_radii=radii,
        pair_scale=scale,
        query_polymer=polymer_mask,
        target_polymer=polymer_mask,
        excluded_codes=excluded_codes,
        pair_code_size=n_atom,
    )
    if neighbours.shape[-1] == 0:
        return neighbours, neighbours.astype(active.dtype)
    source = jnp.arange(n_atom, dtype=jnp.int32).reshape((1, n_atom, 1))
    valid = jnp.isfinite(best_dist2)

    batch_idx = jnp.arange(batch.shape[0], dtype=jnp.int32).reshape((-1, 1, 1))
    reverse_neighbours = neighbours[batch_idx, neighbours]
    reverse_valid = valid[batch_idx, neighbours]
    reverse = jnp.any(
        (reverse_neighbours == source[..., None]) & reverse_valid, axis=-1
    )
    pair_factor = valid.astype(active.dtype) / (1.0 + reverse.astype(active.dtype))
    return neighbours, pair_factor


def _build_fixed_vdw_pairs(
    active, bg_pos, lig_local, dmax, max_neighbors, lig_r=None, bg_r=None, scale=None
):
    """Build moving-ligand to fixed-background neighbours for the current CG block."""

    n_active = active.shape[-2]
    batch = jax.lax.stop_gradient(active.reshape((-1, n_active, 3)))
    lig = batch[:, lig_local, :]
    neighbours, best_dist2 = _build_cell_pairs_jax(
        lig,
        bg_pos,
        dmax,
        max_neighbors,
        query_radii=lig_r,
        target_radii=bg_r,
        pair_scale=scale,
    )
    return neighbours, jnp.isfinite(best_dist2).astype(active.dtype)


def _safe_vdw_diff_jax(diff, source, target, canonical):
    if canonical:
        lo = jnp.minimum(source, target)
        hi = jnp.maximum(source, target)
        code = lo * 31 + hi
        orientation = jnp.where(source <= target, 1.0, -1.0)
    else:
        code = source * 31 + target
        orientation = 1.0
    axis = code % 3
    base_sign = jnp.where((code // 3) % 2 == 0, 1.0, -1.0)
    unit = jnp.stack((axis == 0, axis == 1, axis == 2), axis=-1).astype(diff.dtype)
    fallback = VDW_OVERLAP_EPS * (base_sign * orientation)[..., None] * unit
    effective = diff + jax.lax.stop_gradient(fallback - diff)
    norm2 = jnp.sum(diff * diff, axis=-1)
    return jnp.where((norm2 < VDW_OVERLAP_EPS**2)[..., None], effective, diff)


def _vdw_pair_energy(
    active,
    bg_pos,
    lig_local,
    neighbours,
    pair_mask,
    lig_r,
    bg_r,
    scale,
    weight,
):
    """Fixed-background VdW energy over the per-step neighbour list."""

    n_active, n_bg = active.shape[-2], bg_pos.shape[-2]
    batch = active.reshape((-1, n_active, 3))
    background = bg_pos.reshape((-1, n_bg, 3))
    lig = batch[:, lig_local, :]
    batch_idx = jnp.arange(batch.shape[0], dtype=jnp.int32).reshape((-1, 1, 1))
    other = background[batch_idx, neighbours]
    diff = lig[:, :, None, :] - other
    source = lig_local.reshape((1, -1, 1))
    diff = _safe_vdw_diff_jax(diff, source, neighbours, canonical=False)
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + EPS)
    r_min = scale * (lig_r[None, :, None] + bg_r[neighbours])
    delta = jnp.minimum(dist - r_min, 0.0)
    return weight * jnp.sum(pair_mask * delta**2)


def _active_vdw_pair_energy(active, neighbours, pair_factor, radii, scale, weight):
    """VdW energy over the per-step active-active neighbour list."""

    n_atom = active.shape[-2]
    batch = active.reshape((-1, n_atom, 3))
    batch_idx = jnp.arange(batch.shape[0], dtype=jnp.int32).reshape((-1, 1, 1))
    other = batch[batch_idx, neighbours]
    diff = batch[:, :, None, :] - other
    source = jnp.arange(n_atom, dtype=jnp.int32).reshape((1, n_atom, 1))
    diff = _safe_vdw_diff_jax(diff, source, neighbours, canonical=True)
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
    has_builtin = spec.has_conformer() or spec.has_per_entry()
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
        vdw_dmax = jnp.asarray(float(_vc.dmax))
        vdw_max_neighbors = int(_vc.max_neighbors)
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
    has_static_vdw = spec.has_array_term("vdw")
    has_any_vdw = has_static_vdw or has_vdw or has_active_vdw
    vdw_step_limit = float(getattr(spec, "vdw_max_atom_step", 0.1))
    vdw_rebuild_interval = int(getattr(spec, "vdw_neighbor_rebuild_interval", 10))
    if has_any_vdw:
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
        if has_builtin or has_vdw or has_active_vdw or has_custom:
            if has_any_vdw:
                _s = jnp.asarray(sigma)
                _st = jnp.asarray(step)
                in_win = (
                    (_s <= conf_ss)
                    & (_s >= conf_stop)
                    & (_st >= conf_sstep)
                    & (_st <= conf_estep)
                )
                step_cap = jnp.where(in_win, vdw_step_limit, jnp.inf)
            else:
                step_cap = None
            if has_vdw:
                bg_pos = coords[..., vdw_bg_global, :]
                vdw_w = jnp.where(in_win, vdw_weight, 0.0)
            if has_active_vdw:
                active_vdw_w = jnp.where(in_win, active_vdw_weight, 0.0)

            def build_dynamic_pairs(a, fixed_cutoff, active_cutoff):
                fixed_neighbours = fixed_mask = None
                active_neighbours = active_factor = None
                if has_vdw:
                    fixed_neighbours, fixed_mask = _build_fixed_vdw_pairs(
                        a,
                        bg_pos,
                        vdw_lig_local,
                        fixed_cutoff,
                        vdw_max_neighbors,
                        vdw_lig_r,
                        vdw_bg_r,
                        vdw_scale,
                    )
                if has_active_vdw:
                    active_neighbours, active_factor = _build_active_vdw_pairs(
                        a,
                        active_vdw_radii,
                        active_vdw_polymer,
                        active_vdw_excluded,
                        active_cutoff,
                        active_vdw_max_neighbors,
                        active_vdw_scale,
                    )
                return (
                    fixed_neighbours,
                    fixed_mask,
                    active_neighbours,
                    active_factor,
                )

            def energy_fn(
                a,
                fixed_neighbours,
                fixed_mask,
                active_neighbours,
                active_factor,
            ):
                e = jax_energy.total_energy(a, prepared, sigma, step)
                if has_vdw:
                    e = e + _vdw_pair_energy(
                        a,
                        bg_pos,
                        vdw_lig_local,
                        fixed_neighbours,
                        fixed_mask,
                        vdw_lig_r,
                        vdw_bg_r,
                        vdw_scale,
                        vdw_w,
                    )
                if has_active_vdw:
                    e = e + _active_vdw_pair_energy(
                        a,
                        active_neighbours,
                        active_factor,
                        active_vdw_radii,
                        active_vdw_scale,
                        active_vdw_w,
                    )
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

            if is_cg and (has_vdw or has_active_vdw):
                n_blocks = (max_iter + vdw_rebuild_interval - 1) // vdw_rebuild_interval

                def block_body(block_index, state):
                    current, stopped = state

                    def run_block(a):
                        block_iters = jnp.minimum(
                            vdw_rebuild_interval,
                            max_iter - block_index * vdw_rebuild_interval,
                        )
                        movement = vdw_step_limit * block_iters
                        fixed_cutoff = None
                        active_cutoff = None
                        if has_vdw:
                            max_r_min = vdw_scale * (
                                jnp.max(vdw_lig_r) + jnp.max(vdw_bg_r)
                            )
                            fixed_cutoff = jnp.maximum(vdw_dmax, max_r_min + movement)
                        if has_active_vdw:
                            max_r_min = (
                                active_vdw_scale * 2.0 * jnp.max(active_vdw_radii)
                            )
                            active_cutoff = jnp.maximum(
                                active_vdw_dmax, max_r_min + 2.0 * movement
                            )
                        pairs = build_dynamic_pairs(a, fixed_cutoff, active_cutoff)
                        updated = _cg_minimize(
                            lambda x: energy_fn(x, *pairs),
                            a,
                            block_iters,
                            max_atom_step=step_cap,
                        )
                        return updated, jnp.all(updated == a)

                    return jax.lax.cond(
                        stopped, lambda a: (a, stopped), run_block, current
                    )

                opt, _stopped = jax.lax.fori_loop(
                    0, n_blocks, block_body, (active, jnp.asarray(False))
                )
            elif is_cg:
                opt = _cg_minimize(
                    lambda a: energy_fn(a, None, None, None, None),
                    active,
                    max_iter,
                    max_atom_step=step_cap,
                )
            else:
                pairs = build_dynamic_pairs(
                    active,
                    vdw_dmax if has_vdw else None,
                    active_vdw_dmax if has_active_vdw else None,
                )
                import jaxopt  # only the non-default l-bfgs method needs jaxopt

                opt = (
                    jaxopt.LBFGS(
                        fun=lambda a: energy_fn(a, *pairs),
                        maxiter=max_iter,
                        linesearch="backtracking",
                        implicit_diff=False,
                    )
                    .run(active)
                    .params
                )
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
