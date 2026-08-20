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
default ``max_iter=100`` they agree on the standard fixtures. (That stall caveat was
measured against the cold-start line search, which restarted every iteration at step 1.0
and so exhausted ``MAX_LS`` more readily; the warm start makes it rarer, but the caveat
is kept until it is re-measured.)

The dynamic VdW neighbour lists are rebuilt on measured displacement against a Verlet
skin, not on a fixed cadence. ``lax.fori_loop`` needs a static trip count, so the number
of blocks stays fixed and it is the REBUILD that is ``lax.cond``-gated; the neighbour
arrays and the CG state ride in the loop carry, so a block that does not rebuild neither
re-evaluates the energy nor loses the conjugate direction. Keep this in step with
``torch_optim`` — the displacement metric, the 1x/2x skin asymmetry and the constant
movement bound are deliberately identical in both files.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp

from rgi_utils._array_ops import VDW_OVERLAP_EPS
from rgi_utils._config_util import (
    VDW_MAX_ATOM_STEP_DEFAULT,
    VDW_NEIGHBOR_REBUILD_INTERVAL_DEFAULT,
    VDW_NEIGHBOR_SKIN_DEFAULT,
    VDW_SCALE_DEFAULT,
)
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
    LS_STEP_GROW,
    LS_STEP_MAX,
    LS_STEP_MIN,
    MAX_LS,
)
from rgi_utils.spec import check_active_vdw_int32_safe

logger = logging.getLogger(__name__)


def _max_disp(current, reference):
    """Largest per-atom Euclidean displacement between two coordinate sets.

    The Euclidean norm, NOT a component-wise max: the latter underestimates the true
    displacement by up to sqrt(3) and would let a neighbour list go stale unnoticed. The
    torch optimizer computes the same quantity — keep the two in step.
    """
    return jnp.max(jnp.linalg.norm(current - reference, axis=-1))


def _cg_minimize(
    energy_fn,
    x0,
    max_iter,
    max_ls=MAX_LS,
    gtol=GTOL,
    ftol=FTOL,
    max_atom_step=None,
    state=None,
    return_state=False,
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

    def line_search(x_base, d, f, g_proto, slope, step0):
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
            step0,
            jnp.asarray(False),
            x_base,
            f,
            g_proto,
            jnp.asarray(0),
        )
        # On acceptance `body` leaves the slot at the accepted step; on exhaustion it holds
        # the already-shrunk `step * BACKTRACK`. The caller only carries it when `accepted`.
        s, accepted, xt, ft, gt, _i = jax.lax.while_loop(cond, body, init)
        return xt, ft, gt, accepted, s

    def _fresh():
        f_, g_ = vg(x0)
        return (
            f_,
            g_,
            -g_,
            jnp.sum(g_ * g_),
            jnp.asarray(LS_STEP_MAX),
            jnp.max(jnp.abs(g_)) < gtol,
        )

    if state is None:
        f0, g0, d0, gg0, s0, stop0 = _fresh()
    else:
        # Resume a previous block. `lax.cond` EXECUTES only the taken branch, so a carried
        # block genuinely pays no re-entry evaluation. Mirrors the torch `state` argument;
        # the caller must pass None after a neighbour rebuild (the objective changed).
        f_c, g_c, d_c, gg_c, s_c, valid = state
        f0, g0, d0, gg0, s0, stop0 = jax.lax.cond(
            valid,
            lambda: (f_c, g_c, d_c, gg_c, s_c, jnp.asarray(False)),
            _fresh,
        )

    def cond(st):
        _x, _f, _g, _d, _gg, it, stop, _step = st
        return jnp.logical_and(jnp.logical_not(stop), it < max_iter)

    def body(st):
        x, f, g, d, gg, it, _stop, carried = st
        bad = jnp.logical_or(jnp.logical_not(jnp.isfinite(gg)), gg <= GG_FLOOR)
        d = jnp.where(jnp.sum(d * g) >= 0.0, -g, d)  # restart if not a descent dir
        # On a degenerate iteration (non-finite / underflowed gg) zero the search
        # direction so the line search accepts step 0 in ONE eval (no wasted
        # backtracking) — matching torch's pre-line-search break; `use` discards it.
        d = jnp.where(bad, jnp.zeros_like(d), d)
        slope = jnp.sum(d * g)
        step0 = jnp.minimum(
            LS_STEP_MAX, jnp.maximum(carried, LS_STEP_MIN) * LS_STEP_GROW
        )
        xt, ft, gt, accepted, acc_step = line_search(x, d, f, g, slope, step0)
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
        # Carry the accepted step only on a real iteration: on a `bad` one `d` was zeroed so
        # the step is meaningless, and on exhaustion the slot holds the shrunk trial value.
        nstep = jnp.where(use, acc_step, carried)
        stop_next = jnp.logical_or(bad, jnp.logical_or(jnp.logical_not(accepted), conv))
        return (nx, nf, ng, nd, ngg, it + 1, stop_next, nstep)

    init = (x0, f0, g0, d0, gg0, jnp.asarray(0), stop0, s0)
    xf, ff, gf, df, ggf, _it, stopf, sf = jax.lax.while_loop(cond, body, init)
    if return_state:
        # `stop` is False exactly when the loop exited on the iteration budget, i.e. the
        # solver is still live and the next block can resume it.
        return xf, (ff, gf, df, ggf, sf, jnp.logical_not(stopf))
    return xf


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
    scale=VDW_SCALE_DEFAULT,
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
    vdw_step_limit = float(
        getattr(spec, "vdw_max_atom_step", VDW_MAX_ATOM_STEP_DEFAULT)
    )
    # How often a dynamic list is CHECKED for staleness; the rebuild itself is triggered by
    # measured displacement against `vdw_skin` (see torch_optim, kept deliberately in step).
    vdw_rebuild_interval = int(
        getattr(
            spec,
            "vdw_neighbor_rebuild_interval",
            VDW_NEIGHBOR_REBUILD_INTERVAL_DEFAULT,
        )
    )
    vdw_skin = float(getattr(spec, "vdw_neighbor_skin", VDW_NEIGHBOR_SKIN_DEFAULT))
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

            def empty_dynamic_pairs(a):
                """Static-shape zero neighbour lists for an inactive VdW window."""
                n_batch = a.reshape((-1, a.shape[-2], 3)).shape[0]
                fixed_neighbours = fixed_mask = None
                active_neighbours = active_factor = None
                if has_vdw:
                    n_neighbour = min(vdw_max_neighbors, bg_pos.shape[-2])
                    shape = (
                        n_batch,
                        vdw_lig_local.shape[0],
                        n_neighbour,
                    )
                    fixed_neighbours = jnp.zeros(shape, dtype=jnp.int32)
                    fixed_mask = jnp.zeros(shape, dtype=a.dtype)
                if has_active_vdw:
                    n_neighbour = min(active_vdw_max_neighbors, max(0, a.shape[-2] - 1))
                    shape = (n_batch, a.shape[-2], n_neighbour)
                    active_neighbours = jnp.zeros(shape, dtype=jnp.int32)
                    active_factor = jnp.zeros(shape, dtype=a.dtype)
                return (
                    fixed_neighbours,
                    fixed_mask,
                    active_neighbours,
                    active_factor,
                )

            def initial_dynamic_pairs(a, fixed_cutoff, active_cutoff):
                # Another restraint can keep _descend active after the conformer window
                # closes. Avoid the O(N log N) cell-list build when both dynamic VdW
                # weights are zero while preserving the exact static carry shapes.
                return jax.lax.cond(
                    in_win,
                    lambda x: build_dynamic_pairs(x, fixed_cutoff, active_cutoff),
                    empty_dynamic_pairs,
                    a,
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
                # `movement` is the worst-case travel between two staleness CHECKS, so it
                # uses the CONSTANT interval, not the traced per-block count: a list now
                # survives many blocks and the bound must cover one unchecked stretch.
                # Keep this expression identical to torch_optim's, including the 1x / 2x
                # asymmetry (only the ligand moves against the fixed background).
                movement = vdw_step_limit * vdw_rebuild_interval
                fixed_cutoff = None
                active_cutoff = None
                if has_vdw:
                    _mr = vdw_scale * (jnp.max(vdw_lig_r) + jnp.max(vdw_bg_r))
                    fixed_cutoff = jnp.maximum(vdw_dmax, _mr + movement + vdw_skin)
                if has_active_vdw:
                    _mr = active_vdw_scale * 2.0 * jnp.max(active_vdw_radii)
                    active_cutoff = jnp.maximum(
                        active_vdw_dmax, _mr + 2.0 * movement + vdw_skin
                    )

                # Build once BEFORE the loop: the carry then has concrete shapes and there
                # is no "block 0 must always build" special case. (Seeding the references
                # with inf does not work -- inf - inf is nan and nan > T is False.)
                _n0, _m0, _a0, _f0 = initial_dynamic_pairs(
                    active, fixed_cutoff, active_cutoff
                )
                # The CG state's dtypes must match the loop carry EXACTLY (fori_loop demands
                # an invariant carry), and the energy dtype is not simply the coord dtype --
                # it depends on x64 and on the prepared arrays. Take the exact structure from
                # an abstract trace rather than guessing: eval_shape runs no computation.
                _probe = jax.eval_shape(
                    lambda a: _cg_minimize(
                        lambda x: energy_fn(x, _n0, _m0, _a0, _f0),
                        a,
                        0,
                        max_atom_step=step_cap,
                        return_state=True,
                    ),
                    active,
                )
                carry = {
                    "x": active,
                    "stopped": jnp.asarray(False),
                    # CG state carried across blocks; the trailing flag marks it invalid
                    # until a block has actually produced one.
                    "cg": tuple(jnp.zeros(s.shape, s.dtype) for s in _probe[1]),
                }
                if has_vdw:
                    carry["fn"], carry["fm"] = _n0, _m0
                    carry["fref"] = active[..., vdw_lig_local, :]
                if has_active_vdw:
                    carry["an"], carry["af"] = _a0, _f0
                    carry["aref"] = active

                def block_body(block_index, c):
                    def run_block(c):
                        a = c["x"]
                        new = dict(c)
                        rebuilt = jnp.asarray(False)
                        if has_vdw:
                            lig = a[..., vdw_lig_local, :]
                            # `in_win` gate: outside the conformer window step_cap is inf,
                            # so displacement is unbounded and every block would rebuild --
                            # while the VdW weight is 0, so the list does not matter. torch
                            # gets this structurally (no dynamic list => a single block).
                            need = jnp.logical_and(
                                in_win, _max_disp(lig, c["fref"]) > vdw_skin
                            )
                            nb, mask = jax.lax.cond(
                                need,
                                lambda: _build_fixed_vdw_pairs(
                                    a,
                                    bg_pos,
                                    vdw_lig_local,
                                    fixed_cutoff,
                                    vdw_max_neighbors,
                                    vdw_lig_r,
                                    vdw_bg_r,
                                    vdw_scale,
                                ),
                                lambda: (c["fn"], c["fm"]),
                            )
                            new["fn"], new["fm"] = nb, mask
                            new["fref"] = jnp.where(need, lig, c["fref"])
                            rebuilt = jnp.logical_or(rebuilt, need)
                        if has_active_vdw:
                            # both endpoints move, so half the budget each
                            need = jnp.logical_and(
                                in_win, _max_disp(a, c["aref"]) > 0.5 * vdw_skin
                            )
                            nb, factor = jax.lax.cond(
                                need,
                                lambda: _build_active_vdw_pairs(
                                    a,
                                    active_vdw_radii,
                                    active_vdw_polymer,
                                    active_vdw_excluded,
                                    active_cutoff,
                                    active_vdw_max_neighbors,
                                    active_vdw_scale,
                                ),
                                lambda: (c["an"], c["af"]),
                            )
                            new["an"], new["af"] = nb, factor
                            new["aref"] = jnp.where(need, a, c["aref"])
                            rebuilt = jnp.logical_or(rebuilt, need)

                        pairs = (
                            new.get("fn"),
                            new.get("fm"),
                            new.get("an"),
                            new.get("af"),
                        )
                        block_iters = jnp.minimum(
                            vdw_rebuild_interval,
                            max_iter - block_index * vdw_rebuild_interval,
                        )
                        # a rebuild changed the objective -> the carried state describes the
                        # OLD pair list, so invalidate it (one evaluation per REBUILD, which
                        # is now rare, instead of one per block)
                        cg_in = c["cg"][:-1] + (
                            jnp.logical_and(c["cg"][-1], jnp.logical_not(rebuilt)),
                        )
                        updated, cg_out = _cg_minimize(
                            lambda x: energy_fn(x, *pairs),
                            a,
                            block_iters,
                            max_atom_step=step_cap,
                            state=cg_in,
                            return_state=True,
                        )
                        new["x"] = updated
                        new["cg"] = cg_out
                        new["stopped"] = jnp.all(updated == a)
                        return new

                    return jax.lax.cond(c["stopped"], lambda c: c, run_block, c)

                opt = jax.lax.fori_loop(0, n_blocks, block_body, carry)["x"]
            elif is_cg:
                opt = _cg_minimize(
                    lambda a: energy_fn(a, None, None, None, None),
                    active,
                    max_iter,
                    max_atom_step=step_cap,
                )
            else:
                pairs = initial_dynamic_pairs(
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


def dynamic_vdw_energy(spec, coords) -> float:
    """Return both optimizer-only VdW residuals for host-side diagnostics."""
    if not spec.is_active():
        return 0.0
    vc = getattr(spec, "vdw_config", None)
    ac = getattr(spec, "active_vdw_config", None)
    has_vdw = vc is not None and vc.weight > 0
    has_active_vdw = ac is not None and ac.weight > 0
    if not (has_vdw or has_active_vdw):
        return 0.0

    coords = jnp.asarray(coords)
    active = coords[..., jnp.asarray(spec.active_sites, dtype=jnp.int32), :]
    dtype = active.dtype
    total = 0.0
    if has_vdw:
        lig_local = jnp.asarray(vc.ligand_local, dtype=jnp.int32)
        lig_r = jnp.asarray(vc.ligand_radii, dtype=dtype)
        bg_r = jnp.asarray(vc.background_radii, dtype=dtype)
        scale = jnp.asarray(float(vc.scale), dtype=dtype)
        bg_pos = coords[..., jnp.asarray(vc.background_global, dtype=jnp.int32), :]
        neighbours, pair_mask = _build_fixed_vdw_pairs(
            active,
            bg_pos,
            lig_local,
            jnp.asarray(float(vc.dmax), dtype=dtype),
            int(vc.max_neighbors),
            lig_r,
            bg_r,
            scale,
        )
        total += float(
            _vdw_pair_energy(
                active,
                bg_pos,
                lig_local,
                neighbours,
                pair_mask,
                lig_r,
                bg_r,
                scale,
                jnp.asarray(float(vc.weight), dtype=dtype),
            )
        )
    if has_active_vdw:
        check_active_vdw_int32_safe(int(ac.radii.shape[0]))
        radii = jnp.asarray(ac.radii, dtype=dtype)
        scale = jnp.asarray(float(ac.scale), dtype=dtype)
        neighbours, pair_factor = _build_active_vdw_pairs(
            active,
            radii,
            jnp.asarray(ac.polymer_mask, dtype=bool),
            jnp.asarray(ac.excluded_codes, dtype=jnp.int32),
            jnp.asarray(float(ac.dmax), dtype=dtype),
            int(ac.max_neighbors),
            scale,
        )
        total += float(
            _active_vdw_pair_energy(
                active,
                neighbours,
                pair_factor,
                radii,
                scale,
                jnp.asarray(float(ac.weight), dtype=dtype),
            )
        )
    return total
