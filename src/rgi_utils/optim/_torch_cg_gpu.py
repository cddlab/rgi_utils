"""GPU conjugate-gradient for the torch backend: same correct early-exit CG as the CPU
path, but with a ``torch.compile``-d (inductor-fused) energy+grad so it stops losing to CPU.

Why this shape. A single restraint energy+grad eval is **launch-bound** on GPU: it is
dozens of tiny kernels whose launch latency dwarfs their compute (a 690-atom RMSD Kabsch,
or ~200 conformer terms), so the eager CG (`torch_optim._minimize_cg`) ran ~1.6-2.8x
slower than torch-on-CPU. A naive "sync-free" eager rewrite does NOT help: a vmap'd
fixed-width line search either over-computes (large width -> compute-bound RMSD loses) or
cannot reach the fine backtracking steps a stiff term needs (small width -> the chiral
term silently fails to converge).

The torch analogue that is BOTH fast AND correct:
  * keep the proven SEQUENTIAL early-exit line search (reaches arbitrarily fine steps, so
    stiff chiral converges exactly like the CPU/jax path) -- it has host syncs, but they
    are cheap once the eval itself is cheap;
  * make the eval cheap: ``torch.func.grad_and_value(_energy)`` wrapped ONCE in
    ``torch.compile`` so inductor FUSES the dozens of tiny fwd+bwd kernels into far fewer
    launches -- which is what kills the launch-bound cost (measured: conformer x0.32 /
    combined x0.49 vs torch-CPU; rmsd ~parity). It is compiled a single time at module
    scope and reused: the changing data (``prepared`` masks, with the noise gate
    pre-folded in by the caller so there is no python-float ``sigma``) is passed as an
    ARGUMENT, not a closure, so dynamo guards on shapes only and the artifact is reused
    across diffusion steps and structures of the same shape.

NB: the default (inductor) compile mode is deliberate -- ``mode="reduce-overhead"``
(CUDA graphs) is ~5x SLOWER here because the CG feeds a freshly-allocated trial tensor
every line-search step, which makes the CUDA-graph tree re-record each call; plain
inductor fusion has no such static-input requirement.

The two DYNAMIC VdW terms -- the fixed background (default boltz/protenix conformer) and the
active-active polymer neighbour list -- are folded into further compiled energies, one per
combination, so those paths are JIT-compiled too: ``_ENERGY_BY_MODE`` maps the 2-bit mode
(bit 0 = fixed background, bit 1 = active-active) onto ``_energy`` / ``_energy_vdw`` /
``_energy_active_vdw`` / ``_energy_both_vdw``. ``torch_optim._get_custom_cvg`` wraps the SAME
table and adds the custom-restraint closures on top (its artifact must be per-optimizer,
since the closures are spec-specific), so the with- and without-custom compiled paths cannot
drift apart. Used only for CUDA coords; CPU keeps ``_minimize_cg``. Shared ``_cg_config``
constants keep the convergence contract identical to CPU/jax. Any compile failure degrades
to the eager functional CG -- still the correct early-exit algorithm.
"""

from __future__ import annotations

import logging
import math
import os

import torch

from rgi_utils._array_ops import VDW_OVERLAP_EPS
from rgi_utils._config_util import VDW_SCALE_DEFAULT
from rgi_utils.energy import torch_energy
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

logger = logging.getLogger(__name__)

# set RGI_DISABLE_COMPILE=1 to force the eager functional CG (debugging / unsupported env)
_COMPILE_DISABLED = os.environ.get("RGI_DISABLE_COMPILE", "") not in ("", "0", "false")
# Per-artifact mode: bit 0=fixed-background VdW, bit 1=active-active VdW. One
# artifact failing must not disable the independent modes for the rest of the batch.
_compile_failed = {0: False, 1: False, 2: False, 3: False}
_CVG_BY_MODE = {}

# Defense-in-depth: the compiled energy is module-global and reused across all structures
# in a process, so an unforeseen value-specialized leaf must not silently trip dynamo's
# recompile cap and drop to PERMANENT eager. Raise the limits once (best-effort across
# torch versions). The real fix is keeping python-float scalars OUT of the compiled pytree.
try:
    import torch._dynamo as _dynamo

    for _attr in (
        "recompile_limit",
        "accumulated_recompile_limit",
        "cache_size_limit",
        "accumulated_cache_size_limit",
    ):
        if hasattr(_dynamo.config, _attr):
            setattr(_dynamo.config, _attr, max(int(getattr(_dynamo.config, _attr)), 64))
except Exception:
    pass


def _energy(a, prepared):
    """Pure pre-gated restraint energy (distance + conformer + RMSD + group; every active
    term, distance now an autodiff CG term). ``prepared`` carries the noise gate folded into
    its masks, so there is no ``sigma`` argument -- which is what lets the compiled graph be
    reused across steps."""
    return torch_energy.total_energy(a, prepared, sigma=None)


def _cell_hash_torch(cells):
    """Return an int32 spatial hash; callers verify cells after hash lookup."""
    return (
        cells[..., 0] * CELL_HASH_PRIMES[0]
        ^ cells[..., 1] * CELL_HASH_PRIMES[1]
        ^ cells[..., 2] * CELL_HASH_PRIMES[2]
    )


def _merge_cell_candidates_torch(best_dist2, best_idx, dist2, candidate, k):
    """Keep the lexicographically smallest ``(distance squared, atom index)`` rows."""
    merged_dist2 = torch.cat((best_dist2, dist2), dim=-1)
    merged_idx = torch.cat((best_idx, candidate), dim=-1)

    # Two stable sorts implement a deterministic lexicographic key without perturbing
    # distances: index is the secondary key and squared distance the primary key.
    by_index = torch.argsort(merged_idx, dim=-1, stable=True)
    merged_dist2 = torch.gather(merged_dist2, -1, by_index)
    merged_idx = torch.gather(merged_idx, -1, by_index)
    by_distance = torch.argsort(merged_dist2, dim=-1, stable=True)
    by_distance = by_distance[..., :k]
    return (
        torch.gather(merged_dist2, -1, by_distance),
        torch.gather(merged_idx, -1, by_distance),
    )


def _build_cell_pairs_torch(
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
    query_batch = query.reshape(-1, n_query, 3).detach()
    target_batch = target.reshape(-1, n_target, 3).detach()
    if query_batch.shape[0] != target_batch.shape[0]:
        raise ValueError("query and target VdW batch dimensions must match")
    max_candidates = n_target - int(exclude_self)
    if n_query == 0 or max_candidates <= 0:
        empty_idx = torch.zeros(
            (query_batch.shape[0], n_query, 0),
            dtype=torch.long,
            device=query.device,
        )
        return empty_idx, empty_idx.to(query.dtype)
    k = min(int(max_neighbors), max_candidates)
    n_batch = query_batch.shape[0]
    device = query.device

    dmax_value = torch.as_tensor(dmax, dtype=query_batch.dtype, device=device)
    cell_width = torch.where(dmax_value > 0, dmax_value, dmax_value.new_tensor(1.0))
    safe_query = torch.nan_to_num(query_batch, nan=0.0, posinf=0.0, neginf=0.0)
    safe_target = torch.nan_to_num(target_batch, nan=0.0, posinf=0.0, neginf=0.0)
    query_cells = torch.floor(safe_query / cell_width).to(torch.int32)
    target_cells = torch.floor(safe_target / cell_width).to(torch.int32)
    target_hashes = _cell_hash_torch(target_cells)
    order = torch.argsort(target_hashes, dim=-1, stable=True)
    sorted_hashes = torch.gather(target_hashes, -1, order)

    own_start = torch.searchsorted(
        sorted_hashes, target_hashes.contiguous(), right=False
    )
    own_end = torch.searchsorted(sorted_hashes, target_hashes.contiguous(), right=True)
    max_bucket = int(torch.max(own_end - own_start).item())

    best_score = torch.full(
        (n_batch, n_query, k),
        float("inf"),
        dtype=query_batch.dtype,
        device=device,
    )
    best_idx = torch.zeros((n_batch, n_query, k), dtype=torch.long, device=device)
    batch_idx = torch.arange(n_batch, device=device).view(-1, 1, 1)
    source = torch.arange(n_query, device=device).view(1, n_query, 1)
    offsets = torch.tensor(CELL_OFFSETS, dtype=torch.int32, device=device)
    chunk_offsets = torch.arange(CELL_CHUNK_SIZE, device=device).view(1, 1, -1)
    cutoff2 = dmax_value * dmax_value

    for offset in offsets:
        adjacent_cells = query_cells + offset
        query_hashes = _cell_hash_torch(adjacent_cells).contiguous()
        starts = torch.searchsorted(sorted_hashes, query_hashes, right=False)
        ends = torch.searchsorted(sorted_hashes, query_hashes, right=True)
        for base in range(0, max_bucket, CELL_CHUNK_SIZE):
            positions = starts[..., None] + base + chunk_offsets
            position_valid = positions < ends[..., None]
            safe_positions = torch.clamp(positions, max=n_target - 1)
            candidate = order[batch_idx, safe_positions]
            candidate_cells = target_cells[batch_idx, candidate]
            same_cell = torch.all(
                candidate_cells == adjacent_cells[..., None, :], dim=-1
            )
            candidate_coords = target_batch[batch_idx, candidate]
            delta = query_batch[:, :, None, :] - candidate_coords
            dist2 = torch.sum(delta * delta, dim=-1)
            valid_candidate = (
                position_valid
                & same_cell
                & torch.isfinite(dist2)
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
                score = torch.sqrt(dist2 + EPS) - pair_scale * (source_r + target_r)
            if query_polymer is not None:
                valid_candidate = valid_candidate & (
                    query_polymer[source] | target_polymer[candidate]
                )
            if excluded_codes is not None and excluded_codes.numel() > 0:
                lo = torch.minimum(source, candidate)
                hi = torch.maximum(source, candidate)
                codes = lo * pair_code_size + hi
                positions = torch.searchsorted(excluded_codes, codes)
                positions = torch.clamp(positions, max=excluded_codes.numel() - 1)
                valid_candidate = valid_candidate & (excluded_codes[positions] != codes)
            score = torch.where(
                valid_candidate, score, torch.full_like(score, float("inf"))
            )
            candidate = torch.where(
                valid_candidate, candidate, torch.zeros_like(candidate)
            )
            best_score, best_idx = _merge_cell_candidates_torch(
                best_score, best_idx, score, candidate, k
            )

    return best_idx, best_score


def build_active_vdw_pairs(
    active,
    radii,
    polymer_mask,
    excluded_codes,
    dmax,
    max_neighbors,
    scale=VDW_SCALE_DEFAULT,
):
    """Build a fixed-width directed neighbour list with a sorted spatial cell list.

    Exclusions are applied before K candidates are selected by smallest VdW clearance.
    Mutual directed rows receive weight 1/2 and one-sided KNN rows weight 1, so every
    physical pair contributes once. The caller holds the list fixed for one CG block.
    Sorting is O(N log N); the 27 neighbouring hash buckets are traversed in bounded
    chunks without a capacity cutoff. Normal molecular density is therefore O(N log N)
    time and O(N*K) memory, while a fully collapsed cell correctly degrades to O(N^2).
    """

    n_atom = active.shape[-2]
    batch = active.reshape(-1, n_atom, 3).detach()
    neighbours, best_score = _build_cell_pairs_torch(
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
        return neighbours, neighbours.to(active.dtype)
    batch_idx = torch.arange(batch.shape[0], device=active.device).view(-1, 1, 1)
    source = torch.arange(n_atom, device=active.device).view(1, n_atom, 1)
    valid = torch.isfinite(best_score)

    reverse_neighbours = neighbours[batch_idx, neighbours]
    reverse_valid = valid[batch_idx, neighbours]
    reverse = ((reverse_neighbours == source.unsqueeze(-1)) & reverse_valid).any(-1)
    pair_factor = valid.to(active.dtype) / (1.0 + reverse.to(active.dtype))
    return neighbours, pair_factor


def build_fixed_vdw_pairs(
    active, bg_pos, lig_local, dmax, max_neighbors, lig_r=None, bg_r=None, scale=None
):
    """Build moving-ligand to fixed-background neighbours for the current CG block."""

    n_active = active.shape[-2]
    batch = active.reshape(-1, n_active, 3).detach()
    lig = batch[:, lig_local, :]
    neighbours, best_score = _build_cell_pairs_torch(
        lig,
        bg_pos,
        dmax,
        max_neighbors,
        query_radii=lig_r,
        target_radii=bg_r,
        pair_scale=scale,
    )
    return neighbours, torch.isfinite(best_score).to(active.dtype)


def _safe_vdw_diff_torch(diff, source, target, canonical):
    if canonical:
        lo = torch.minimum(source, target)
        hi = torch.maximum(source, target)
        code = lo * 31 + hi
        orientation = torch.where(source <= target, 1.0, -1.0)
    else:
        code = source * 31 + target
        orientation = 1.0
    axis = code % 3
    base_sign = torch.where((code // 3) % 2 == 0, 1.0, -1.0)
    unit = torch.stack((axis == 0, axis == 1, axis == 2), dim=-1).to(diff.dtype)
    fallback = VDW_OVERLAP_EPS * (base_sign * orientation)[..., None] * unit
    effective = diff + (fallback - diff).detach()
    norm2 = torch.sum(diff * diff, dim=-1)
    return torch.where((norm2 < VDW_OVERLAP_EPS**2)[..., None], effective, diff)


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
    """Fixed-background VdW energy over a fixed per-step neighbour list."""

    n_active, n_bg = active.shape[-2], bg_pos.shape[-2]
    batch = active.reshape(-1, n_active, 3)
    background = bg_pos.reshape(-1, n_bg, 3)
    if neighbours.shape[-1] == 0:
        return torch.sum(batch) * 0.0
    lig = batch[:, lig_local, :]
    batch_idx = torch.arange(batch.shape[0], device=active.device).view(-1, 1, 1)
    other = background[batch_idx, neighbours]
    diff = lig[:, :, None, :] - other
    source = lig_local.reshape(1, -1, 1)
    diff = _safe_vdw_diff_torch(diff, source, neighbours, canonical=False)
    dist = torch.sqrt(torch.sum(diff**2, dim=-1) + EPS)
    r_min = scale * (lig_r[None, :, None] + bg_r[neighbours])
    delta = torch.clamp(dist - r_min, max=0.0)
    return weight * torch.sum(pair_mask * delta**2)


def active_vdw_pair_energy(active, neighbours, pair_factor, radii, scale, weight):
    """VdW energy over a fixed per-step active-active neighbour list."""

    n_atom = active.shape[-2]
    batch = active.reshape(-1, n_atom, 3)
    if neighbours.shape[-1] == 0:
        return torch.sum(batch) * 0.0
    batch_idx = torch.arange(batch.shape[0], device=active.device).view(-1, 1, 1)
    other = batch[batch_idx, neighbours]
    diff = batch[:, :, None, :] - other
    source = torch.arange(n_atom, device=active.device).reshape(1, n_atom, 1)
    diff = _safe_vdw_diff_torch(diff, source, neighbours, canonical=True)
    dist = torch.sqrt(torch.sum(diff**2, dim=-1) + EPS)
    r_min = scale * (radii[None, :, None] + radii[neighbours])
    delta = torch.clamp(dist - r_min, max=0.0)
    return weight * torch.sum(pair_factor * delta**2)


def _energy_vdw(
    a,
    prepared,
    bg_pos,
    lig_local,
    neighbours,
    pair_mask,
    lig_r,
    bg_r,
    scale,
    weight,
):
    """``_energy`` + the dynamic fixed-background VdW term, as one compiled energy so the
    default boltz/protenix conformer (which uses the dynamic VdW) is JIT-compiled too."""
    return _energy(a, prepared) + _vdw_pair_energy(
        a,
        bg_pos,
        lig_local,
        neighbours,
        pair_mask,
        lig_r,
        bg_r,
        scale,
        weight,
    )


def _energy_active_vdw(a, prepared, neighbours, pair_factor, radii, scale, weight):
    return _energy(a, prepared) + active_vdw_pair_energy(
        a, neighbours, pair_factor, radii, scale, weight
    )


def _energy_both_vdw(
    a,
    prepared,
    bg_pos,
    lig_local,
    fixed_neighbours,
    fixed_pair_mask,
    lig_r,
    bg_r,
    fixed_scale,
    fixed_weight,
    neighbours,
    pair_factor,
    radii,
    active_scale,
    active_weight,
):
    return (
        _energy(a, prepared)
        + _vdw_pair_energy(
            a,
            bg_pos,
            lig_local,
            fixed_neighbours,
            fixed_pair_mask,
            lig_r,
            bg_r,
            fixed_scale,
            fixed_weight,
        )
        + active_vdw_pair_energy(
            a,
            neighbours,
            pair_factor,
            radii,
            active_scale,
            active_weight,
        )
    )


# VdW mode -> energy fn, keyed by the same bits as ``_compile_failed`` (0=fixed-background,
# 1=active-active). Shared with ``torch_optim._get_custom_cvg``, which wraps the SAME base
# energy and adds the custom closures on top, so the two compiled paths cannot drift.
_ENERGY_BY_MODE = {
    0: _energy,
    1: _energy_vdw,
    2: _energy_active_vdw,
    3: _energy_both_vdw,
}


def _get_cvg(mode=0):
    """Return the compiled grad/value artifact for the requested VdW mode."""

    if _COMPILE_DISABLED or _compile_failed[mode]:
        return None
    try:
        if mode not in _CVG_BY_MODE:
            _CVG_BY_MODE[mode] = torch.compile(
                torch.func.grad_and_value(_ENERGY_BY_MODE[mode], argnums=0),
                fullgraph=False,
            )
        return _CVG_BY_MODE[mode]
    except Exception as exc:
        logger.warning("torch.compile of the GPU CG energy failed (%s); eager", exc)
        _compile_failed[mode] = True
        return None


def _cg_minimize_torch(
    vg, x0, max_iter, max_ls=MAX_LS, gtol=GTOL, ftol=FTOL, max_atom_step=None
):
    """Sequential nonlinear CG (Polak-Ribiere+, backtracking Armijo line search, restart
    on non-descent) — the exact algorithm of ``torch_optim._minimize_cg`` but functional:
    ``vg(x) -> (grad, value)``. ``x0`` is the active-site coords. Returns the optimized
    coords (keeps ``x0`` if non-finite). The early-exit line search reaches arbitrarily
    fine steps, so stiff terms (chiral) converge as on CPU; the host scalar reads are
    cheap because each ``vg`` is a fused compiled call."""
    x = x0
    g, e = vg(x)
    f = float(e)
    if float(g.abs().max()) < gtol:
        return x
    d = -g
    gg = torch.sum(g * g)
    carried = LS_STEP_MAX  # warm-started trial step (see _cg_config)
    for _ in range(max_iter):
        gg_v, dg_v = torch.stack((gg, torch.sum(d * g))).tolist()
        if not math.isfinite(gg_v) or gg_v <= GG_FLOOR:
            break
        if dg_v >= 0.0:  # not a descent direction -> restart (rare)
            d = g.neg()
            dg_v = float(torch.sum(d * g))
        slope = dg_v
        xbase = x
        step = min(LS_STEP_MAX, max(carried, LS_STEP_MIN) * LS_STEP_GROW)
        accepted = False
        gt = g
        for _ in range(max_ls):
            delta = step * d
            if max_atom_step is not None:
                atom_norm = torch.sqrt(
                    torch.sum(delta * delta, dim=-1, keepdim=True) + EPS
                )
                delta = delta * torch.clamp(max_atom_step / atom_norm, max=1.0)
            xt = xbase + delta
            gt, e = vg(xt)
            if max_atom_step is None:
                f_new, predicted = float(e), step * slope
            else:
                f_new, predicted = torch.stack((e, torch.sum(g * delta))).tolist()
            if f_new <= f + ARMIJO_C1 * predicted:
                accepted = True
                break
            step *= BACKTRACK
        if not accepted:  # line search exhausted -> converged / stuck
            x = xbase
            break
        carried = step
        x = xt
        gmax_v, pr_num_v = torch.stack(
            (gt.abs().max(), torch.sum(gt * (gt - g)))
        ).tolist()
        converged = gmax_v < gtol or abs(f_new - f) < ftol * (1.0 + abs(f))
        beta = max(0.0, pr_num_v / (gg_v + EPS))  # PR+ (gg_v = this iter's gg)
        d = gt.neg() + beta * d
        f, g, gg = f_new, gt, torch.sum(gt * gt)
        if converged:
            break
    if not torch.isfinite(x).all():
        return x0
    return x


def gpu_cg(prepared, x0, max_iter, vdw=None, active_vdw=None, max_atom_step=None):
    """Run the CG on CUDA coords. ``prepared`` is the (stable, pre-gated) energy dict;
    ``x0`` the active coords; ``vdw`` carries the fixed-background positions, atom data,
    and per-step neighbour list; ``active_vdw`` similarly carries the fixed per-step
    polymer neighbour list.
    The energy+grad is inductor-compiled for CUDA coords (the non-GPU tests
    exercise the eager functional CG, the same correct algorithm); any compile/runtime
    failure degrades permanently to eager. Returns optimized coords."""
    mode = (1 if vdw is not None else 0) | (2 if active_vdw is not None else 0)
    cvg = _get_cvg(mode=mode) if x0.is_cuda else None
    if cvg is not None:
        if mode == 0:

            def vg(x):
                return cvg(x, prepared)

        elif mode == 1:

            def vg(x):
                return cvg(x, prepared, *vdw)

        elif mode == 2:

            def vg(x):
                return cvg(x, prepared, *active_vdw)

        else:

            def vg(x):
                return cvg(x, prepared, *vdw, *active_vdw)

        try:
            return _cg_minimize_torch(vg, x0, max_iter, max_atom_step=max_atom_step)
        except (
            Exception
        ) as exc:  # this artifact's runtime failure -> eager, permanently
            logger.warning("GPU CG (compiled) failed at runtime (%s); eager", exc)
            _compile_failed[mode] = True

    # eager fallback (CPU coords, compile disabled, or compiled artifact failed): the same
    # correct early-exit CG on the same (vdw-augmented) energy used by the compiled path.
    base = _ENERGY_BY_MODE[mode]
    extra = (vdw or ()) + (active_vdw or ())

    def e_of(a):
        return base(a, prepared, *extra)

    return _cg_minimize_torch(
        torch.func.grad_and_value(e_of), x0, max_iter, max_atom_step=max_atom_step
    )
