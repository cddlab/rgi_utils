"""Closed-form centroid-distance restraint (no iterative solver).

A distance restraint's energy depends ONLY on the two groups' centroids
(``E = f(|centroid1 - centroid2|)``), so its minimiser is a rigid translation of each group
along the centroid-centroid axis until the separation hits the target. That is a 1-DOF
problem solvable in closed form -- there is no need to run a CG/LBFGS optimiser
over every group atom (which is what the energy-layer ``distance_energy`` term
forced when it was part of the joint optimisation).

For each restraint d (applied only when ``sigma <= start_sigma[d]``):
  centroid1, centroid2  -> dist = |centroid2-centroid1|, unit axis u
  delta (type): harmonic -> target1-dist; flat-bottomed -> nearest-bound gap;
                lower(2) -> max(0,target1-dist); upper(3) -> min(0,target2-dist)
  split by move_mode (the `move` config key): both(0) = minimal-displacement
                s1 = -delta*N2/(N1+N2), s2 = +delta*N1/(N1+N2); move 1 = -delta,0
                (only G1 moves); move 2 = 0,+delta (only G2 moves) -- see _split
  shift every G1 atom by s1*u and every G2 atom by s2*u (masked, scatter-add).

A SINGLE restraint (or several with DISJOINT groups) is solved exactly in one
pass -- this is exactly the configuration the per-atom CG used to converge to (the
distance gradient is uniform within a group, so CG just rigid-translates), but in
one O(N) step instead of ~max_iter iterations. When MULTIPLE restraints SHARE an
atom/group the per-restraint shifts couple, so one pass would satisfy none of
them; we then iterate (Jacobi) ``_COUPLED_PASSES`` times to converge (disjoint
restraints reach the fixed point on pass 1, so later passes are no-ops).

The three backends share ONE body (``_apply_distance_shift``); a small per-backend
``ops`` facade carries the gather/scatter/cast primitives that differ (the same
ops-facade split as ``energy/_terms.py`` and ``custom/backends.py`` -- the arithmetic is
identical across array libraries via operator overloading, so only sqrt/where/zeros_like/
index/cast/scatter need a backend). ``prepared`` is the ``distance`` dict that
``energy.*_energy.prepare_spec`` already builds (grp1_idx/grp2_idx local indices,
grp1_mask/grp2_mask, target1/target2, dist_type, move_mode, mask, start_sigma, stop_sigma,
start_step, stop_step). The gate is the active sigma window AND the active step window (a
restraint uses one or the other; the unused axis is always-on). ``move_mode`` / the
``*_step`` keys may be absent in a hand-built dict -> ``move_mode`` treated as 0 (both),
step window treated as always-on.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

_EPS = 1e-12
# Jacobi passes when restraints may couple (n_dist > 1); 1 restraint never couples.
_COUPLED_PASSES = 16

# Per-backend primitive facade. ``copy_in``: clone the coord tensor (numpy must, the
# functional torch/jax are no-ops); ``index``: cast the local-index arrays (torch needs
# .long()); ``asarray``: lift a sigma/step scalar to a backend array WITHOUT float()
# (so the jax wrapper's traced scalars survive lax.scan); ``cast``: bool/array -> the
# reference's float dtype (.astype vs torch's .to); ``scatter_add``: index_add return.
_Ops = namedtuple(
    "_Ops", "copy_in index asarray cast sqrt where zeros_like scatter_add"
)


def _apply_distance_shift(active, d, sigma, step, ops):
    """Closed-form centroid-distance shift shared by the numpy/torch/jax wrappers.

    The maths is written with plain operators (``*``/``+``/``.sum``/``/``), which every
    array library overloads identically; only the primitives in ``ops`` vary per backend.
    The jax wrapper passes TRACED ``sigma``/``step`` scalars, so this body must never
    coerce them with ``float()`` -- it lifts them through ``ops.asarray`` instead, keeping
    the whole function JIT/scan-able.
    """
    active = ops.copy_in(active)
    idx1, idx2 = ops.index(d["grp1_idx"]), ops.index(d["grp2_idx"])
    m1, m2 = d["grp1_mask"], d["grp2_mask"]
    t1, t2, dt = d["target1"], d["target2"], d["dist_type"]
    n1 = m1.sum(-1)
    n2 = m2.sum(-1)
    denom = n1 + n2 + _EPS
    mm = d.get("move_mode")  # 0=both / 1=grp1 only / 2=grp2 only (absent -> both)
    mm = ops.zeros_like(dt) if mm is None else mm
    c1, c2 = _split(mm, n1, n2, denom)  # constant across passes (no coord dependence)
    gate = ops.cast(d["mask"], m1)
    if sigma is not None:
        s = ops.asarray(sigma)
        gate = gate * ops.cast(s <= d["start_sigma"], m1)
        # release below stop_sigma (-1 default -> s>=-1 always true -> never released)
        gate = gate * ops.cast(s >= d["stop_sigma"], m1)
    if (
        step is not None and "start_step" in d
    ):  # step window (ANDed; -inf/+inf = always)
        st = ops.asarray(step)
        gate = gate * ops.cast(st >= d["start_step"], m1)
        gate = gate * ops.cast(st <= d["stop_step"], m1)
    fidx1, fidx2 = idx1.reshape(-1), idx2.reshape(-1)
    passes = 1 if idx1.shape[0] <= 1 else _COUPLED_PASSES
    for _ in range(passes):
        g1 = active[..., idx1, :]
        g2 = active[..., idx2, :]
        centroid1 = (g1 * m1[..., None]).sum(-2) / (n1[..., None] + _EPS)
        centroid2 = (g2 * m2[..., None]).sum(-2) / (n2[..., None] + _EPS)
        diff = centroid2 - centroid1
        dist = ops.sqrt((diff * diff).sum(-1) + _EPS)
        u = diff / (dist[..., None] + _EPS)
        delta = _delta(dist, t1, t2, dt, ops.where, ops.zeros_like) * gate
        centroid1_shift = (-delta * c1)[..., None] * u
        centroid2_shift = (delta * c2)[..., None] * u
        pa1 = _per_atom_shift(centroid1_shift, m1)
        pa2 = _per_atom_shift(centroid2_shift, m2)
        active = ops.scatter_add(active, fidx1, pa1)
        active = ops.scatter_add(active, fidx2, pa2)
    return active


def apply_distance_shift_numpy(active, d, sigma=None, step=None):
    """active: (..., n_active, 3) float64. Returns a shifted copy."""
    ops = _Ops(
        copy_in=lambda a: np.array(a, dtype=np.float64, copy=True),
        index=lambda i: i,
        asarray=np.asarray,
        cast=lambda x, ref: np.asarray(x).astype(ref.dtype),
        sqrt=np.sqrt,
        where=np.where,
        zeros_like=np.zeros_like,
        scatter_add=_np_scatter_add,
    )
    return _apply_distance_shift(active, d, sigma, step, ops)


def apply_distance_shift_torch(active, d, sigma=None, step=None):
    """active: (..., n_active, 3) torch tensor. Returns a shifted tensor (the caller
    runs this under no_grad / inference_mode(False); pure arithmetic, no autograd)."""
    import torch

    ops = _Ops(
        copy_in=lambda a: a,
        index=lambda i: i.long(),
        asarray=torch.as_tensor,
        cast=lambda x, ref: x.to(ref.dtype),
        sqrt=torch.sqrt,
        where=torch.where,
        zeros_like=torch.zeros_like,
        scatter_add=lambda a, idx, vals: a.index_add(-2, idx, vals.to(a.dtype)),
    )
    return _apply_distance_shift(active, d, sigma, step, ops)


def apply_distance_shift_jax(active, d, sigma, step=None):
    """active: (..., n_active, 3) jax array. Pure + JIT/scan-able. ``sigma`` (and the
    optional ``step``) are (traced) scalars; gating folds into the per-restraint delta."""
    import jax.numpy as jnp

    ops = _Ops(
        copy_in=lambda a: a,
        index=lambda i: i,
        asarray=jnp.asarray,
        cast=lambda x, ref: jnp.asarray(x).astype(ref.dtype),
        sqrt=jnp.sqrt,
        where=jnp.where,
        zeros_like=jnp.zeros_like,
        scatter_add=lambda a, idx, vals: a.at[..., idx, :].add(vals),
    )
    return _apply_distance_shift(active, d, sigma, step, ops)


def _delta(dist, t1, t2, dt, where, zeros_like):
    """Per-restraint centroid-distance correction by type (0=harmonic, 1=flat-bottomed,
    2=lower-bound, 3=upper-bound). ``where``/``zeros_like`` are the backend's ops, so
    this one formula serves numpy/torch/jax."""
    zero = zeros_like(dist)
    dh = t1 - dist
    df = where(dist < t1, t1 - dist, where(dist > t2, t2 - dist, zero))
    dl = where(dist < t1, t1 - dist, zero)
    du = where(dist > t2, t2 - dist, zero)
    return where(dt == 0, dh, where(dt == 1, df, where(dt == 2, dl, du)))


def _split(mm, n1, n2, denom):
    """Per-restraint shift-distribution coefficients ``(c1, c2)`` for the move mode
    ``mm`` (0=both, 1=group1 only, 2=group2 only): ``centroid1_shift = -delta*c1*u``,
    ``centroid2_shift = +delta*c2*u``. "both" is the minimal-displacement split (each group
    moves inversely to its size); 1/2 put the WHOLE shift on group1 / group2 so the
    other group stays fixed (e.g. pull only a ligand toward a fixed pocket). EVERY mode
    changes the centroid separation by ``delta`` (only the distribution differs), so the
    target is reached in every mode. Pure arithmetic (``==``/``*``/``+``) on the masks
    (bool -> float), so one helper serves numpy/torch/jax with no backend ``where``."""
    both = mm == 0
    c1 = (mm == 1) + both * (n2 / denom)  # 1 when move==1, n2/denom when both, else 0
    c2 = (mm == 2) + both * (n1 / denom)  # 1 when move==2, n1/denom when both, else 0
    return c1, c2


def _per_atom_shift(centroid_shift, m):
    """Broadcast a per-restraint centroid shift ``(..., n_dist, 3)`` to per-(restraint,
    atom) via the group mask, then flatten the ``(n_dist, max_grp)`` dims so it lines
    up with the flattened scatter index. Pure ``*``/``reshape``, so the one helper
    works on every backend (numpy/torch/jax arrays alike)."""
    return (centroid_shift[..., :, None, :] * m[..., None]).reshape(
        *centroid_shift.shape[:-2], -1, 3
    )


def _scatter_add_numpy(active, flat_idx, vals):
    """active[..., flat_idx, :] += vals with accumulation over repeated indices,
    for any leading batch shape."""
    n_active = active.shape[-2]
    batch = active.shape[:-2]
    B = int(np.prod(batch)) if batch else 1
    flat = active.reshape(B, n_active, 3)
    k = flat_idx.shape[0]
    b_idx = np.repeat(np.arange(B), k)
    a_idx = np.tile(flat_idx, B)
    np.add.at(flat, (b_idx, a_idx), vals.reshape(B, k, 3).reshape(-1, 3))
    active[...] = flat.reshape(active.shape)


def _np_scatter_add(active, flat_idx, vals):
    """numpy ``ops.scatter_add``: mutate ``active`` in place (the wrapper already cloned
    it) then return it, so the shared body's ``active = ops.scatter_add(...)`` reassign
    matches the functional torch/jax backends."""
    _scatter_add_numpy(active, flat_idx, vals)
    return active
