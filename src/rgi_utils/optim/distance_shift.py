"""Closed-form COM-distance restraint (no iterative solver).

A distance restraint's energy depends ONLY on the two groups' centres of mass
(``E = f(|COM1 - COM2|)``), so its minimiser is a rigid translation of each group
along the COM-COM axis until the separation hits the target. That is a 1-DOF
problem solvable in closed form -- there is no need to run a CG/LBFGS optimiser
over every group atom (which is what the energy-layer ``distance_energy`` term
forced when it was part of the joint optimisation).

For each restraint d (applied only when ``sigma <= start_sigma[d]``):
  COM1, COM2  -> dist = |COM2-COM1|, unit axis u
  delta (type): harmonic -> target1-dist; flat-bottomed -> nearest-bound gap;
                lower(2) -> max(0,target1-dist); upper(3) -> min(0,target2-dist)
  minimal-displacement split: s1 = -delta*N2/(N1+N2), s2 = +delta*N1/(N1+N2)
  shift every G1 atom by s1*u and every G2 atom by s2*u (masked, scatter-add).

A SINGLE restraint (or several with DISJOINT groups) is solved exactly in one
pass -- this is exactly the configuration the per-atom CG used to converge to (the
distance gradient is uniform within a group, so CG just rigid-translates), but in
one O(N) step instead of ~max_iter iterations. When MULTIPLE restraints SHARE an
atom/group the per-restraint shifts couple, so one pass would satisfy none of
them; we then iterate (Jacobi) ``_COUPLED_PASSES`` times to converge (disjoint
restraints reach the fixed point on pass 1, so later passes are no-ops).

The three backends share the maths; only the gather/scatter primitives differ.
``prepared`` is the ``distance`` dict that ``energy.*_energy.prepare_spec`` already
builds (grp1_idx/grp2_idx local indices, grp1_mask/grp2_mask, target1/target2,
dist_type, mask, start_sigma).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12
# Jacobi passes when restraints may couple (n_dist > 1); 1 restraint never couples.
_COUPLED_PASSES = 16


def apply_distance_shift_numpy(active, d, sigma=None):
    """active: (..., n_active, 3) float64. Returns a shifted copy."""
    active = np.array(active, dtype=np.float64, copy=True)
    idx1, idx2 = d["grp1_idx"], d["grp2_idx"]
    m1, m2 = d["grp1_mask"], d["grp2_mask"]
    t1, t2, dt = d["target1"], d["target2"], d["dist_type"]
    n1 = m1.sum(-1)
    n2 = m2.sum(-1)
    denom = n1 + n2 + _EPS
    gate = d["mask"].astype(np.float64)
    if sigma is not None:
        s = float(sigma)
        gate = gate * (s <= d["start_sigma"]).astype(np.float64)
        # release below stop_sigma (-1 default -> s>=-1 always true -> never released)
        gate = gate * (s >= d["stop_sigma"]).astype(np.float64)
    passes = 1 if idx1.shape[0] <= 1 else _COUPLED_PASSES
    for _ in range(passes):
        g1 = active[..., idx1, :]
        g2 = active[..., idx2, :]
        com1 = (g1 * m1[..., None]).sum(-2) / (n1[..., None] + _EPS)
        com2 = (g2 * m2[..., None]).sum(-2) / (n2[..., None] + _EPS)
        diff = com2 - com1
        dist = np.sqrt((diff * diff).sum(-1) + _EPS)
        u = diff / (dist[..., None] + _EPS)
        delta = _delta(dist, t1, t2, dt, np.where, np.zeros_like) * gate
        com1_shift = (-delta * (n2 / denom))[..., None] * u
        com2_shift = (delta * (n1 / denom))[..., None] * u
        pa1 = _per_atom_shift(com1_shift, m1)
        pa2 = _per_atom_shift(com2_shift, m2)
        _scatter_add_numpy(active, idx1.reshape(-1), pa1)
        _scatter_add_numpy(active, idx2.reshape(-1), pa2)
    return active


def _delta(dist, t1, t2, dt, where, zeros_like):
    """Per-restraint COM-distance correction by type (0=harmonic, 1=flat-bottomed,
    2=lower-bound, 3=upper-bound). ``where``/``zeros_like`` are the backend's ops, so
    this one formula serves numpy/torch/jax."""
    zero = zeros_like(dist)
    dh = t1 - dist
    df = where(dist < t1, t1 - dist, where(dist > t2, t2 - dist, zero))
    dl = where(dist < t1, t1 - dist, zero)
    du = where(dist > t2, t2 - dist, zero)
    return where(dt == 0, dh, where(dt == 1, df, where(dt == 2, dl, du)))


def _per_atom_shift(com_shift, m):
    """Broadcast a per-restraint COM shift ``(..., n_dist, 3)`` to per-(restraint,
    atom) via the group mask, then flatten the ``(n_dist, max_grp)`` dims so it lines
    up with the flattened scatter index. Pure ``*``/``reshape``, so the one helper
    works on every backend (numpy/torch/jax arrays alike)."""
    return (com_shift[..., :, None, :] * m[..., None]).reshape(
        *com_shift.shape[:-2], -1, 3
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


def apply_distance_shift_torch(active, d, sigma=None):
    """active: (..., n_active, 3) torch tensor. Returns a shifted tensor (the caller
    runs this under no_grad / inference_mode(False); pure arithmetic, no autograd)."""
    import torch

    idx1, idx2 = d["grp1_idx"].long(), d["grp2_idx"].long()
    m1, m2 = d["grp1_mask"], d["grp2_mask"]
    t1, t2, dt = d["target1"], d["target2"], d["dist_type"]
    n1 = m1.sum(-1)
    n2 = m2.sum(-1)
    denom = n1 + n2 + _EPS
    gate = d["mask"].to(m1.dtype)
    if sigma is not None:
        s = float(sigma)
        gate = gate * (s <= d["start_sigma"]).to(m1.dtype)
        # release below stop_sigma (-1 default -> s>=-1 always true -> never released)
        gate = gate * (s >= d["stop_sigma"]).to(m1.dtype)
    fidx1, fidx2 = idx1.reshape(-1), idx2.reshape(-1)
    passes = 1 if idx1.shape[0] <= 1 else _COUPLED_PASSES
    for _ in range(passes):
        g1 = active[..., idx1, :]
        g2 = active[..., idx2, :]
        com1 = (g1 * m1[..., None]).sum(-2) / (n1[..., None] + _EPS)
        com2 = (g2 * m2[..., None]).sum(-2) / (n2[..., None] + _EPS)
        diff = com2 - com1
        dist = torch.sqrt((diff * diff).sum(-1) + _EPS)
        u = diff / (dist[..., None] + _EPS)
        delta = _delta(dist, t1, t2, dt, torch.where, torch.zeros_like) * gate
        com1_shift = (-delta * (n2 / denom))[..., None] * u
        com2_shift = (delta * (n1 / denom))[..., None] * u
        pa1 = _per_atom_shift(com1_shift, m1)
        pa2 = _per_atom_shift(com2_shift, m2)
        active = active.index_add(-2, fidx1, pa1.to(active.dtype))
        active = active.index_add(-2, fidx2, pa2.to(active.dtype))
    return active


def apply_distance_shift_jax(active, d, sigma):
    """active: (..., n_active, 3) jax array. Pure + JIT/scan-able. ``sigma`` is a
    (traced) scalar; gating folds into the per-restraint delta."""
    import jax.numpy as jnp

    idx1, idx2 = d["grp1_idx"], d["grp2_idx"]
    m1, m2 = d["grp1_mask"], d["grp2_mask"]
    t1, t2, dt = d["target1"], d["target2"], d["dist_type"]
    n1 = m1.sum(-1)
    n2 = m2.sum(-1)
    denom = n1 + n2 + _EPS
    gate = d["mask"]
    if sigma is not None:
        s = jnp.asarray(sigma)
        gate = gate * (s <= d["start_sigma"]).astype(gate.dtype)
        # release below stop_sigma (-1 default -> s>=-1 always true -> never released)
        gate = gate * (s >= d["stop_sigma"]).astype(gate.dtype)
    fidx1, fidx2 = idx1.reshape(-1), idx2.reshape(-1)
    passes = 1 if idx1.shape[0] <= 1 else _COUPLED_PASSES
    for _ in range(passes):
        g1 = active[..., idx1, :]
        g2 = active[..., idx2, :]
        com1 = (g1 * m1[..., None]).sum(-2) / (n1[..., None] + _EPS)
        com2 = (g2 * m2[..., None]).sum(-2) / (n2[..., None] + _EPS)
        diff = com2 - com1
        dist = jnp.sqrt((diff * diff).sum(-1) + _EPS)
        u = diff / (dist[..., None] + _EPS)
        delta = _delta(dist, t1, t2, dt, jnp.where, jnp.zeros_like) * gate
        com1_shift = (-delta * (n2 / denom))[..., None] * u
        com2_shift = (delta * (n1 / denom))[..., None] * u
        pa1 = _per_atom_shift(com1_shift, m1)
        pa2 = _per_atom_shift(com2_shift, m2)
        active = active.at[..., fidx1, :].add(pa1)
        active = active.at[..., fidx2, :].add(pa2)
    return active
