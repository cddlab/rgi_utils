"""Pure JAX restraint energies — differentiable via ``jax.grad``.

All functions take ``positions`` of shape ``(..., n_active, 3)`` and return a
scalar (leading batch dims are summed over). Indices are local indices into
active_sites. Adapted from the AlphaFold 3 restraint prototype.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rgi_utils.energy._terms import BREAKDOWN_KEYS, pack_spec, term_energies

_EPS = 1e-12


def bond_energy(positions, idx, r0, slack, weight, half, mask):
    """Flat-bottomed bond length energy; ``half`` penalizes stretch only."""
    ai, aj = idx[:, 0], idx[:, 1]
    diff = positions[..., ai, :] - positions[..., aj, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + _EPS)
    r_upper = r0 + slack
    r_lower = r0 - slack
    delta_full = jnp.where(
        dist > r_upper,
        dist - r_upper,
        jnp.where(dist < r_lower, dist - r_lower, 0.0),
    )
    delta_half = jnp.maximum(0.0, dist - r_upper)
    delta = jnp.where(half > 0.5, delta_half, delta_full)
    return jnp.sum(weight * delta**2 * mask)


def angle_energy(positions, idx, th0, slack, weight, mask):
    """Flat-bottomed bond angle energy; vertex is column 1."""
    ai, aj, ak = idx[:, 0], idx[:, 1], idx[:, 2]
    rij = positions[..., ai, :] - positions[..., aj, :]
    rkj = positions[..., ak, :] - positions[..., aj, :]
    nij = jnp.sqrt(jnp.sum(rij**2, axis=-1) + _EPS)
    nkj = jnp.sqrt(jnp.sum(rkj**2, axis=-1) + _EPS)
    cos_th = jnp.sum(rij * rkj, axis=-1) / (nij * nkj)
    cos_th = jnp.clip(cos_th, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = jnp.arccos(cos_th)
    th_u = th0 + slack
    th_l = th0 - slack
    delta = jnp.where(
        theta > th_u, theta - th_u, jnp.where(theta < th_l, theta - th_l, 0.0)
    )
    return jnp.sum(weight * delta**2 * mask)


def chiral_energy(positions, idx, vol0, slack, weight, mask):
    """Flat-bottomed chiral volume energy; center is column 0. Zero within ``±slack``
    of ``vol0`` (reference geometry has zero energy), quadratic outside. Mirrors
    ``numpy_energy.chiral_energy``."""
    a0 = positions[..., idx[:, 0], :]
    a1 = positions[..., idx[:, 1], :]
    a2 = positions[..., idx[:, 2], :]
    a3 = positions[..., idx[:, 3], :]
    v1 = a1 - a0
    v2 = a2 - a0
    v3 = a3 - a0
    vol = jnp.sum(v1 * jnp.cross(v2, v3), axis=-1)
    d = vol - vol0
    delta = jnp.where(d > slack, d - slack, jnp.where(d < -slack, d + slack, 0.0))
    return jnp.sum(weight * delta**2 * mask)


def cistrans_energy(positions, idx, phi0, slack, weight, mask):
    """Flat-bottomed cis/trans (E/Z) torsion energy; bond axis is columns 1-2.

    Periodicity-safe: the deviation ``phi - phi0`` is wrapped to [-pi, pi] before
    the flat-bottomed square penalty. Mirrors ``numpy_energy.cistrans_energy``.
    Pure jnp so it stays JIT/vmap-able inside the AF3 scan.
    """
    p0 = positions[..., idx[:, 0], :]
    p1 = positions[..., idx[:, 1], :]
    p2 = positions[..., idx[:, 2], :]
    p3 = positions[..., idx[:, 3], :]
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    b2n = b2 / jnp.sqrt(
        jnp.sum(b2**2, axis=-1, keepdims=True) + _EPS
    )  # _EPS inside: finite grad at b2=0
    m1 = jnp.cross(n1, b2n)
    x = jnp.sum(n1 * n2, axis=-1)
    y = jnp.sum(m1 * n2, axis=-1)
    # Avoid atan2(0, 0) at exactly-degenerate geometry: jax's bare arctan2(0,0) has a
    # NaN gradient (0*NaN=NaN survives even a masked row) where torch gives 0, so the
    # backends diverged. Nudging x makes it finite and equal to numpy/torch.
    x = jnp.where((x == 0.0) & (y == 0.0), x + _EPS, x)
    phi = jnp.arctan2(y, x)
    d = phi - phi0
    d = jnp.arctan2(jnp.sin(d), jnp.cos(d))  # wrap to [-pi, pi]
    delta = jnp.where(d > slack, d - slack, jnp.where(d < -slack, d + slack, 0.0))
    return jnp.sum(weight * delta**2 * mask)


def vdw_energy(positions, idx, r_min, weight, mask):
    """VdW repulsion (lower-bound only): penalize d < r_min."""
    ai, aj = idx[:, 0], idx[:, 1]
    diff = positions[..., ai, :] - positions[..., aj, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + _EPS)
    delta = jnp.minimum(0.0, dist - r_min)
    return jnp.sum(weight * delta**2 * mask)


def distance_energy(
    positions,
    grp1_idx,
    grp2_idx,
    grp1_mask,
    grp2_mask,
    target1,
    target2,
    dist_type,
    move_mode,
    weight,
    mask,
):
    """Centroid distance energy between two atom groups. dist_type: 0=harmonic,
    1=flat-bottomed, 2=lower-bound, 3=upper-bound. Mirrors ``torch_energy.distance_energy``:
    an autodiff CG term whose two centroids go through ``_move_centroid`` with the REDUCED-MASS
    scale ``mu = N1*N2/(N1+N2)`` (minimal-displacement split N2:N1 + O(1) separation step);
    ``move_mode`` (0=both / 1=group1 only / 2=group2 only) pins the other group via ``free=0``.
    Value == numpy's plain-centroid distance_energy (parity); grad is rescaled (torch-vs-jax)."""
    n1 = jnp.sum(grp1_mask, axis=-1)  # (..., n_dist) group sizes
    n2 = jnp.sum(grp2_mask, axis=-1)
    mu = n1 * n2 / (n1 + n2 + _EPS)  # reduced mass: minimal-displacement + O(1) step
    both = move_mode == 0
    scale1 = jnp.where(both, mu, n1)  # both -> mu; else the moving group uses its own N
    scale2 = jnp.where(both, mu, n2)
    free1 = (move_mode != 2).astype(grp1_mask.dtype)  # group1 free unless move_mode==2
    free2 = (move_mode != 1).astype(grp2_mask.dtype)  # group2 free unless move_mode==1
    centroid1 = _move_centroid(positions, grp1_idx, grp1_mask, free1, scale1)
    centroid2 = _move_centroid(positions, grp2_idx, grp2_mask, free2, scale2)
    diff = centroid2 - centroid1
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + _EPS)
    delta_harmonic = dist - target1
    delta_flat = jnp.where(
        dist < target1, dist - target1, jnp.where(dist > target2, dist - target2, 0.0)
    )
    delta_lower = jnp.minimum(0.0, dist - target1)
    delta_upper = jnp.maximum(0.0, dist - target2)
    delta = jnp.where(
        dist_type == 0,
        delta_harmonic,
        jnp.where(
            dist_type == 1,
            delta_flat,
            jnp.where(dist_type == 2, delta_lower, delta_upper),
        ),
    )
    return jnp.sum(weight * delta**2 * mask)


def _group_centroid(positions, grp_idx, grp_mask):
    """Masked-mean geometric centroid (unweighted; NOT mass-weighted) of a padded atom group (mirrors
    ``numpy_energy._group_centroid``). Pure jnp so it stays JIT/scan-able."""
    pos = positions[..., grp_idx, :]  # (..., n, max_grp, 3)
    m = grp_mask[..., None]
    return jnp.sum(pos * m, axis=-2) / (jnp.sum(grp_mask, axis=-1)[..., None] + _EPS)


def _move_centroid(positions, grp_idx, grp_mask, free, scale=None):
    """Centroid of a group for the restraint gradient (mirrors ``torch_energy._move_centroid``):
    ``centroid_eff = centroid_d + scale*(centroid - centroid_d)`` rescales the per-atom gradient to
    ``scale/N`` x the centroid gradient. ``scale=None`` -> ``scale=N`` (group-restraint default:
    cancels 1/N so the group translates rigidly by the full step, weight=1 drives ANY size). The
    distance restraint passes the REDUCED MASS ``N1*N2/(N1+N2)`` to BOTH groups (ratio N2:N1
    minimal-displacement split + O(1), N-independent separation step). ``free``=0 PINS a group
    (stop-gradient, the move knob). Value == centroid, so energy value parity holds; distance/group
    grad parity is torch-vs-jax."""
    centroid = _group_centroid(positions, grp_idx, grp_mask)
    centroid_d = jax.lax.stop_gradient(centroid)
    if scale is None:
        scale = jnp.sum(grp_mask, axis=-1)  # group size N (..., n_restr) (rigid step)
    centroid_eff = centroid_d + scale[..., None] * (
        centroid - centroid_d
    )  # un-suppress the 1/N centroid gradient (rigid step)
    return jnp.where((free > 0.5)[..., None], centroid_eff, centroid_d)


def _group_delta(val, harmonic_dev, target1, target2, geom_type):
    """Distance-style flat-bottom delta (mirrors ``distance_energy``). geom_type 0=
    harmonic (uses ``harmonic_dev``), 1=flat-bottomed, 2=lower, 3=upper."""
    d_flat = jnp.where(
        val < target1, val - target1, jnp.where(val > target2, val - target2, 0.0)
    )
    d_lower = jnp.minimum(0.0, val - target1)
    d_upper = jnp.maximum(0.0, val - target2)
    return jnp.where(
        geom_type == 0,
        harmonic_dev,
        jnp.where(geom_type == 1, d_flat, jnp.where(geom_type == 2, d_lower, d_upper)),
    )


def _dihedral_angle(p0, p1, p2, p3):
    """Torsion angle (radians) about the p1-p2 axis (mirrors ``cistrans_energy``)."""
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    b2n = b2 / jnp.sqrt(jnp.sum(b2**2, axis=-1, keepdims=True) + _EPS)
    m1 = jnp.cross(n1, b2n)
    x = jnp.sum(n1 * n2, axis=-1)
    y = jnp.sum(m1 * n2, axis=-1)
    x = jnp.where((x == 0.0) & (y == 0.0), x + _EPS, x)
    return jnp.arctan2(y, x)


def group_angle_energy(
    positions,
    grp1_idx,
    grp2_idx,
    grp3_idx,
    grp1_mask,
    grp2_mask,
    grp3_mask,
    target1,
    target2,
    geom_type,
    move_free,
    weight,
    mask,
):
    """Distance-style flat-bottomed angle between three group centroids (vertex = group 2).
    Mirrors ``numpy_energy.group_angle_energy``; ``move_free`` (n,3) pins groups via the
    stop-gradient select in ``_move_centroid``. Pure jnp (JIT/scan/vmap-able)."""
    centroid1 = _move_centroid(positions, grp1_idx, grp1_mask, move_free[..., 0])
    centroid2 = _move_centroid(positions, grp2_idx, grp2_mask, move_free[..., 1])
    centroid3 = _move_centroid(positions, grp3_idx, grp3_mask, move_free[..., 2])
    rij = centroid1 - centroid2
    rkj = centroid3 - centroid2
    nij = jnp.sqrt(jnp.sum(rij**2, axis=-1) + _EPS)
    nkj = jnp.sqrt(jnp.sum(rkj**2, axis=-1) + _EPS)
    cos_th = jnp.sum(rij * rkj, axis=-1) / (nij * nkj)
    cos_th = jnp.clip(cos_th, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = jnp.arccos(cos_th)
    delta = _group_delta(theta, theta - target1, target1, target2, geom_type)
    return jnp.sum(weight * delta**2 * mask)


def group_dihedral_energy(
    positions,
    grp1_idx,
    grp2_idx,
    grp3_idx,
    grp4_idx,
    grp1_mask,
    grp2_mask,
    grp3_mask,
    grp4_mask,
    target1,
    target2,
    geom_type,
    move_free,
    weight,
    mask,
):
    """Distance-style flat-bottomed dihedral between 4 group centroids (axis = centroid2-centroid3).
    harmonic (geom_type 0) is periodicity-safe (deviation wrapped). Mirrors
    ``numpy_energy.group_dihedral_energy``; ``move_free`` pins via ``_move_centroid``."""
    p0 = _move_centroid(positions, grp1_idx, grp1_mask, move_free[..., 0])
    p1 = _move_centroid(positions, grp2_idx, grp2_mask, move_free[..., 1])
    p2 = _move_centroid(positions, grp3_idx, grp3_mask, move_free[..., 2])
    p3 = _move_centroid(positions, grp4_idx, grp4_mask, move_free[..., 3])
    phi = _dihedral_angle(p0, p1, p2, p3)
    dev = phi - target1
    harmonic_dev = jnp.arctan2(jnp.sin(dev), jnp.cos(dev))  # wrap to [-pi, pi]
    delta = _group_delta(phi, harmonic_dev, target1, target2, geom_type)
    return jnp.sum(weight * delta**2 * mask)


# Improper uses the same ordered four-centroid torsion as dihedral.
group_improper_energy = group_dihedral_energy


def _kabsch_R(Q0, P0):
    """Optimal proper rotation R (det +1) s.t. R Q0 ~ P0 (Kabsch). ``H`` (and thus R)
    is wrapped in ``stop_gradient`` so ``jax.grad`` flows only through the moving atoms
    — no SVD backward (unstable at degenerate geometry). Pure jnp so it stays
    JIT/scan-able. Mirrors ``numpy_energy._kabsch_R``."""
    H = jax.lax.stop_gradient(jnp.swapaxes(Q0, -1, -2) @ P0)  # (..., 3, 3)
    U, _S, Vt = jnp.linalg.svd(H)
    V = jnp.swapaxes(Vt, -1, -2)
    d = jnp.sign(jnp.linalg.det(V @ jnp.swapaxes(U, -1, -2)))
    d = jnp.where(d == 0.0, 1.0, d)
    Vd = V.at[..., :, 2].multiply(d[..., None])  # V @ diag(1,1,d)
    R = Vd @ jnp.swapaxes(U, -1, -2)
    return jax.lax.stop_gradient(R)


def rmsd_energy(
    positions,
    fit_idx,
    fit_mask,
    fit_ref,
    calc_idx,
    calc_mask,
    calc_ref,
    target1,
    target2,
    geom_type,
    weight,
    mask,
):
    """Fit/calc Kabsch RMSD restraint (mirrors ``numpy_energy.rmsd_energy``). R +
    centroids from the FIT atoms; RMSD over the CALC atoms (distance-style flat-bottom
    delta on the RMSD via ``geom_type``). R is stop-gradient'd."""
    Pf = positions[..., fit_idx, :]
    mf = fit_mask[..., None]
    nf = jnp.sum(fit_mask, axis=-1)
    Pfc = jnp.sum(Pf * mf, axis=-2) / (nf[..., None] + _EPS)
    Qfc = jnp.sum(fit_ref * mf, axis=-2) / (nf[..., None] + _EPS)
    Pf0 = (Pf - Pfc[..., None, :]) * mf
    Qf0 = (fit_ref - Qfc[..., None, :]) * mf
    R = _kabsch_R(Qf0, Pf0)
    Pc = positions[..., calc_idx, :]
    mc = calc_mask[..., None]
    nc = jnp.sum(calc_mask, axis=-1)
    Pc0 = (Pc - Pfc[..., None, :]) * mc
    Qc0 = (calc_ref - Qfc[..., None, :]) * mc
    # Qc0 is reference-only (no `positions` dependency) and R is already stop-gradient'd
    # in _kabsch_R, so an extra stop_gradient on Yc is inert -- drop it to stay textually
    # identical to torch_energy/numpy_energy (the "parity is structural" invariant).
    Yc = Qc0 @ jnp.swapaxes(R, -1, -2)
    resid = (Pc0 - Yc) * mc
    msd = jnp.sum(resid**2, axis=(-2, -1)) / (nc + _EPS)
    rmsd = jnp.sqrt(msd + _EPS)
    delta = _group_delta(rmsd, rmsd - target1, target1, target2, geom_type)
    return jnp.sum(weight * delta**2 * mask)


def _plane_normal(cov):
    """Best-fit-plane unit normal per group = smallest-eigenvalue eigenvector of ``cov``
    (..., 3, 3). ``stop_gradient``'d so ``jax.grad`` flows only through the moving atoms —
    no eigh backward (unstable at degenerate geometry). Pure jnp so it stays JIT/scan-able.
    Mirrors ``numpy_energy._plane_normal`` + the ``_kabsch_R`` carve-out."""
    _w, vecs = jnp.linalg.eigh(
        jax.lax.stop_gradient(cov)
    )  # ascending; columns = vectors
    return jax.lax.stop_gradient(vecs[..., :, 0])


def _plane_rms(grp_pos, grp_mask):
    """Out-of-plane RMS deviation of each padded atom group from its own best-fit plane
    (mirrors ``numpy_energy._plane_rms``). Shared by ``plane_energy`` (conformer,
    ``slack``) and ``group_plane_energy`` (standalone, four restraint types); pure jnp so
    both stay JIT/``lax.scan``-able."""
    m = grp_mask[..., None]
    n_eff = jnp.sum(grp_mask, axis=-1)
    centroid = jnp.sum(grp_pos * m, axis=-2) / (n_eff[..., None] + _EPS)
    x0 = (grp_pos - centroid[..., None, :]) * m
    cov = jnp.swapaxes(x0, -1, -2) @ x0  # (..., n, 3, 3)
    normal = _plane_normal(cov)
    dev = jnp.sum(x0 * normal[..., None, :], axis=-1)
    msd = jnp.sum(dev**2 * grp_mask, axis=-1) / (n_eff + _EPS)
    return jnp.sqrt(msd + _EPS)


def plane_energy(positions, idx, grp_mask, slack, weight, mask):
    """Best-fit-plane restraint over padded atom groups (mirrors
    ``numpy_energy.plane_energy``). Penalise each group's out-of-plane RMS deviation
    beyond ``slack`` (target 0). The plane normal is stop-gradient'd."""
    rms = _plane_rms(positions[..., idx, :], grp_mask)
    delta = jnp.maximum(0.0, rms - slack)
    return jnp.sum(weight * delta**2 * mask)


def group_plane_energy(
    positions,
    idx,
    grp_mask,
    free,
    target1,
    target2,
    geom_type,
    weight,
    mask,
):
    """Standalone best-fit-plane restraint (``plane_restraints_config``; mirrors
    ``numpy_energy.group_plane_energy``). Same measured quantity as ``plane_energy``,
    shaped by the four distance-style types and gated per entry.

    ``free`` (..., n, max_atoms) {0,1} is the per-atom ``move`` mask: a pinned atom keeps
    its VALUE in the plane fit but is stop-gradient'd, so the CG does not move it for this
    restraint. No N-times gradient rescale — see the numpy docstring."""
    grp_pos = positions[..., idx, :]
    grp_pos = jnp.where(free[..., None] > 0, grp_pos, jax.lax.stop_gradient(grp_pos))
    rms = _plane_rms(grp_pos, grp_mask)
    delta = _group_delta(rms, rms - target1, target1, target2, geom_type)
    return jnp.sum(weight * delta**2 * mask)


# leaf energy functions by name, for the shared term_energies dispatch
_LEAF_FNS = {
    "bond_energy": bond_energy,
    "angle_energy": angle_energy,
    "chiral_energy": chiral_energy,
    "plane_energy": plane_energy,
    "cistrans_energy": cistrans_energy,
    "vdw_energy": vdw_energy,
    "distance_energy": distance_energy,
    "rmsd_energy": rmsd_energy,
    "group_angle_energy": group_angle_energy,
    "group_dihedral_energy": group_dihedral_energy,
    "group_improper_energy": group_improper_energy,
    "group_plane_energy": group_plane_energy,
}


def _gates(prepared, positions, sigma, step=None):
    """The conformer gate ``cg`` and a per-restraint ``sigma_gate``. jnp comparisons
    (tracer-safe inside ``lax.scan``). Each restraint is gated on EITHER a sigma window OR
    a step window (mutually exclusive at config time); the gate ANDs both axes, the unused
    axis always-on. ``sigma is None`` / ``step is None`` each disable their own axis (both
    None -> identity gate)."""
    s = None if sigma is None else jnp.asarray(sigma)
    st = None if step is None else jnp.asarray(step)
    cg = jnp.asarray(1.0, dtype=positions.dtype)
    if s is not None:  # conformer window conf_stop <= sigma <= conf_start
        cg = cg * (
            (s <= prepared.get("conf_start_sigma", 1e30))
            & (s >= prepared.get("conf_stop_sigma", -1.0))
        ).astype(positions.dtype)
    if st is not None:  # AND the conformer STEP window conf_start_step..conf_stop_step
        cg = cg * (
            (st >= prepared.get("conf_start_step", float("-inf")))
            & (st <= prepared.get("conf_stop_step", float("inf")))
        ).astype(positions.dtype)

    def sigma_gate(start_sigma, stop_sigma, start_step, stop_step, mask):
        g = mask
        if s is not None:
            g = g * (s <= start_sigma).astype(mask.dtype)
            if stop_sigma is not None:  # released below stop_sigma (rmsd terminus fix)
                g = g * (s >= stop_sigma).astype(mask.dtype)
        if st is not None:  # step window (the alternative gate axis), ANDed in
            if start_step is not None:
                g = g * (st >= start_step).astype(mask.dtype)
            if stop_step is not None:
                g = g * (st <= stop_step).astype(mask.dtype)
        return g

    return cg, sigma_gate


def total_energy(positions, prepared, sigma=None, step=None):
    """Sum all restraint energies. ``sigma`` (noise level) and ``step`` (diffusion step
    index) gate each restraint via its active sigma window AND step window folded into the
    mask (conformer terms share the conf window; distance/RMSD/group each have their own
    per-restraint window). ``sigma=None``/``step=None`` disable that axis's gating. Pure jnp
    so it stays JIT/vmap-able."""
    cg, sigma_gate = _gates(prepared, positions, sigma, step)
    ene = jnp.asarray(0.0, dtype=positions.dtype)
    for v in term_energies(_LEAF_FNS, prepared, positions, cg, sigma_gate).values():
        ene = ene + v
    return ene


def energy_breakdown(positions, prepared, sigma=None, step=None):
    """Per-term restraint energies (same maths + gating as ``total_energy``), as a
    ``{bond, angle, chiral, plane, cistrans, vdw, distance, rmsd}`` python-float dict. Not
    for use inside JIT (the floats force a device->host sync); for diagnostics."""
    cg, sigma_gate = _gates(prepared, positions, sigma, step)
    out = dict.fromkeys(BREAKDOWN_KEYS, 0.0)
    for k, v in term_energies(_LEAF_FNS, prepared, positions, cg, sigma_gate).items():
        out[k] = float(v)
    return out


def prepare_spec(spec):
    """Convert a backend-agnostic ``RestraintSpec`` into jnp arrays."""
    return pack_spec(
        spec, lambda x: jnp.asarray(x, dtype=jnp.int32), lambda x: jnp.asarray(x)
    )
