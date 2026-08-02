"""PyTorch restraint energies — differentiable via ``torch.autograd``.

Mirrors ``jax_energy.py`` exactly so the two backends agree numerically (checked
by ``tests/test_backend_parity.py``). ``positions`` has shape ``(..., n_active, 3)``;
the result is a scalar. Gradients come from autograd, so no hand-written grad.
"""

from __future__ import annotations

import numpy as np
import torch

from rgi_utils.energy._terms import BREAKDOWN_KEYS, pack_spec, term_energies

_EPS = 1e-12


def bond_energy(positions, idx, r0, slack, weight, half, mask):
    """Flat-bottomed bond length energy; ``half`` penalizes stretch only."""
    ai, aj = idx[:, 0], idx[:, 1]
    diff = positions[..., ai, :] - positions[..., aj, :]
    dist = torch.sqrt(torch.sum(diff**2, dim=-1) + _EPS)
    r_upper = r0 + slack
    r_lower = r0 - slack
    zero = torch.zeros_like(dist)
    delta_full = torch.where(
        dist > r_upper,
        dist - r_upper,
        torch.where(dist < r_lower, dist - r_lower, zero),
    )
    delta_half = torch.clamp(dist - r_upper, min=0.0)
    delta = torch.where(half > 0.5, delta_half, delta_full)
    return torch.sum(weight * delta**2 * mask)


def angle_energy(positions, idx, th0, slack, weight, mask):
    """Flat-bottomed bond angle energy; vertex is column 1."""
    ai, aj, ak = idx[:, 0], idx[:, 1], idx[:, 2]
    rij = positions[..., ai, :] - positions[..., aj, :]
    rkj = positions[..., ak, :] - positions[..., aj, :]
    nij = torch.sqrt(torch.sum(rij**2, dim=-1) + _EPS)
    nkj = torch.sqrt(torch.sum(rkj**2, dim=-1) + _EPS)
    cos_th = torch.sum(rij * rkj, dim=-1) / (nij * nkj)
    cos_th = torch.clamp(cos_th, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.arccos(cos_th)
    th_u = th0 + slack
    th_l = th0 - slack
    zero = torch.zeros_like(theta)
    delta = torch.where(
        theta > th_u, theta - th_u, torch.where(theta < th_l, theta - th_l, zero)
    )
    return torch.sum(weight * delta**2 * mask)


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
    vol = torch.sum(v1 * torch.linalg.cross(v2, v3, dim=-1), dim=-1)
    d = vol - vol0
    zero = torch.zeros_like(d)
    delta = torch.where(d > slack, d - slack, torch.where(d < -slack, d + slack, zero))
    return torch.sum(weight * delta**2 * mask)


def cistrans_energy(positions, idx, phi0, slack, weight, mask):
    """Flat-bottomed cis/trans (E/Z) torsion energy; bond axis is columns 1-2.

    Periodicity-safe: the deviation ``phi - phi0`` is wrapped to [-pi, pi] before
    the flat-bottomed square penalty. Mirrors ``numpy_energy.cistrans_energy``.
    """
    p0 = positions[..., idx[:, 0], :]
    p1 = positions[..., idx[:, 1], :]
    p2 = positions[..., idx[:, 2], :]
    p3 = positions[..., idx[:, 3], :]
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = torch.linalg.cross(b1, b2, dim=-1)
    n2 = torch.linalg.cross(b2, b3, dim=-1)
    b2n = b2 / torch.sqrt(
        torch.sum(b2**2, dim=-1, keepdim=True) + _EPS
    )  # _EPS inside: finite grad at b2=0
    m1 = torch.linalg.cross(n1, b2n, dim=-1)
    x = torch.sum(n1 * n2, dim=-1)
    y = torch.sum(m1 * n2, dim=-1)
    # Avoid atan2(0, 0) at exactly-degenerate geometry so the gradient stays finite
    # and matches numpy/jax (jax's bare atan2(0,0) gradient is NaN). See numpy ref.
    x = torch.where((x == 0.0) & (y == 0.0), x + _EPS, x)
    phi = torch.atan2(y, x)
    d = phi - phi0
    d = torch.atan2(torch.sin(d), torch.cos(d))  # wrap to [-pi, pi]
    zero = torch.zeros_like(d)
    delta = torch.where(d > slack, d - slack, torch.where(d < -slack, d + slack, zero))
    return torch.sum(weight * delta**2 * mask)


def vdw_energy(positions, idx, r_min, weight, mask):
    """VdW repulsion (lower-bound only): penalize d < r_min."""
    ai, aj = idx[:, 0], idx[:, 1]
    diff = positions[..., ai, :] - positions[..., aj, :]
    dist = torch.sqrt(torch.sum(diff**2, dim=-1) + _EPS)
    delta = torch.clamp(dist - r_min, max=0.0)
    return torch.sum(weight * delta**2 * mask)


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
    """Centroid distance energy. dist_type: 0=harmonic, 1=flat-bottomed, 2=lower, 3=upper.

    Autodiff CG term (no longer closed-form): both group centroids go through
    ``_move_centroid`` with the REDUCED-MASS scale ``mu = N1*N2/(N1+N2)`` so the CG step
    translates each group rigidly with the minimal-displacement split (ratio N2:N1) and an
    O(1), N-independent change in separation — reproducing the old closed-form ``move=both``.
    ``move_mode`` (0=both / 1=group1 only / 2=group2 only) pins the other group via ``free=0``
    (stop-gradient), reproducing the closed-form ``move`` key; the moving group of a pinned
    pair uses its own N so it still reaches the target in one rigid step. Value == numpy's
    plain-centroid distance_energy (parity); only the gradient is rescaled (torch-vs-jax)."""
    n1 = torch.sum(grp1_mask, dim=-1)  # (..., n_dist) group sizes
    n2 = torch.sum(grp2_mask, dim=-1)
    mu = n1 * n2 / (n1 + n2 + _EPS)  # reduced mass: minimal-displacement + O(1) step
    both = move_mode == 0
    scale1 = torch.where(
        both, mu, n1
    )  # both -> mu; else the moving group uses its own N
    scale2 = torch.where(both, mu, n2)
    free1 = (move_mode != 2).to(
        grp1_mask.dtype
    )  # group1 free unless move_mode==2 (pin g1)
    free2 = (move_mode != 1).to(
        grp2_mask.dtype
    )  # group2 free unless move_mode==1 (pin g2)
    centroid1 = _move_centroid(positions, grp1_idx, grp1_mask, free1, scale1)
    centroid2 = _move_centroid(positions, grp2_idx, grp2_mask, free2, scale2)
    diff = centroid2 - centroid1
    dist = torch.sqrt(torch.sum(diff**2, dim=-1) + _EPS)
    zero = torch.zeros_like(dist)
    delta_harmonic = dist - target1
    delta_flat = torch.where(
        dist < target1,
        dist - target1,
        torch.where(dist > target2, dist - target2, zero),
    )
    delta_lower = torch.clamp(dist - target1, max=0.0)
    delta_upper = torch.clamp(dist - target2, min=0.0)
    delta = torch.where(
        dist_type == 0,
        delta_harmonic,
        torch.where(
            dist_type == 1,
            delta_flat,
            torch.where(dist_type == 2, delta_lower, delta_upper),
        ),
    )
    return torch.sum(weight * delta**2 * mask)


def _group_centroid(positions, grp_idx, grp_mask):
    """Masked-mean geometric centroid (unweighted; NOT mass-weighted) of a padded atom group (mirrors
    ``numpy_energy._group_centroid``). ``grp_idx``/``grp_mask`` are (..., n, max_grp)."""
    pos = positions[..., grp_idx, :]  # (..., n, max_grp, 3)
    m = grp_mask[..., None]
    return torch.sum(pos * m, dim=-2) / (torch.sum(grp_mask, dim=-1)[..., None] + _EPS)


def _move_centroid(positions, grp_idx, grp_mask, free, scale=None):
    """Centroid of a group for the restraint gradient, adjusted so the group moves as a RIGID
    UNIT at any weight (like the closed-form distance shift), independent of group size N:

    (1) ``centroid_eff = centroid_d + scale*(centroid - centroid_d)`` (``centroid_d`` = stop-gradient'd
        centroid). Value == ``centroid``; the gradient is ``scale x``: dcentroid/datom = 1/N, so the
        per-atom gradient becomes ``scale/N`` x the centroid gradient. ``scale=None`` -> ``scale=N``
        (the group-restraint default): cancels the 1/N so the whole group translates rigidly by the
        full step, and weight=1 drives ANY group size. The distance restraint instead passes the
        REDUCED MASS ``scale = N1*N2/(N1+N2)`` to BOTH groups so per-atom grad = ``N2/(N1+N2)`` (group1)
        / ``N1/(N1+N2)`` (group2) -> ratio N2:N1 (minimal-displacement split) AND O(1) separation step
        (N-independent); a pinned group passes its own N (only the moving group's scale matters).
    (2) ``free`` ((..., n) {0,1}) = 0 PINS a group: its centroid is stop-gradient'd (the move
        knob; mirrors rmsd ``_kabsch_R``).

    Value is unchanged either way, so the energy (all-backend value parity) is unaffected;
    only the gradient is rescaled, so distance/group grad parity is torch-vs-jax, not numpy-FD."""
    centroid = _group_centroid(positions, grp_idx, grp_mask)
    centroid_d = centroid.detach()
    if scale is None:
        scale = torch.sum(grp_mask, dim=-1)  # group size N (..., n_restr) (rigid step)
    centroid_eff = centroid_d + scale[..., None] * (
        centroid - centroid_d
    )  # un-suppress the 1/N centroid gradient (rigid step)
    return torch.where((free > 0.5)[..., None], centroid_eff, centroid_d)


def _group_delta(val, harmonic_dev, target1, target2, geom_type):
    """Distance-style flat-bottom delta (mirrors ``distance_energy``). geom_type 0=
    harmonic (uses ``harmonic_dev`` so the dihedral can pass a wrapped deviation), 1=
    flat-bottomed, 2=lower, 3=upper."""
    zero = torch.zeros_like(val)
    d_flat = torch.where(
        val < target1, val - target1, torch.where(val > target2, val - target2, zero)
    )
    d_lower = torch.clamp(val - target1, max=0.0)
    d_upper = torch.clamp(val - target2, min=0.0)
    return torch.where(
        geom_type == 0,
        harmonic_dev,
        torch.where(
            geom_type == 1, d_flat, torch.where(geom_type == 2, d_lower, d_upper)
        ),
    )


def _dihedral_angle(p0, p1, p2, p3):
    """Torsion angle (radians) about the p1-p2 axis (mirrors ``cistrans_energy``)."""
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = torch.linalg.cross(b1, b2, dim=-1)
    n2 = torch.linalg.cross(b2, b3, dim=-1)
    b2n = b2 / torch.sqrt(torch.sum(b2**2, dim=-1, keepdim=True) + _EPS)
    m1 = torch.linalg.cross(n1, b2n, dim=-1)
    x = torch.sum(n1 * n2, dim=-1)
    y = torch.sum(m1 * n2, dim=-1)
    x = torch.where((x == 0.0) & (y == 0.0), x + _EPS, x)
    return torch.atan2(y, x)


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
    detach-select in ``_move_centroid``."""
    centroid1 = _move_centroid(positions, grp1_idx, grp1_mask, move_free[..., 0])
    centroid2 = _move_centroid(positions, grp2_idx, grp2_mask, move_free[..., 1])
    centroid3 = _move_centroid(positions, grp3_idx, grp3_mask, move_free[..., 2])
    rij = centroid1 - centroid2
    rkj = centroid3 - centroid2
    nij = torch.sqrt(torch.sum(rij**2, dim=-1) + _EPS)
    nkj = torch.sqrt(torch.sum(rkj**2, dim=-1) + _EPS)
    cos_th = torch.sum(rij * rkj, dim=-1) / (nij * nkj)
    cos_th = torch.clamp(cos_th, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.arccos(cos_th)
    delta = _group_delta(theta, theta - target1, target1, target2, geom_type)
    return torch.sum(weight * delta**2 * mask)


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
    harmonic_dev = torch.atan2(torch.sin(dev), torch.cos(dev))  # wrap to [-pi, pi]
    delta = _group_delta(phi, harmonic_dev, target1, target2, geom_type)
    return torch.sum(weight * delta**2 * mask)


# Improper uses the same ordered four-centroid torsion as dihedral.
group_improper_energy = group_dihedral_energy


def _kabsch_R(Q0, P0):
    """Optimal proper rotation R (det +1) s.t. R Q0 ~ P0 (Kabsch). Computed under
    ``no_grad`` and detached, so the gradient flows only through the moving atoms in
    the caller — no SVD backward (unstable at degenerate geometry). Mirrors
    ``numpy_energy._kabsch_R``."""
    with torch.no_grad():
        H = torch.swapaxes(Q0, -1, -2) @ P0  # (..., 3, 3)
        U, _S, Vt = torch.linalg.svd(H)
        V = torch.swapaxes(Vt, -1, -2)
        d = torch.sign(torch.linalg.det(V @ torch.swapaxes(U, -1, -2)))
        d = torch.where(d == 0.0, torch.ones_like(d), d)
        Vd = V.clone()
        Vd[..., :, 2] = Vd[..., :, 2] * d[..., None]  # V @ diag(1,1,d)
        R = Vd @ torch.swapaxes(U, -1, -2)
    return R.detach()


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
    centroids from the FIT atoms; RMSD measured over the CALC atoms (distance-style
    flat-bottom delta on the RMSD via ``geom_type``). R is detached so autograd
    differentiates only the moving atoms."""
    Pf = positions[..., fit_idx, :]
    mf = fit_mask[..., None]
    nf = torch.sum(fit_mask, dim=-1)
    Pfc = torch.sum(Pf * mf, dim=-2) / (nf[..., None] + _EPS)
    Qfc = torch.sum(fit_ref * mf, dim=-2) / (nf[..., None] + _EPS)
    Pf0 = (Pf - Pfc[..., None, :]) * mf
    Qf0 = (fit_ref - Qfc[..., None, :]) * mf
    R = _kabsch_R(Qf0, Pf0)
    Pc = positions[..., calc_idx, :]
    mc = calc_mask[..., None]
    nc = torch.sum(calc_mask, dim=-1)
    Pc0 = (Pc - Pfc[..., None, :]) * mc
    Qc0 = (calc_ref - Qfc[..., None, :]) * mc
    Yc = Qc0 @ torch.swapaxes(R, -1, -2)
    resid = (Pc0 - Yc) * mc
    msd = torch.sum(resid**2, dim=(-2, -1)) / (nc + _EPS)
    rmsd = torch.sqrt(msd + _EPS)
    delta = _group_delta(rmsd, rmsd - target1, target1, target2, geom_type)
    return torch.sum(weight * delta**2 * mask)


def _plane_normal(cov):
    """Best-fit-plane unit normal per group = smallest-eigenvalue eigenvector of ``cov``
    (..., 3, 3). Computed under ``no_grad`` and detached, so the gradient flows only
    through the moving atoms in the caller — no eigh backward (unstable at degenerate
    geometry). Mirrors ``numpy_energy._plane_normal`` + the ``_kabsch_R`` carve-out."""
    with torch.no_grad():
        _w, vecs = torch.linalg.eigh(cov)  # ascending eigenvalues; columns = vectors
        normal = vecs[..., :, 0]
    return normal.detach()


def _plane_rms(grp_pos, grp_mask):
    """Out-of-plane RMS deviation of each padded atom group from its own best-fit plane
    (mirrors ``numpy_energy._plane_rms``). Shared by ``plane_energy`` (conformer,
    ``slack``) and ``group_plane_energy`` (standalone, four restraint types) so the
    detached-normal maths exists once per backend."""
    m = grp_mask[..., None]
    n_eff = torch.sum(grp_mask, dim=-1)
    centroid = torch.sum(grp_pos * m, dim=-2) / (n_eff[..., None] + _EPS)
    x0 = (grp_pos - centroid[..., None, :]) * m
    cov = torch.swapaxes(x0, -1, -2) @ x0  # (..., n, 3, 3)
    normal = _plane_normal(cov)
    dev = torch.sum(x0 * normal[..., None, :], dim=-1)
    msd = torch.sum(dev**2 * grp_mask, dim=-1) / (n_eff + _EPS)
    return torch.sqrt(msd + _EPS)


def plane_energy(positions, idx, grp_mask, slack, weight, mask):
    """Best-fit-plane restraint over padded atom groups (mirrors
    ``numpy_energy.plane_energy``). Penalise each group's out-of-plane RMS deviation
    beyond ``slack`` (target 0). The plane normal is detached, so autograd
    differentiates only the moving atoms."""
    rms = _plane_rms(positions[..., idx, :], grp_mask)
    delta = torch.clamp(rms - slack, min=0.0)
    return torch.sum(weight * delta**2 * mask)


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
    its VALUE in the plane fit but its gradient is detached, so the CG does not move it
    for this restraint (the ``_move_centroid(free=0)`` mechanism, value-preserving so the
    numpy reference still matches). No N-times gradient rescale — see the numpy docstring.
    """
    grp_pos = positions[..., idx, :]
    keep = free[..., None] > 0
    grp_pos = torch.where(keep, grp_pos, grp_pos.detach())
    rms = _plane_rms(grp_pos, grp_mask)
    delta = _group_delta(rms, rms - target1, target1, target2, geom_type)
    return torch.sum(weight * delta**2 * mask)


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


def _gates(prepared, sigma, step=None):
    """The conformer gate ``cg`` (a python scalar) and a per-restraint ``sigma_gate``
    (casting the comparison to the mask dtype). Each restraint is gated on EITHER a
    sigma window OR a step window (mutually exclusive at config time); the gate ANDs both
    axes, with the unused axis always-on by default. ``sigma is None`` / ``step is None``
    each disable their own axis (so finalize, which passes both None, stays ungated)."""
    if sigma is None:
        cg = 1.0
    else:  # conformer window conf_stop <= sigma <= conf_start (conf_stop=-1 -> never)
        cg = (sigma <= prepared.get("conf_start_sigma", 1e30)) and (
            sigma >= prepared.get("conf_stop_sigma", -1.0)
        )
    if (
        step is not None
    ):  # AND the conformer STEP window (conf_start_step..conf_stop_step)
        cg = (
            cg
            and (step >= prepared.get("conf_start_step", float("-inf")))
            and (step <= prepared.get("conf_stop_step", float("inf")))
        )

    def sigma_gate(start_sigma, stop_sigma, start_step, stop_step, mask):
        g = mask
        if sigma is not None:
            g = g * (sigma <= start_sigma).to(mask.dtype)
            if stop_sigma is not None:  # released below stop_sigma (rmsd terminus fix)
                g = g * (sigma >= stop_sigma).to(mask.dtype)
        if step is not None:  # step window (the alternative gate axis), ANDed in
            if start_step is not None:
                g = g * (step >= start_step).to(mask.dtype)
            if stop_step is not None:
                g = g * (step <= stop_step).to(mask.dtype)
        return g

    return cg, sigma_gate


def total_energy(positions, prepared, sigma=None, step=None):
    """Sum all restraint energies. ``sigma`` (noise level) and ``step`` (diffusion step
    index) gate each restraint: it contributes only inside its active sigma window AND
    its active step window (folded into the mask). Conformer terms share the conf window;
    distance/RMSD/group each have their own per-restraint window. ``sigma=None``/``step=None``
    disable that axis's gating."""
    cg, sigma_gate = _gates(prepared, sigma, step)
    ene = torch.zeros((), dtype=positions.dtype, device=positions.device)
    for v in term_energies(_LEAF_FNS, prepared, positions, cg, sigma_gate).values():
        ene = ene + v
    return ene


def energy_breakdown(positions, prepared, sigma=None, step=None):
    """Per-term restraint energies (same maths + gating as ``total_energy``), as a
    ``{bond, angle, chiral, plane, cistrans, vdw, distance, rmsd}`` python-float dict."""
    cg, sigma_gate = _gates(prepared, sigma, step)
    out = dict.fromkeys(BREAKDOWN_KEYS, 0.0)
    for k, v in term_energies(_LEAF_FNS, prepared, positions, cg, sigma_gate).items():
        out[k] = float(v)
    return out


def prepare_spec(spec, device="cpu", dtype=torch.float32):
    """Convert a backend-agnostic ``RestraintSpec`` into torch tensors on ``device``."""

    def _f(x):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

    def _i(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.long, device=device)

    return pack_spec(spec, _i, _f)
