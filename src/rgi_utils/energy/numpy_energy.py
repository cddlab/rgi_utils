"""NumPy restraint energies — the reference implementation.

Mirrors ``jax_energy.py`` / ``torch_energy.py`` exactly. Kept solely as the pure-
numpy reference that ``tests/test_backend_parity.py`` checks the torch/jax energies
and autodiff gradients against — there is no numpy optimizer (the old scipy/numpy
optimization backend was removed). ``positions`` has shape ``(..., n_active, 3)``;
the result is a scalar.
"""

from __future__ import annotations

import numpy as np

from rgi_utils.energy._terms import BREAKDOWN_KEYS, pack_spec, term_energies

_EPS = 1e-12


def bond_energy(positions, idx, r0, slack, weight, half, mask):
    """Flat-bottomed bond length energy; ``half`` penalizes stretch only."""
    ai, aj = idx[:, 0], idx[:, 1]
    diff = positions[..., ai, :] - positions[..., aj, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1) + _EPS)
    r_upper = r0 + slack
    r_lower = r0 - slack
    delta_full = np.where(
        dist > r_upper,
        dist - r_upper,
        np.where(dist < r_lower, dist - r_lower, 0.0),
    )
    delta_half = np.maximum(0.0, dist - r_upper)
    delta = np.where(half > 0.5, delta_half, delta_full)
    return np.sum(weight * delta**2 * mask)


def angle_energy(positions, idx, th0, slack, weight, mask):
    """Flat-bottomed bond angle energy; vertex is column 1."""
    ai, aj, ak = idx[:, 0], idx[:, 1], idx[:, 2]
    rij = positions[..., ai, :] - positions[..., aj, :]
    rkj = positions[..., ak, :] - positions[..., aj, :]
    nij = np.sqrt(np.sum(rij**2, axis=-1) + _EPS)
    nkj = np.sqrt(np.sum(rkj**2, axis=-1) + _EPS)
    cos_th = np.sum(rij * rkj, axis=-1) / (nij * nkj)
    cos_th = np.clip(cos_th, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = np.arccos(cos_th)
    th_u = th0 + slack
    th_l = th0 - slack
    delta = np.where(
        theta > th_u, theta - th_u, np.where(theta < th_l, theta - th_l, 0.0)
    )
    return np.sum(weight * delta**2 * mask)


def chiral_energy(positions, idx, vol0, slack, weight, mask):
    """Flat-bottomed chiral volume (scalar triple product) energy; center is column
    0. Zero within ``±slack`` of the reference volume ``vol0`` (so the reference
    geometry itself has zero energy), quadratic outside — same flat-bottom shape as
    bond/angle. ``slack=0`` reduces to a pure harmonic toward ``vol0``."""
    a0 = positions[..., idx[:, 0], :]
    a1 = positions[..., idx[:, 1], :]
    a2 = positions[..., idx[:, 2], :]
    a3 = positions[..., idx[:, 3], :]
    v1 = a1 - a0
    v2 = a2 - a0
    v3 = a3 - a0
    vol = np.sum(v1 * np.cross(v2, v3), axis=-1)
    d = vol - vol0
    delta = np.where(d > slack, d - slack, np.where(d < -slack, d + slack, 0.0))
    return np.sum(weight * delta**2 * mask)


def dihedral_energy(positions, idx, phi0, slack, weight, mask):
    """Flat-bottomed dihedral (torsion) energy; bond axis is columns 1-2.

    Periodicity-safe: the deviation ``phi - phi0`` is wrapped to [-pi, pi] before
    the flat-bottomed square penalty, so e.g. +179 deg and -179 deg read as a 2 deg
    difference (not 358). Used to hold acyclic double bonds at their reference
    (input-conformer) cis/trans geometry.
    """
    p0 = positions[..., idx[:, 0], :]
    p1 = positions[..., idx[:, 1], :]
    p2 = positions[..., idx[:, 2], :]
    p3 = positions[..., idx[:, 3], :]
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2n = b2 / np.sqrt(
        np.sum(b2**2, axis=-1, keepdims=True) + _EPS
    )  # _EPS inside: finite grad at b2=0
    m1 = np.cross(n1, b2n)
    x = np.sum(n1 * n2, axis=-1)
    y = np.sum(m1 * n2, axis=-1)
    # Avoid atan2(0, 0) at exactly-degenerate geometry (collinear i-j-k / j-k-l, or
    # coincident j==k): a bare atan2(0,0) has a NaN gradient in jax and 0 in torch, so
    # the backends diverge (and 0*NaN=NaN survives even a masked row). Nudging x makes
    # the gradient finite AND identical across all three backends. Near-collinear
    # sensitivity is inherent to the dihedral and is handled by the optimizers' line
    # search + non-finite guards, not here.
    x = np.where((x == 0.0) & (y == 0.0), x + _EPS, x)
    phi = np.arctan2(y, x)
    d = phi - phi0
    d = np.arctan2(np.sin(d), np.cos(d))  # wrap to [-pi, pi]
    delta = np.where(d > slack, d - slack, np.where(d < -slack, d + slack, 0.0))
    return np.sum(weight * delta**2 * mask)


def vdw_energy(positions, idx, r_min, weight, mask):
    """VdW repulsion (lower-bound only): penalize d < r_min."""
    ai, aj = idx[:, 0], idx[:, 1]
    diff = positions[..., ai, :] - positions[..., aj, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1) + _EPS)
    delta = np.minimum(0.0, dist - r_min)
    return np.sum(weight * delta**2 * mask)


def distance_energy(
    positions,
    grp1_idx,
    grp2_idx,
    grp1_mask,
    grp2_mask,
    target1,
    target2,
    dist_type,
    mask,
):
    """COM distance energy. dist_type: 0=harmonic, 1=flat-bottomed, 2=lower, 3=upper."""
    grp1_pos = positions[..., grp1_idx, :]  # (..., n_dist, max_grp, 3)
    grp2_pos = positions[..., grp2_idx, :]
    m1 = grp1_mask[..., None]
    m2 = grp2_mask[..., None]
    com1 = np.sum(grp1_pos * m1, axis=-2) / (
        np.sum(grp1_mask, axis=-1)[..., None] + _EPS
    )
    com2 = np.sum(grp2_pos * m2, axis=-2) / (
        np.sum(grp2_mask, axis=-1)[..., None] + _EPS
    )
    diff = com2 - com1
    dist = np.sqrt(np.sum(diff**2, axis=-1) + _EPS)
    delta_harmonic = dist - target1
    delta_flat = np.where(
        dist < target1, dist - target1, np.where(dist > target2, dist - target2, 0.0)
    )
    delta_lower = np.minimum(0.0, dist - target1)
    delta_upper = np.maximum(0.0, dist - target2)
    delta = np.where(
        dist_type == 0,
        delta_harmonic,
        np.where(
            dist_type == 1,
            delta_flat,
            np.where(dist_type == 2, delta_lower, delta_upper),
        ),
    )
    return np.sum(delta**2 * mask)


def _kabsch_R(Q0, P0):
    """Optimal proper rotation ``R`` (det +1) minimising ``sum_a ||R Q0_a - P0_a||^2``
    (Kabsch). ``Q0``/``P0`` are centred, padding-zeroed ``(..., A, 3)``. Returns
    ``(..., 3, 3)``. The torch/jax mirrors STOP the gradient through ``R`` (numpy has
    no autodiff), so the SVD is never differentiated — see ``rmsd_energy``."""
    H = np.swapaxes(Q0, -1, -2) @ P0  # (..., 3, 3) cross-covariance sum_a Q0_a x P0_a
    U, _S, Vt = np.linalg.svd(H)
    V = np.swapaxes(Vt, -1, -2)
    d = np.sign(np.linalg.det(V @ np.swapaxes(U, -1, -2)))
    d = np.where(d == 0.0, 1.0, d)  # degenerate H (det 0) -> no reflection flip
    Vd = V.copy()
    Vd[..., :, 2] = Vd[..., :, 2] * d[..., None]  # V @ diag(1,1,d)
    return Vd @ np.swapaxes(U, -1, -2)


def rmsd_energy(
    positions, fit_idx, fit_mask, fit_ref, calc_idx, calc_mask, calc_ref,
    target_rmsd, weight, mask,
):
    """Fit/calc Kabsch RMSD restraint: ``sum_r weight_r (rmsd_r - target_rmsd_r)^2``.

    The optimal rotation R + centroids come from the FIT atoms (``fit_idx`` local
    indices, ``fit_ref`` paired reference); that superposition is applied to the CALC
    atoms and the residual RMSD is measured over them (fit==calc -> plain superposed
    RMSD). R is treated as constant per evaluation (numpy: no autodiff; torch/jax:
    stop-gradient), so the SVD is never differentiated and the gradient flows through
    the moving fit+calc atoms."""
    Pf = positions[..., fit_idx, :]
    mf = fit_mask[..., None]
    nf = np.sum(fit_mask, axis=-1)
    Pfc = np.sum(Pf * mf, axis=-2) / (nf[..., None] + _EPS)  # target fit centroid
    Qfc = np.sum(fit_ref * mf, axis=-2) / (nf[..., None] + _EPS)  # ref fit centroid
    Pf0 = (Pf - Pfc[..., None, :]) * mf
    Qf0 = (fit_ref - Qfc[..., None, :]) * mf
    R = _kabsch_R(Qf0, Pf0)  # R Qf0 ~ Pf0
    # measure on calc atoms, superposed by the FIT transform (centroids + R)
    Pc = positions[..., calc_idx, :]
    mc = calc_mask[..., None]
    nc = np.sum(calc_mask, axis=-1)
    Pc0 = (Pc - Pfc[..., None, :]) * mc
    Qc0 = (calc_ref - Qfc[..., None, :]) * mc
    Yc = Qc0 @ np.swapaxes(R, -1, -2)
    resid = (Pc0 - Yc) * mc
    msd = np.sum(resid**2, axis=(-2, -1)) / (nc + _EPS)  # (..., n_rmsd)
    rmsd = np.sqrt(msd + _EPS)
    return np.sum(weight * (rmsd - target_rmsd) ** 2 * mask)


# leaf energy functions by name, for the shared term_energies dispatch
_LEAF_FNS = {
    "bond_energy": bond_energy,
    "angle_energy": angle_energy,
    "chiral_energy": chiral_energy,
    "dihedral_energy": dihedral_energy,
    "vdw_energy": vdw_energy,
    "distance_energy": distance_energy,
    "rmsd_energy": rmsd_energy,
}


def _gates(prepared, sigma):
    """The conformer gate ``cg`` and a per-restraint ``sigma_gate`` for this noise
    level (numpy: a plain boolean multiplier; identity when ``sigma is None``)."""
    if sigma is None:
        cg = 1.0
    else:  # conformer window conf_stop <= sigma <= conf_start (conf_stop=-1 -> never)
        cg = (sigma <= prepared.get("conf_start_sigma", 1e30)) and (
            sigma >= prepared.get("conf_stop_sigma", -1.0)
        )

    def sigma_gate(start_sigma, stop_sigma, mask):
        if sigma is None:
            return mask
        g = mask * (sigma <= start_sigma)
        if stop_sigma is not None:  # released below stop_sigma (e.g. rmsd terminus fix)
            g = g * (sigma >= stop_sigma)
        return g

    return cg, sigma_gate


def total_energy(positions, prepared, sigma=None, include_distance=True):
    """Sum all restraint energies. ``prepared`` is the dict from ``prepare_spec``.

    ``sigma`` is the current diffusion noise level: each restraint contributes only
    when ``sigma <= start_sigma`` (a 0/1 gate folded into its mask). The conformer
    terms (bond/angle/chiral/dihedral/vdw) share one ``conf_start_sigma``; distance
    and RMSD have their own per-restraint gate. RMSD is summed regardless of
    ``include_distance`` (the CG solver calls with ``include_distance=False``).
    ``sigma=None`` disables gating (all active).
    """
    cg, sigma_gate = _gates(prepared, sigma)
    ene = 0.0
    for v in term_energies(
        _LEAF_FNS, prepared, positions, cg, sigma_gate, include_distance
    ).values():
        ene = ene + v
    return ene


def energy_breakdown(positions, prepared, sigma=None):
    """Per-term restraint energies (same maths + gating as ``total_energy``), as a
    ``{bond, angle, chiral, dihedral, vdw, distance, rmsd}`` float dict for callers
    that report each term's contribution (e.g. ``finalize`` logging)."""
    cg, sigma_gate = _gates(prepared, sigma)
    out = dict.fromkeys(BREAKDOWN_KEYS, 0.0)
    for k, v in term_energies(
        _LEAF_FNS, prepared, positions, cg, sigma_gate, include_distance=True
    ).items():
        out[k] = float(v)
    return out


def prepare_spec(spec):
    """Convert a backend-agnostic ``RestraintSpec`` into NumPy arrays (dict form)."""
    return pack_spec(
        spec,
        lambda x: np.asarray(x, dtype=np.int64),
        lambda x: np.asarray(x, dtype=np.float64),
    )
