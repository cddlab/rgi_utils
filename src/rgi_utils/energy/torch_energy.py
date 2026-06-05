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


def dihedral_energy(positions, idx, phi0, slack, weight, mask):
    """Flat-bottomed dihedral (torsion) energy; bond axis is columns 1-2.

    Periodicity-safe: the deviation ``phi - phi0`` is wrapped to [-pi, pi] before
    the flat-bottomed square penalty. Mirrors ``numpy_energy.dihedral_energy``.
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
    mask,
):
    """COM distance energy. dist_type: 0=harmonic, 1=flat-bottomed, 2=lower, 3=upper."""
    grp1_pos = positions[..., grp1_idx, :]  # (..., n_dist, max_grp, 3)
    grp2_pos = positions[..., grp2_idx, :]
    m1 = grp1_mask[..., None]
    m2 = grp2_mask[..., None]
    com1 = torch.sum(grp1_pos * m1, dim=-2) / (
        torch.sum(grp1_mask, dim=-1)[..., None] + _EPS
    )
    com2 = torch.sum(grp2_pos * m2, dim=-2) / (
        torch.sum(grp2_mask, dim=-1)[..., None] + _EPS
    )
    diff = com2 - com1
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
    return torch.sum(delta**2 * mask)


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
    positions, fit_idx, fit_mask, fit_ref, calc_idx, calc_mask, calc_ref,
    target_rmsd, weight, mask,
):
    """Fit/calc Kabsch RMSD restraint (mirrors ``numpy_energy.rmsd_energy``). R +
    centroids from the FIT atoms; RMSD measured over the CALC atoms. R is detached so
    autograd differentiates only the moving atoms."""
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
    return torch.sum(weight * (rmsd - target_rmsd) ** 2 * mask)


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
    """The conformer gate ``cg`` (a python scalar) and a per-restraint ``sigma_gate``
    (casting the comparison to the mask dtype); identity when ``sigma is None``."""
    cg = 1.0 if sigma is None else (sigma <= prepared.get("conf_start_sigma", 1e30))

    def sigma_gate(start_sigma, mask):
        return mask if sigma is None else mask * (sigma <= start_sigma).to(mask.dtype)

    return cg, sigma_gate


def total_energy(positions, prepared, sigma=None, include_distance=True):
    """Sum all restraint energies. ``sigma`` (current noise level) gates each
    restraint: it contributes only when ``sigma <= start_sigma`` (folded into the
    mask). Conformer terms share ``conf_start_sigma``; distance/RMSD have their own.
    RMSD is summed regardless of ``include_distance``. ``sigma=None`` disables gating."""
    cg, sigma_gate = _gates(prepared, sigma)
    ene = torch.zeros((), dtype=positions.dtype, device=positions.device)
    for v in term_energies(
        _LEAF_FNS, prepared, positions, cg, sigma_gate, include_distance
    ).values():
        ene = ene + v
    return ene


def energy_breakdown(positions, prepared, sigma=None):
    """Per-term restraint energies (same maths + gating as ``total_energy``), as a
    ``{bond, angle, chiral, dihedral, vdw, distance, rmsd}`` python-float dict."""
    cg, sigma_gate = _gates(prepared, sigma)
    out = dict.fromkeys(BREAKDOWN_KEYS, 0.0)
    for k, v in term_energies(
        _LEAF_FNS, prepared, positions, cg, sigma_gate, include_distance=True
    ).items():
        out[k] = float(v)
    return out


def prepare_spec(spec, device="cpu", dtype=torch.float32):
    """Convert a backend-agnostic ``RestraintSpec`` into torch tensors on ``device``."""

    def _f(x):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

    def _i(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.long, device=device)

    return pack_spec(spec, _i, _f)
