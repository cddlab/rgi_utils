"""PyTorch restraint energies — differentiable via ``torch.autograd``.

Mirrors ``jax_energy.py`` exactly so the two backends agree numerically (checked
by ``tests/test_backend_parity.py``). ``positions`` has shape ``(..., n_active, 3)``;
the result is a scalar. Gradients come from autograd, so no hand-written grad.
"""

from __future__ import annotations

import numpy as np
import torch

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
    """Chiral volume (scalar triple product) energy; center is column 0."""
    a0 = positions[..., idx[:, 0], :]
    a1 = positions[..., idx[:, 1], :]
    a2 = positions[..., idx[:, 2], :]
    a3 = positions[..., idx[:, 3], :]
    v1 = a1 - a0
    v2 = a2 - a0
    v3 = a3 - a0
    vol = torch.sum(v1 * torch.linalg.cross(v2, v3, dim=-1), dim=-1)
    thr = torch.where(vol0 > 0, vol0 - slack, vol0 + slack)
    delta = vol - thr
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


def _kabsch_aligned_ref(Q0, P0):
    """Optimal-rotation-aligned reference ``Y0_a = R Q0_a`` (Kabsch). The rotation is
    computed under ``no_grad`` and the result detached, so the gradient flows only
    through ``P0`` in the caller — no SVD backward (which is unstable at degenerate /
    planar geometry). Mirrors ``numpy_energy._kabsch_aligned_ref``."""
    with torch.no_grad():
        H = torch.swapaxes(Q0, -1, -2) @ P0  # (..., 3, 3)
        U, _S, Vt = torch.linalg.svd(H)
        V = torch.swapaxes(Vt, -1, -2)
        d = torch.sign(torch.linalg.det(V @ torch.swapaxes(U, -1, -2)))
        d = torch.where(d == 0.0, torch.ones_like(d), d)
        Vd = V.clone()
        Vd[..., :, 2] = Vd[..., :, 2] * d[..., None]  # V @ diag(1,1,d)
        R = Vd @ torch.swapaxes(U, -1, -2)
        Y0 = Q0 @ torch.swapaxes(R, -1, -2)
    return Y0.detach()


def rmsd_energy(positions, target_idx, target_mask, ref_coords, target_rmsd, weight, mask):
    """Kabsch-superposed RMSD restraint (mirrors ``numpy_energy.rmsd_energy``); the
    optimal rotation is detached so autograd differentiates only the moving atoms."""
    P = positions[..., target_idx, :]  # (..., n_rmsd, max_atoms, 3)
    m = target_mask[..., None]
    n = torch.sum(target_mask, dim=-1)
    Pc = torch.sum(P * m, dim=-2) / (n[..., None] + _EPS)
    Qc = torch.sum(ref_coords * m, dim=-2) / (n[..., None] + _EPS)
    P0 = (P - Pc[..., None, :]) * m
    Q0 = (ref_coords - Qc[..., None, :]) * m
    Y0 = _kabsch_aligned_ref(Q0, P0)
    resid = (P0 - Y0) * m
    msd = torch.sum(resid**2, dim=(-2, -1)) / (n + _EPS)
    rmsd = torch.sqrt(msd + _EPS)
    return torch.sum(weight * (rmsd - target_rmsd) ** 2 * mask)


def total_energy(positions, prepared, sigma=None, include_distance=True):
    """Sum all restraint energies. ``sigma`` (current noise level) gates each
    restraint: it contributes only when ``sigma <= start_sigma`` (folded into the
    mask). Conformer terms share ``conf_start_sigma``; distances have their own.
    ``sigma=None`` disables gating."""
    cg = 1.0 if sigma is None else (sigma <= prepared.get("conf_start_sigma", 1e30))
    ene = torch.zeros((), dtype=positions.dtype, device=positions.device)
    if "bond" in prepared:
        b = prepared["bond"]
        ene = ene + bond_energy(
            positions,
            b["idx"],
            b["r0"],
            b["slack"],
            b["weight"],
            b["half"],
            b["mask"] * cg,
        )
    if "angle" in prepared:
        a = prepared["angle"]
        ene = ene + angle_energy(
            positions, a["idx"], a["th0"], a["slack"], a["weight"], a["mask"] * cg
        )
    if "chiral" in prepared:
        c = prepared["chiral"]
        ene = ene + chiral_energy(
            positions, c["idx"], c["vol0"], c["slack"], c["weight"], c["mask"] * cg
        )
    if "dihedral" in prepared:
        dh = prepared["dihedral"]
        ene = ene + dihedral_energy(
            positions, dh["idx"], dh["phi0"], dh["slack"], dh["weight"], dh["mask"] * cg
        )
    if "vdw" in prepared:
        v = prepared["vdw"]
        ene = ene + vdw_energy(
            positions, v["idx"], v["r_min"], v["weight"], v["mask"] * cg
        )
    if include_distance and "distance" in prepared:
        d = prepared["distance"]
        dmask = d["mask"]
        if sigma is not None:
            dmask = dmask * (sigma <= d["start_sigma"]).to(dmask.dtype)
        ene = ene + distance_energy(
            positions,
            d["grp1_idx"],
            d["grp2_idx"],
            d["grp1_mask"],
            d["grp2_mask"],
            d["target1"],
            d["target2"],
            d["dist_type"],
            dmask,
        )
    # RMSD: optimised by the CG solver, so summed regardless of include_distance.
    if "rmsd" in prepared:
        r = prepared["rmsd"]
        rmask = r["mask"]
        if sigma is not None:
            rmask = rmask * (sigma <= r["start_sigma"]).to(rmask.dtype)
        ene = ene + rmsd_energy(
            positions,
            r["target_idx"],
            r["target_mask"],
            r["ref_coords"],
            r["target_rmsd"],
            r["weight"],
            rmask,
        )
    return ene


def energy_breakdown(positions, prepared, sigma=None):
    """Per-term restraint energies (same maths + gating as ``total_energy``).

    Returns ``{bond, angle, chiral, vdw, distance}`` python floats (host-side).
    """
    cg = 1.0 if sigma is None else (sigma <= prepared.get("conf_start_sigma", 1e30))
    out = {
        "bond": 0.0,
        "angle": 0.0,
        "chiral": 0.0,
        "dihedral": 0.0,
        "vdw": 0.0,
        "distance": 0.0,
        "rmsd": 0.0,
    }
    if "bond" in prepared:
        b = prepared["bond"]
        out["bond"] = float(
            bond_energy(
                positions,
                b["idx"],
                b["r0"],
                b["slack"],
                b["weight"],
                b["half"],
                b["mask"] * cg,
            )
        )
    if "angle" in prepared:
        a = prepared["angle"]
        out["angle"] = float(
            angle_energy(
                positions, a["idx"], a["th0"], a["slack"], a["weight"], a["mask"] * cg
            )
        )
    if "chiral" in prepared:
        c = prepared["chiral"]
        out["chiral"] = float(
            chiral_energy(
                positions, c["idx"], c["vol0"], c["slack"], c["weight"], c["mask"] * cg
            )
        )
    if "dihedral" in prepared:
        dh = prepared["dihedral"]
        out["dihedral"] = float(
            dihedral_energy(
                positions,
                dh["idx"],
                dh["phi0"],
                dh["slack"],
                dh["weight"],
                dh["mask"] * cg,
            )
        )
    if "vdw" in prepared:
        v = prepared["vdw"]
        out["vdw"] = float(
            vdw_energy(positions, v["idx"], v["r_min"], v["weight"], v["mask"] * cg)
        )
    if "distance" in prepared:
        d = prepared["distance"]
        dmask = d["mask"]
        if sigma is not None:
            dmask = dmask * (sigma <= d["start_sigma"]).to(dmask.dtype)
        out["distance"] = float(
            distance_energy(
                positions,
                d["grp1_idx"],
                d["grp2_idx"],
                d["grp1_mask"],
                d["grp2_mask"],
                d["target1"],
                d["target2"],
                d["dist_type"],
                dmask,
            )
        )
    if "rmsd" in prepared:
        r = prepared["rmsd"]
        rmask = r["mask"]
        if sigma is not None:
            rmask = rmask * (sigma <= r["start_sigma"]).to(rmask.dtype)
        out["rmsd"] = float(
            rmsd_energy(
                positions,
                r["target_idx"],
                r["target_mask"],
                r["ref_coords"],
                r["target_rmsd"],
                r["weight"],
                rmask,
            )
        )
    return out


def prepare_spec(spec, device="cpu", dtype=torch.float32):
    """Convert a backend-agnostic ``RestraintSpec`` into torch tensors on ``device``."""

    def _f(x):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

    def _i(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.long, device=device)

    prepared = {"conf_start_sigma": float(getattr(spec, "conf_start_sigma", -1.0))}
    if spec.bond is not None and spec.bond.mask.sum() > 0:
        b = spec.bond
        prepared["bond"] = {
            "idx": _i(b.idx),
            "r0": _f(b.r0),
            "slack": _f(b.slack),
            "weight": _f(b.weight),
            "half": _f(b.half),
            "mask": _f(b.mask),
        }
    if spec.angle is not None and spec.angle.mask.sum() > 0:
        a = spec.angle
        prepared["angle"] = {
            "idx": _i(a.idx),
            "th0": _f(a.th0),
            "slack": _f(a.slack),
            "weight": _f(a.weight),
            "mask": _f(a.mask),
        }
    if spec.chiral is not None and spec.chiral.mask.sum() > 0:
        c = spec.chiral
        prepared["chiral"] = {
            "idx": _i(c.idx),
            "vol0": _f(c.vol0),
            "slack": _f(c.slack),
            "weight": _f(c.weight),
            "mask": _f(c.mask),
        }
    if spec.dihedral is not None and spec.dihedral.mask.sum() > 0:
        dh = spec.dihedral
        prepared["dihedral"] = {
            "idx": _i(dh.idx),
            "phi0": _f(dh.phi0),
            "slack": _f(dh.slack),
            "weight": _f(dh.weight),
            "mask": _f(dh.mask),
        }
    if spec.vdw is not None and spec.vdw.mask.sum() > 0:
        v = spec.vdw
        prepared["vdw"] = {
            "idx": _i(v.idx),
            "r_min": _f(v.r_min),
            "weight": _f(v.weight),
            "mask": _f(v.mask),
        }
    if spec.distance is not None and spec.distance.mask.sum() > 0:
        d = spec.distance
        prepared["distance"] = {
            "grp1_idx": _i(d.grp1_idx),
            "grp2_idx": _i(d.grp2_idx),
            "grp1_mask": _f(d.grp1_mask),
            "grp2_mask": _f(d.grp2_mask),
            "target1": _f(d.target1),
            "target2": _f(d.target2),
            "dist_type": _i(d.dist_type),
            "mask": _f(d.mask),
            "start_sigma": _f(d.start_sigma),
        }
    if spec.rmsd is not None and spec.rmsd.mask.sum() > 0:
        r = spec.rmsd
        prepared["rmsd"] = {
            "target_idx": _i(r.target_local_idx),
            "target_mask": _f(r.target_mask),
            "ref_coords": _f(r.ref_coords),
            "target_rmsd": _f(r.target_rmsd),
            "weight": _f(r.weight),
            "mask": _f(r.mask),
            "start_sigma": _f(r.start_sigma),
        }
    return prepared
