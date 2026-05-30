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


def total_energy(positions, prepared, sigma=None):
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
    if "vdw" in prepared:
        v = prepared["vdw"]
        ene = ene + vdw_energy(
            positions, v["idx"], v["r_min"], v["weight"], v["mask"] * cg
        )
    if "distance" in prepared:
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
    return ene


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
    return prepared
