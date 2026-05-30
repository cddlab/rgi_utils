"""NumPy restraint energies — the reference implementation.

Mirrors ``jax_energy.py`` / ``torch_energy.py`` exactly. Used as the ground-truth
for ``tests/test_backend_parity.py`` and for the CPU fallback optimizer. Gradients
for the CPU path are obtained from this energy via the optim layer.
``positions`` has shape ``(..., n_active, 3)``; the result is a scalar.
"""

from __future__ import annotations

import numpy as np

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
    """Chiral volume (scalar triple product) energy; center is column 0."""
    a0 = positions[..., idx[:, 0], :]
    a1 = positions[..., idx[:, 1], :]
    a2 = positions[..., idx[:, 2], :]
    a3 = positions[..., idx[:, 3], :]
    v1 = a1 - a0
    v2 = a2 - a0
    v3 = a3 - a0
    vol = np.sum(v1 * np.cross(v2, v3), axis=-1)
    thr = np.where(vol0 > 0, vol0 - slack, vol0 + slack)
    delta = vol - thr
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


def total_energy(positions, prepared, sigma=None):
    """Sum all restraint energies. ``prepared`` is the dict from ``prepare_spec``.

    ``sigma`` is the current diffusion noise level: each restraint contributes
    only when ``sigma <= start_sigma`` (a 0/1 gate folded into its mask). The
    conformer terms (bond/angle/chiral/vdw) share one ``conf_start_sigma``; each
    distance restraint has its own. ``sigma=None`` disables gating (all active).
    """
    cg = 1.0 if sigma is None else (sigma <= prepared.get("conf_start_sigma", 1e30))
    ene = 0.0
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
            dmask = dmask * (sigma <= d["start_sigma"])
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


def energy_breakdown(positions, prepared, sigma=None):
    """Per-term restraint energies (same maths + gating as ``total_energy``).

    Returns ``{bond, angle, chiral, vdw, distance}`` floats so callers can report
    how much each restraint type contributes (e.g. ``finalize`` logging). Terms
    absent from ``prepared`` stay 0.0.
    """
    cg = 1.0 if sigma is None else (sigma <= prepared.get("conf_start_sigma", 1e30))
    out = {"bond": 0.0, "angle": 0.0, "chiral": 0.0, "vdw": 0.0, "distance": 0.0}
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
    if "vdw" in prepared:
        v = prepared["vdw"]
        out["vdw"] = float(
            vdw_energy(positions, v["idx"], v["r_min"], v["weight"], v["mask"] * cg)
        )
    if "distance" in prepared:
        d = prepared["distance"]
        dmask = d["mask"]
        if sigma is not None:
            dmask = dmask * (sigma <= d["start_sigma"])
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
    return out


def prepare_spec(spec):
    """Convert a backend-agnostic ``RestraintSpec`` into NumPy arrays (dict form)."""
    prepared = {"conf_start_sigma": float(getattr(spec, "conf_start_sigma", -1.0))}
    if spec.bond is not None and spec.bond.mask.sum() > 0:
        b = spec.bond
        prepared["bond"] = {
            "idx": np.asarray(b.idx, dtype=np.int64),
            "r0": np.asarray(b.r0, dtype=np.float64),
            "slack": np.asarray(b.slack, dtype=np.float64),
            "weight": np.asarray(b.weight, dtype=np.float64),
            "half": np.asarray(b.half, dtype=np.float64),
            "mask": np.asarray(b.mask, dtype=np.float64),
        }
    if spec.angle is not None and spec.angle.mask.sum() > 0:
        a = spec.angle
        prepared["angle"] = {
            "idx": np.asarray(a.idx, dtype=np.int64),
            "th0": np.asarray(a.th0, dtype=np.float64),
            "slack": np.asarray(a.slack, dtype=np.float64),
            "weight": np.asarray(a.weight, dtype=np.float64),
            "mask": np.asarray(a.mask, dtype=np.float64),
        }
    if spec.chiral is not None and spec.chiral.mask.sum() > 0:
        c = spec.chiral
        prepared["chiral"] = {
            "idx": np.asarray(c.idx, dtype=np.int64),
            "vol0": np.asarray(c.vol0, dtype=np.float64),
            "slack": np.asarray(c.slack, dtype=np.float64),
            "weight": np.asarray(c.weight, dtype=np.float64),
            "mask": np.asarray(c.mask, dtype=np.float64),
        }
    if spec.vdw is not None and spec.vdw.mask.sum() > 0:
        v = spec.vdw
        prepared["vdw"] = {
            "idx": np.asarray(v.idx, dtype=np.int64),
            "r_min": np.asarray(v.r_min, dtype=np.float64),
            "weight": np.asarray(v.weight, dtype=np.float64),
            "mask": np.asarray(v.mask, dtype=np.float64),
        }
    if spec.distance is not None and spec.distance.mask.sum() > 0:
        d = spec.distance
        prepared["distance"] = {
            "grp1_idx": np.asarray(d.grp1_idx, dtype=np.int64),
            "grp2_idx": np.asarray(d.grp2_idx, dtype=np.int64),
            "grp1_mask": np.asarray(d.grp1_mask, dtype=np.float64),
            "grp2_mask": np.asarray(d.grp2_mask, dtype=np.float64),
            "target1": np.asarray(d.target1, dtype=np.float64),
            "target2": np.asarray(d.target2, dtype=np.float64),
            "dist_type": np.asarray(d.dist_type, dtype=np.int64),
            "mask": np.asarray(d.mask, dtype=np.float64),
            "start_sigma": np.asarray(d.start_sigma, dtype=np.float64),
        }
    return prepared
