"""Pure JAX restraint energies — differentiable via ``jax.grad``.

All functions take ``positions`` of shape ``(..., n_active, 3)`` and return a
scalar (leading batch dims are summed over). Indices are local indices into
active_sites. Adapted from the AlphaFold 3 restraint prototype.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

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


def dihedral_energy(positions, idx, phi0, slack, weight, mask):
    """Flat-bottomed dihedral (torsion) energy; bond axis is columns 1-2.

    Periodicity-safe: the deviation ``phi - phi0`` is wrapped to [-pi, pi] before
    the flat-bottomed square penalty. Mirrors ``numpy_energy.dihedral_energy``.
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
    mask,
):
    """COM distance energy between two atom groups. dist_type: 0=harmonic,
    1=flat-bottomed, 2=lower-bound, 3=upper-bound."""
    grp1_pos = positions[..., grp1_idx, :]  # (..., n_dist, max_grp, 3)
    grp2_pos = positions[..., grp2_idx, :]
    m1 = grp1_mask[..., None]
    m2 = grp2_mask[..., None]
    com1 = jnp.sum(grp1_pos * m1, axis=-2) / (
        jnp.sum(grp1_mask, axis=-1)[..., None] + _EPS
    )
    com2 = jnp.sum(grp2_pos * m2, axis=-2) / (
        jnp.sum(grp2_mask, axis=-1)[..., None] + _EPS
    )
    diff = com2 - com1
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
    return jnp.sum(delta**2 * mask)


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
    positions, fit_idx, fit_mask, fit_ref, calc_idx, calc_mask, calc_ref,
    target_rmsd, weight, mask,
):
    """Fit/calc Kabsch RMSD restraint (mirrors ``numpy_energy.rmsd_energy``). R +
    centroids from the FIT atoms; RMSD over the CALC atoms. R is stop-gradient'd."""
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
    Yc = jax.lax.stop_gradient(Qc0 @ jnp.swapaxes(R, -1, -2))
    resid = (Pc0 - Yc) * mc
    msd = jnp.sum(resid**2, axis=(-2, -1)) / (nc + _EPS)
    rmsd = jnp.sqrt(msd + _EPS)
    return jnp.sum(weight * (rmsd - target_rmsd) ** 2 * mask)


def total_energy(positions, prepared, sigma=None, include_distance=True):
    """Sum all restraint energies. ``sigma`` (current noise level) gates each
    restraint via ``sigma <= start_sigma`` folded into the mask (conformer terms
    share ``conf_start_sigma``; distances have their own). ``sigma=None`` disables
    gating. Pure jnp so it stays JIT/vmap-able."""
    if sigma is None:
        cg = 1.0
    else:
        cg = (jnp.asarray(sigma) <= prepared.get("conf_start_sigma", 1e30)).astype(
            positions.dtype
        )
    ene = jnp.asarray(0.0, dtype=positions.dtype)
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
            dmask = dmask * (jnp.asarray(sigma) <= d["start_sigma"]).astype(dmask.dtype)
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
            rmask = rmask * (jnp.asarray(sigma) <= r["start_sigma"]).astype(rmask.dtype)
        ene = ene + rmsd_energy(
            positions,
            r["fit_idx"], r["fit_mask"], r["fit_ref"],
            r["calc_idx"], r["calc_mask"], r["calc_ref"],
            r["target_rmsd"], r["weight"], rmask,
        )
    return ene


def energy_breakdown(positions, prepared, sigma=None):
    """Per-term restraint energies (same maths + gating as ``total_energy``).

    Returns ``{bond, angle, chiral, vdw, distance}`` python floats (host-side).
    Not for use inside JIT (the floats force a device->host sync); for diagnostics.
    """
    if sigma is None:
        cg = 1.0
    else:
        cg = (jnp.asarray(sigma) <= prepared.get("conf_start_sigma", 1e30)).astype(
            positions.dtype
        )
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
            dmask = dmask * (jnp.asarray(sigma) <= d["start_sigma"]).astype(dmask.dtype)
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
            rmask = rmask * (jnp.asarray(sigma) <= r["start_sigma"]).astype(rmask.dtype)
        out["rmsd"] = float(
            rmsd_energy(
                positions,
                r["fit_idx"], r["fit_mask"], r["fit_ref"],
                r["calc_idx"], r["calc_mask"], r["calc_ref"],
                r["target_rmsd"], r["weight"], rmask,
            )
        )
    return out


def prepare_spec(spec):
    """Convert a backend-agnostic ``RestraintSpec`` into jnp arrays."""
    prepared = {"conf_start_sigma": float(getattr(spec, "conf_start_sigma", -1.0))}
    if spec.bond is not None and spec.bond.mask.sum() > 0:
        b = spec.bond
        prepared["bond"] = {
            "idx": jnp.asarray(b.idx, dtype=jnp.int32),
            "r0": jnp.asarray(b.r0),
            "slack": jnp.asarray(b.slack),
            "weight": jnp.asarray(b.weight),
            "half": jnp.asarray(b.half),
            "mask": jnp.asarray(b.mask),
        }
    if spec.angle is not None and spec.angle.mask.sum() > 0:
        a = spec.angle
        prepared["angle"] = {
            "idx": jnp.asarray(a.idx, dtype=jnp.int32),
            "th0": jnp.asarray(a.th0),
            "slack": jnp.asarray(a.slack),
            "weight": jnp.asarray(a.weight),
            "mask": jnp.asarray(a.mask),
        }
    if spec.chiral is not None and spec.chiral.mask.sum() > 0:
        c = spec.chiral
        prepared["chiral"] = {
            "idx": jnp.asarray(c.idx, dtype=jnp.int32),
            "vol0": jnp.asarray(c.vol0),
            "slack": jnp.asarray(c.slack),
            "weight": jnp.asarray(c.weight),
            "mask": jnp.asarray(c.mask),
        }
    if spec.dihedral is not None and spec.dihedral.mask.sum() > 0:
        dh = spec.dihedral
        prepared["dihedral"] = {
            "idx": jnp.asarray(dh.idx, dtype=jnp.int32),
            "phi0": jnp.asarray(dh.phi0),
            "slack": jnp.asarray(dh.slack),
            "weight": jnp.asarray(dh.weight),
            "mask": jnp.asarray(dh.mask),
        }
    if spec.vdw is not None and spec.vdw.mask.sum() > 0:
        v = spec.vdw
        prepared["vdw"] = {
            "idx": jnp.asarray(v.idx, dtype=jnp.int32),
            "r_min": jnp.asarray(v.r_min),
            "weight": jnp.asarray(v.weight),
            "mask": jnp.asarray(v.mask),
        }
    if spec.distance is not None and spec.distance.mask.sum() > 0:
        d = spec.distance
        prepared["distance"] = {
            "grp1_idx": jnp.asarray(d.grp1_idx, dtype=jnp.int32),
            "grp2_idx": jnp.asarray(d.grp2_idx, dtype=jnp.int32),
            "grp1_mask": jnp.asarray(d.grp1_mask),
            "grp2_mask": jnp.asarray(d.grp2_mask),
            "target1": jnp.asarray(d.target1),
            "target2": jnp.asarray(d.target2),
            "dist_type": jnp.asarray(d.dist_type, dtype=jnp.int32),
            "mask": jnp.asarray(d.mask),
            "start_sigma": jnp.asarray(d.start_sigma),
        }
    if spec.rmsd is not None and spec.rmsd.mask.sum() > 0:
        r = spec.rmsd
        prepared["rmsd"] = {
            "fit_idx": jnp.asarray(r.fit_idx, dtype=jnp.int32),
            "fit_mask": jnp.asarray(r.fit_mask),
            "fit_ref": jnp.asarray(r.fit_ref),
            "calc_idx": jnp.asarray(r.calc_idx, dtype=jnp.int32),
            "calc_mask": jnp.asarray(r.calc_mask),
            "calc_ref": jnp.asarray(r.calc_ref),
            "target_rmsd": jnp.asarray(r.target_rmsd),
            "weight": jnp.asarray(r.weight),
            "mask": jnp.asarray(r.mask),
            "start_sigma": jnp.asarray(r.start_sigma),
        }
    return prepared
