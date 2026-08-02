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


def cistrans_energy(positions, idx, phi0, slack, weight, mask):
    """Flat-bottomed cis/trans (E/Z) torsion energy; bond axis is columns 1-2.

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
    move_mode,
    weight,
    mask,
):
    """Centroid distance energy. dist_type: 0=harmonic, 1=flat-bottomed, 2=lower, 3=upper.

    numpy is the VALUE reference (no autodiff): plain centroids, ``move_mode`` ignored (it
    only shapes the torch/jax gradient via ``_move_centroid``'s scale/pin). The value equals
    the torch/jax ``distance_energy`` (whose ``centroid_eff`` is value-identical), so energy
    parity holds; distance grad parity is torch-vs-jax (the rescaled gradient does not match
    a numpy finite-difference). For a satisfied restraint ``delta -> 0`` so weight/move are
    invisible anyway.
    """
    grp1_pos = positions[..., grp1_idx, :]  # (..., n_dist, max_grp, 3)
    grp2_pos = positions[..., grp2_idx, :]
    m1 = grp1_mask[..., None]
    m2 = grp2_mask[..., None]
    centroid1 = np.sum(grp1_pos * m1, axis=-2) / (
        np.sum(grp1_mask, axis=-1)[..., None] + _EPS
    )
    centroid2 = np.sum(grp2_pos * m2, axis=-2) / (
        np.sum(grp2_mask, axis=-1)[..., None] + _EPS
    )
    diff = centroid2 - centroid1
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
    return np.sum(weight * delta**2 * mask)


def _group_centroid(positions, grp_idx, grp_mask):
    """Masked-mean geometric centroid (unweighted; NOT mass-weighted) of a padded atom group.

    ``grp_idx`` (..., n, max_grp) gathers atoms; ``grp_mask`` (..., n, max_grp) {0,1}
    zeroes padding columns. Returns (..., n, 3). Plain geometric centre (NOT
    mass-weighted), identical to the centroid in ``distance_energy``."""
    pos = positions[..., grp_idx, :]  # (..., n, max_grp, 3)
    m = grp_mask[..., None]
    return np.sum(pos * m, axis=-2) / (np.sum(grp_mask, axis=-1)[..., None] + _EPS)


def _move_centroid(positions, grp_idx, grp_mask, free):
    """Centroid of a group. ``free`` ((..., n) {0,1}, 1=free) is gradient-only: numpy is the
    VALUE reference (no autodiff), so move is a no-op and the value uses every group;
    the torch/jax mirrors stop-gradient this centroid where ``free`` is 0 (pinned)."""
    return _group_centroid(positions, grp_idx, grp_mask)


def _group_delta(val, harmonic_dev, target1, target2, geom_type):
    """Distance-style flat-bottom delta (mirrors ``distance_energy``). ``geom_type``
    0=harmonic (uses ``harmonic_dev`` so the dihedral can pass a wrapped deviation),
    1=flat-bottomed, 2=lower bound, 3=upper bound."""
    d_flat = np.where(
        val < target1, val - target1, np.where(val > target2, val - target2, 0.0)
    )
    d_lower = np.minimum(0.0, val - target1)
    d_upper = np.maximum(0.0, val - target2)
    return np.where(
        geom_type == 0,
        harmonic_dev,
        np.where(geom_type == 1, d_flat, np.where(geom_type == 2, d_lower, d_upper)),
    )


def _dihedral_angle(p0, p1, p2, p3):
    """Torsion angle (radians) about the p1-p2 axis; periodicity handled by the caller.
    Degenerate (collinear) geometry is nudged to keep the gradient finite + equal across
    backends, as in ``cistrans_energy``."""
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2n = b2 / np.sqrt(np.sum(b2**2, axis=-1, keepdims=True) + _EPS)
    m1 = np.cross(n1, b2n)
    x = np.sum(n1 * n2, axis=-1)
    y = np.sum(m1 * n2, axis=-1)
    x = np.where((x == 0.0) & (y == 0.0), x + _EPS, x)
    return np.arctan2(y, x)


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
    ``geom_type`` 0=harmonic / 1=flat-bottomed / 2=lower / 3=upper, bounds
    ``target1``/``target2`` in radians. The centroid-only energy gives every atom in a free
    group the same gradient (the group translates rigidly); ``move_free`` (n,3) pins the
    groups whose column is 0 in torch/jax (no-op in this numpy value reference)."""
    centroid1 = _move_centroid(positions, grp1_idx, grp1_mask, move_free[..., 0])
    centroid2 = _move_centroid(positions, grp2_idx, grp2_mask, move_free[..., 1])
    centroid3 = _move_centroid(positions, grp3_idx, grp3_mask, move_free[..., 2])
    rij = centroid1 - centroid2
    rkj = centroid3 - centroid2
    nij = np.sqrt(np.sum(rij**2, axis=-1) + _EPS)
    nkj = np.sqrt(np.sum(rkj**2, axis=-1) + _EPS)
    cos_th = np.sum(rij * rkj, axis=-1) / (nij * nkj)
    cos_th = np.clip(cos_th, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = np.arccos(cos_th)
    delta = _group_delta(theta, theta - target1, target1, target2, geom_type)
    return np.sum(weight * delta**2 * mask)


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
    """Distance-style flat-bottomed dihedral between four group centroids (axis = group2-3).
    ``harmonic`` (geom_type 0) is periodicity-safe: the deviation ``phi - target1`` is
    wrapped to [-pi, pi] before the square. ``flat-bottomed``/lower/upper use the raw
    angle (``target1 < target2`` is enforced, so a window cannot straddle +-180).
    ``move_free`` (n,4) as in ``group_angle_energy``."""
    p0 = _move_centroid(positions, grp1_idx, grp1_mask, move_free[..., 0])
    p1 = _move_centroid(positions, grp2_idx, grp2_mask, move_free[..., 1])
    p2 = _move_centroid(positions, grp3_idx, grp3_mask, move_free[..., 2])
    p3 = _move_centroid(positions, grp4_idx, grp4_mask, move_free[..., 3])
    phi = _dihedral_angle(p0, p1, p2, p3)
    dev = phi - target1
    harmonic_dev = np.arctan2(np.sin(dev), np.cos(dev))  # wrap to [-pi, pi]
    delta = _group_delta(phi, harmonic_dev, target1, target2, geom_type)
    return np.sum(weight * delta**2 * mask)


# Improper uses the same ordered four-centroid torsion as dihedral.
group_improper_energy = group_dihedral_energy


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
    """Fit/calc Kabsch RMSD restraint: ``sum_r weight_r * delta_r^2`` where ``delta`` is
    the distance-style flat-bottom deviation of the RMSD value (``geom_type`` 0=harmonic /
    1=flat-bottomed / 2=flat-bottomed1 lower bound / 3=flat-bottomed2 upper bound, bounds
    ``target1``/``target2`` in Angstrom).

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
    delta = _group_delta(rmsd, rmsd - target1, target1, target2, geom_type)
    return np.sum(weight * delta**2 * mask)


def _plane_normal(cov):
    """Best-fit-plane unit normal per group = eigenvector of the SMALLEST eigenvalue of
    the (masked, centred) covariance ``cov`` (..., 3, 3). ``np.linalg.eigh`` returns
    eigenvalues ascending with eigenvectors as columns, so column 0 is the normal. The
    torch/jax mirrors STOP the gradient through this normal (numpy has no autodiff), so
    the eigendecomposition is never differentiated — the ``_kabsch_R`` pattern."""
    _w, vecs = np.linalg.eigh(cov)
    return vecs[..., :, 0]


def _plane_rms(grp_pos, grp_mask):
    """Out-of-plane RMS deviation (Angstrom) of each padded atom group from its OWN
    best-fit plane. ``grp_pos`` (..., n, max_atoms, 3), ``grp_mask`` (..., n, max_atoms)
    {0,1} zeroing padding columns; returns (..., n).

    Shared by ``plane_energy`` (conformer term, ``slack`` flat-bottom) and
    ``group_plane_energy`` (standalone ``plane_restraints_config`` term, four restraint
    types) so the eigendecomposition maths lives in ONE place per backend — that is what
    keeps the three-backend parity structural rather than a copy that can drift."""
    m = grp_mask[..., None]
    n_eff = np.sum(grp_mask, axis=-1)  # (..., n) real atoms per group
    centroid = np.sum(grp_pos * m, axis=-2) / (n_eff[..., None] + _EPS)
    x0 = (grp_pos - centroid[..., None, :]) * m  # centred, padding-zeroed
    cov = np.swapaxes(x0, -1, -2) @ x0  # (..., n, 3, 3) covariance sum
    normal = _plane_normal(cov)  # (..., n, 3); stop-grad in torch/jax
    dev = np.sum(x0 * normal[..., None, :], axis=-1)  # signed out-of-plane distance
    msd = np.sum(dev**2 * grp_mask, axis=-1) / (n_eff + _EPS)  # per-atom mean-square
    return np.sqrt(msd + _EPS)


def plane_energy(positions, idx, grp_mask, slack, weight, mask):
    """Best-fit-plane restraint over padded atom groups: penalise each group's
    out-of-plane RMS deviation (Angstrom) beyond ``slack`` (target 0 = planar). ``idx``
    (..., n_plane, max_atoms) gathers each group's atoms; ``grp_mask`` zeroes padding
    columns so variable group size is static-shape safe. The plane normal is the
    smallest-eigenvalue eigenvector of the masked centred covariance, treated as
    constant per evaluation (numpy: no autodiff; torch/jax: stop-gradient), so the
    eigendecomposition is never differentiated and the gradient flows through the moving
    atoms — same carve-out as ``rmsd_energy``'s Kabsch rotation. ``slack=0`` reduces to a
    pure harmonic on the RMS out-of-plane distance."""
    rms = _plane_rms(positions[..., idx, :], grp_mask)
    delta = np.maximum(0.0, rms - slack)  # one-sided flat-bottom (rms >= 0, target 0)
    return np.sum(weight * delta**2 * mask)


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
    """Standalone best-fit-plane restraint over selection-resolved atom groups
    (``plane_restraints_config``) — the same measured quantity as ``plane_energy`` (the
    group's out-of-plane RMS deviation) but shaped by the four distance-style restraint
    types (``geom_type`` 0=harmonic / 1=flat-bottomed / 2=lower / 3=upper, bounds
    ``target1``/``target2`` in Angstrom) and gated per entry rather than by the shared
    conformer gate.

    One entry may pool SEVERAL selection groups into one plane (a shared best-fit plane,
    e.g. two stacked nucleobases); ``free`` (..., n, max_atoms) {0,1} is the per-ATOM
    ``move`` mask — a pinned atom still contributes its position to the plane fit but
    receives no gradient (torch/jax stop-gradient; a value no-op here, as numpy is the
    value reference). Per-atom rather than per-group because the number of groups varies
    per entry, unlike ``group_angle``/``group_dihedral``.

    NOTE there is deliberately NO ``_move_centroid``-style N-times gradient rescale: the
    plane RMS is a genuine least-squares fit rather than a rigid-body translation, so the
    natural ``1/N`` per-atom gradient is correct and CG's line search absorbs the scale.
    """
    rms = _plane_rms(positions[..., idx, :], grp_mask)
    delta = _group_delta(rms, rms - target1, target1, target2, geom_type)
    return np.sum(weight * delta**2 * mask)


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
    """The conformer gate ``cg`` and a per-restraint ``sigma_gate`` for this noise level
    + step (numpy: plain boolean multipliers). Each restraint is gated on EITHER a sigma
    window OR a step window (mutually exclusive at config time); the gate ANDs both axes,
    the unused axis always-on. ``sigma is None`` / ``step is None`` disable their own axis."""
    if sigma is None:
        cg = 1.0
    else:  # conformer window conf_stop <= sigma <= conf_start (conf_stop=-1 -> never)
        cg = (sigma <= prepared.get("conf_start_sigma", 1e30)) and (
            sigma >= prepared.get("conf_stop_sigma", -1.0)
        )
    if step is not None:  # AND the conformer STEP window
        cg = (
            cg
            and (step >= prepared.get("conf_start_step", float("-inf")))
            and (step <= prepared.get("conf_stop_step", float("inf")))
        )

    def sigma_gate(start_sigma, stop_sigma, start_step, stop_step, mask):
        g = mask
        if sigma is not None:
            g = g * (sigma <= start_sigma)
            if stop_sigma is not None:  # released below stop_sigma (rmsd terminus fix)
                g = g * (sigma >= stop_sigma)
        if step is not None:  # step window (the alternative gate axis), ANDed in
            if start_step is not None:
                g = g * (step >= start_step)
            if stop_step is not None:
                g = g * (step <= stop_step)
        return g

    return cg, sigma_gate


def total_energy(positions, prepared, sigma=None, step=None):
    """Sum all restraint energies. ``prepared`` is the dict from ``prepare_spec``.

    ``sigma`` (noise level) and ``step`` (diffusion step index) gate each restraint: it
    contributes only inside its active sigma window AND step window (a 0/1 gate folded
    into its mask). Conformer terms share the conf window; distance/RMSD/group each have
    their own per-restraint window. ``sigma=None``/``step=None`` disable that axis's gating.
    """
    cg, sigma_gate = _gates(prepared, sigma, step)
    ene = 0.0
    for v in term_energies(_LEAF_FNS, prepared, positions, cg, sigma_gate).values():
        ene = ene + v
    return ene


def energy_breakdown(positions, prepared, sigma=None, step=None):
    """Per-term restraint energies (same maths + gating as ``total_energy``), as a
    ``{bond, angle, chiral, plane, cistrans, vdw, distance, rmsd}`` float dict for callers
    that report each term's contribution (e.g. ``finalize`` logging)."""
    cg, sigma_gate = _gates(prepared, sigma, step)
    out = dict.fromkeys(BREAKDOWN_KEYS, 0.0)
    for k, v in term_energies(_LEAF_FNS, prepared, positions, cg, sigma_gate).items():
        out[k] = float(v)
    return out


def prepare_spec(spec):
    """Convert a backend-agnostic ``RestraintSpec`` into NumPy arrays (dict form)."""
    return pack_spec(
        spec,
        lambda x: np.asarray(x, dtype=np.int64),
        lambda x: np.asarray(x, dtype=np.float64),
    )
