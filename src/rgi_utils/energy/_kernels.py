"""Shared restraint-energy kernels parameterized by an array-operation backend."""

from __future__ import annotations

from rgi_utils import _geometry as G
from rgi_utils._array_ops import EPS


def bond_energy(ops, positions, idx, r0, slack, weight, half, mask):
    first, second = idx[:, 0], idx[:, 1]
    distance = G.distance_points(
        ops, positions[..., first, :], positions[..., second, :]
    )
    full = G.symmetric_flat_bottom_delta(ops, distance - r0, slack)
    stretch = ops.maximum(0.0, distance - (r0 + slack))
    delta = ops.where(half > 0.5, stretch, full)
    return ops.sum(weight * delta * delta * mask)


def angle_energy(ops, positions, idx, th0, slack, weight, mask):
    theta = G.angle_points(
        ops,
        positions[..., idx[:, 0], :],
        positions[..., idx[:, 1], :],
        positions[..., idx[:, 2], :],
    )
    delta = G.symmetric_flat_bottom_delta(ops, theta - th0, slack)
    return ops.sum(weight * delta * delta * mask)


def chiral_energy(ops, positions, idx, vol0, slack, weight, mask):
    center = positions[..., idx[:, 0], :]
    first = positions[..., idx[:, 1], :] - center
    second = positions[..., idx[:, 2], :] - center
    third = positions[..., idx[:, 3], :] - center
    volume = ops.vdot(first, ops.cross(second, third))
    delta = G.symmetric_flat_bottom_delta(ops, volume - vol0, slack)
    return ops.sum(weight * delta * delta * mask)


def cistrans_energy(ops, positions, idx, phi0, slack, weight, mask):
    phi = G.dihedral_points(
        ops,
        positions[..., idx[:, 0], :],
        positions[..., idx[:, 1], :],
        positions[..., idx[:, 2], :],
        positions[..., idx[:, 3], :],
    )
    delta = G.symmetric_flat_bottom_delta(ops, G.wrap(ops, phi - phi0), slack)
    return ops.sum(weight * delta * delta * mask)


def vdw_energy(ops, positions, idx, r_min, weight, mask):
    distance = G.distance_points(
        ops, positions[..., idx[:, 0], :], positions[..., idx[:, 1], :]
    )
    delta = ops.minimum(0.0, distance - r_min)
    return ops.sum(weight * delta * delta * mask)


def _group_centroid(ops, positions, group_idx, group_mask):
    return G.centroid(ops, positions[..., group_idx, :], group_mask)


def _move_centroid(ops, positions, group_idx, group_mask, free, scale=None):
    """Value-preserving rigid-centroid gradient rescale and optional pinning."""
    value = _group_centroid(ops, positions, group_idx, group_mask)
    fixed = ops.stop_gradient(value)
    if scale is None:
        scale = ops.sum(group_mask, axis=-1)
    effective = fixed + scale[..., None] * (value - fixed)
    return ops.where((free > 0.5)[..., None], effective, fixed)


def distance_energy(
    ops,
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
    count1 = ops.sum(grp1_mask, axis=-1)
    count2 = ops.sum(grp2_mask, axis=-1)
    reduced = count1 * count2 / (count1 + count2 + EPS)
    both = move_mode == 0
    scale1 = ops.where(both, reduced, count1)
    scale2 = ops.where(both, reduced, count2)
    free1 = ops.astype_like(move_mode != 2, grp1_mask)
    free2 = ops.astype_like(move_mode != 1, grp2_mask)
    centroid1 = _move_centroid(ops, positions, grp1_idx, grp1_mask, free1, scale1)
    centroid2 = _move_centroid(ops, positions, grp2_idx, grp2_mask, free2, scale2)
    distance = G.distance_points(ops, centroid1, centroid2)
    delta = G.restraint_delta(ops, distance, target1, target2, dist_type)
    return ops.sum(weight * delta * delta * mask)


def group_angle_energy(
    ops,
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
    first = _move_centroid(ops, positions, grp1_idx, grp1_mask, move_free[..., 0])
    vertex = _move_centroid(ops, positions, grp2_idx, grp2_mask, move_free[..., 1])
    third = _move_centroid(ops, positions, grp3_idx, grp3_mask, move_free[..., 2])
    theta = G.angle_points(ops, first, vertex, third)
    delta = G.restraint_delta(ops, theta, target1, target2, geom_type)
    return ops.sum(weight * delta * delta * mask)


def group_dihedral_energy(
    ops,
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
    groups = (
        (grp1_idx, grp1_mask),
        (grp2_idx, grp2_mask),
        (grp3_idx, grp3_mask),
        (grp4_idx, grp4_mask),
    )
    points = [
        _move_centroid(ops, positions, idx, group_mask, move_free[..., index])
        for index, (idx, group_mask) in enumerate(groups)
    ]
    phi = G.dihedral_points(ops, *points)
    harmonic = G.wrap(ops, phi - target1)
    delta = G.restraint_delta(
        ops, phi, target1, target2, geom_type, harmonic_deviation=harmonic
    )
    return ops.sum(weight * delta * delta * mask)


group_improper_energy = group_dihedral_energy


def rmsd_energy(
    ops,
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
    fit_position = positions[..., fit_idx, :]
    fit_position_center = G.centroid(ops, fit_position, fit_mask)
    fit_ref_center = G.centroid(ops, fit_ref, fit_mask)
    fit_mask_expanded = fit_mask[..., None]
    fit_position_zero = (
        fit_position - fit_position_center[..., None, :]
    ) * fit_mask_expanded
    fit_ref_zero = (fit_ref - fit_ref_center[..., None, :]) * fit_mask_expanded
    rotation = G.kabsch_rotation(ops, fit_ref_zero, fit_position_zero)

    calc_position = positions[..., calc_idx, :]
    calc_mask_expanded = calc_mask[..., None]
    calc_position_zero = (
        calc_position - fit_position_center[..., None, :]
    ) * calc_mask_expanded
    calc_ref_zero = (calc_ref - fit_ref_center[..., None, :]) * calc_mask_expanded
    aligned_ref = ops.matmul(calc_ref_zero, ops.swapaxes_last2(rotation))
    residual = (calc_position_zero - aligned_ref) * calc_mask_expanded
    count = ops.sum(calc_mask, axis=-1)
    mean_squared = ops.sum(residual * residual, axis=(-2, -1)) / (count + EPS)
    value = ops.sqrt(mean_squared + EPS)
    delta = G.restraint_delta(ops, value, target1, target2, geom_type)
    return ops.sum(weight * delta * delta * mask)


def plane_energy(ops, positions, idx, grp_mask, slack, weight, mask):
    value = G.plane_rms(ops, positions[..., idx, :], mask=grp_mask)
    delta = ops.maximum(0.0, value - slack)
    return ops.sum(weight * delta * delta * mask)


def group_plane_energy(
    ops,
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
    group_position = positions[..., idx, :]
    fixed = ops.stop_gradient(group_position)
    group_position = ops.where(free[..., None] > 0, group_position, fixed)
    value = G.plane_rms(ops, group_position, mask=grp_mask)
    delta = G.restraint_delta(ops, value, target1, target2, geom_type)
    return ops.sum(weight * delta * delta * mask)
