"""Backend-independent geometry and restraint-penalty primitives."""

from __future__ import annotations

from rgi_utils._array_ops import EPS


def centroid(ops, block, mask=None):
    """Geometric centroid over the atom axis, optionally excluding padded atoms."""
    if mask is None:
        return ops.mean_atoms(block)
    expanded = mask[..., None]
    count = ops.sum(mask, axis=-1)
    return ops.sum(block * expanded, axis=-2) / (count[..., None] + EPS)


def distance_points(ops, first, second):
    """Euclidean distance between coordinate vectors."""
    return ops.vnorm(first - second)


def angle_points(ops, first, vertex, third):
    """Angle in radians for three coordinate vectors."""
    first_arm = first - vertex
    second_arm = third - vertex
    cosine = ops.vdot(first_arm, second_arm) / (
        ops.vnorm(first_arm) * ops.vnorm(second_arm)
    )
    cosine = ops.clip(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
    return ops.arccos(cosine)


def dihedral_points(ops, p0, p1, p2, p3):
    """Signed ordered torsion in radians about the p1-p2 axis."""
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = ops.cross(b1, b2)
    n2 = ops.cross(b2, b3)
    b2_normalized = b2 / ops.sqrt(ops.sum(b2 * b2, axis=-1, keepdims=True) + EPS)
    m1 = ops.cross(n1, b2_normalized)
    x = ops.vdot(n1, n2)
    y = ops.vdot(m1, n2)
    x = ops.where((x == 0.0) & (y == 0.0), x + EPS, x)
    return ops.arctan2(y, x)


def wrap(ops, value):
    """Wrap an angle or angular deviation into [-pi, pi]."""
    return ops.arctan2(ops.sin(value), ops.cos(value))


def kabsch_rotation(ops, moving_centered, target_centered):
    """Optimal proper rotation from centered moving points to centered target points."""
    covariance = ops.stop_gradient(
        ops.matmul(ops.swapaxes_last2(moving_centered), target_centered)
    )
    u, _singular_values, vt = ops.svd(covariance)
    v = ops.swapaxes_last2(vt)
    ut = ops.swapaxes_last2(u)
    determinant = ops.sign(ops.det(ops.matmul(v, ut)))
    determinant = ops.where(determinant == 0.0, ops.ones_like(determinant), determinant)
    one = ops.ones_like(determinant)
    scale = ops.stack([one, one, determinant], axis=-1)
    fixed_v = v * scale[..., None, :]
    return ops.stop_gradient(ops.matmul(fixed_v, ut))


def superpose(ops, moving, target):
    """Rigid-body superpose a row-aligned moving coordinate block onto a target."""
    moving_center = centroid(ops, moving)
    target_center = centroid(ops, target)
    moving_centered = moving - moving_center[..., None, :]
    target_centered = target - target_center[..., None, :]
    rotation = kabsch_rotation(ops, moving_centered, target_centered)
    return (
        ops.matmul(moving_centered, ops.swapaxes_last2(rotation))
        + target_center[..., None, :]
    )


def aligned_rmsd(ops, moving, target):
    """Kabsch-superposed RMSD for row-aligned coordinate blocks."""
    difference = superpose(ops, moving, target) - target
    mean_squared = ops.mean_last(ops.vdot(difference, difference))
    return ops.sqrt(mean_squared + EPS)


def plane_normal(ops, centered):
    """Stop-gradient best-fit-plane normal for centered coordinate blocks."""
    covariance = ops.stop_gradient(ops.matmul(ops.swapaxes_last2(centered), centered))
    _values, vectors = ops.eigh(covariance)
    return ops.stop_gradient(vectors[..., :, 0])


def plane_rms(ops, block, mask=None, reference=None, reference_mask=None):
    """RMS distance of ``block`` from its own or a reference block's best-fit plane."""
    reference = block if reference is None else reference
    reference_mask = (
        mask if reference_mask is None and reference is block else reference_mask
    )
    reference_center = centroid(ops, reference, reference_mask)
    reference_centered = reference - reference_center[..., None, :]
    if reference_mask is not None:
        reference_centered = reference_centered * reference_mask[..., None]
    normal = plane_normal(ops, reference_centered)
    offset = block - reference_center[..., None, :]
    deviation = ops.vdot(offset, normal[..., None, :])
    if mask is None:
        mean_squared = ops.mean_last(deviation * deviation)
    else:
        count = ops.sum(mask, axis=-1)
        mean_squared = ops.sum(deviation * deviation * mask, axis=-1) / (count + EPS)
    return ops.sqrt(mean_squared + EPS)


def symmetric_flat_bottom_delta(ops, deviation, slack):
    """Signed deviation outside a symmetric flat-bottom interval."""
    return ops.where(
        deviation > slack,
        deviation - slack,
        ops.where(deviation < -slack, deviation + slack, 0.0),
    )


def restraint_delta(ops, value, target1, target2, type_code, harmonic_deviation=None):
    """Distance-style harmonic/interval/lower/upper deviation selected by type code."""
    if harmonic_deviation is None:
        harmonic_deviation = value - target1
    flat = ops.where(
        value < target1,
        value - target1,
        ops.where(value > target2, value - target2, 0.0),
    )
    lower = ops.minimum(0.0, value - target1)
    upper = ops.maximum(0.0, value - target2)
    return ops.where(
        type_code == 0,
        harmonic_deviation,
        ops.where(type_code == 1, flat, ops.where(type_code == 2, lower, upper)),
    )


def harmonic_penalty(ops, value, target):
    deviation = value - target
    return deviation * deviation


def flat_bottomed_penalty(ops, value, lo, hi):
    deviation = restraint_delta(ops, value, lo, hi, 1)
    return deviation * deviation


def lower_bound_penalty(ops, value, lo):
    deviation = restraint_delta(ops, value, lo, 0.0, 2)
    return deviation * deviation


def upper_bound_penalty(ops, value, hi):
    deviation = restraint_delta(ops, value, 0.0, hi, 3)
    return deviation * deviation
