"""Backend-independent geometry and penalty primitives for custom restraints.

Selection identifiers are resolved to prediction or external-reference coordinate
blocks by ``custom.context`` before these functions run. Angular quantities are
in radians.
"""

from __future__ import annotations

from rgi_utils.custom.backends import EPS

GEOMETRY = (
    "distance",
    "angle",
    "dihedral",
    "centroid",
    "rg",
    "norm",
    "dot",
    "coords",
    "kabsch",
    "rmsd",
)


def centroid(ops, block):
    """Plain geometric centroid of a coordinate block."""
    return ops.mean_atoms(block)


def distance(ops, block_a, block_b):
    """Centroid-to-centroid distance."""
    return ops.vnorm(centroid(ops, block_a) - centroid(ops, block_b))


def angle(ops, block_a, block_b, block_c):
    """Angle at block B's centroid."""
    a = centroid(ops, block_a)
    b = centroid(ops, block_b)
    c = centroid(ops, block_c)
    first = a - b
    second = c - b
    cosine = ops.vdot(first, second) / (ops.vnorm(first) * ops.vnorm(second))
    cosine = ops.clip(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
    return ops.arccos(cosine)


def dihedral(ops, block_a, block_b, block_c, block_d):
    """Signed dihedral about the B-C centroid axis."""
    p0 = centroid(ops, block_a)
    p1 = centroid(ops, block_b)
    p2 = centroid(ops, block_c)
    p3 = centroid(ops, block_d)
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = ops.cross(b1, b2)
    n2 = ops.cross(b2, b3)
    b2_normalized = b2 / (ops.vnorm(b2)[..., None] + EPS)
    m1 = ops.cross(n1, b2_normalized)
    x = ops.vdot(n1, n2)
    y = ops.vdot(m1, n2)
    x = ops.where((x == 0.0) & (y == 0.0), x + EPS, x)
    return ops.arctan2(y, x)


def _kabsch_R(ops, moving_centered, target_centered):
    """Optimal proper rotation with the SVD path stop-gradient'd."""
    covariance = ops.stop_gradient(
        ops.matmul(ops.swapaxes_last2(moving_centered), target_centered)
    )
    u, _singular_values, vt = ops.svd(covariance)
    v = ops.swapaxes_last2(vt)
    ut = ops.swapaxes_last2(u)
    determinant = ops.sign(ops.det(ops.matmul(v, ut)))
    determinant = ops.where(determinant == 0.0, ops.ones_like(determinant), determinant)
    one = ops.ones_like(determinant)
    scale = ops.stack([one, one, determinant], -1)
    fixed_v = v * scale[..., None, :]
    return ops.stop_gradient(ops.matmul(fixed_v, ut))


def kabsch(ops, block_a, block_b):
    """Rigid-body superpose block A onto block B."""
    center_a = ops.mean_atoms(block_a)
    center_b = ops.mean_atoms(block_b)
    centered_a = block_a - center_a[..., None, :]
    centered_b = block_b - center_b[..., None, :]
    rotation = _kabsch_R(ops, centered_a, centered_b)
    return ops.matmul(centered_a, ops.swapaxes_last2(rotation)) + center_b[..., None, :]


def rmsd(ops, block_a, ref_block):
    """Kabsch-superposed RMSD to a constant, row-aligned reference block."""
    superposed = kabsch(ops, block_a, ref_block)
    difference = superposed - ref_block
    mean_squared = ops.mean_last(ops.vdot(difference, difference))
    return ops.sqrt(mean_squared + EPS)


def superpose_ref(ops, ref_block, fit_ref, prediction_fit):
    """Place a constant reference block in the prediction frame.

    The complete fitted result is stop-gradient'd so the reference selection
    tracks its fit anchor without pulling that anchor.
    """
    prediction_center = ops.mean_atoms(prediction_fit)
    ref_center = ops.mean_atoms(fit_ref)
    ref_centered = fit_ref - ref_center[..., None, :]
    prediction_centered = prediction_fit - prediction_center[..., None, :]
    rotation = _kabsch_R(ops, ref_centered, prediction_centered)
    group_centered = ref_block - ref_center[..., None, :]
    fitted = (
        ops.matmul(group_centered, ops.swapaxes_last2(rotation))
        + prediction_center[..., None, :]
    )
    return ops.stop_gradient(fitted)


def wrap(ops, value):
    """Wrap an angle or deviation into [-pi, pi]."""
    return ops.arctan2(ops.sin(value), ops.cos(value))


def rg(ops, block):
    """Radius of gyration."""
    center = ops.mean_atoms(block)
    difference = block - center[..., None, :]
    mean_squared = ops.mean_last(ops.vdot(difference, difference))
    return ops.sqrt(mean_squared + EPS)


def norm(ops, value):
    """Euclidean norm over the last axis."""
    return ops.vnorm(value)


def dot(ops, first, second):
    """Dot product over the last axis."""
    return ops.vdot(first, second)


PENALTY = ("harmonic", "flat_bottomed", "flat_bottomed1", "flat_bottomed2")


def harmonic(ops, value, target):
    """Quadratic target penalty."""
    return (value - target) ** 2


def flat_bottomed(ops, value, lo, hi):
    """Penalty outside [lo, hi]."""
    below = ops.clamp_max(value - lo, 0.0)
    above = ops.clamp_min(value - hi, 0.0)
    return below * below + above * above


def flat_bottomed1(ops, value, lo):
    """Lower-bound penalty."""
    below = ops.clamp_max(value - lo, 0.0)
    return below * below


def flat_bottomed2(ops, value, hi):
    """Upper-bound penalty."""
    above = ops.clamp_min(value - hi, 0.0)
    return above * above
