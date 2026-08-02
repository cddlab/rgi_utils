"""Backend-independent geometry and penalty primitives for custom restraints."""

from __future__ import annotations

from rgi_utils import _geometry as G
from rgi_utils._array_ops import EPS

GEOMETRY = (
    "distance",
    "angle",
    "dihedral",
    "improper",
    "centroid",
    "rg",
    "norm",
    "dot",
    "coords",
    "kabsch",
    "rmsd",
    "plane",
)


def centroid(ops, block):
    """Plain geometric centroid of a coordinate block."""
    return G.centroid(ops, block)


def distance(ops, block_a, block_b):
    """Centroid-to-centroid distance."""
    return G.distance_points(ops, centroid(ops, block_a), centroid(ops, block_b))


def angle(ops, block_a, block_b, block_c):
    """Angle at block B's centroid."""
    return G.angle_points(
        ops,
        centroid(ops, block_a),
        centroid(ops, block_b),
        centroid(ops, block_c),
    )


def dihedral(ops, block_a, block_b, block_c, block_d):
    """Signed dihedral about the B-C centroid axis."""
    return G.dihedral_points(
        ops,
        centroid(ops, block_a),
        centroid(ops, block_b),
        centroid(ops, block_c),
        centroid(ops, block_d),
    )


def improper(ops, block_a, block_b, block_c, block_d):
    """Signed improper angle using the ordered dihedral convention."""
    return dihedral(ops, block_a, block_b, block_c, block_d)


def _kabsch_R(ops, moving_centered, target_centered):
    """Optimal proper rotation with the SVD path stop-gradient'd."""
    return G.kabsch_rotation(ops, moving_centered, target_centered)


def kabsch(ops, block_a, block_b):
    """Rigid-body superpose block A onto block B."""
    return G.superpose(ops, block_a, block_b)


def rmsd(ops, block_a, ref_block):
    """Kabsch-superposed RMSD to a constant, row-aligned reference block."""
    return G.aligned_rmsd(ops, block_a, ref_block)


def _plane_normal(ops, block):
    """Stop-gradient best-fit-plane normal of a coordinate block."""
    center = centroid(ops, block)
    return G.plane_normal(ops, block - center[..., None, :])


def plane(ops, block, plane_block=None):
    """RMS deviation from the block's own or another block's best-fit plane."""
    return G.plane_rms(ops, block, reference=plane_block)


def superpose_ref(ops, ref_block, fit_ref, prediction_fit):
    """Place a constant reference block in the prediction frame."""
    prediction_center = centroid(ops, prediction_fit)
    ref_center = centroid(ops, fit_ref)
    ref_centered = fit_ref - ref_center[..., None, :]
    prediction_centered = prediction_fit - prediction_center[..., None, :]
    rotation = G.kabsch_rotation(ops, ref_centered, prediction_centered)
    group_centered = ref_block - ref_center[..., None, :]
    fitted = (
        ops.matmul(group_centered, ops.swapaxes_last2(rotation))
        + prediction_center[..., None, :]
    )
    return ops.stop_gradient(fitted)


def wrap(ops, value):
    """Wrap an angle or deviation into [-pi, pi]."""
    return G.wrap(ops, value)


def rg(ops, block):
    """Radius of gyration."""
    difference = block - centroid(ops, block)[..., None, :]
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
    return G.harmonic_penalty(ops, value, target)


def flat_bottomed(ops, value, lo, hi):
    """Penalty outside [lo, hi]."""
    return G.flat_bottomed_penalty(ops, value, lo, hi)


def flat_bottomed1(ops, value, lo):
    """Lower-bound penalty."""
    return G.lower_bound_penalty(ops, value, lo)


def flat_bottomed2(ops, value, hi):
    """Upper-bound penalty."""
    return G.upper_bound_penalty(ops, value, hi)
