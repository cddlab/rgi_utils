"""The custom-restraint vocabulary: geometry + penalty primitives.

Written ONCE against the ``ops`` facade (``backends.py``), so the same code runs on
numpy / torch / jax and parity is structural. Each geometry primitive takes
``(ops, coords, *idx)`` where ``coords`` is ``(..., n_active, 3)`` and each ``idx`` is a
backend int array of LOCAL indices (into active_sites) for one atom-group selection; it
returns a scalar-per-leading-dim quantity. The centroid is the plain (unweighted) mean of
the selected atoms — no padding/mask, since each custom restraint is its own closure.

These are exposed to the user as ``ctx`` methods and as DSL-callable names (the DSL call
whitelist == this vocabulary + the math passthroughs in ``context.py``). Angular results
are in RADIANS (the config layer converts target degrees when a user writes them).
"""

from __future__ import annotations

from rgi_utils.custom.backends import EPS

# names a custom formula / ctx may use for atom-group geometry. The DSL maps Call(name)
# to ctx.<name>; ctx geometry methods delegate here.
GEOMETRY = ("distance", "angle", "dihedral", "centroid", "rg", "norm", "dot")


def centroid(ops, coords, idx):
    """Plain geometric centroid (mean) of the selected atoms -> (..., 3)."""
    return ops.mean_atoms(ops.gather(coords, idx))


def distance(ops, coords, idx_a, idx_b):
    """Centroid-to-centroid distance of two selections -> (...)."""
    return ops.vnorm(centroid(ops, coords, idx_a) - centroid(ops, coords, idx_b))


def angle(ops, coords, idx_a, idx_b, idx_c):
    """Angle (radians) at selection B's centroid between A's and C's centroids."""
    a = centroid(ops, coords, idx_a)
    b = centroid(ops, coords, idx_b)
    c = centroid(ops, coords, idx_c)
    rij = a - b
    rkj = c - b
    cos_th = ops.vdot(rij, rkj) / (ops.vnorm(rij) * ops.vnorm(rkj))
    cos_th = ops.clip(cos_th, -1.0 + 1e-7, 1.0 - 1e-7)
    return ops.arccos(cos_th)


def dihedral(ops, coords, idx_a, idx_b, idx_c, idx_d):
    """Dihedral (radians) about the B-C centroid axis (A-B-C-D centroids). Degenerate
    (collinear) geometry is nudged to keep the gradient finite + equal across backends,
    mirroring the built-in ``_dihedral_angle``."""
    p0 = centroid(ops, coords, idx_a)
    p1 = centroid(ops, coords, idx_b)
    p2 = centroid(ops, coords, idx_c)
    p3 = centroid(ops, coords, idx_d)
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = ops.cross(b1, b2)
    n2 = ops.cross(b2, b3)
    b2n = b2 / (ops.vnorm(b2)[..., None] + EPS)
    m1 = ops.cross(n1, b2n)
    x = ops.vdot(n1, n2)
    y = ops.vdot(m1, n2)
    x = ops.where((x == 0.0) & (y == 0.0), x + EPS, x)
    return ops.arctan2(y, x)


def rg(ops, coords, idx):
    """Radius of gyration: RMS distance of the selected atoms from their centroid -> (...)."""
    pos = ops.gather(coords, idx)  # (..., k, 3)
    c = ops.mean_atoms(pos)  # (..., 3)
    d = pos - c[..., None, :]  # (..., k, 3)
    msd = ops.mean_last(ops.vdot(d, d))  # mean over atoms of |d|^2 -> (...)
    return ops.sqrt(msd + EPS)


def norm(ops, v):
    """Euclidean norm over the last axis of a vector quantity -> (...)."""
    return ops.vnorm(v)


def dot(ops, u, v):
    """Dot product over the last axis -> (...)."""
    return ops.vdot(u, v)


# --- penalty helpers (convenience; a user may also write the algebra directly) ---
PENALTY = ("harmonic", "flat_bottom", "lower", "upper")


def harmonic(ops, x, target):
    """``(x - target)**2`` — quadratic toward a target."""
    return (x - target) ** 2


def flat_bottom(ops, x, lo, hi):
    """Zero inside ``[lo, hi]``, quadratic outside."""
    below = ops.clamp_max(x - lo, 0.0)  # min(x-lo, 0): active when x < lo
    above = ops.clamp_min(x - hi, 0.0)  # max(x-hi, 0): active when x > hi
    return below * below + above * above


def lower(ops, x, lo):
    """Penalize only ``x < lo``."""
    below = ops.clamp_max(x - lo, 0.0)
    return below * below


def upper(ops, x, hi):
    """Penalize only ``x > hi``."""
    above = ops.clamp_min(x - hi, 0.0)
    return above * above
