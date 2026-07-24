"""The custom-restraint vocabulary: geometry + penalty primitives.

Written ONCE against the ``ops`` facade (``backends.py``), so the same code runs on
numpy / torch / jax and parity is structural. Each geometry primitive takes
``(ops, *blocks)`` where a ``block`` is a coordinate array ``(..., k, 3)`` for one
atom-group selection (the context layer gathers the selection's atoms before calling, so
primitives compose: a block may come straight from a selection OR from ``kabsch`` output).
The centroid is the plain (unweighted) mean of the block's atoms — no padding/mask, since
each custom restraint is its own closure.

These are exposed to the user as ``ctx`` methods and as DSL-callable names (the DSL call
whitelist == this vocabulary + the math passthroughs in ``context.py``). Angular results
are in RADIANS (the config layer converts target degrees when a user writes them).

``kabsch``/``rmsd`` add Kabsch superposition: ``kabsch(A, B)`` returns the ``(..., k, 3)``
coordinates of block A rigid-body-superposed onto block B (so it can be composed further,
e.g. ``norm(kabsch(A, B) - coords(B))``); ``rmsd(A, ref)`` returns the scalar superposed
RMSD of a moving block against a constant reference block. Both freeze the optimal rotation
with ``stop_gradient`` (the SVD is never differentiated), exactly like the built-in
``energy/*_energy.py::_kabsch_R`` term — so their gradient is torch/jax-consistent but does
NOT match a numpy finite-difference (see the parity carve-out in the tests).
"""

from __future__ import annotations

from rgi_utils.custom.backends import EPS

# names a custom formula / ctx may use for atom-group geometry. The DSL maps Call(name)
# to ctx.<name>; ctx geometry methods gather selections to blocks then delegate here.
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
    "ref",
)


def centroid(ops, block):
    """Plain geometric centroid (mean) of a coordinate block -> (..., 3)."""
    return ops.mean_atoms(block)


def distance(ops, block_a, block_b):
    """Centroid-to-centroid distance of two blocks -> (...)."""
    return ops.vnorm(centroid(ops, block_a) - centroid(ops, block_b))


def angle(ops, block_a, block_b, block_c):
    """Angle (radians) at block B's centroid between A's and C's centroids."""
    a = centroid(ops, block_a)
    b = centroid(ops, block_b)
    c = centroid(ops, block_c)
    rij = a - b
    rkj = c - b
    cos_th = ops.vdot(rij, rkj) / (ops.vnorm(rij) * ops.vnorm(rkj))
    cos_th = ops.clip(cos_th, -1.0 + 1e-7, 1.0 - 1e-7)
    return ops.arccos(cos_th)


def dihedral(ops, block_a, block_b, block_c, block_d):
    """Dihedral (radians) about the B-C centroid axis (A-B-C-D centroids). Degenerate
    (collinear) geometry is nudged to keep the gradient finite + equal across backends,
    mirroring the built-in ``_dihedral_angle``."""
    p0 = centroid(ops, block_a)
    p1 = centroid(ops, block_b)
    p2 = centroid(ops, block_c)
    p3 = centroid(ops, block_d)
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = ops.cross(b1, b2)
    n2 = ops.cross(b2, b3)
    b2n = b2 / (ops.vnorm(b2)[..., None] + EPS)
    m1 = ops.cross(n1, b2n)
    x = ops.vdot(n1, n2)
    y = ops.vdot(m1, n2)
    x = ops.where((x == 0.0) & (y == 0.0), x + EPS, x)
    return ops.arctan2(y, x)


def _kabsch_R(ops, q0, p0):
    """Optimal proper rotation ``R`` (det +1) s.t. ``R q0 ~ p0`` (Kabsch), written ONCE
    against the ops facade. ``q0``/``p0`` are centred ``(..., k, 3)``. ``R`` is
    stop-gradient'd (the SVD is never differentiated — unstable at degenerate geometry), so
    the caller's gradient flows only through the moving atoms. Mirrors
    ``energy/{numpy,torch,jax}_energy.py::_kabsch_R``; the ``V @ diag(1,1,d)`` reflection fix
    is written as a column-scale (``V * s[..., None, :]``) so the one implementation runs on
    all three backends (no in-place / ``.at`` split)."""
    h = ops.stop_gradient(ops.matmul(ops.swapaxes_last2(q0), p0))  # (..., 3, 3)
    u, _s, vt = ops.svd(h)
    v = ops.swapaxes_last2(vt)
    ut = ops.swapaxes_last2(u)
    d = ops.sign(ops.det(ops.matmul(v, ut)))
    d = ops.where(d == 0.0, ops.ones_like(d), d)  # degenerate H (det 0) -> no flip
    one = ops.ones_like(d)
    s = ops.stack([one, one, d], -1)  # diag(1,1,d) as a column-scale vector
    vd = v * s[..., None, :]  # V @ diag(1,1,d)
    return ops.stop_gradient(ops.matmul(vd, ut))


def kabsch(ops, block_a, block_b):
    """Rigid-body superpose block A onto block B (positional atom correspondence, so
    ``|A| == |B|``); return A's superposed coordinates ``(..., k, 3)``. The rotation is
    stop-gradient'd, so the gradient flows through the moving atoms of A (via the frozen
    rotation) and B's centroid, not through the SVD."""
    ca = ops.mean_atoms(block_a)
    cb = ops.mean_atoms(block_b)
    a0 = block_a - ca[..., None, :]
    b0 = block_b - cb[..., None, :]
    r = _kabsch_R(ops, a0, b0)  # R a0 ~ b0
    return ops.matmul(a0, ops.swapaxes_last2(r)) + cb[..., None, :]


def rmsd(ops, block_a, ref_block):
    """Kabsch-superposed RMSD (scalar per leading dim) of a moving block against a CONSTANT
    reference block (row-aligned to A). ``= sqrt(mean_atoms ||superpose(A -> ref) - ref||^2)``.
    The reference does not move (grad flows through A only); the rotation is stop-gradient'd."""
    a_sup = kabsch(ops, block_a, ref_block)
    diff = a_sup - ref_block
    msd = ops.mean_last(ops.vdot(diff, diff))  # mean over atoms of |diff|^2 -> (...)
    return ops.sqrt(msd + EPS)


def superpose_ref(ops, ref_block, fit_ref, p_fit):
    """Place a CONSTANT reference block ``ref_block`` (``(k, 3)``, atoms selected on the
    reference) into the PREDICTION frame, by the Kabsch superposition that best maps the
    constant reference anchor ``fit_ref`` (``(m, 3)``) onto the MOVING prediction anchor
    ``p_fit`` (``(..., m, 3)``). Returns the fitted block ``(..., k, 3)``.

    This is the INVERSE of ``kabsch`` (which moves a group onto a constant reference); here a
    constant reference is moved onto the moving prediction, so a reference atom becomes a fixed
    landmark in the prediction's frame. The WHOLE transform is stop-gradient'd (the rotation via
    ``_kabsch_R`` is already frozen; the result is ``stop_gradient``'d as well), so the fitted
    landmark TRACKS the anchor's current value but exerts NO force on it — the gradient of a
    restraint using ``ref(...)`` flows only through the *other* (prediction) group. Its grad is
    therefore torch/jax-consistent but not numpy-FD-equal (the same carve-out as
    ``kabsch``/``rmsd``)."""
    cp = ops.mean_atoms(p_fit)  # (..., 3) — moving anchor centroid
    cq = ops.mean_atoms(fit_ref)  # (3,) — constant reference anchor centroid
    q0 = fit_ref - cq[..., None, :]  # (m, 3)
    p0 = p_fit - cp[..., None, :]  # (..., m, 3)
    r = _kabsch_R(ops, q0, p0)  # R q0 ~ p0 (rotation frozen inside)
    g0 = ref_block - cq[..., None, :]  # (k, 3) in the reference's centred frame
    fitted = ops.matmul(g0, ops.swapaxes_last2(r)) + cp[..., None, :]
    return ops.stop_gradient(fitted)


def wrap(ops, x):
    """Wrap an angle / angular deviation (radians) into ``[-pi, pi]`` via
    ``arctan2(sin x, cos x)``. This is the SAME periodicity fold the built-in
    ``cistrans_energy`` / ``group_dihedral_energy`` apply internally to a dihedral
    deviation, exposed here so a custom formula can be periodicity-safe:
    ``wrap(dihedral(A,B,C,D) - target)**2`` treats +179deg and -179deg as a 2deg
    difference (not 358). Unlike those built-ins it needs NO atan2(0,0) guard, because
    ``sin(x)**2 + cos(x)**2 == 1`` so the two args are never simultaneously zero (the
    guard there is for a raw cross product that can vanish at collinear geometry)."""
    return ops.arctan2(ops.sin(x), ops.cos(x))


def rg(ops, block):
    """Radius of gyration: RMS distance of the block's atoms from their centroid -> (...)."""
    c = ops.mean_atoms(block)  # (..., 3)
    d = block - c[..., None, :]  # (..., k, 3)
    msd = ops.mean_last(ops.vdot(d, d))  # mean over atoms of |d|^2 -> (...)
    return ops.sqrt(msd + EPS)


def norm(ops, v):
    """Euclidean norm over the last axis of a vector quantity -> (...)."""
    return ops.vnorm(v)


def dot(ops, u, v):
    """Dot product over the last axis -> (...)."""
    return ops.vdot(u, v)


# --- penalty helpers (convenience; a user may also write the algebra directly) ---
PENALTY = ("harmonic", "flat_bottomed", "flat_bottomed1", "flat_bottomed2")


def harmonic(ops, x, target):
    """``(x - target)**2`` — quadratic toward a target."""
    return (x - target) ** 2


def flat_bottomed(ops, x, lo, hi):
    """Zero inside ``[lo, hi]``, quadratic outside — both a lower and an upper bound
    (mirrors the built-in distance/angle ``flat-bottomed`` block)."""
    below = ops.clamp_max(x - lo, 0.0)  # min(x-lo, 0): active when x < lo
    above = ops.clamp_min(x - hi, 0.0)  # max(x-hi, 0): active when x > hi
    return below * below + above * above


def flat_bottomed1(ops, x, lo):
    """Lower bound — penalize only ``x < lo`` (mirrors ``flat-bottomed1``)."""
    below = ops.clamp_max(x - lo, 0.0)
    return below * below


def flat_bottomed2(ops, x, hi):
    """Upper bound — penalize only ``x > hi`` (mirrors ``flat-bottomed2``)."""
    above = ops.clamp_min(x - hi, 0.0)
    return above * above
