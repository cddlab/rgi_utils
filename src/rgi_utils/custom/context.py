"""``RestraintContext`` (ctx) — what a custom energy function (config DSL or code) sees.

A custom restraint is a function ``energy(ctx) -> scalar``. ``ctx`` exposes the shared
vocabulary (geometry + penalty + math) bound to one backend + the live coords + the
restraint's resolved atom-group selections, so the SAME function runs on numpy / torch /
jax. A selection is referenced by an identifier: a config name (from the entry's
``selections`` map) or a raw selection string (code calls ``ctx.distance("chain A", ...)``)
— both are keys in the resolved-selection dict.

``ResolveContext`` is the setup-time twin: it returns plain-numpy shaped dummies and only
records which selection identifiers the function touches, so the engine can resolve them
to atoms (and build active_sites) without a real backend or coordinates.
"""

from __future__ import annotations

import numpy as np

from rgi_utils.custom import vocabulary as V

# every name a custom formula / ctx function may call (the DSL Call whitelist). geometry +
# penalty come from the vocabulary; the rest are elementwise math passthroughs.
MATH_CALLS = (
    "sqrt",
    "exp",
    "log",
    "abs",
    "sin",
    "cos",
    "clip",
    "minimum",
    "maximum",
    "where",
    "sum",
)
CALL_NAMES = frozenset(V.GEOMETRY + V.PENALTY + MATH_CALLS)


class RestraintContext:
    """Backend-bound evaluation context. ``selections`` maps identifier -> backend int
    index array (LOCAL indices into active_sites). ``coords`` is ``(..., n_active, 3)``."""

    def __init__(self, ops, selections, coords):
        self._ops = ops
        self._sel = selections
        self._coords = coords

    def _idx(self, sel):
        try:
            return self._sel[sel]
        except KeyError:
            raise KeyError(
                f"custom restraint references selection {sel!r} which was not resolved "
                "(config: add it to the entry's 'selections' map; code: pass a selection "
                "string to the ctx method)"
            ) from None

    # --- geometry (vocabulary) ---
    def distance(self, a, b):
        return V.distance(self._ops, self._coords, self._idx(a), self._idx(b))

    def angle(self, a, b, c):
        return V.angle(
            self._ops, self._coords, self._idx(a), self._idx(b), self._idx(c)
        )

    def dihedral(self, a, b, c, d):
        return V.dihedral(
            self._ops,
            self._coords,
            self._idx(a),
            self._idx(b),
            self._idx(c),
            self._idx(d),
        )

    def centroid(self, a):
        return V.centroid(self._ops, self._coords, self._idx(a))

    def rg(self, a):
        return V.rg(self._ops, self._coords, self._idx(a))

    def norm(self, v):
        return V.norm(self._ops, v)

    def dot(self, u, v):
        return V.dot(self._ops, u, v)

    # --- penalty (vocabulary) ---
    def harmonic(self, x, target):
        return V.harmonic(self._ops, x, target)

    def flat_bottom(self, x, lo, hi):
        return V.flat_bottom(self._ops, x, lo, hi)

    def lower(self, x, lo):
        return V.lower(self._ops, x, lo)

    def upper(self, x, hi):
        return V.upper(self._ops, x, hi)

    # --- elementwise math passthrough ---
    def sqrt(self, x):
        return self._ops.sqrt(x)

    def exp(self, x):
        return self._ops.exp(x)

    def log(self, x):
        return self._ops.log(x)

    def abs(self, x):
        return self._ops.abs(x)

    def sin(self, x):
        return self._ops.sin(x)

    def cos(self, x):
        return self._ops.cos(x)

    def clip(self, x, lo, hi):
        return self._ops.clip(x, lo, hi)

    def minimum(self, a, b):
        return self._ops.minimum(a, b)

    def maximum(self, a, b):
        return self._ops.maximum(a, b)

    def where(self, c, a, b):
        return self._ops.where(c, a, b)

    def sum(self, x):
        return self._ops.sum(x)


class ResolveContext:
    """Setup-time twin: returns plain-numpy shaped dummies and records the selection
    identifiers the energy function uses (so they can be resolved to atoms). Dummies are
    DISTINCT + positive so differences / norms / divisions stay finite while the function
    runs (it only needs to execute, not be numerically meaningful)."""

    def __init__(self):
        self.selections: list = []  # ordered-unique identifiers seen

    def _rec(self, *sels):
        for s in sels:
            if s not in self.selections:
                self.selections.append(s)

    def _centroid_dummy(self, sel):
        self._rec(sel)
        # distinct per identifier so centroid(A)-centroid(B) != 0
        return np.array([float(self.selections.index(sel) + 1), 1.0, 1.0])

    # geometry
    def distance(self, a, b):
        self._rec(a, b)
        return 1.0

    def angle(self, a, b, c):
        self._rec(a, b, c)
        return 1.0

    def dihedral(self, a, b, c, d):
        self._rec(a, b, c, d)
        return 1.0

    def centroid(self, a):
        return self._centroid_dummy(a)

    def rg(self, a):
        self._rec(a)
        return 1.0

    def norm(self, v):
        return float(np.sqrt(np.sum(np.asarray(v, dtype=float) ** 2))) + 1.0

    def dot(self, u, v):
        return 1.0

    # penalty + math: operate on the plain-float / np-array dummies natively
    def harmonic(self, x, target):
        return (x - target) ** 2

    def flat_bottom(self, x, lo, hi):
        return 0.0

    def lower(self, x, lo):
        return 0.0

    def upper(self, x, hi):
        return 0.0

    def sqrt(self, x):
        return np.sqrt(np.abs(x) + 1.0)

    def exp(self, x):
        return 1.0

    def log(self, x):
        return 0.0

    def abs(self, x):
        return np.abs(x)

    def sin(self, x):
        return 0.0

    def cos(self, x):
        return 1.0

    def clip(self, x, lo, hi):
        return x

    def minimum(self, a, b):
        return a

    def maximum(self, a, b):
        return a

    def where(self, c, a, b):
        return a

    def sum(self, x):
        return float(np.sum(x))
