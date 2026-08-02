"""Backend and resolve contexts for custom restraint functions."""

from __future__ import annotations

import numpy as np

from rgi_utils.custom import vocabulary as V

MATH_CALLS = (
    "sqrt",
    "exp",
    "log",
    "abs",
    "sin",
    "cos",
    "wrap",
    "clip",
    "minimum",
    "maximum",
    "where",
    "sum",
)
CALL_NAMES = frozenset(V.GEOMETRY + V.PENALTY + MATH_CALLS)


class RestraintContext:
    """Evaluate custom geometry against live prediction and fixed ref blocks."""

    def __init__(
        self,
        ops,
        selections,
        coords,
        refs=None,
        selection_refs=None,
        ref_fits=None,
        ref_blocks=None,
        move_free=None,
    ):
        self._ops = ops
        self._sel = selections
        self._coords = coords
        self._refs = refs or {}
        self._selection_refs = selection_refs or {}
        self._ref_fits = ref_fits or {}
        self._ref_blocks = ref_blocks or {}
        self._move_free = move_free or {}

    def _idx(self, selection):
        try:
            return self._sel[selection]
        except KeyError:
            raise KeyError(
                f"custom restraint references prediction selection {selection!r} "
                "which was not resolved"
            ) from None

    def _reference_coords(self, selection, ref_name):
        try:
            block_np = self._ref_blocks[(selection, ref_name)]
        except KeyError:
            raise KeyError(
                f"custom reference selection {ref_name!r} and {selection!r} "
                "was not resolved"
            ) from None
        block = self._ops.const_like(block_np, self._coords)
        fit = self._ref_fits.get(ref_name)
        if fit is None:
            return block
        fit_idx, fit_ref_np = fit
        fit_ref = self._ops.const_like(fit_ref_np, self._coords)
        prediction_fit = self._ops.gather(self._coords, fit_idx)
        return V.superpose_ref(self._ops, block, fit_ref, prediction_fit)

    def _coords_of(self, value):
        if not isinstance(value, str):
            return value
        ref_source = self._selection_refs.get(value)
        if ref_source is not None:
            return self._reference_coords(*ref_source)
        block = self._ops.gather(self._coords, self._idx(value))
        if not self._move_free.get(value, True):
            block = self._ops.stop_gradient(block)
        return block

    def distance(self, a, b):
        return V.distance(self._ops, self._coords_of(a), self._coords_of(b))

    def angle(self, a, b, c):
        return V.angle(
            self._ops,
            self._coords_of(a),
            self._coords_of(b),
            self._coords_of(c),
        )

    def dihedral(self, a, b, c, d):
        return V.dihedral(
            self._ops,
            self._coords_of(a),
            self._coords_of(b),
            self._coords_of(c),
            self._coords_of(d),
        )

    def improper(self, a, b, c, d):
        return V.improper(
            self._ops,
            self._coords_of(a),
            self._coords_of(b),
            self._coords_of(c),
            self._coords_of(d),
        )

    def centroid(self, a):
        return V.centroid(self._ops, self._coords_of(a))

    def rg(self, a):
        return V.rg(self._ops, self._coords_of(a))

    def coords(self, a):
        return self._coords_of(a)

    def kabsch(self, a, b):
        return V.kabsch(self._ops, self._coords_of(a), self._coords_of(b))

    def plane(self, a, b=None):
        """Out-of-plane RMS deviation of ``a`` — from its own best-fit plane, or from the
        plane fitted to ``b``. Both go through ``_coords_of``, so a reference-backed
        selection (``refN and ...``) and ``move`` pinning work with no extra handling."""
        return V.plane(
            self._ops,
            self._coords_of(a),
            None if b is None else self._coords_of(b),
        )

    def rmsd(self, a, b):
        """Kabsch RMSD from prediction selection ``a`` to ref selection ``b``."""
        if not isinstance(a, str) or not isinstance(b, str):
            raise TypeError("custom rmsd(A, B) requires two bare selection identifiers")
        try:
            target_idx, ref_np = self._refs[(a, b)]
        except KeyError:
            raise KeyError(
                f"custom rmsd({a!r}, {b!r}) was not resolved; B must map to "
                "'refN and <atom selection>'"
            ) from None
        target = self._ops.gather(self._coords, target_idx)
        if not self._move_free.get(a, True):
            target = self._ops.stop_gradient(target)
        ref = self._ops.const_like(ref_np, target)
        return V.rmsd(self._ops, target, ref)

    def norm(self, value):
        return V.norm(self._ops, value)

    def dot(self, first, second):
        return V.dot(self._ops, first, second)

    def harmonic(self, value, target):
        return V.harmonic(self._ops, value, target)

    def flat_bottomed(self, value, lo, hi):
        return V.flat_bottomed(self._ops, value, lo, hi)

    def flat_bottomed1(self, value, lo):
        return V.flat_bottomed1(self._ops, value, lo)

    def flat_bottomed2(self, value, hi):
        return V.flat_bottomed2(self._ops, value, hi)

    def sqrt(self, value):
        return self._ops.sqrt(value)

    def exp(self, value):
        return self._ops.exp(value)

    def log(self, value):
        return self._ops.log(value)

    def abs(self, value):
        return self._ops.abs(value)

    def sin(self, value):
        return self._ops.sin(value)

    def cos(self, value):
        return self._ops.cos(value)

    def wrap(self, value):
        return V.wrap(self._ops, value)

    def clip(self, value, lo, hi):
        return self._ops.clip(value, lo, hi)

    def minimum(self, first, second):
        return self._ops.minimum(first, second)

    def maximum(self, first, second):
        return self._ops.maximum(first, second)

    def where(self, condition, first, second):
        return self._ops.where(condition, first, second)

    def sum(self, value):
        return self._ops.sum(value)


class ResolveContext:
    """Record selection identifiers touched by a custom energy."""

    def __init__(self):
        self.selections: list[str] = []
        self.rmsd_pairs: list[tuple[str, str]] = []
        self.kabsch_pairs: list[tuple[str, str]] = []

    def _rec(self, *selections):
        for selection in selections:
            if selection not in self.selections:
                self.selections.append(selection)

    def _block_of(self, value):
        if isinstance(value, str):
            self._rec(value)
            index = float(self.selections.index(value) + 1)
            return np.array([[index, 1.0, 1.0], [index, 2.0, 2.0]])
        return np.asarray(value, dtype=float)

    def distance(self, a, b):
        ca = np.mean(self._block_of(a), axis=-2)
        cb = np.mean(self._block_of(b), axis=-2)
        return float(np.linalg.norm(ca - cb)) + 1.0

    def angle(self, a, b, c):
        self._block_of(a), self._block_of(b), self._block_of(c)
        return 1.0

    def dihedral(self, a, b, c, d):
        self._block_of(a), self._block_of(b), self._block_of(c), self._block_of(d)
        return 1.0

    def improper(self, a, b, c, d):
        self._block_of(a), self._block_of(b), self._block_of(c), self._block_of(d)
        return 1.0

    def centroid(self, a):
        return np.mean(self._block_of(a), axis=-2)

    def rg(self, a):
        self._block_of(a)
        return 1.0

    def coords(self, a):
        return self._block_of(a)

    def kabsch(self, a, b):
        self._block_of(a), self._block_of(b)
        if isinstance(a, str) and isinstance(b, str):
            self.kabsch_pairs.append((a, b))
        return np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    def plane(self, a, b=None):
        self._block_of(a)
        if b is not None:
            self._block_of(b)
        return 1.0

    def rmsd(self, a, b):
        if not isinstance(a, str) or not isinstance(b, str):
            raise TypeError("custom rmsd(A, B) requires two bare selection identifiers")
        self._rec(a, b)
        if (a, b) not in self.rmsd_pairs:
            self.rmsd_pairs.append((a, b))
        return 1.0

    def norm(self, value):
        return float(np.sqrt(np.sum(np.asarray(value, dtype=float) ** 2))) + 1.0

    def dot(self, first, second):
        return 1.0

    def harmonic(self, value, target):
        return (value - target) ** 2

    def flat_bottomed(self, value, lo, hi):
        return 0.0

    def flat_bottomed1(self, value, lo):
        return 0.0

    def flat_bottomed2(self, value, hi):
        return 0.0

    def sqrt(self, value):
        return np.sqrt(np.abs(value) + 1.0)

    def exp(self, value):
        return 1.0

    def log(self, value):
        return 0.0

    def abs(self, value):
        return np.abs(value)

    def sin(self, value):
        return 0.0

    def cos(self, value):
        return 1.0

    def wrap(self, value):
        return value

    def clip(self, value, lo, hi):
        return value

    def minimum(self, first, second):
        return first

    def maximum(self, first, second):
        return first

    def where(self, condition, first, second):
        return first

    def sum(self, value):
        return np.sum(value)
