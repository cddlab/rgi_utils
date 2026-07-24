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
# penalty come from the vocabulary; the rest are mostly elementwise math passthroughs to
# ``ops`` — EXCEPT ``wrap``, which is a small composite (arctan2(sin, cos)) living in the
# vocabulary so it runs on every backend.
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
    """Backend-bound evaluation context. ``selections`` maps identifier -> backend int
    index array (LOCAL indices into active_sites). ``coords`` is ``(..., n_active, 3)``.
    ``refs`` maps ``(selection_identifier, ref_name) -> constant reference coord block`` (a
    backend array row-aligned to the selection's atoms) for the ``rmsd`` primitive.

    Reference-anchored geometry (the ``ref`` primitive) uses two more maps: ``ref_fits`` maps
    ``ref_name -> (fit_target_local_idx, fit_ref_coords)`` (the prediction anchor's LOCAL
    indices + the constant reference anchor block, for the per-eval Kabsch fit), and
    ``ref_blocks`` maps ``(reference_selection, ref_name) -> ref_group_coords`` (the constant
    ``(k, 3)`` block of atoms selected ON THE REFERENCE, in the reference frame). A ref with no
    fit selections has no ``ref_fits`` entry and is used in its own frame."""

    def __init__(
        self, ops, selections, coords, refs=None, ref_fits=None, ref_blocks=None
    ):
        self._ops = ops
        self._sel = selections
        self._coords = coords
        self._refs = refs or {}
        self._ref_fits = ref_fits or {}
        self._ref_blocks = ref_blocks or {}

    def _idx(self, sel):
        try:
            return self._sel[sel]
        except KeyError:
            raise KeyError(
                f"custom restraint references selection {sel!r} which was not resolved "
                "(config: add it to the entry's 'selections' map; code: pass a selection "
                "string to the ctx method)"
            ) from None

    def _coords_of(self, a):
        """A selection identifier (str) -> gather its atoms into a ``(..., k, 3)`` block; an
        already-computed coord block (e.g. ``kabsch`` output) -> pass through. This is what
        lets geometry primitives compose over both selections and superposed blocks."""
        if isinstance(a, str):
            return self._ops.gather(self._coords, self._idx(a))
        return a

    def _ref_entry(self, sel_name, ref_name):
        """(backend int index array of the MATCHED target-atom subset, numpy ref coords)."""
        try:
            return self._refs[(sel_name, ref_name)]
        except KeyError:
            raise KeyError(
                f"custom rmsd references ref {ref_name!r} for selection {sel_name!r} which "
                "was not resolved (add it to the entry's 'refs' map)"
            ) from None

    # --- geometry (vocabulary) ---
    def distance(self, a, b):
        return V.distance(self._ops, self._coords_of(a), self._coords_of(b))

    def angle(self, a, b, c):
        return V.angle(
            self._ops, self._coords_of(a), self._coords_of(b), self._coords_of(c)
        )

    def dihedral(self, a, b, c, d):
        return V.dihedral(
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
        """The raw ``(..., k, 3)`` coordinate block of a selection — needed to compose a
        bare selection with ``kabsch`` output in arithmetic (``kabsch(A, B) - coords(B)``),
        since a bare name in an expression is otherwise a selection identifier, not coords."""
        return self._coords_of(a)

    def kabsch(self, a, b):
        return V.kabsch(self._ops, self._coords_of(a), self._coords_of(b))

    def rmsd(self, a, r):
        """Superposed RMSD of moving selection ``a`` onto the constant reference ``r`` (from
        the entry's ``refs`` map). ``a`` MUST be a bare selection identifier — the reference
        block was aligned to it row-for-row at setup, so a coord expression can't be used."""
        if not isinstance(a, str):
            raise TypeError(
                "custom rmsd(A, ref): the first argument must be a bare selection "
                "identifier (the reference is aligned to it at setup), not a coord "
                "expression such as kabsch(...)"
            )
        tgt_idx, ref_np = self._ref_entry(a, r)
        # gather the MATCHED subset of the selection's atoms (best-effort pairing may drop
        # some), which the reference block is row-aligned to; convert the ref to the live
        # coords' dtype/device only now (so float32/float64 can't mismatch mid-op).
        block = self._ops.gather(self._coords, tgt_idx)
        ref = self._ops.const_like(ref_np, block)
        return V.rmsd(self._ops, block, ref)

    def _ref_block_entry(self, sel, r):
        try:
            return self._ref_blocks[(sel, r)]
        except KeyError:
            raise KeyError(
                f"custom ref({sel!r}, {r!r}) was not resolved (define {r!r} in the entry's "
                "'refs' map and reference it in the formula)"
            ) from None

    def ref(self, sel, r):
        """The ``(..., k, 3)`` coordinate block of atoms selected by ``sel`` ON REFERENCE ``r``,
        placed into the PREDICTION frame by ``r``'s Kabsch fit (defined in the entry's ``refs``
        map via ``atom_selection_target_fit`` / ``atom_selection_ref_fit``). ``sel`` is a raw
        selection string (or a name from ``selections``) evaluated on the REFERENCE, not the
        prediction — so the returned block is a FIXED landmark that tracks the fit anchor but
        exerts no force on it. Composes with every geometry primitive:
        ``distance(A, ref("chain B", r))``, ``angle(A, ref(B, r), C)``, …. A ref with no fit
        selections yields the block in its own frame (still fixed)."""
        if not isinstance(sel, str):
            raise TypeError(
                "custom ref(sel, r): the first argument must be a bare reference selection "
                "string (evaluated on the reference structure), not a coord expression"
            )
        block_np = self._ref_block_entry(
            sel, r
        )  # (k, 3) reference-frame coords (const)
        block = self._ops.const_like(block_np, self._coords)
        fit = self._ref_fits.get(r)
        if fit is None:  # no fit selection -> use the reference block in its own frame
            return block
        fit_idx, fit_ref_np = fit
        fit_ref = self._ops.const_like(fit_ref_np, self._coords)
        p_fit = self._ops.gather(self._coords, fit_idx)  # (..., m, 3) moving anchor
        return V.superpose_ref(self._ops, block, fit_ref, p_fit)

    def norm(self, v):
        return V.norm(self._ops, v)

    def dot(self, u, v):
        return V.dot(self._ops, u, v)

    # --- penalty (vocabulary) ---
    def harmonic(self, x, target):
        return V.harmonic(self._ops, x, target)

    def flat_bottomed(self, x, lo, hi):
        return V.flat_bottomed(self._ops, x, lo, hi)

    def flat_bottomed1(self, x, lo):
        return V.flat_bottomed1(self._ops, x, lo)

    def flat_bottomed2(self, x, hi):
        return V.flat_bottomed2(self._ops, x, hi)

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

    def wrap(self, x):
        # composite (not a bare ops passthrough): fold an angle/deviation into [-pi, pi]
        return V.wrap(self._ops, x)

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
    runs (it only needs to execute, not be numerically meaningful). A ``kabsch``/``coords``
    result is a ``(2, 3)`` dummy BLOCK, so it composes into ``centroid``/``norm`` etc. — the
    geometry methods accept either a selection identifier (str, recorded) or such a block.
    ``rmsd`` additionally records the ``(selection, ref)`` pair so the engine can load +
    align that reference at setup."""

    def __init__(self):
        self.selections: list = []  # ordered-unique identifiers seen
        self.refs: list = []  # ordered-unique (selection_identifier, ref_name) pairs
        self.kabsch_pairs: list = []  # (a, b) selection-identifier pairs for count checks
        # ordered-unique (reference_selection, ref_name) pairs for the ref() primitive. The
        # selection is evaluated on the REFERENCE, so it is NOT a prediction selection (not
        # recorded in self.selections); the fit's prediction anchor lives in the refs map.
        self.ref_groups: list = []

    def _rec(self, *sels):
        for s in sels:
            if s not in self.selections:
                self.selections.append(s)

    def _block_of(self, a):
        """A selection identifier (str) -> record it + a DISTINCT ``(2, 3)`` dummy block (so
        centroids of different identifiers differ); an already-computed block (kabsch/coords
        output) -> pass through as a numpy array. Mirrors ``RestraintContext._coords_of``."""
        if isinstance(a, str):
            self._rec(a)
            i = float(self.selections.index(a) + 1)
            return np.array([[i, 1.0, 1.0], [i, 2.0, 2.0]])
        return np.asarray(a, dtype=float)

    # geometry
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

    def centroid(self, a):
        return np.mean(self._block_of(a), axis=-2)

    def rg(self, a):
        self._block_of(a)
        return 1.0

    def coords(self, a):
        return self._block_of(a)

    def kabsch(self, a, b):
        self._block_of(a), self._block_of(b)
        # record a bare selection-vs-selection pair so resolve can check |A| == |B| (Kabsch
        # needs a positional 1:1 correspondence); a nested block arg (str-free) is skipped.
        if isinstance(a, str) and isinstance(b, str):
            self.kabsch_pairs.append((a, b))
        # a (2, 3) dummy standing in for the superposed block (values only need to stay
        # finite while the resolve pass runs).
        return np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    def rmsd(self, a, r):
        if not isinstance(a, str):
            raise TypeError(
                "custom rmsd(A, ref): the first argument must be a bare selection "
                "identifier, not a coord expression such as kabsch(...)"
            )
        if not isinstance(r, str):
            raise TypeError(
                "custom rmsd(A, ref): the second argument (ref) must be a bare reference "
                "name from the entry's 'refs' map"
            )
        self._rec(a)
        if (a, r) not in self.refs:
            self.refs.append((a, r))
        return 1.0

    def ref(self, sel, r):
        if not isinstance(sel, str):
            raise TypeError(
                "custom ref(sel, r): the first argument must be a bare reference selection "
                "string (evaluated on the reference), not a coord expression such as kabsch(...)"
            )
        if not isinstance(r, str):
            raise TypeError(
                "custom ref(sel, r): the second argument (ref) must be a bare reference name "
                "from the entry's 'refs' map"
            )
        if (sel, r) not in self.ref_groups:
            self.ref_groups.append((sel, r))
        # a (2, 3) dummy BLOCK standing in for the fitted reference group (so it composes into
        # centroid/distance/angle/…). `sel` is a REFERENCE selection — deliberately NOT recorded
        # as a prediction selection (self.selections); its atoms are resolved on the ref instead.
        return np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    def norm(self, v):
        return float(np.sqrt(np.sum(np.asarray(v, dtype=float) ** 2))) + 1.0

    def dot(self, u, v):
        return 1.0

    # penalty + math: operate on the plain-float / np-array dummies natively
    def harmonic(self, x, target):
        return (x - target) ** 2

    def flat_bottomed(self, x, lo, hi):
        return 0.0

    def flat_bottomed1(self, x, lo):
        return 0.0

    def flat_bottomed2(self, x, hi):
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

    def wrap(self, x):
        # identity dummy: only needs to stay finite while selections are recorded
        return x

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
