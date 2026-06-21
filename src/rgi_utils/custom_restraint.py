"""Built-in ``custom`` restraint — the config-only authoring pattern (pattern B).

This is ONE registered :class:`~rgi_utils.registry.RestraintType` (registered at import
time) whose data class *interprets a declarative config*: the user picks a geometric
``measure`` from a fixed vocabulary and a penalty ``form``, and supplies atom selections
+ target(s). No Python is written — an original restraint is added by editing the config
alone. Because the measures are engine code (the leaf functions in
``energy/{numpy,torch,jax}_energy.py:custom_energy``, shared by all three backends),
3-backend parity is satisfied automatically.

Config surface (one entry per restraint, under ``custom_restraints_config``)::

    custom_restraints_config:
      - measure: radius_of_gyration   # distance | angle | dihedral | radius_of_gyration
        atom_selection: "chain A"      # 1-group measures (rg); else atom_selection1..N
        form: harmonic                 # harmonic | flat-bottomed | flat-bottomed1/2
        target: 12.0                   # harmonic / single-bound target
        # target1, target2: flat-bottomed window bounds
        weight: 1.0
        # move: which groups are free (default all); start_sigma / stop_sigma optional

Measures and their group counts (centroids are the unweighted geometric mean of each
selection; the angular measures' targets are DEGREES, distance/rg are Angstrom):

* ``distance`` (2 groups)        — ``|centroid1 - centroid2|``
* ``angle`` (3 groups)           — angle at ``centroid2`` between ``centroid1`` / ``centroid3`` (degrees)
* ``dihedral`` (4 groups)        — dihedral about the ``centroid2-centroid3`` axis (degrees, periodicity-safe harmonic)
* ``radius_of_gyration`` (1 grp) — RMS spread of the group's atoms about their centroid

For distance/angle/dihedral the selected groups translate **rigidly** (the
``_move_centroid`` rescale, like the built-in group restraints), so ``weight: 1`` drives
any group size; ``move`` pins groups (default: all free). ``radius_of_gyration`` is an
internal spread, so it uses the plain centroid and the natural per-atom gradient.

The restraint is a per-entry, CG-solved energy term (the registry forbids the closed-form
``dist`` gate), so it shares the per-restraint ``start_sigma`` / ``stop_sigma`` window and
is summed by the same solver as conformer / rmsd / group restraints — on every backend.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from rgi_utils import registry
from rgi_utils._config_util import warn_unknown_keys
from rgi_utils.group_geom_restr_data import _parse_move, _resolve_group_sites
from rgi_utils.spec import DIST_TYPE_CODES

logger = logging.getLogger(__name__)

# measure name -> integer code consumed by the leaf ``custom_energy`` (the leaf computes
# every measure and selects by this code, like ``geom_type`` in the distance energy).
MEASURE_DISTANCE = 0
MEASURE_ANGLE = 1
MEASURE_DIHEDRAL = 2
MEASURE_RG = 3

# name -> (code, n_selection_groups). ``rg`` is an alias of radius_of_gyration,
# ``centroid_distance`` of distance.
_MEASURES = {
    "distance": (MEASURE_DISTANCE, 2),
    "centroid_distance": (MEASURE_DISTANCE, 2),
    "angle": (MEASURE_ANGLE, 3),
    "dihedral": (MEASURE_DIHEDRAL, 4),
    "radius_of_gyration": (MEASURE_RG, 1),
    "rg": (MEASURE_RG, 1),
}
# measures whose targets are angles (DEGREES in config -> RADIANS in the spec).
_ANGULAR_CODES = frozenset({MEASURE_ANGLE, MEASURE_DIHEDRAL})

_MAX_GROUPS = 4  # widest measure (dihedral); shorter measures pad the rest empty

_CUSTOM_KEYS = {
    "measure",
    "atom_selection",
    "atom_selection1",
    "atom_selection2",
    "atom_selection3",
    "atom_selection4",
    "form",
    "target",
    "target1",
    "target2",
    "weight",
    "move",
    "start_sigma",
    "stop_sigma",
}


def _parse_custom_form(config: dict, angular: bool):
    """Parse the penalty ``form`` of a custom entry into
    ``(form_str, target1, target2)``. ``form`` defaults to ``harmonic``. Targets are read
    from the flat keys ``target`` (harmonic / single bound) or ``target1`` / ``target2``
    (flat-bottomed window); for an angular measure they are converted DEGREES -> RADIANS.
    Returns the four distance-style ``form`` strings (mapped to ``DIST_TYPE_CODES`` later);
    raises on a missing/ill-ordered target so a misconfigured restraint never silently
    becomes a no-op."""
    form = str(config.get("form", "harmonic")).strip().lower()
    conv = math.radians if angular else (lambda x: float(x))

    def _need(key):
        if key not in config:
            raise ValueError(f"custom form {form!r} needs '{key}'")
        return float(config[key])

    if form == "harmonic":
        return "harmonic", conv(_need("target")), 0.0
    if form == "flat-bottomed":
        t1, t2 = float(_need("target1")), float(_need("target2"))
        if t1 >= t2:
            raise ValueError("custom flat-bottomed needs target1 < target2")
        return "flat-bottomed", conv(t1), conv(t2)
    if form == "flat-bottomed1":
        return "flat-bottomed1", conv(_need("target1")), 0.0
    if form == "flat-bottomed2":
        return "flat-bottomed2", 0.0, conv(_need("target2"))
    raise ValueError(
        f"unknown custom form {form!r} (harmonic / flat-bottomed / flat-bottomed1 / "
        "flat-bottomed2)"
    )


class CustomData:
    """One ``custom_restraints_config`` entry: parse + per-structure site resolution.

    Implements the registry's data-class contract (``set_config`` / ``resolve_sites`` /
    ``run_restr`` / ``start_sigma`` / ``stop_sigma`` / ``iter_global_sites``). Exposes
    ``target_sites1..4`` (unused groups are empty lists) so the data builder can pad every
    measure to a uniform 4-group layout."""

    def __init__(self) -> None:
        self.measure: str | None = None
        self.measure_type: int = MEASURE_DISTANCE
        self.n_groups: int = 0
        self.geom_type: str | None = None
        self.target1: float = 0.0
        self.target2: float = 0.0
        self.weight: float = 1.0
        self.move_free: tuple = (True, True, True, True)
        self.start_sigma: float | None = None
        self.stop_sigma: float = -1.0
        self.run_restr: bool = False
        self._selections: list[str] = []
        for g in range(_MAX_GROUPS):
            setattr(self, f"target_sites{g + 1}", [])

    def set_config(self, config: dict) -> None:
        warn_unknown_keys(
            config, _CUSTOM_KEYS, "custom_restraints_config entry", logger
        )
        measure = config.get("measure")
        if measure is None:
            raise ValueError("custom restraint entry needs a 'measure'")
        measure = str(measure).strip().lower()
        if measure not in _MEASURES:
            raise ValueError(
                f"unknown custom measure {measure!r}; known: {sorted(_MEASURES)}"
            )
        self.measure = measure
        self.measure_type, n = _MEASURES[measure]
        self.n_groups = n
        # selections: a 1-group measure accepts the bare 'atom_selection' (or
        # 'atom_selection1'); multi-group measures need atom_selection1..n.
        sels: list[str] = []
        if n == 1:
            s = config.get("atom_selection", config.get("atom_selection1"))
            if s is None:
                raise ValueError(f"measure {measure!r} needs 'atom_selection'")
            sels = [s]
        else:
            for g in range(n):
                s = config.get(f"atom_selection{g + 1}")
                if s is None:
                    raise ValueError(
                        f"measure {measure!r} needs 'atom_selection{g + 1}'"
                    )
                sels.append(s)
        self._selections = sels
        angular = self.measure_type in _ANGULAR_CODES
        self.geom_type, self.target1, self.target2 = _parse_custom_form(config, angular)
        _w = config.get("weight")
        if _w is not None:
            self.weight = float(_w)
        # default all groups free; pad the move mask out to 4 groups (unused groups are
        # pinned but also empty, so they never contribute).
        free = _parse_move(config, n, tuple([True] * n))
        self.move_free = tuple(free) + tuple([False] * (_MAX_GROUPS - n))
        _ss = config.get("start_sigma")
        self.start_sigma = float(_ss) if _ss is not None else None
        _stop = config.get("stop_sigma")
        self.stop_sigma = float(_stop) if _stop is not None else -1.0
        self.run_restr = self.geom_type is not None

    def resolve_sites(self, adapter) -> None:
        if not self.run_restr:
            return
        sites = _resolve_group_sites(adapter, self._selections)
        for g in range(_MAX_GROUPS):
            setattr(self, f"target_sites{g + 1}", sites[g] if g < len(sites) else [])
        logger.info(
            "custom restraint (%s) resolved: groups %s atoms",
            self.measure,
            "/".join(str(len(s)) for s in sites),
        )

    def iter_global_sites(self):
        out: list[int] = []
        for g in range(_MAX_GROUPS):
            out.extend(int(s) for s in (getattr(self, f"target_sites{g + 1}") or []))
        return out


@dataclass
class CustomArrays:
    """Padded custom-restraint arrays (one row per entry, 4 groups, local indices).

    Unused groups (a measure narrower than 4 groups) carry an all-zero mask. ``measure_type``
    selects the geometry in the leaf; ``form_type`` is a ``DIST_TYPE_CODES`` penalty shape.
    """

    grp1_idx: np.ndarray
    grp2_idx: np.ndarray
    grp3_idx: np.ndarray
    grp4_idx: np.ndarray
    grp1_mask: np.ndarray
    grp2_mask: np.ndarray
    grp3_mask: np.ndarray
    grp4_mask: np.ndarray
    measure_type: np.ndarray  # (n,) int (MEASURE_*)
    target1: np.ndarray  # (n,) float (radians for angular measures, else Angstrom)
    target2: np.ndarray  # (n,) float
    form_type: np.ndarray  # (n,) int (DIST_TYPE_CODES: 0=harmonic..3=upper)
    move_free: np.ndarray  # (n, 4) {0,1}: 1 = group free to move
    weight: np.ndarray  # (n,)
    mask: np.ndarray  # (n,) {0,1}: 1 = valid restraint
    start_sigma: np.ndarray  # (n,)
    stop_sigma: np.ndarray  # (n,)


# (field, kind) per CustomArrays field, in pack_spec order. Must include mask +
# start_sigma + stop_sigma (the per-entry gate the dispatch reads).
SPEC_SCHEMA = (
    ("grp1_idx", "i"),
    ("grp2_idx", "i"),
    ("grp3_idx", "i"),
    ("grp4_idx", "i"),
    ("grp1_mask", "f"),
    ("grp2_mask", "f"),
    ("grp3_mask", "f"),
    ("grp4_mask", "f"),
    ("measure_type", "i"),
    ("target1", "f"),
    ("target2", "f"),
    ("form_type", "i"),
    ("move_free", "f"),
    ("weight", "f"),
    ("mask", "f"),
    ("start_sigma", "f"),
    ("stop_sigma", "f"),
)

# leaf-call args (positional, before the trailing gated mask appended by term_energies).
TERM_ARGS = (
    "grp1_idx",
    "grp2_idx",
    "grp3_idx",
    "grp4_idx",
    "grp1_mask",
    "grp2_mask",
    "grp3_mask",
    "grp4_mask",
    "measure_type",
    "target1",
    "target2",
    "form_type",
    "move_free",
    "weight",
)


def _pad_groups4(items, g2l):
    """Pad each item's 4 (some empty) groups into (n, max_grp) local-index + {0,1} mask
    arrays. Like ``featurizer._pad_groups`` but tolerant of empty groups (a measure with
    < 4 groups leaves the rest as all-zero-mask padding)."""
    n = len(items)
    max_grp = 1
    for it in items:
        for g in range(_MAX_GROUPS):
            max_grp = max(max_grp, len(getattr(it, f"target_sites{g + 1}") or []))
    idx = [np.zeros((n, max_grp), dtype=np.int64) for _ in range(_MAX_GROUPS)]
    msk = [np.zeros((n, max_grp)) for _ in range(_MAX_GROUPS)]
    for ri, it in enumerate(items):
        for g in range(_MAX_GROUPS):
            local = [g2l[int(s)] for s in (getattr(it, f"target_sites{g + 1}") or [])]
            idx[g][ri, : len(local)] = local
            msk[g][ri, : len(local)] = 1.0
    return idx, msk


def build_custom_arrays(items, g2l) -> CustomArrays:
    """data_builder: resolved CustomData items + global->local map -> CustomArrays."""
    n = len(items)
    (g1, g2, g3, g4), (m1, m2, m3, m4) = _pad_groups4(items, g2l)
    return CustomArrays(
        grp1_idx=g1,
        grp2_idx=g2,
        grp3_idx=g3,
        grp4_idx=g4,
        grp1_mask=m1,
        grp2_mask=m2,
        grp3_mask=m3,
        grp4_mask=m4,
        measure_type=np.array([it.measure_type for it in items], dtype=np.int64),
        target1=np.array([float(it.target1) for it in items]),
        target2=np.array([float(it.target2) for it in items]),
        form_type=np.array(
            [DIST_TYPE_CODES[it.geom_type] for it in items], dtype=np.int64
        ),
        move_free=np.array([[1.0 if f else 0.0 for f in it.move_free] for it in items]),
        weight=np.array([float(it.weight) for it in items]),
        mask=np.ones(n),
        start_sigma=np.array(
            [
                float(it.start_sigma) if it.start_sigma is not None else float("inf")
                for it in items
            ]
        ),
        stop_sigma=np.array([float(getattr(it, "stop_sigma", -1.0)) for it in items]),
    )


CUSTOM_RESTRAINT = registry.RestraintType(
    name="custom",
    config_section="custom_restraints_config",
    data_class=CustomData,
    data_builder=build_custom_arrays,
    spec_schema=SPEC_SCHEMA,
    term_args=TERM_ARGS,
    # lazy dotted paths -> resolved only when that backend runs, so importing this module
    # (at ``import rgi_utils``) never pulls torch/jax.
    leaf_fns={
        "numpy": "rgi_utils.energy.numpy_energy:custom_energy",
        "torch": "rgi_utils.energy.torch_energy:custom_energy",
        "jax": "rgi_utils.energy.jax_energy:custom_energy",
    },
    n_groups=_MAX_GROUPS,
)


def register() -> None:
    """Register the built-in ``custom`` restraint (idempotent). Called at package import."""
    if registry.get_registered("custom") is None:
        registry.register_restraint(CUSTOM_RESTRAINT)
