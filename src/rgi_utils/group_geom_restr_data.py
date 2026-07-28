"""Parse + resolve group-centroid angle / dihedral restraints.

These restrain the angle (3 groups) / dihedral (4 groups) formed by the centroids
of atom groups — the angular analogue of the centroid-distance restraint in
``distance_restr_data.py``, and they mirror its config surface: the four restraint
types ``harmonic`` / ``flat-bottomed`` / ``flat-bottomed1`` / ``flat-bottomed2`` and the
``move`` key. Targets are given in DEGREES in the config by default (set
``unit: radians`` on the entry to give them in radians) and stored in RADIANS here
(matching the energy layer, and symmetric with the distance restraint storing
Angstroms).

Each entry is gated on EITHER a sigma window (``start_sigma`` / ``stop_sigma``, noise
level) OR a step window (``start_step`` / ``stop_step``, diffusion step index) — the two
are mutually exclusive (see ``check_window_exclusive``); an omitted window is always-on.

Geometry convention (mirrors the conformer angle/dihedral terms):
  * angle    — groups 1-2-3, the VERTEX is group 2 (``centroid1 - centroid2 - centroid3``);
  * dihedral — groups 1-2-3-4, the rotatable AXIS is the ``centroid2-centroid3`` line.

``move``: which group(s) the solver may move; the rest are pinned (their centroids are
stop-gradient'd in the energy, so the CG holds them fixed FOR THIS restraint — another
restraint sharing those atoms still moves them, like the distance ``move``). The DEFAULT
(omitted) moves the "arms" and pins the anchor: an ANGLE frees groups 1 + 3 and pins the
vertex (group 2); a DIHEDRAL frees groups 1 + 4 and pins the axis (groups 2 + 3) — the
geometrically natural choice. Override with ``"all"`` (every group free), a single int
``k`` (only group ``k``), or a list / comma-string of 1-based indices (e.g. ``[1, 4]`` /
``"1,4"``). (Distance's ``both`` is a 2-group word; here it is a synonym for ``all``.)
Freeing a vertex/axis group instead admits multiple solutions — the angle/dihedral still
reaches its target, but the exact pose is not unique; pinning the anchor (the default)
avoids that.

For a dihedral, ``flat-bottomed`` windows cannot straddle +-180 deg (``target1 <
target2`` is enforced); use ``harmonic`` (periodicity-safe) for a target near +-180.
"""

from __future__ import annotations

import logging
import math

from rgi_utils._config_util import (
    apply_window_params,
    parse_geom_type,
    parse_move_indices,
    warn_unknown_keys,
)
from rgi_utils.atom_context import FrameworkAdapter, candidate_dict
from rgi_utils.selection import AtomSelector

logger = logging.getLogger(__name__)

_KNOWN_ANGLE_KEYS = {
    "atom_selection1",
    "atom_selection2",
    "atom_selection3",
    "weight",
    "start_sigma",
    "stop_sigma",
    "start_step",
    "stop_step",
    "unit",
    "move",
    "harmonic",
    "flat-bottomed",
    "flat-bottomed1",
    "flat-bottomed2",
}
_KNOWN_DIHEDRAL_KEYS = _KNOWN_ANGLE_KEYS | {"atom_selection4"}


def resolve_group_sites(
    adapter: FrameworkAdapter, selections: list[str]
) -> list[list[int]]:
    """Resolve each selection string to a list of global atom indices.

    Makes ONE pass over ``adapter.iter_atoms()`` and evaluates every selector per atom
    (so resolving N groups does not iterate the structure N times). Returns a parallel
    list of index lists (same order as ``selections``); raises if any group is empty —
    a group that selects nothing is a config error, not a silent no-op.
    """
    selectors = [AtomSelector(s) for s in selections]
    sites: list[list[int]] = [[] for _ in selectors]
    for atom in adapter.iter_atoms():
        candidate = candidate_dict(atom)
        for gi, sel in enumerate(selectors):
            if sel.eval(candidate):
                sites[gi].append(atom.index)
    for gi, s in enumerate(sites):
        if len(s) == 0:
            raise ValueError(f"group {gi + 1} selection matched no atoms")
    return sites


def _parse_move(config: dict, n_groups: int, default_free: tuple) -> tuple:
    """Parse the ``move`` key into a per-group "is free to move" tuple of ``n_groups``
    bools (the others are pinned). Unlike the 2-group distance restraint (where ``both``
    means the two groups), an angle has 3 groups and a dihedral 4, so ``move`` selects
    WHICH groups may move:

      * omitted -> ``default_free`` — the geometrically natural default: an ANGLE moves
        its two arms (groups 1 + 3) and pins the vertex (group 2); a DIHEDRAL moves its
        two end groups (1 + 4) and pins the axis (groups 2 + 3);
      * ``"all"`` -> every group free;
      * a single int ``k`` -> only group ``k`` free;
      * a list / comma- or space-separated string of 1-based indices (e.g. ``[1, 4]`` or
        ``"1,4"``) -> exactly those groups free.

    ``"both"`` is accepted as a synonym for ``"all"`` (it is the distance restraint's
    2-group word). Out-of-range / empty / non-integer raises — a silent fallback would
    move the wrong groups. The ``move`` vocabulary (int / list / comma-string / all/both)
    is parsed by the shared ``parse_move_indices`` so it stays in lockstep with the
    distance restraint's ``move`` key; here we only map the index set onto a free mask.
    """
    idx = parse_move_indices(config.get("move"), n_groups)
    if idx is None:
        return default_free
    return tuple((g + 1) in idx for g in range(n_groups))


def _parse_common(
    self, config: dict, n_groups: int, base: str, default_free: tuple
) -> None:
    """Shared parse of weight / start_sigma / stop_sigma / move / type for both classes
    (``self`` is the AngleRestraintData / DihedralRestraintData being filled).
    ``default_free`` is the per-group free mask used when ``move`` is omitted."""
    # weight + the sigma/step gate windows: one shared parse (so the null/zero handling
    # can't drift across distance/rmsd/angle/dihedral). The windows default to always-on
    # (set in __init__); start_sigma None -> +inf is filled by config.from_dict.
    apply_window_params(self, config, "angle/dihedral_restraints_config entry")
    self.move_free = _parse_move(config, n_groups, default_free)
    # angle/dihedral targets are DEGREES by default; `unit: radians` makes conv the
    # identity. The flat-bottomed `t1 < t2` check inside parse_geom_type runs on the raw
    # (pre-conv) values, so it is unit-agnostic. RMSD/distance pass `float` (native A).
    unit = str(config.get("unit", "degrees")).strip().lower()
    if unit not in ("degrees", "radians"):
        raise ValueError(
            f"angle/dihedral 'unit' must be 'degrees' or 'radians' "
            f"(got {config.get('unit')!r})"
        )
    conv = float if unit == "radians" else (lambda x: math.radians(float(x)))
    self.geom_type, self.target1, self.target2 = parse_geom_type(config, base, conv)


class AngleRestraintData:
    """Centroid angle restraint between three atom groups (vertex = group 2).

    Restraint types mirror the distance restraint (``harmonic`` / ``flat-bottomed`` /
    ``flat-bottomed1`` / ``flat-bottomed2``) on the angle value; ``target1``/``target2``
    are stored in RADIANS (from the config's degrees). ``start_sigma`` defaults to None
    here; ``RestraintsConfig.from_dict`` turns an omitted value into +inf.
    """

    atom_selection1: str
    atom_selection2: str
    atom_selection3: str
    target1: float  # radians
    target2: float  # radians
    geom_type: str | None  # harmonic / flat-bottomed / flat-bottomed1 / flat-bottomed2
    move_free: tuple  # per-group "free to move" bools (the rest are pinned)
    weight: float
    target_sites1: list
    target_sites2: list
    target_sites3: list
    run_restr: bool
    start_sigma: float
    stop_sigma: float
    start_step: float  # step-window lower bound (-inf = always); XOR the sigma window
    stop_step: float  # step-window upper bound (+inf = always)

    def __init__(self):
        self.atom_selection1 = None
        self.atom_selection2 = None
        self.atom_selection3 = None
        self.target1 = None
        self.target2 = None
        self.geom_type = None
        self.move_free = None  # set by set_config -> _parse_move (default all free)
        self.weight = 1.0
        self.target_sites1 = None
        self.target_sites2 = None
        self.target_sites3 = None
        self.run_restr = None
        self.start_sigma = None  # from_dict defaults None -> +inf (every step)
        self.stop_sigma = -1.0  # never released (active down to sigma=0)
        self.start_step = float(
            "-inf"
        )  # step-window (omitted -> always); XOR sigma win
        self.stop_step = float("inf")

    def set_config(self, config: dict):
        warn_unknown_keys(
            config, _KNOWN_ANGLE_KEYS, "angle_restraints_config entry", logger
        )
        self.atom_selection1 = config.get("atom_selection1", None)
        self.atom_selection2 = config.get("atom_selection2", None)
        self.atom_selection3 = config.get("atom_selection3", None)
        # default move: arms (groups 1 + 3) free, vertex (group 2) pinned
        _parse_common(
            self,
            config,
            n_groups=3,
            base="target_angle",
            default_free=(True, False, True),
        )
        self.run_restr = (
            self.atom_selection1 is not None
            and self.atom_selection2 is not None
            and self.atom_selection3 is not None
            and self.geom_type is not None
        )
        if not self.run_restr:
            raise ValueError(
                "angle restraint needs atom_selection1/2/3 and a type block "
                "(harmonic / flat-bottomed / flat-bottomed1 / flat-bottomed2)"
            )

    def resolve_sites(self, adapter: FrameworkAdapter) -> None:
        if not self.run_restr:
            return
        (
            self.target_sites1,
            self.target_sites2,
            self.target_sites3,
        ) = resolve_group_sites(
            adapter,
            [self.atom_selection1, self.atom_selection2, self.atom_selection3],
        )
        logger.info(
            "angle restraint resolved: groups %d/%d/%d atoms",
            len(self.target_sites1),
            len(self.target_sites2),
            len(self.target_sites3),
        )

    def is_valid(self) -> bool:
        return self.run_restr


class DihedralRestraintData:
    """Centroid dihedral restraint between four atom groups (axis = centroid2-centroid3 line).

    Restraint types mirror the distance restraint on the dihedral value;
    ``target1``/``target2`` are stored in RADIANS (from the config's degrees). The
    ``harmonic`` type is periodicity-safe (deviation wrapped to +-180); flat-bottomed
    windows cannot straddle +-180 (``target1 < target2`` enforced). ``start_sigma`` None
    -> +inf in ``from_dict``.
    """

    atom_selection1: str
    atom_selection2: str
    atom_selection3: str
    atom_selection4: str
    target1: float  # radians
    target2: float  # radians
    geom_type: str | None
    move_free: tuple  # per-group "free to move" bools (the rest are pinned)
    weight: float
    target_sites1: list
    target_sites2: list
    target_sites3: list
    target_sites4: list
    run_restr: bool
    start_sigma: float
    stop_sigma: float
    start_step: float  # step-window lower bound (-inf = always); XOR the sigma window
    stop_step: float  # step-window upper bound (+inf = always)

    def __init__(self):
        self.atom_selection1 = None
        self.atom_selection2 = None
        self.atom_selection3 = None
        self.atom_selection4 = None
        self.target1 = None
        self.target2 = None
        self.geom_type = None
        self.move_free = None  # set by set_config -> _parse_move (default all free)
        self.weight = 1.0
        self.target_sites1 = None
        self.target_sites2 = None
        self.target_sites3 = None
        self.target_sites4 = None
        self.run_restr = None
        self.start_sigma = None  # from_dict defaults None -> +inf (every step)
        self.stop_sigma = -1.0  # never released
        self.start_step = float(
            "-inf"
        )  # step-window (omitted -> always); XOR sigma win
        self.stop_step = float("inf")

    def set_config(self, config: dict):
        warn_unknown_keys(
            config, _KNOWN_DIHEDRAL_KEYS, "dihedral_restraints_config entry", logger
        )
        self.atom_selection1 = config.get("atom_selection1", None)
        self.atom_selection2 = config.get("atom_selection2", None)
        self.atom_selection3 = config.get("atom_selection3", None)
        self.atom_selection4 = config.get("atom_selection4", None)
        # default move: end groups (1 + 4) free, axis (groups 2 + 3) pinned
        _parse_common(
            self,
            config,
            n_groups=4,
            base="target_dihedral",
            default_free=(True, False, False, True),
        )
        self.run_restr = (
            self.atom_selection1 is not None
            and self.atom_selection2 is not None
            and self.atom_selection3 is not None
            and self.atom_selection4 is not None
            and self.geom_type is not None
        )
        if not self.run_restr:
            raise ValueError(
                "dihedral restraint needs atom_selection1..4 and a type block "
                "(harmonic / flat-bottomed / flat-bottomed1 / flat-bottomed2)"
            )

    def resolve_sites(self, adapter: FrameworkAdapter) -> None:
        if not self.run_restr:
            return
        (
            self.target_sites1,
            self.target_sites2,
            self.target_sites3,
            self.target_sites4,
        ) = resolve_group_sites(
            adapter,
            [
                self.atom_selection1,
                self.atom_selection2,
                self.atom_selection3,
                self.atom_selection4,
            ],
        )
        logger.info(
            "dihedral restraint resolved: groups %d/%d/%d/%d atoms",
            len(self.target_sites1),
            len(self.target_sites2),
            len(self.target_sites3),
            len(self.target_sites4),
        )

    def is_valid(self) -> bool:
        return self.run_restr
