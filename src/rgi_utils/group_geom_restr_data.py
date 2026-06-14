"""Parse + resolve group-COM angle / dihedral restraints.

These restrain the angle (3 groups) / dihedral (4 groups) formed by the centres of
mass of atom groups — the angular analogue of the COM-distance restraint in
``distance_restr_data.py``, and they mirror its config surface: the four restraint
types ``harmonic`` / ``flat-bottomed`` / ``flat-bottomed1`` / ``flat-bottomed2`` and the
``move`` key. Targets are given in DEGREES in the config and stored in RADIANS here
(matching the energy layer, and symmetric with the distance restraint storing
Angstroms).

Geometry convention (mirrors the conformer angle/dihedral terms):
  * angle    — groups 1-2-3, the VERTEX is group 2 (``COM1 - COM2 - COM3``);
  * dihedral — groups 1-2-3-4, the rotatable AXIS is the ``COM2-COM3`` line.

``move``: which group(s) the solver may move; the rest are pinned (their COMs are
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

from rgi_utils.atom_context import FrameworkAdapter
from rgi_utils.selection import AtomSelector

logger = logging.getLogger(__name__)


def _resolve_group_sites(
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
        candidate = {
            "chain": atom.chain,
            "resid": atom.resid,
            "index": atom.index,
            "name": atom.name,
            "mol_type": atom.mol_type,
            "resname": atom.resname,
        }
        for gi, sel in enumerate(selectors):
            if sel.eval(candidate):
                sites[gi].append(atom.index)
    for gi, s in enumerate(sites):
        assert len(s) != 0, f"group {gi + 1} selection matched no atoms"
    return sites


def _parse_geom_type(config: dict, base: str):
    """Parse the distance-style restraint-type block of an angle/dihedral entry.

    ``base`` is ``"target_angle"`` or ``"target_dihedral"``; the harmonic block carries
    ``base``, the others ``base + "1"`` / ``base + "2"``. Values are DEGREES in the
    config; returns ``(geom_type_str, target1_rad, target2_rad)`` (unused -> 0.0), or
    ``(None, None, None)`` if no type block is present. Mirrors ``DistanceData``.
    """
    if "harmonic" in config:
        t = config["harmonic"].get(base)
        if t is None:
            raise ValueError(f"harmonic needs {base}")
        return "harmonic", math.radians(float(t)), 0.0
    if "flat-bottomed" in config:
        t1 = config["flat-bottomed"].get(f"{base}1")
        t2 = config["flat-bottomed"].get(f"{base}2")
        if t1 is None or t2 is None:
            raise ValueError(f"flat-bottomed needs {base}1 and {base}2")
        t1, t2 = float(t1), float(t2)
        if t1 >= t2:
            raise ValueError(f"{base}1 must be smaller than {base}2")
        return "flat-bottomed", math.radians(t1), math.radians(t2)
    if "flat-bottomed1" in config:
        t1 = config["flat-bottomed1"].get(f"{base}1")
        if t1 is None:
            raise ValueError(f"flat-bottomed1 needs {base}1")
        return "flat-bottomed1", math.radians(float(t1)), 0.0
    if "flat-bottomed2" in config:
        t2 = config["flat-bottomed2"].get(f"{base}2")
        if t2 is None:
            raise ValueError(f"flat-bottomed2 needs {base}2")
        return "flat-bottomed2", 0.0, math.radians(float(t2))
    return None, None, None


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
    move the wrong groups.
    """
    mv = config.get("move")
    if mv is None:
        return default_free
    if isinstance(mv, str):
        s = mv.strip().lower()
        if s in ("all", "both"):
            return tuple([True] * n_groups)
        items = [p for p in s.replace(",", " ").split() if p]
    elif isinstance(mv, (list, tuple)):
        items = list(mv)
    else:
        items = [mv]  # a bare int / float
    try:
        idx = {int(x) for x in items}
    except (TypeError, ValueError):
        raise ValueError(
            f"'move' must be 'all' or group indices 1..{n_groups} (got {mv!r})"
        )
    if not idx or any(k < 1 or k > n_groups for k in idx):
        raise ValueError(
            f"'move' indices must be within 1..{n_groups} (got {mv!r})"
        )
    return tuple((g + 1) in idx for g in range(n_groups))


def _parse_common(
    self, config: dict, n_groups: int, base: str, default_free: tuple
) -> None:
    """Shared parse of weight / start_sigma / stop_sigma / move / type for both classes
    (``self`` is the AngleRestraintData / DihedralRestraintData being filled).
    ``default_free`` is the per-group free mask used when ``move`` is omitted."""
    _w = config.get("weight")
    if _w is not None:
        self.weight = float(_w)
    _ss = config.get("start_sigma")
    if _ss is not None:
        self.start_sigma = float(_ss)
    _stop = config.get("stop_sigma")
    if _stop is not None:
        self.stop_sigma = float(_stop)
    self.move_free = _parse_move(config, n_groups, default_free)
    self.geom_type, self.target1, self.target2 = _parse_geom_type(config, base)


class AngleRestraintData:
    """COM angle restraint between three atom groups (vertex = group 2).

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

    def set_config(self, config: dict):
        self.atom_selection1 = config.get("atom_selection1", None)
        self.atom_selection2 = config.get("atom_selection2", None)
        self.atom_selection3 = config.get("atom_selection3", None)
        # default move: arms (groups 1 + 3) free, vertex (group 2) pinned
        _parse_common(
            self, config, n_groups=3, base="target_angle",
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
        ) = _resolve_group_sites(
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
    """COM dihedral restraint between four atom groups (axis = COM2-COM3 line).

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

    def set_config(self, config: dict):
        self.atom_selection1 = config.get("atom_selection1", None)
        self.atom_selection2 = config.get("atom_selection2", None)
        self.atom_selection3 = config.get("atom_selection3", None)
        self.atom_selection4 = config.get("atom_selection4", None)
        # default move: end groups (1 + 4) free, axis (groups 2 + 3) pinned
        _parse_common(
            self, config, n_groups=4, base="target_dihedral",
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
        ) = _resolve_group_sites(
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
