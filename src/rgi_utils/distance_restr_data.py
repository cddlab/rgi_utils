from __future__ import annotations

import logging
from dataclasses import dataclass

from rgi_utils._config_util import (
    apply_window_params,
    parse_geom_type,
    parse_move_indices,
    warn_unknown_keys,
)
from rgi_utils.atom_context import FrameworkAdapter, candidate_dict
from rgi_utils.selection import AtomSelector

logger = logging.getLogger(__name__)

_KNOWN_DISTANCE_KEYS = {
    "atom_selection1",
    "atom_selection2",
    "calc_method",
    "weight",
    "start_sigma",
    "stop_sigma",
    "start_step",
    "stop_step",
    "move",
    "harmonic",
    "flat-bottomed",
    "flat-bottomed1",
    "flat-bottomed2",
}


@dataclass
class DistanceData:
    atom_selection1: str
    atom_selection2: str
    target_distance: float
    target_distance1: float  # used in flat-bottomed, flat-bottomed1
    target_distance2: float  # used in flat-bottomed, flat-bottomed2
    distance_restraint_type: str | None  # harmonic / flat-bottomed /
    # flat-bottomed1 / flat-bottomed2 (assigned by set_config)
    target_sites1: list
    target_sites2: list
    calc_method: str  # ["unfixed-absolute"]
    run_restr: bool
    start_sigma: float  # apply this restraint only when noise level <= start_sigma
    stop_sigma: float  # RELEASE this restraint when noise level < stop_sigma (-1=never)
    start_step: float  # step-window lower bound (-inf = always); XOR the sigma window
    stop_step: float  # step-window upper bound (+inf = always)
    move_mode: int  # 0=both / 1=grp1 only / 2=grp2 only (the 'move' config key)
    # per-restraint least-squares weight (the CG jointly minimises Σ wᵢ·δᵢ²). A NO-OP
    # for a single restraint / restraints with disjoint groups (each reaches its exact
    # target regardless); only changes the outcome for OVER-CONSTRAINED coupled restraints
    # whose shared atom is their sole mover, where it balances the competition (2:1 etc).
    weight: float

    def __init__(self):
        self.atom_selection1 = None
        self.atom_selection2 = None
        self.target_distance = None
        self.target_distance1 = None
        self.target_distance2 = None
        # the attribute actually read by set_config/featurizer; init to None so a
        # config missing all type keys falls through to the clear
        # ValueError("distance restraints not run") instead of an AttributeError.
        self.distance_restraint_type = None
        self.target_sites1 = None
        self.target_sites2 = None
        self.calc_method = None
        self.run_restr = None
        self.start_sigma = (
            None  # per-restraint; optional (from_dict defaults None -> +inf)
        )
        self.stop_sigma = -1.0  # per-restraint lower bound; -1 = never released (off)
        # step-window: active for start_step <= step <= stop_step (omitted -> -inf/+inf =
        # always). Mutually exclusive with the sigma window; ANDed with it in the gate.
        self.start_step = float("-inf")
        self.stop_step = float("inf")
        self.move_mode = 0  # which group moves: 0=both (default) / 1=grp1 / 2=grp2
        self.weight = 1.0  # relative strength (no-op unless over-constrained coupling)

    def set_config(self, config: dict):
        warn_unknown_keys(
            config, _KNOWN_DISTANCE_KEYS, "distance_restraints_config entry", logger
        )
        self.atom_selection1 = config.get("atom_selection1", None)
        self.atom_selection2 = config.get("atom_selection2", None)
        self.calc_method = config.get("calc_method", "unfixed-absolute")
        # weight + the sigma/step gate windows: one shared parse (so the null/zero handling
        # can't drift across distance/rmsd/angle/dihedral). weight default 1.0 is a no-op
        # unless over-constrained coupling (see the field comment); the windows default to
        # always-on (set in __init__), and start_sigma None -> +inf is filled by from_dict.
        apply_window_params(self, config, "distance_restraints_config entry")
        # per-distance move mode (OPTIONAL; default "both"): which group(s) the centroid
        # shift moves. The `move` vocabulary is parsed by the shared parse_move_indices so
        # it stays in lockstep with the angle/dihedral `move` key; a 2-group distance maps
        # the returned index set onto the 0/1/2 move_mode enum. Accepts both/all/omitted ->
        # 0 (split, both move); 1 / [1] / "1" -> 1 (only atom_selection1's group); 2 / [2]
        # -> 2 (only atom_selection2's group); [1, 2] / "1,2" -> 0. Out-of-range ([1, 3],
        # 3) / empty raises (silent fallback would move wrong groups).
        idx = parse_move_indices(config.get("move"), 2)  # {1}, {2}, or {1, 2}
        if idx is not None:
            self.move_mode = {
                frozenset({1, 2}): 0,
                frozenset({1}): 1,
                frozenset({2}): 2,
            }[frozenset(idx)]
        # Restraint type + target(s) via the shared helper — the SAME parse rmsd /
        # angle / dihedral use, so the four type keys (harmonic / flat-bottomed /
        # flat-bottomed1 / flat-bottomed2) and their error messages can't drift. It maps
        # the returned (type, t1, t2) into distance's three target fields by type;
        # `_dist_params` reads only the field belonging to the resolved type, so the
        # unused ones stay at their __init__ defaults. conv=float (native Angstroms; the
        # flat-bottomed t1<t2 check lives inside parse_geom_type). No type key present ->
        # (None, None, None), which falls through to the run_restr=False raise below.
        gtype, t1, t2 = parse_geom_type(config, "target_distance", float)
        self.distance_restraint_type = gtype
        if gtype == "harmonic":
            self.target_distance = t1
        elif gtype == "flat-bottomed":
            self.target_distance1, self.target_distance2 = t1, t2
        elif gtype == "flat-bottomed1":
            self.target_distance1 = t1
        elif gtype == "flat-bottomed2":
            self.target_distance2 = t2
        self.run_restr = (
            (self.atom_selection1 is not None)
            and (self.atom_selection2 is not None)
            and (self.distance_restraint_type is not None)
        )

        if self.calc_method not in ["unfixed-absolute"]:
            raise ValueError("calc_method must be unfixed-absolute")

        if not self.run_restr:
            raise ValueError("distance restraints not run")

        logger.info(f"{self.distance_restraint_type=}")

    def resolve_sites(self, adapter: FrameworkAdapter) -> None:
        """Resolve atom indices for distance restraint sites using a framework
        adapter."""
        if not self.run_restr:
            return

        self.target_sites1 = []
        self.target_sites2 = []

        atom_selector1 = AtomSelector(self.atom_selection1)
        atom_selector2 = AtomSelector(self.atom_selection2)

        for atom in adapter.iter_atoms():
            candidate = candidate_dict(atom)
            if atom_selector1.eval(candidate):
                self.target_sites1.append(atom.index)
            if atom_selector2.eval(candidate):
                self.target_sites2.append(atom.index)

        if len(self.target_sites1) == 0:
            raise ValueError(
                f"distance restraint atom_selection1 matched no atoms: "
                f"{self.atom_selection1!r}"
            )
        if len(self.target_sites2) == 0:
            raise ValueError(
                f"distance restraint atom_selection2 matched no atoms: "
                f"{self.atom_selection2!r}"
            )

        logger.info(
            "distance restraint resolved: group1=%d atoms, group2=%d atoms",
            len(self.target_sites1),
            len(self.target_sites2),
        )

    def is_valid(self) -> bool:
        return self.run_restr
