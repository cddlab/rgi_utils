from __future__ import annotations

import logging
from dataclasses import dataclass

from rgi_utils._config_util import warn_unknown_keys
from rgi_utils.atom_context import FrameworkAdapter
from rgi_utils.selection import AtomSelector

logger = logging.getLogger(__name__)

_KNOWN_DISTANCE_KEYS = {
    "atom_selection1",
    "atom_selection2",
    "calc_method",
    "start_sigma",
    "stop_sigma",
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
    move_mode: int  # 0=both / 1=grp1 only / 2=grp2 only (the 'move' config key)

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
        self.move_mode = 0  # which group moves: 0=both (default) / 1=grp1 / 2=grp2

    def set_config(self, config: dict):
        warn_unknown_keys(
            config, _KNOWN_DISTANCE_KEYS, "distance_restraints_config entry", logger
        )
        self.atom_selection1 = config.get("atom_selection1", None)
        self.atom_selection2 = config.get("atom_selection2", None)
        self.calc_method = config.get("calc_method", "unfixed-absolute")
        # per-distance start_sigma (OPTIONAL; from_dict defaults None -> +inf = every
        # step). Guard float() against an explicit null so a `start_sigma:` /
        # `"start_sigma": null` entry is treated as omitted (-> the default) not a crash.
        _ss = config.get("start_sigma")
        if _ss is not None:
            self.start_sigma = float(_ss)
        # per-distance stop_sigma (OPTIONAL; default -1 = never released). Releasing a
        # distance restraint late lets the model relax the centroid-driven pull in its final
        # steps; off by default like rmsd/conformer.
        _stop = config.get("stop_sigma")
        if _stop is not None:
            self.stop_sigma = float(_stop)
        # per-distance move mode (OPTIONAL; default "both"): which group(s) the closed-
        # form centroid shift moves. both/omitted -> 0 (split, both move); 1 -> only
        # atom_selection1's group; 2 -> only atom_selection2's group. Accepts int or str
        # (1 / "1"); unknown value raises (silent fallback would move wrong groups).
        _mv = config.get("move")
        if _mv is not None:
            _mv_s = str(_mv).strip().lower()
            if _mv_s == "both":
                self.move_mode = 0
            elif _mv_s == "1":
                self.move_mode = 1
            elif _mv_s == "2":
                self.move_mode = 2
            else:
                raise ValueError(
                    f"distance 'move' must be 'both', 1, or 2 (got {_mv!r})"
                )
        if "harmonic" in config:
            self.target_distance = config["harmonic"].get("target_distance", None)
            if self.target_distance is not None:
                self.distance_restraint_type = "harmonic"
                self.target_distance = float(self.target_distance)
            else:
                raise ValueError("target_distance is None")
        elif "flat-bottomed" in config:
            self.target_distance1 = config["flat-bottomed"].get(
                "target_distance1", None
            )
            self.target_distance2 = config["flat-bottomed"].get(
                "target_distance2", None
            )
            if self.target_distance1 is not None and self.target_distance2 is not None:
                self.distance_restraint_type = "flat-bottomed"
                self.target_distance1 = float(self.target_distance1)
                self.target_distance2 = float(self.target_distance2)
                if self.target_distance1 >= self.target_distance2:
                    raise ValueError(
                        "target_distance1 must be smaller than target_distance2"
                    )
            else:
                raise ValueError("target_distance1 or 2 is None")
        elif "flat-bottomed1" in config:
            self.target_distance1 = config["flat-bottomed1"].get(
                "target_distance1", None
            )
            if self.target_distance1 is not None:
                self.distance_restraint_type = "flat-bottomed1"
                self.target_distance1 = float(self.target_distance1)
            else:
                raise ValueError("target_distance1 is None")
        elif "flat-bottomed2" in config:
            self.target_distance2 = config["flat-bottomed2"].get(
                "target_distance2", None
            )
            if self.target_distance2 is not None:
                self.distance_restraint_type = "flat-bottomed2"
                self.target_distance2 = float(self.target_distance2)
            else:
                raise ValueError("target_distance2 is None")
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
            candidate = {
                "chain": atom.chain,
                "resid": atom.resid,
                "index": atom.index,
                "name": atom.name,
                "mol_type": atom.mol_type,
                "resname": atom.resname,
            }
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
