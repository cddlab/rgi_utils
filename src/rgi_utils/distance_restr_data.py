from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from rgi_utils.atom_context import FrameworkAdapter
from rgi_utils.selection import AtomSelector

logger = logging.getLogger(__name__)


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
    target_local_sites1: list
    target_local_sites2: list
    calc_method: str  # ["unfixed-absolute"]
    run_restr: bool
    start_sigma: float  # apply this restraint only when noise level <= start_sigma

    def __init__(self):
        self.atom_selection1 = None
        self.atom_selection2 = None
        self.target_distance = None
        self.target_distance1 = None
        self.target_distance2 = None
        # the attribute actually read by set_config/calc/grad/featurizer; init to
        # None so a config missing all type keys falls through to the clear
        # ValueError("distance restraints not run") instead of an AttributeError.
        self.distance_restraint_type = None
        self.target_sites1 = None
        self.target_sites2 = None
        self.target_local_sites1 = None
        self.target_local_sites2 = None
        self.calc_method = None
        self.run_restr = None
        self.start_sigma = None  # per-restraint; optional (from_dict defaults None -> +inf)

    def set_config(self, config: dict):
        self.atom_selection1 = config.get("atom_selection1", None)
        self.atom_selection2 = config.get("atom_selection2", None)
        self.calc_method = config.get("calc_method", "unfixed-absolute")
        # per-distance start_sigma (OPTIONAL; from_dict defaults None -> +inf = every
        # step). Guard float() against an explicit null so a `start_sigma:` /
        # `"start_sigma": null` entry is treated as omitted (-> the default) not a crash.
        _ss = config.get("start_sigma")
        if _ss is not None:
            self.start_sigma = float(_ss)
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
                if self.target_distance1 > self.target_distance2:
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
        self.target_local_sites1 = []
        self.target_local_sites2 = []

        atom_selector1 = AtomSelector(self.atom_selection1)
        atom_selector2 = AtomSelector(self.atom_selection2)

        for atom in adapter.iter_atoms():
            candidate = {
                "chain": atom.chain,
                "resid": atom.resid,
                "index": atom.index,
            }
            if atom_selector1.eval(candidate):
                self.target_sites1.append(atom.index)
            if atom_selector2.eval(candidate):
                self.target_sites2.append(atom.index)

        assert len(self.target_sites1) != 0, "target_sites1 is empty"
        assert len(self.target_sites2) != 0, "target_sites2 is empty"

        logger.info(
            "distance restraint resolved: group1=%d atoms, group2=%d atoms",
            len(self.target_sites1),
            len(self.target_sites2),
        )

    def _calculate_com_vector(self, crds: np.ndarray) -> np.ndarray:
        com1 = np.mean(crds[self.target_local_sites1, :], axis=0)
        com2 = np.mean(crds[self.target_local_sites2, :], axis=0)
        return com2 - com1

    def calc(self, crds_in: np.ndarray) -> float:
        if self.calc_method == "unfixed-absolute":
            com_vector = self._calculate_com_vector(crds_in)
            dist = np.linalg.norm(com_vector)
            delta = 0.0
            restraint_type = self.distance_restraint_type
            if restraint_type == "harmonic":
                delta = dist - self.target_distance
            elif (
                restraint_type in ("flat-bottomed", "flat-bottomed1")
                and dist < self.target_distance1
            ):
                delta = dist - self.target_distance1
            elif (
                restraint_type in ("flat-bottomed", "flat-bottomed2")
                and dist > self.target_distance2
            ):
                delta = dist - self.target_distance2
        else:
            raise NotImplementedError
        return delta**2

    def grad(self, crds: np.ndarray, grad: np.ndarray) -> None:
        if self.calc_method == "unfixed-absolute":
            com_vector = self._calculate_com_vector(crds)
            dist = np.linalg.norm(com_vector)
            if dist < 1e-8:
                return
            delta = 0.0
            restraint_type = self.distance_restraint_type
            if restraint_type == "harmonic":
                delta = dist - self.target_distance
            elif (
                restraint_type in ("flat-bottomed", "flat-bottomed1")
                and dist < self.target_distance1
            ):
                delta = dist - self.target_distance1
            elif (
                restraint_type in ("flat-bottomed", "flat-bottomed2")
                and dist > self.target_distance2
            ):
                delta = dist - self.target_distance2
            if abs(delta) < 1e-9:
                return
            coeff = 2 * delta
            grad_com = coeff * com_vector / dist
            grad_atom1 = -grad_com / len(self.target_local_sites1)
            grad_atom2 = grad_com / len(self.target_local_sites2)
            grad[self.target_local_sites1, :] += grad_atom1
            grad[self.target_local_sites2, :] += grad_atom2
        else:
            raise NotImplementedError

    def is_valid(self) -> bool:
        return self.run_restr

    def distance(self, crds: np.ndarray) -> float:
        return np.linalg.norm(self._calculate_com_vector(crds))

    def print(self, crds: np.ndarray) -> None:
        logger.info(
            f"COM distance of '{self.atom_selection1}'"
            f" - '{self.atom_selection2}': {self.distance(crds)}"
        )

    def calc_sd(self, crds: np.ndarray) -> float:
        dist = self.distance(crds)
        delta = 0.0
        restraint_type = self.distance_restraint_type

        if restraint_type == "harmonic":
            delta = dist - self.target_distance
        elif (
            restraint_type in ("flat-bottomed", "flat-bottomed1")
            and dist < self.target_distance1
        ):
            delta = dist - self.target_distance1
        elif (
            restraint_type in ("flat-bottomed", "flat-bottomed2")
            and dist > self.target_distance2
        ):
            delta = dist - self.target_distance2

        return delta**2
