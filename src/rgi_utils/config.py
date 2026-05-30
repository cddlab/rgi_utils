"""Parse the ``restraints_config`` dict shared by all tools.

One source of truth for defaults and the distance-restraint encoding, so boltz
(YAML), protenix (JSON) and AF3 only need to extract the dict from their input
format and hand it here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rgi_utils.distance_restr_data import DistanceData

logger = logging.getLogger(__name__)


@dataclass
class RestraintsConfig:
    verbose: bool = False
    gpu: bool = False
    backend: str | None = None  # "numpy" | "torch" | "jax"; None = auto
    method: str = "CG"
    max_iter: int = 100
    start_sigma: float = -1.0
    learning_rate: float = 0.01  # jax gradient-descent step size
    conformer_config: dict = field(default_factory=dict)
    distance_data: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, config: dict | None) -> "RestraintsConfig":
        config = config or {}
        cfg = cls(
            verbose=config.get("verbose", False),
            gpu=config.get("gpu", False),
            backend=config.get("backend", None),
            method=config.get("method", "CG"),
            max_iter=config.get("max_iter", 100),
            start_sigma=config.get("start_sigma", -1.0),
            learning_rate=config.get("learning_rate", 0.01),
            conformer_config=config.get("conformer_restraints_config", {}) or {},
        )
        for entry in config.get("distance_restraints_config", []) or []:
            dd = DistanceData()
            dd.set_config(entry)
            cfg.distance_data.append(dd)
        return cfg

    def resolve_backend(self) -> str:
        """Auto-select backend: torch if gpu else numpy; explicit wins."""
        if self.backend:
            return self.backend
        return "torch" if self.gpu else "numpy"
