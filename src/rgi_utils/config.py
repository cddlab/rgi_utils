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
    start_sigma: float = -1.0  # global default for every restraint
    conf_start_sigma: float = -1.0  # one value for all conformer (ligand) restraints
    learning_rate: float = 0.01  # jax gradient-descent step size
    conformer_config: dict = field(default_factory=dict)
    distance_data: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, config: dict | None) -> "RestraintsConfig":
        config = config or {}
        # Guard float() against an explicit null (YAML `start_sigma:` / JSON null,
        # both -> None): treat present-but-null like "unset" and fall back instead
        # of crashing with float(None).
        _gss = config.get("start_sigma")
        global_start_sigma = float(_gss) if _gss is not None else -1.0
        conformer_config = config.get("conformer_restraints_config", {}) or {}
        _css = conformer_config.get("start_sigma")
        conf_start_sigma = float(_css) if _css is not None else global_start_sigma
        # Coerce gpu to a real bool: a quoted/string value (e.g. "false"/"no"/"off"/
        # "0") is truthy in Python and would otherwise pick the torch (GPU) backend
        # for a CPU-intended run.
        gpu_raw = config.get("gpu", False)
        gpu = (
            gpu_raw
            if isinstance(gpu_raw, bool)
            else str(gpu_raw).strip().lower() in ("1", "true", "yes", "on")
        )
        # Normalize/validate the explicit backend, and surface a gpu/backend conflict.
        backend = config.get("backend", None)
        if backend is not None:
            backend = str(backend).strip().lower()
            if backend not in ("numpy", "torch", "jax"):
                logger.warning(
                    "unknown restraints backend %r (expected numpy/torch/jax)", backend
                )
            elif gpu and backend == "numpy":
                logger.warning(
                    "gpu:true but backend='numpy' is set explicitly; restraints will "
                    "run on the CPU (explicit backend overrides gpu)"
                )
        cfg = cls(
            verbose=config.get("verbose", False),
            gpu=gpu,
            backend=backend,
            method=config.get("method", "CG"),
            max_iter=config.get("max_iter", 100),
            start_sigma=global_start_sigma,
            # one conformer start_sigma for all ligands (falls back to global)
            conf_start_sigma=conf_start_sigma,
            learning_rate=config.get("learning_rate", 0.01),
            conformer_config=conformer_config,
        )
        for entry in config.get("distance_restraints_config", []) or []:
            dd = DistanceData()
            dd.set_config(entry)
            # each distance restraint may set its own start_sigma; else global
            if dd.start_sigma is None:
                dd.start_sigma = global_start_sigma
            cfg.distance_data.append(dd)
        return cfg

    def resolve_backend(self) -> str:
        """Auto-select backend: torch if gpu else numpy; explicit wins."""
        if self.backend:
            return self.backend
        return "torch" if self.gpu else "numpy"
