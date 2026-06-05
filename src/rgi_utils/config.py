"""Parse the ``restraints_config`` dict shared by all tools.

One source of truth for defaults and the distance-restraint encoding, so boltz
(YAML), protenix (JSON) and AF3 only need to extract the dict from their input
format and hand it here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rgi_utils.distance_restr_data import DistanceData
from rgi_utils.rmsd_restr_data import RmsdData

logger = logging.getLogger(__name__)


@dataclass
class RestraintsConfig:
    verbose: bool = False
    gpu: bool = False
    backend: str | None = None  # "numpy" | "torch" | "jax"; None = auto
    method: str = "CG"
    max_iter: int = 100
    conf_start_sigma: float = -1.0  # one value for all conformer (ligand) restraints
    learning_rate: float = 0.01  # jax gradient-descent step size
    conformer_config: dict = field(default_factory=dict)
    distance_data: list = field(default_factory=list)
    rmsd_data: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, config: dict | None) -> "RestraintsConfig":
        config = config or {}
        # start_sigma is NOT a global key (the old global + per-entry override scheme was
        # confusing): it is set per distance entry and once for all conformer terms. A
        # top-level 'start_sigma' is rejected. It is OPTIONAL per restraint — when omitted
        # it defaults to +inf, i.e. the restraint is active at EVERY diffusion step. Set it
        # (e.g. 1.0) to apply a restraint only late (low-noise) in denoising.
        if "start_sigma" in config:
            raise ValueError(
                "restraints_config: top-level 'start_sigma' is not supported — set it on "
                "each distance_restraints_config entry and inside conformer_restraints_config "
                "(or omit it: a restraint with no start_sigma is active at every step)."
            )
        _ALWAYS_ON = float("inf")  # omitted start_sigma -> active at every step
        conformer_config = config.get("conformer_restraints_config", {}) or {}
        _css = conformer_config.get("start_sigma")
        conf_start_sigma = float(_css) if _css is not None else _ALWAYS_ON
        # Coerce gpu to a real bool: a quoted/string value (e.g. "false"/"no"/"off"/
        # "0") is truthy in Python and would otherwise pick the torch (GPU) backend
        # for a CPU-intended run.
        gpu_raw = config.get("gpu", False)
        if isinstance(gpu_raw, bool):
            gpu = gpu_raw
        elif isinstance(gpu_raw, (int, float)):
            # numeric flag: 1 / 1.0 / 2 -> True, 0 / 0.0 -> False (str("1.0") would
            # otherwise miss the literal set and wrongly pick the CPU backend).
            gpu = bool(gpu_raw)
        else:
            gpu = str(gpu_raw).strip().lower() in ("1", "true", "yes", "on")
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
            # one start_sigma for all conformer terms (omitted -> +inf = every step)
            conf_start_sigma=conf_start_sigma,
            learning_rate=config.get("learning_rate", 0.01),
            conformer_config=conformer_config,
        )
        for entry in config.get("distance_restraints_config", []) or []:
            dd = DistanceData()
            dd.set_config(entry)
            # start_sigma is optional; omitted -> active at every step (+inf gate).
            if dd.start_sigma is None:
                dd.start_sigma = _ALWAYS_ON
            cfg.distance_data.append(dd)
        for entry in config.get("rmsd_restraints_config", []) or []:
            rr = RmsdData()
            rr.set_config(entry)
            # start_sigma is optional; omitted -> active at every step (+inf gate).
            if rr.start_sigma is None:
                rr.start_sigma = _ALWAYS_ON
            cfg.rmsd_data.append(rr)
        return cfg

    def resolve_backend(self) -> str:
        """Auto-select backend: torch if gpu else numpy; explicit wins."""
        if self.backend:
            return self.backend
        return "torch" if self.gpu else "numpy"
