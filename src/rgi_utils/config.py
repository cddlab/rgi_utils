"""Parse the ``restraints_config`` dict shared by all tools.

One source of truth for defaults and the distance-restraint encoding, so boltz
(YAML), protenix (JSON) and AF3 only need to extract the dict from their input
format and hand it here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rgi_utils.distance_restr_data import DistanceData
from rgi_utils.group_geom_restr_data import AngleRestraintData, DihedralRestraintData
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
    conf_stop_sigma: float = -1.0  # shared conformer lower bound; -1 = never released
    conformer_config: dict = field(default_factory=dict)
    distance_data: list = field(default_factory=list)
    rmsd_data: list = field(default_factory=list)
    angle_data: list = field(default_factory=list)  # group-centroid angle restraints
    dihedral_data: list = field(default_factory=list)  # group-centroid dihedral restraints

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
        # shared conformer lower bound: release conformer terms below this noise level
        # (omitted -> -1 = off). Window: conf_stop <= sigma <= conf_start
        _csstop = conformer_config.get("stop_sigma")
        conf_stop_sigma = float(_csstop) if _csstop is not None else -1.0
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
        # Normalize/validate the explicit backend. The numpy/scipy optimizer backend
        # was removed (gpu:false now runs torch on CPU), so reject it loudly rather
        # than silently doing something else.
        backend = config.get("backend", None)
        if backend is not None:
            backend = str(backend).strip().lower()
            if backend == "numpy":
                raise ValueError(
                    "the numpy/scipy restraint backend has been removed; use the "
                    "default torch backend (gpu:false runs torch on CPU) or "
                    "backend: jax"
                )
            if backend not in ("torch", "jax"):
                logger.warning(
                    "unknown restraints backend %r (expected torch/jax)", backend
                )
        cfg = cls(
            verbose=config.get("verbose", False),
            gpu=gpu,
            backend=backend,
            method=config.get("method", "CG"),
            max_iter=config.get("max_iter", 100),
            # one start_sigma for all conformer terms (omitted -> +inf = every step)
            conf_start_sigma=conf_start_sigma,
            conf_stop_sigma=conf_stop_sigma,
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
        # group-centroid angle (3 groups) / dihedral (4 groups) restraints — same
        # per-entry start_sigma convention as distance/rmsd (None -> +inf = every step).
        for entry in config.get("angle_restraints_config", []) or []:
            ad = AngleRestraintData()
            ad.set_config(entry)
            if ad.start_sigma is None:
                ad.start_sigma = _ALWAYS_ON
            cfg.angle_data.append(ad)
        for entry in config.get("dihedral_restraints_config", []) or []:
            dd = DihedralRestraintData()
            dd.set_config(entry)
            if dd.start_sigma is None:
                dd.start_sigma = _ALWAYS_ON
            cfg.dihedral_data.append(dd)
        return cfg

    def resolve_backend(self) -> str:
        """Default to torch; ``gpu`` selects the DEVICE (CPU when false, the
        accelerator when true), not the backend — so ``gpu:false`` runs the same
        torch optimizer on CPU. The only other backend is jax (AF3); the numpy/scipy
        optimizer was removed (an explicit ``backend: numpy`` is rejected in
        ``from_dict``)."""
        if self.backend:
            return self.backend
        return "torch"
