"""Parse the ``restraints_config`` dict shared by all tools.

One source of truth for defaults and the distance-restraint encoding, so boltz
(YAML), protenix (JSON) and AF3 only need to extract the dict from their input
format and hand it here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rgi_utils._config_util import check_window_exclusive, coerce_bool
from rgi_utils.custom.data import CustomData
from rgi_utils.distance_restr_data import DistanceData
from rgi_utils.group_geom_restr_data import AngleRestraintData, DihedralRestraintData
from rgi_utils.rmsd_restr_data import RmsdData

logger = logging.getLogger(__name__)


@dataclass
class RestraintsConfig:
    verbose: bool = False
    gpu: bool = True
    backend: str | None = None  # "numpy" | "torch" | "jax"; None = auto
    method: str = "CG"
    max_iter: int = 100
    conf_start_sigma: float = -1.0  # one value for all conformer (ligand) restraints
    conf_stop_sigma: float = -1.0  # shared conformer lower bound; -1 = never released
    # shared conformer STEP window (the alternative gate axis to the sigma window above):
    # active for conf_start_step <= step <= conf_stop_step. -inf/+inf = always (default).
    # Mutually exclusive with the conformer sigma window.
    conf_start_step: float = float("-inf")
    conf_stop_step: float = float("inf")
    conformer_config: dict = field(default_factory=dict)
    distance_data: list = field(default_factory=list)
    rmsd_data: list = field(default_factory=list)
    angle_data: list = field(default_factory=list)  # group-centroid angle restraints
    dihedral_data: list = field(
        default_factory=list
    )  # group-centroid dihedral restraints
    custom_data: list = field(
        default_factory=list
    )  # custom restraints (rgi_utils.custom)

    @classmethod
    def from_dict(cls, config: dict | None) -> "RestraintsConfig":
        config = config or {}
        # Validate top-level keys against a fixed whitelist and RAISE on anything
        # unknown. The dangerous case this catches: a misspelled SECTION name (e.g.
        # 'distance_restraint_config' instead of 'distance_restraints_config') makes
        # config.get(<correct name>, []) return nothing and silently drops the WHOLE
        # restraint block -> a valid-looking unrestrained run (wasted GPU + wrong
        # conclusions). A warning would be muted by the package NullHandler, so this
        # must raise. 'start_sigma' is excluded so its dedicated migration message
        # below fires instead of this generic one.
        _KNOWN_TOP_LEVEL = {
            "verbose",
            "gpu",
            "backend",
            "method",
            "max_iter",
            "conformer_restraints_config",
            "distance_restraints_config",
            "rmsd_restraints_config",
            "angle_restraints_config",
            "dihedral_restraints_config",
            "custom_restraints_config",
        }
        _unknown_top = set(config) - _KNOWN_TOP_LEVEL - {"start_sigma"}
        if _unknown_top:
            raise ValueError(
                f"restraints_config: unknown top-level key(s) {sorted(_unknown_top)}. "
                f"Known keys: {sorted(_KNOWN_TOP_LEVEL)}. A misspelled section name "
                f"(e.g. 'distance_restraint_config') would silently drop the whole "
                f"restraint block, so it is rejected here."
            )
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
        # The conformer cis/trans term was renamed dihedral -> cistrans. Reject the old
        # key loudly (like the start_sigma / backend:numpy guards) rather than silently
        # falling back to the default weight, which would weaken or re-enable the term.
        if "dihedral" in conformer_config:
            raise ValueError(
                "conformer_restraints_config: 'dihedral' was renamed to 'cistrans' "
                "(it restrains acyclic double bonds' cis/trans (E/Z) geometry)."
            )
        # conformer terms share ONE gate window. Like every other restraint it is EITHER
        # a sigma window OR a step window (mutually exclusive); reuse the shared check.
        check_window_exclusive(conformer_config, "conformer_restraints_config")
        _css = conformer_config.get("start_sigma")
        conf_start_sigma = float(_css) if _css is not None else _ALWAYS_ON
        # shared conformer lower bound: release conformer terms below this noise level
        # (omitted -> -1 = off). Window: conf_stop <= sigma <= conf_start
        _csstop = conformer_config.get("stop_sigma")
        conf_stop_sigma = float(_csstop) if _csstop is not None else -1.0
        # shared conformer STEP window (omitted -> -inf/+inf = always): active for
        # conf_start_step <= step <= conf_stop_step (the alternative gate axis).
        _csa = conformer_config.get("start_step")
        conf_start_step = float(_csa) if _csa is not None else float("-inf")
        _cso = conformer_config.get("stop_step")
        conf_stop_step = float(_cso) if _cso is not None else float("inf")
        # Default GPU (gpu:true): minimize on whatever device the coords live on. Coerce
        # via the shared helper so a quoted/string value (e.g. "false"/"no"/"off"/"0") --
        # truthy in plain Python -- correctly turns the GPU off for a CPU-intended run.
        gpu = coerce_bool(config.get("gpu", True))
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
            conf_start_step=conf_start_step,
            conf_stop_step=conf_stop_step,
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
        # custom restraints (expression DSL / code fn). start_sigma None -> +inf (active
        # every step) is applied when the CustomSpec is built (CustomData.build_spec).
        for entry in config.get("custom_restraints_config", []) or []:
            cd = CustomData()
            cd.set_config(entry)
            cfg.custom_data.append(cd)
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
