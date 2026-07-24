"""Parse the ``restraints_config`` dict shared by all tools.

One source of truth for defaults and the distance-restraint encoding, so boltz
(YAML), protenix (JSON) and AF3 only need to extract the dict from their input
format and hand it here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rgi_utils._config_util import check_window_exclusive, coerce_bool
from rgi_utils.base_pair_restr_data import BasePairData
from rgi_utils.custom.data import CustomData
from rgi_utils.distance_restr_data import DistanceData
from rgi_utils.group_geom_restr_data import AngleRestraintData, DihedralRestraintData
from rgi_utils.ref_geom_restr_data import RefGeomData, is_ref_anchored
from rgi_utils.rmsd_restr_data import RmsdData


@dataclass
class RestraintsConfig:
    verbose: bool = False
    gpu: bool = True
    method: str = "CG"
    max_iter: int = 100
    # one value for all conformer restraints; +inf = active every step (the
    # documented "omitted start_sigma" default, matching build_spec). from_dict always
    # passes this explicitly, so the default only applies to a bare RestraintsConfig().
    conf_start_sigma: float = float("inf")
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
    base_pair_data: list = field(
        default_factory=list
    )  # nucleic-acid base-pair restraints (expand to distance + plane)

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
            "method",
            "max_iter",
            "conformer_restraints_config",
            "distance_restraints_config",
            "rmsd_restraints_config",
            "angle_restraints_config",
            "dihedral_restraints_config",
            "custom_restraints_config",
            "base_pair_restraints_config",
        }
        _unknown_top = set(config) - _KNOWN_TOP_LEVEL - {"start_sigma"}
        if _unknown_top:
            # 'backend' was removed as a config key: it is now inferred from how the
            # engine is invoked (get_minimizer() -> jax; minimize(coords) with a
            # torch/numpy array -> torch). Give a targeted migration hint instead of
            # the generic unknown-key error so stale configs get an actionable message.
            hint = ""
            if "backend" in _unknown_top:
                hint = (
                    " Note: 'backend' is no longer configurable — it is inferred from "
                    "how the engine is invoked (get_minimizer() => jax; minimize() "
                    "with a torch/numpy array => torch)."
                )
            raise ValueError(
                f"restraints_config: unknown top-level key(s) {sorted(_unknown_top)}. "
                f"Known keys: {sorted(_KNOWN_TOP_LEVEL)}. A misspelled section name "
                f"(e.g. 'distance_restraint_config') would silently drop the whole "
                f"restraint block, so it is rejected here.{hint}"
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
        # The conformer plane term was renamed improper -> planarity -> plane. Reject both
        # old keys loudly (same policy as dihedral above) instead of silently ignoring
        # them, which would leave the opt-in plane term OFF without warning.
        if "improper" in conformer_config:
            raise ValueError(
                "conformer_restraints_config: 'improper' was renamed to 'plane' "
                "(a best-fit-plane restraint over aromatic rings + sp2 groups)."
            )
        if "planarity" in conformer_config:
            raise ValueError(
                "conformer_restraints_config: 'planarity' was renamed to 'plane' "
                "(it is now a servalcat-style best-fit-plane restraint over whole planar "
                "atom groups — aromatic/conjugated rings + non-ring sp2 groups — not just "
                "per-centre sp2 signed volume)."
            )
        known_conformer_keys = {
            "start_sigma",
            "stop_sigma",
            "start_step",
            "stop_step",
            "bond",
            "angle",
            "chiral",
            "plane",
            "cistrans",
            "vdw",
        }
        unknown_conformer = {
            key
            for key in conformer_config
            if not str(key).startswith("_") and key not in known_conformer_keys
        }
        if unknown_conformer:
            raise ValueError(
                "conformer_restraints_config: unknown key(s) "
                f"{sorted(unknown_conformer)}. Known keys: "
                f"{sorted(known_conformer_keys)}"
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
        # Validate `method` against the solver whitelist. An unrecognized value used to
        # SILENTLY fall back to L-BFGS (torch_optim._is_cg / jax_optim route any non-CG
        # string to L-BFGS), so a typo ('CGG', 'bfgs') ran a different, untested solver
        # with no error. Raise instead (a warning would be muted by the NullHandler).
        method = config.get("method", "CG")
        _valid_methods = {"cg", "ncg", "nonlinear-cg", "nonlinearcg", "l-bfgs", "lbfgs"}
        if str(method).lower() not in _valid_methods:
            raise ValueError(
                f"unknown method {method!r}: expected a CG alias "
                "(cg/ncg/nonlinear-cg/nonlinearcg) or l-bfgs (l-bfgs/lbfgs)"
            )
        cfg = cls(
            # coerce so a quoted/string value (e.g. "false"/"no"/"off"/"0") -- truthy in
            # plain Python -- correctly turns verbose OFF (mirrors the gpu coercion above).
            verbose=coerce_bool(config.get("verbose", False)),
            gpu=gpu,
            method=method,
            max_iter=config.get("max_iter", 100),
            # one start_sigma for all conformer terms (omitted -> +inf = every step)
            conf_start_sigma=conf_start_sigma,
            conf_stop_sigma=conf_stop_sigma,
            conf_start_step=conf_start_step,
            conf_stop_step=conf_stop_step,
            conformer_config=conformer_config,
        )
        # a distance/angle/dihedral entry carrying reference keys (ref_pdb/ref_cif or any
        # atom_selectionN_ref) is REFERENCE-ANCHORED: it fits the reference onto the prediction
        # and measures between prediction + fitted-reference groups. Route it to RefGeomData
        # (a kind="ref_geom" CustomSpec on the custom-closure path) instead of the array term.
        for entry in config.get("distance_restraints_config", []) or []:
            if is_ref_anchored(entry):
                rg = RefGeomData("distance")
                rg.set_config(entry)
                cfg.custom_data.append(rg)
                continue
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
            if is_ref_anchored(entry):
                rg = RefGeomData("angle")
                rg.set_config(entry)
                cfg.custom_data.append(rg)
                continue
            ad = AngleRestraintData()
            ad.set_config(entry)
            if ad.start_sigma is None:
                ad.start_sigma = _ALWAYS_ON
            cfg.angle_data.append(ad)
        for entry in config.get("dihedral_restraints_config", []) or []:
            if is_ref_anchored(entry):
                rg = RefGeomData("dihedral")
                rg.set_config(entry)
                cfg.custom_data.append(rg)
                continue
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
        # nucleic-acid base-pair restraints: a config-time macro that expands into WC
        # H-bond distance restraints (+ optional coplanarity plane) at resolve time. The
        # per-entry start_sigma applies to the generated distance restraints, so it uses
        # the same None -> +inf (every step) default as distance/rmsd.
        for entry in config.get("base_pair_restraints_config", []) or []:
            bp = BasePairData()
            bp.set_config(entry)
            if bp.start_sigma is None:
                bp.start_sigma = _ALWAYS_ON
            cfg.base_pair_data.append(bp)
        return cfg
