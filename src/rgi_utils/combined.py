"""``CombinedRestraints`` — the single entry point used by every tool.

Lifecycle:
    restr = CombinedRestraints.get_instance()
    restr.set_config(config_dict)          # parse YAML/JSON restraints_config
    restr.setup(adapter, nbatch)           # resolve distances + build conformer spec
    restr.minimize(coords, step, sigma)    # called each denoising step
    restr.finalize(coords, step)           # optional stats

The backend (numpy/torch/jax) is chosen from the config; torch/jax optimizers are
imported lazily so importing this module needs neither. JAX tools that run inside
``jax.lax.scan`` should grab the pure minimizer via ``get_minimizer()`` instead of
calling ``minimize`` per step.
"""

from __future__ import annotations

import logging

from rgi_utils.config import RestraintsConfig
from rgi_utils.featurizer import build_spec

logger = logging.getLogger(__name__)


def _enable_verbose_logging() -> None:
    """Attach a stdout handler to the rgi_utils logger for verbose runs.

    Libraries normally stay silent (NullHandler in __init__). When the user sets
    verbose=true we surface restraint stats on stdout for debugging.
    """
    pkg = logging.getLogger("rgi_utils")
    if not any(isinstance(h, logging.StreamHandler) for h in pkg.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[rgi_utils] %(levelname)s %(message)s"))
        pkg.addHandler(handler)
    pkg.setLevel(logging.INFO)


class CombinedRestraints:
    _instance = None

    @classmethod
    def get_instance(cls) -> "CombinedRestraints":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (call between independent uses / tests)."""
        cls._instance = None

    def __init__(self) -> None:
        self.config = RestraintsConfig()
        self.spec = None
        self._backend = None
        self._optimizer = None
        self._minimize_fn = None

    def set_config(self, config: dict) -> None:
        self.config = RestraintsConfig.from_dict(config)
        if self.config.verbose:
            _enable_verbose_logging()

    def setup(self, adapter, nbatch: int = 1) -> None:
        """Resolve distance selections and build the conformer spec from the adapter."""
        cfg = self.config
        for dr in cfg.distance_data:
            dr.resolve_sites(adapter)

        ligand_confs = []
        if hasattr(adapter, "iter_ligand_confs"):
            ligand_confs = list(adapter.iter_ligand_confs())

        elements = None
        if hasattr(adapter, "get_elements"):
            try:
                elements = adapter.get_elements()
            except Exception as exc:  # element info is optional (VdW only)
                logger.warning("get_elements failed, VdW disabled: %s", exc)

        self.spec = build_spec(
            ligand_confs,
            cfg.distance_data,
            cfg.conformer_config,
            elements=elements,
            conf_start_sigma=cfg.conf_start_sigma,
        )
        self._backend = cfg.resolve_backend()
        self._optimizer = None
        self._minimize_fn = None

        if not self.spec.is_active():
            logger.info("CombinedRestraints: no active restraints")
            if cfg.verbose:
                # print() (not just logging) so it survives host logging configs
                print("[rgi_utils] setup: NO ACTIVE RESTRAINTS", flush=True)
            return
        self._build_optimizer()
        if cfg.verbose:
            d = self.spec.distance
            n_dist = 0 if d is None else int(d.mask.sum())
            vc = self.spec.vdw_config
            vdw_s = (
                "off"
                if vc is None
                else f"{len(vc.ligand_local)}lig/{len(vc.protein_global)}prot"
            )
            # Per-restraint start_sigma: conformer terms share conf_start_sigma;
            # each distance restraint has its own (show the observed range).
            if d is None or n_dist == 0:
                dist_ss = "n/a"
            else:
                ss = d.start_sigma
                dist_ss = (
                    f"{float(ss.min()):g}"
                    if float(ss.min()) == float(ss.max())
                    else f"{float(ss.min()):g}..{float(ss.max()):g}"
                )
            msg = (
                f"[rgi_utils] setup: backend={self._backend} "
                f"n_active={self.spec.n_active} "
                f"conformer={self.spec.has_conformer()} n_distance={n_dist} "
                f"vdw={vdw_s} conf_start_sigma={self.spec.conf_start_sigma:g} "
                f"dist_start_sigma={dist_ss}"
            )
            logger.info(msg)
            print(msg, flush=True)

    def _build_optimizer(self) -> None:
        b = self._backend
        # Dynamic ligand-protein VdW (vdw_config) is implemented in the torch
        # optimizer only; warn loudly rather than silently dropping it elsewhere.
        vc = self.spec.vdw_config
        if vc is not None and getattr(vc, "weight", 0) > 0 and b != "torch":
            logger.warning(
                "VdW restraint requested but backend=%s ignores it "
                "(dynamic ligand-protein VdW is implemented for the torch "
                "backend only); no VdW term will be applied.",
                b,
            )
        if b == "torch":
            from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

            self._optimizer = TorchRestraintOptimizer(
                self.spec, max_iter=self.config.max_iter, method=self.config.method
            )
        elif b == "jax":
            from rgi_utils.optim.jax_optim import make_minimizer

            self._minimize_fn = make_minimizer(
                self.spec,
                max_iter=self.config.max_iter,
                learning_rate=self.config.learning_rate,
                start_sigma=self.config.start_sigma,
                method=self.config.method,
            )
        elif b == "numpy":
            from rgi_utils.optim.numpy_optim import NumpyRestraintOptimizer

            self._optimizer = NumpyRestraintOptimizer(
                self.spec, max_iter=self.config.max_iter, method=self.config.method
            )
        else:
            raise ValueError(f"unknown backend: {b}")

    def is_active(self) -> bool:
        return self.spec is not None and self.spec.is_active()

    def get_minimizer(self):
        """Return the pure ``(coords, sigma) -> coords`` jax minimizer (jax backend)."""
        return self._minimize_fn

    def minimize(self, coords, istep: int = 0, sigma=None):
        """Optimize coordinates for one denoising step. Returns the (possibly new)
        coordinate object. torch/numpy mutate in place; jax returns a new array."""
        if not self.is_active():
            return coords
        b = self._backend
        # Per-restraint gating now lives in the energy (sigma <= start_sigma per
        # term); each optimizer additionally skips the step when sigma exceeds
        # every restraint's start_sigma (spec.max_start_sigma()).
        if b == "jax":
            s = sigma if sigma is not None else 1e30
            return self._minimize_fn(coords, s)
        if b == "torch":
            self._optimizer.minimize(coords, sigma=sigma)
            return coords
        if b == "numpy":
            return self._minimize_numpy(coords, sigma)
        return coords

    def _minimize_numpy(self, coords, sigma=None):
        import numpy as np

        if hasattr(coords, "detach"):  # torch tensor on the numpy backend (gpu:false)
            import torch

            arr = coords.detach().cpu().numpy().astype(np.float64)
            self._optimizer.minimize(arr, sigma=sigma)
            with torch.no_grad():
                coords.copy_(
                    torch.as_tensor(arr, device=coords.device, dtype=coords.dtype)
                )
            return coords
        arr = np.asarray(coords)
        self._optimizer.minimize(arr, sigma=sigma)
        return arr

    def finalize(self, coords, istep: int = 0) -> None:
        if not self.is_active() or not self.config.verbose:
            return
        try:
            e = self._restraint_energy(coords)
            msg = f"[rgi_utils] finalize (step {istep}): restraint energy = {e:.5f}"
            logger.info(msg)
            print(msg, flush=True)
        except Exception as exc:  # stats are best-effort
            logger.warning("finalize stats failed: %s", exc)

    def _restraint_energy(self, coords) -> float:
        b = self._backend
        if b == "numpy":
            arr = coords.detach().cpu().numpy() if hasattr(coords, "detach") else coords
            return self._optimizer.energy(arr)
        if b == "jax":
            from rgi_utils.optim.jax_optim import energy_of

            return energy_of(self.spec, coords)
        return self._optimizer.energy(coords)

    # --- Legacy parse-time builders (boltz schema.py back-compat) -------------
    # Ligand conformer restraints are now built from LigandConf via the
    # featurizer, so the ligand-path calls (make_bond/make_angle_restraints and
    # make_chiral without invert) are intentional no-ops. A few restraint kinds
    # are NOT yet ported and would otherwise be lost silently — warn loudly.
    _warned: set = set()

    @classmethod
    def _warn_unsupported(cls, feature: str) -> None:
        if feature not in cls._warned:
            cls._warned.add(feature)
            logger.warning(
                "rgi_utils does not yet support %s restraints; they are ignored. "
                "(distance + ligand bond/angle/chiral/vdw are supported.)",
                feature,
            )

    def make_chiral(self, *args, **kwargs) -> None:
        # invert=True is the polymer D-residue chirality flip (not yet ported);
        # the plain ligand call is handled by the featurizer.
        if kwargs.get("invert"):
            self._warn_unsupported("inverted-chirality (D-residue)")

    def make_bond(self, *args, **kwargs) -> None:
        pass

    def make_angle_restraints(self, *args, **kwargs) -> None:
        pass

    def make_link_bond(self, *args, **kwargs) -> None:
        self._warn_unsupported("inter-chain link-bond")

    def link_bonds_by_conf(self, *args, **kwargs) -> None:
        # only warn when there is actually link-bond config to apply
        cfg = args[1] if len(args) > 1 else kwargs.get("config")
        if cfg:
            self._warn_unsupported("link_bonds_by_conf")
