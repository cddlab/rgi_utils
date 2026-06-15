"""``CombinedRestraints`` — the restraint entry point used by every tool.

Supported (instance-scoped) lifecycle — construct ONE per structure so batch
runs never cross-contaminate:
    restr = CombinedRestraints()
    restr.setup(adapter, nbatch, config=restraints_config)  # config optional
    restr.minimize(coords, step, sigma)    # called each denoising step
    restr.finalize(coords, step)           # optional per-term energy stats

``setup`` clears any prior derived state and (re)builds the spec, so reusing an
instance is safe too. Two calls ``set_config(dict)`` then ``setup(adapter)`` are
equivalent to passing ``config=`` to ``setup``.

The backend (numpy/torch/jax) is chosen from the config; torch/jax optimizers are
imported lazily so importing this module needs neither. JAX tools that run inside
``jax.lax.scan`` should grab the pure minimizer via ``get_minimizer()``.

``get_instance()`` / ``reset()`` are a back-compat singleton shim (kept for
existing tests and boltz's legacy parse-time builders); new code should use the
instance-scoped lifecycle above, not the singleton.
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

    def setup(self, adapter, nbatch: int = 1, config: dict | None = None) -> None:
        """Build the restraint spec from the adapter (and optional ``config``).

        Instance-scoped lifecycle (the supported pattern):
        ``CombinedRestraints() -> setup(adapter, config=...) -> minimize ->
        finalize``. Derived state (spec / optimizers) is cleared up front so a
        reused instance never carries a stale spec, and passing ``config`` folds
        the old two-call ``set_config -> setup`` into one. Constructing a fresh
        instance per structure makes batch runs cross-contamination-free.
        """
        # Clear derived state first: a reused instance must not keep a stale spec.
        self.spec = None
        self._optimizer = None
        self._minimize_fn = None
        if config is not None:
            self.set_config(config)
        cfg = self.config
        for dr in cfg.distance_data:
            dr.resolve_sites(adapter)
        for rr in cfg.rmsd_data:
            rr.resolve_sites(adapter)
        for ar in cfg.angle_data:
            ar.resolve_sites(adapter)
        for dr in cfg.dihedral_data:
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
            conf_stop_sigma=cfg.conf_stop_sigma,
            rmsd_restraints=cfg.rmsd_data,
            angle_restraints=cfg.angle_data,
            dihedral_restraints=cfg.dihedral_data,
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
        self._warn_never_active()
        self._build_optimizer()
        if cfg.verbose:
            d = self.spec.distance
            n_dist = 0 if d is None else int(d.mask.sum())
            rm = self.spec.rmsd
            n_rmsd = 0 if rm is None else int(rm.mask.sum())
            ga = self.spec.group_angle
            n_grp_angle = 0 if ga is None else int(ga.mask.sum())
            gd = self.spec.group_dihedral
            n_grp_dihedral = 0 if gd is None else int(gd.mask.sum())
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
                f"n_rmsd={n_rmsd} n_group_angle={n_grp_angle} "
                f"n_group_dihedral={n_grp_dihedral} "
                f"vdw={vdw_s} conf_start_sigma={self.spec.conf_start_sigma:g} "
                f"dist_start_sigma={dist_ss}"
            )
            logger.info(msg)
            print(msg, flush=True)

    def _warn_never_active(self) -> None:
        """Warn when restraints are built (non-zero weights/counts) but their
        ``start_sigma`` is < 0, so the gate ``sigma <= start_sigma`` never fires —
        a silent no-op the verbose count/energy logs do NOT reveal (counts and the
        ungated finalize energy both read non-zero). ``-1`` is the documented
        never-fires sentinel, so this catches "forgot to set start_sigma"."""
        import numpy as np

        spec = self.spec
        msgs = []
        if spec.has_conformer() and float(spec.conf_start_sigma) < 0:
            msgs.append(
                f"conformer restraints are configured but conf_start_sigma="
                f"{float(spec.conf_start_sigma):g} < 0, so they will NEVER activate "
                f"(gate is sigma <= start_sigma) — set "
                f"conformer_restraints_config.start_sigma to a positive value"
            )
        if spec.has_conformer() and float(
            getattr(spec, "conf_stop_sigma", -1.0)
        ) > float(spec.conf_start_sigma):
            msgs.append(
                "conformer conf_stop_sigma > conf_start_sigma, so the active window "
                "(conf_stop_sigma <= sigma <= conf_start_sigma) is EMPTY and the "
                "conformer terms NEVER activate — set conf_stop_sigma below it"
            )
        d = spec.distance
        if d is not None and d.mask.sum() > 0:
            active = np.asarray(d.mask) > 0
            ss = np.asarray(d.start_sigma)[active]
            if ss.size and float(ss.min()) < 0:
                msgs.append(
                    "one or more distance restraints have start_sigma < 0, so they "
                    "will NEVER activate (gate is sigma <= start_sigma)"
                )
            stop = np.asarray(d.stop_sigma)[active]
            if stop.size and np.any(stop > ss):
                msgs.append(
                    "one or more distance restraints have stop_sigma > start_sigma, so "
                    "their active window is EMPTY and they NEVER activate"
                )
        rm = spec.rmsd
        if rm is not None and rm.mask.sum() > 0:
            active = np.asarray(rm.mask) > 0
            ss = np.asarray(rm.start_sigma)[active]
            if ss.size and float(ss.min()) < 0:
                msgs.append(
                    "one or more RMSD restraints have start_sigma < 0, so they "
                    "will NEVER activate (gate is sigma <= start_sigma)"
                )
            # stop_sigma > start_sigma inverts the window stop_sigma<=sigma<=start_sigma
            # to EMPTY -> the restraint is silently a no-op (counts/finalize still read
            # non-zero), so flag the likely config error loudly like start_sigma < 0.
            stop = np.asarray(rm.stop_sigma)[active]
            if stop.size and np.any(stop > ss):
                msgs.append(
                    "one or more RMSD restraints have stop_sigma > start_sigma, so the "
                    "active window (stop_sigma <= sigma <= start_sigma) is EMPTY and "
                    "they NEVER activate — set stop_sigma below start_sigma"
                )
        # group-centroid angle/dihedral: same per-restraint gate as distance/rmsd, so the
        # same silent-no-op traps apply (start_sigma < 0 never fires; stop > start is an
        # empty window). Counts + ungated finalize energy would both read non-zero.
        for label, arr in (
            ("angle", spec.group_angle),
            ("dihedral", spec.group_dihedral),
        ):
            if arr is None or arr.mask.sum() <= 0:
                continue
            active = np.asarray(arr.mask) > 0
            ss = np.asarray(arr.start_sigma)[active]
            if ss.size and float(ss.min()) < 0:
                msgs.append(
                    f"one or more group {label} restraints have start_sigma < 0, so "
                    f"they will NEVER activate (gate is sigma <= start_sigma)"
                )
            stop = np.asarray(arr.stop_sigma)[active]
            if stop.size and np.any(stop > ss):
                msgs.append(
                    f"one or more group {label} restraints have stop_sigma > "
                    f"start_sigma, so their active window is EMPTY and they NEVER "
                    f"activate — set stop_sigma below start_sigma"
                )
        for m in msgs:
            logger.warning(m)
            if self.config.verbose:
                print(f"[rgi_utils] WARNING: {m}", flush=True)

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
                method=self.config.method,
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
        coordinate object. torch mutates in place; jax returns a new array."""
        if not self.is_active():
            return coords
        b = self._backend
        # Per-restraint gating now lives in the energy (sigma <= start_sigma per
        # term); each optimizer additionally skips the step when sigma exceeds
        # every restraint's start_sigma (spec.max_start_sigma()).
        if b == "jax":
            # sigma=None means "no gating, all restraints active" (matching the torch
            # branch). The gate is `sigma <= start_sigma`, so the None sentinel must be
            # LOW (-inf) to pass every gate; 1e30 would skip everything.
            s = sigma if sigma is not None else float("-inf")
            return self._minimize_fn(coords, s)
        if b == "torch":
            return self._minimize_torch(coords, sigma)
        return coords

    def _minimize_torch(self, coords, sigma=None):
        """Run the torch optimizer (the only CPU/GPU restraint optimizer — the
        numpy/scipy backend was removed).

        - ``torch.Tensor``: optimize on the coords' device, except with ``gpu:false``
          on an accelerator tensor, where we compute on CPU (move to CPU, optimize,
          write the result back to the original device). The optimization is identical
          to the GPU path and the dynamic ligand-protein VdW still applies.
        - numpy array: optimize on a CPU torch tensor and write the result back in
          place (the torch optimizer replaces the old scipy path for array callers)."""
        import torch

        if isinstance(coords, torch.Tensor):
            if not self.config.gpu and coords.device.type != "cpu":
                cpu_coords = coords.detach().to("cpu")
                self._optimizer.minimize(cpu_coords, sigma=sigma)
                with torch.no_grad():
                    coords.copy_(
                        cpu_coords.to(device=coords.device, dtype=coords.dtype)
                    )
            else:
                self._optimizer.minimize(coords, sigma=sigma)
            return coords

        import numpy as np

        arr = np.asarray(coords)
        t = torch.as_tensor(arr, dtype=torch.float64)
        self._optimizer.minimize(t, sigma=sigma)
        out = t.detach().cpu().numpy()
        if isinstance(arr, np.ndarray) and arr.dtype.kind == "f" and arr.flags.writeable:
            arr[...] = out  # update the caller's array in place
            return arr
        return out

    def finalize(self, coords, istep: int = 0) -> None:
        if not self.is_active() or not self.config.verbose:
            return
        try:
            bd = self._restraint_breakdown(coords)
            total = (
                bd["bond"]
                + bd["angle"]
                + bd["chiral"]
                + bd["dihedral"]
                + bd["vdw"]
                + bd["distance"]
                + bd.get("rmsd", 0.0)
                + bd.get("group_angle", 0.0)
                + bd.get("group_dihedral", 0.0)
            )
            # The dynamic ligand-protein VdW (spec.vdw_config) is applied only inside
            # the torch optimizer and is absent from the static per-term breakdown
            # above (energy_breakdown reads only spec.vdw). Add it directly (it is
            # >= 0 by construction) — NOT as (optimizer.energy - static total), which
            # would mix the float64 static breakdown with the float32 optimizer energy
            # and leave the static terms' rounding error in the result (even negative).
            if (
                self._backend == "torch"
                and getattr(self.spec, "vdw_config", None) is not None
                and self._optimizer is not None
            ):
                dyn_vdw = float(self._optimizer.dynamic_vdw_energy(coords))
                bd["vdw"] += dyn_vdw
                total += dyn_vdw
            msg = (
                f"[rgi_utils] finalize (step {istep}): "
                f"bond={bd['bond']:.5f} angle={bd['angle']:.5f} "
                f"chiral={bd['chiral']:.5f} dihedral={bd['dihedral']:.5f} "
                f"vdw={bd['vdw']:.5f} "
                f"distance={bd['distance']:.5f} rmsd={bd.get('rmsd', 0.0):.5f} "
                f"group_angle={bd.get('group_angle', 0.0):.5f} "
                f"group_dihedral={bd.get('group_dihedral', 0.0):.5f} "
                f"total={total:.5f}"
            )
            logger.info(msg)
            print(msg, flush=True)
        except Exception as exc:  # stats are best-effort
            logger.warning("finalize stats failed: %s", exc)

    def _restraint_breakdown(self, coords) -> dict:
        """Per-term restraint energies at ``coords`` (active atoms), backend-aware."""
        import numpy as np

        spec = self.spec
        active_idx = spec.active_sites
        b = self._backend
        if b == "jax":
            import jax.numpy as jnp

            from rgi_utils.energy import jax_energy

            active = jnp.asarray(coords)[..., active_idx, :]
            return jax_energy.energy_breakdown(active, jax_energy.prepare_spec(spec))
        if b == "torch":
            import torch

            from rgi_utils.energy import torch_energy

            c = (
                coords
                if isinstance(coords, torch.Tensor)
                else torch.as_tensor(np.asarray(coords))
            )
            active = c[..., active_idx, :].to(torch.float64)
            prepared = torch_energy.prepare_spec(
                spec, dtype=torch.float64, device=active.device
            )
            return torch_energy.energy_breakdown(active, prepared)
        raise ValueError(f"unknown backend: {b}")

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
