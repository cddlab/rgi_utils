"""GPU restraint optimizer for PyTorch tools (boltz, protenix).

Minimizes the restraint energy on active-site coordinates using autograd for
gradients. ``method`` selects the solver: ``"CG"`` (default) -> a nonlinear
conjugate-gradient solver (Polak-Ribiere+ with a backtracking Armijo line search),
matching the jax backend's pure-jax CG (a port of this solver); ``"l-bfgs"`` ->
``torch.optim.LBFGS`` (strong-Wolfe). Operates in-place on the coordinate tensor
and stays on whatever device the coordinates live on, so ``gpu: true`` runs
entirely on GPU.

The ligand-protein VdW term (``spec.vdw_config``) is handled here rather than in
the static energy layer: the ligand atoms come from the optimised ``active`` set
while the protein atoms are a *fixed background* read from the full coordinate
tensor. The clash penalty is recomputed every closure call, so it tracks the
moving ligand (only the ligand is pushed; the protein is held fixed).
"""

from __future__ import annotations

import logging
import math

import torch

from rgi_utils.energy import torch_energy
from rgi_utils.energy._terms import CONF_KEYS, PER_ENTRY_KEYS
from rgi_utils.optim._cg_config import (
    ARMIJO_C1,
    BACKTRACK,
    EPS,
    FTOL,
    GG_FLOOR,
    GTOL,
    MAX_LS,
)
from rgi_utils.optim.distance_shift import apply_distance_shift_torch

logger = logging.getLogger(__name__)


class TorchRestraintOptimizer:
    def __init__(self, spec, max_iter: int = 100, method: str = "CG"):
        self.spec = spec
        self.max_iter = max_iter
        self.method = method
        self._prepared = None
        self._prepared_g = {}  # cache {gate-state -> stable pre-gated prepared} (GPU CG)
        self._active_idx = None
        self._device = None
        self._vdw = None  # dict of device tensors for the ligand-protein VdW term
        # custom restraints -> torch closures, built lazily in _ensure UNDER
        # inference_mode(False) and on the coords' device, so the baked selection-index
        # tensors are normal + on-device (an inference tensor used in the autograd gather
        # can't be saved for backward -- boltz/Lightning run under inference_mode).
        self._custom_terms = None

    def _custom_energy(self, active, sigma):
        """Per-entry sigma-gated sum of the custom-restraint closure energies at ``active``
        (local coords). Returns ``None`` when no custom term is active so the caller can
        skip adding it. Gate: ``stop_sigma <= sigma <= start_sigma`` (a python-float
        compare, since the eager CG has a python-float sigma)."""
        if not self._custom_terms:
            return None
        total = None
        for _name, start, stop, closure in self._custom_terms:
            if sigma is not None and not (sigma <= start and sigma >= stop):
                continue
            e = closure(active)
            total = e if total is None else total + e
        return total

    def _ensure(self, device, dtype) -> None:
        if self._prepared is not None and self._device == device:
            return
        # Build constant tensors outside inference mode so they are normal (not
        # inference) tensors and can participate in autograd ops with the leaf.
        with torch.inference_mode(False):
            self._active_idx = torch.as_tensor(
                self.spec.active_sites, dtype=torch.long, device=device
            )
            self._prepared = torch_energy.prepare_spec(
                self.spec, device=device, dtype=dtype
            )
            self._prepared_g = {}  # rebuilt lazily for the new device
            self._setup_vdw(device, dtype)
            from rgi_utils.custom.closure import build_terms

            self._custom_terms = build_terms(self.spec.custom, "torch", device=device)
        self._device = device
        if self._custom_terms:
            logger.info(
                "%d custom restraint(s): torch CG runs EAGER (the fused torch.compile GPU "
                "path bypasses closure terms, so it is skipped when customs are present)",
                len(self._custom_terms),
            )

    def _gated_prepared(self, sigma):
        """Stable pre-gated ``prepared`` for the compiled GPU CG, cached by the discrete
        GATE STATE — the conformer gate (``sigma <= conf_start_sigma``) plus the
        per-restraint gate (``stop_sigma <= sigma <= start_sigma``) of every per-entry
        term (rmsd, group_angle, group_dihedral — ``PER_ENTRY_KEYS``). The noise
        gate is folded into the masks here so the compiled energy is called with
        ``sigma=None``; scalar leaves (e.g. ``conf_start_sigma``) are DROPPED because a
        python-float in the compiled energy's pytree makes dynamo guard on its value and
        recompile per distinct value. Stable object identity per gate state lets
        ``torch.compile`` reuse its artifact; sigma decreases monotonically so each gate
        flips at most once -> a few states -> a few compiles, then reuse. distance is
        excluded (closed-form). The conformer-gated key set (``CONF_KEYS``) and the
        per-entry-gated set (``PER_ENTRY_KEYS``) both come from ``_TERMS``, so adding a
        term can't silently leave it ungated on the compiled path."""
        p = self._prepared
        # conformer gate over the window conf_stop <= sigma <= conf_start (conf_stop=-1
        # -> never released, so this reduces to the old sigma<=conf_start gate).
        cg_key = (
            1.0
            if sigma is None
            else float(
                (sigma <= float(self.spec.conf_start_sigma))
                and (sigma >= float(self.spec.conf_stop_sigma))
            )
        )
        # per-restraint on/off for every per-entry term present (one tiny sync each),
        # over the active window stop_sigma <= sigma <= start_sigma (released below
        # stop_sigma so the model's final low-sigma steps re-idealise geometry).
        gates: dict[str, tuple] = {}
        if sigma is not None:
            for gk in PER_ENTRY_KEYS:
                if gk in p:
                    pe = p[gk]
                    on = (sigma <= pe["start_sigma"]) & (sigma >= pe["stop_sigma"])
                    gates[gk] = tuple(bool(b) for b in on.tolist())
        # the cache key carries EVERY gate state, so a step flipping only a group gate
        # gets its own (correct) entry instead of reusing a stale conf/rmsd mask.
        key = (cg_key, tuple(sorted(gates.items())))
        cache = self._prepared_g
        if key not in cache:
            pg = {}
            for k, v in p.items():
                if not isinstance(v, dict):
                    continue  # drop scalar leaves (conf_start_sigma): a python-float in
                    # the compiled energy's pytree forces a per-value dynamo recompile,
                    # and total_energy(sigma=None) never reads it
                elif k in CONF_KEYS:
                    pg[k] = {**v, "mask": v["mask"] * cg_key}
                elif k in gates:
                    rg = torch.tensor(
                        gates[k], dtype=v["mask"].dtype, device=v["mask"].device
                    )
                    pg[k] = {**v, "mask": v["mask"] * rg}
                else:
                    pg[k] = v
            cache[key] = pg
        return cache[key]

    def _setup_vdw(self, device, dtype) -> None:
        vc = getattr(self.spec, "vdw_config", None)
        if vc is None or vc.weight <= 0:
            self._vdw = None
            return
        self._vdw = {
            "lig_local": torch.as_tensor(
                vc.ligand_local, dtype=torch.long, device=device
            ),
            "lig_r": torch.as_tensor(vc.ligand_radii, dtype=dtype, device=device),
            "prot_global": torch.as_tensor(
                vc.protein_global, dtype=torch.long, device=device
            ),
            "prot_r": torch.as_tensor(vc.protein_radii, dtype=dtype, device=device),
            # 0-dim tensors (NOT python floats): passed into the compiled VdW energy, where
            # a python-float arg would make dynamo guard on its value and recompile per
            # distinct scale/weight across structures in a batch.
            "weight": torch.as_tensor(float(vc.weight), dtype=dtype, device=device),
            "scale": torch.as_tensor(float(vc.scale), dtype=dtype, device=device),
        }

    def _vdw_energy(self, active, prot_pos):
        """Ligand-protein VdW repulsion (delegates to the pure ``_vdw_pair_energy`` so the
        all-pairs ``weight*sum(clamp(d - scale*(r_i+r_j), max=0)**2)`` maths lives once —
        the compiled GPU energy folds in the same function). ``active`` (..., n_active, 3)
        is the optimised tensor; ``prot_pos`` (..., n_prot, 3) the fixed background."""
        from rgi_utils.optim._torch_cg_gpu import _vdw_pair_energy

        v = self._vdw
        return _vdw_pair_energy(
            active,
            prot_pos,
            v["lig_local"],
            v["lig_r"],
            v["prot_r"],
            v["scale"],
            v["weight"],
        )

    def minimize(self, coords, sigma=None, start_sigma=None, max_iter=None):
        """Optimize ``coords`` (..., n_atom, 3) in-place. Each restraint is gated
        on ``sigma <= its start_sigma`` inside the energy; the whole step is
        skipped only when ``sigma`` exceeds every restraint's start_sigma."""
        if not self.spec.is_active():
            return coords
        # sigma is the per-step scalar noise level. Coerce to a python float: the skip
        # test below and the GPU pre-gate (which builds the rmsd-gate tuple from it) both
        # assume a scalar; a stray multi-element tensor would otherwise fail GPU-only.
        if sigma is not None:
            sigma = float(sigma)
        if sigma is not None and sigma > self.spec.max_start_sigma():
            return coords
        # Optimize in fp32 even when the model runs the diffusion in bf16/fp16: a
        # half-precision CG line search + autograd gradient is too coarse, so the
        # restraint could silently fail to converge. Work in fp32 and cast the result
        # back to the coord dtype. fp32/fp64 tools are unaffected (work_dtype ==
        # coords.dtype), so this is a safe no-op for them.
        out_dtype = coords.dtype
        work_dtype = (
            torch.float32
            if coords.dtype in (torch.float16, torch.bfloat16)
            else coords.dtype
        )
        self._ensure(coords.device, work_dtype)
        mi = max_iter if max_iter is not None else self.max_iter
        # VdW is a conformer restraint -> gated by the conformer window
        # conf_stop <= sigma <= conf_start (conf_stop=-1 -> never released)
        vdw_active = self._vdw is not None and (
            sigma is None
            or (
                sigma <= float(self.spec.conf_start_sigma)
                and sigma >= float(self.spec.conf_stop_sigma)
            )
        )

        has_dist = self.spec.has_distance()
        has_conf = self.spec.has_conformer()
        has_rmsd = self.spec.has_rmsd()
        # group-centroid angle/dihedral are CG-solved like rmsd (not closed-form), so the
        # solver branch must run when either is present.
        has_group = self.spec.has_group_angle() or self.spec.has_group_dihedral()
        has_custom = self.spec.has_custom()
        prepared = self._prepared

        # boltz / Lightning run prediction under torch.inference_mode, where leaf
        # tensors cannot require grad. Re-enable autograd and copy the active sites
        # into a normal tensor we can mutate / attach a graph to.
        with torch.inference_mode(False), torch.enable_grad():
            active = torch.empty(
                coords[..., self._active_idx, :].shape,
                dtype=work_dtype,
                device=coords.device,
            )
            active.copy_(coords[..., self._active_idx, :])  # casts bf16/fp16 -> fp32

            # 1) Distance restraints: closed-form rigid centroid translation (no solver, no
            #    autograd) -- a centroid-distance restraint is a 1-DOF problem, so iterating
            #    a per-atom CG over it is wasteful. Gated per-restraint inside the shift.
            if has_dist:
                with torch.no_grad():
                    active = apply_distance_shift_torch(
                        active, prepared["distance"], sigma
                    )

            # 2) Conformer (bond/angle/chiral/cistrans/vdw) + RMSD + group-centroid
            #    angle/dihedral restraints: gradient solver on the non-distance energy
            #    (distance already applied above; total_energy(include_distance=False)
            #    covers conformer, RMSD AND the group terms). Skipped for distance-only.
            if has_conf or has_rmsd or has_group or has_custom:
                active = active.detach().clone()
                active.requires_grad_(True)
                prot_pos = None
                if vdw_active:
                    prot_pos = torch.empty(
                        coords[..., self._vdw["prot_global"], :].shape,
                        dtype=work_dtype,
                        device=coords.device,
                    )
                    prot_pos.copy_(coords[..., self._vdw["prot_global"], :])

                def energy_fn():
                    e = torch_energy.total_energy(
                        active, prepared, sigma, include_distance=False
                    )
                    if prot_pos is not None:
                        e = e + self._vdw_energy(active, prot_pos)
                    ce = self._custom_energy(active, sigma)
                    if ce is not None:
                        e = e + ce
                    return e

                # custom restraints are arbitrary closures that the fused gpu_cg energy
                # cannot read, so force the eager early-exit CG (correct on CUDA, unfused)
                # whenever any custom restraint is present.
                if self._is_cg() and active.is_cuda and not has_custom:
                    # GPU: the same early-exit CG as the CPU path, but with an
                    # inductor-fused (NOT CUDA-graph) torch.compile'd energy+grad so the
                    # launch-bound eval stops dominating (the eager _minimize_cg below is
                    # launch/sync-bound on GPU). CPU keeps _minimize_cg (already optimal).
                    from rgi_utils.optim._torch_cg_gpu import gpu_cg

                    # _gated_prepared pre-folds the noise gate into the masks (so the
                    # compiled energy takes sigma=None and carries NO python-float scalar
                    # leaf, which would make dynamo recompile per value) and caches a
                    # stable dict per gate state for compile reuse. dynamic ligand-protein
                    # VdW is folded in via gpu_cg's `vdw` tuple (prot_pos + radii/consts).
                    vdw = None
                    if prot_pos is not None:
                        v = self._vdw
                        vdw = (
                            prot_pos,
                            v["lig_local"],
                            v["lig_r"],
                            v["prot_r"],
                            v["scale"],
                            v["weight"],
                        )
                    prepared_g = self._gated_prepared(sigma)
                    opt = gpu_cg(prepared_g, active.detach(), mi, vdw=vdw)
                    with torch.no_grad():
                        active.copy_(opt)
                elif self._is_cg():
                    self._minimize_cg(active, energy_fn, mi)
                else:
                    opt = torch.optim.LBFGS(
                        [active], max_iter=mi, line_search_fn="strong_wolfe"
                    )

                    def closure():
                        opt.zero_grad()
                        e = energy_fn()
                        e.backward()
                        return e

                    opt.step(closure)
            new_active = active.detach().clone()

        # Robustness (mirror the jax backend's guard in jax_optim.py): a degenerate
        # geometry can make the solver step diverge to non-finite values; keep the
        # input coordinates rather than writing NaN/Inf into the structure.
        if not torch.isfinite(new_active).all():
            # If the INPUT coords were already non-finite, the model (or a broken GPU
            # kernel) diverged upstream — the restraint only inherited the NaN. Reporting
            # input_finite keeps a NaN from being misattributed to the restraint.
            input_finite = bool(torch.isfinite(coords[..., self._active_idx, :]).all())
            logger.warning(
                "restraint step produced non-finite coords; skipping update "
                "(input_finite=%s)",
                input_finite,
            )
            return coords
        # back in the ambient (inference) context: in-place write is allowed
        coords[..., self._active_idx, :] = new_active.to(out_dtype)
        return coords

    def _is_cg(self) -> bool:
        return (self.method or "cg").lower() in (
            "cg",
            "ncg",
            "nonlinear-cg",
            "nonlinearcg",
        )

    def _minimize_cg(
        self,
        active,
        energy_fn,
        max_iter,
        max_ls: int = MAX_LS,
        gtol: float = GTOL,
        ftol: float = FTOL,
    ) -> None:
        """In-place nonlinear conjugate gradient (Polak-Ribiere+, backtracking
        Armijo line search). Matches the jax backend's pure-jax CG (a port of this
        solver), so ``method='CG'`` is the same algorithm on both backends. ``active``
        is a leaf tensor (requires_grad=True); ``energy_fn()`` returns the scalar
        energy with ``active`` in its graph. Stops early on convergence
        (``max|grad| < gtol`` or ``|df| < ftol``, mirroring torch LBFGS's
        tolerance_grad / tolerance_change) or when the line search stalls, so simple
        restraints finish well under ``max_iter``."""

        # GPU note: every float()/.item() on a device scalar is a blocking
        # device->host sync that serialises the GPU; for this tiny CG that dominates
        # the runtime. value_grad returns the energy as a TENSOR and the loop BATCHES
        # the per-iteration scalar reads into one .tolist() each, keeping the maths
        # identical to a plain CG.
        def value_grad():
            if active.grad is not None:
                active.grad = None
            e = energy_fn()
            e.backward()
            return e.detach(), active.grad.detach().clone()

        e_t, g = value_grad()
        f = float(e_t)  # the line search needs the scalar energy
        if float(g.abs().max()) < gtol:
            return
        d = g.neg()
        gg = torch.sum(g * g)
        for _ in range(max_iter):
            # one host read for both top-of-iteration scalars (gg + descent slope)
            dg = torch.sum(d * g)
            gg_v, dg_v = torch.stack((gg, dg)).tolist()
            if not math.isfinite(gg_v) or gg_v <= GG_FLOOR:
                break
            if dg_v >= 0.0:  # not a descent direction -> restart (rare)
                d = g.neg()
                dg_v = float(torch.sum(d * g))
            slope = dg_v
            x0 = active.detach().clone()
            step, accepted = 1.0, False
            for _ in range(max_ls):
                with torch.no_grad():
                    active.copy_(x0 + step * d)
                e_t, g_new = value_grad()
                f_new = float(e_t)  # 1 sync/trial (Armijo needs the value)
                if f_new <= f + ARMIJO_C1 * step * slope:  # Armijo sufficient decrease
                    accepted = True
                    break
                step *= BACKTRACK
            if not accepted:  # line search exhausted -> converged / stuck
                with torch.no_grad():
                    active.copy_(x0)
                break
            # one host read for the post-step scalars (grad-max + PR+ numerator)
            gmax_v, pr_num_v = torch.stack(
                (g_new.abs().max(), torch.sum(g_new * (g_new - g)))
            ).tolist()
            converged = gmax_v < gtol or abs(f_new - f) < ftol * (1.0 + abs(f))
            beta = max(0.0, pr_num_v / (gg_v + EPS))  # gg_v = this iter's gg
            d = g_new.neg() + beta * d  # Polak-Ribiere+ (auto-restart when beta<0)
            f, g, gg = f_new, g_new, torch.sum(g_new * g_new)
            if converged:
                break

    def energy(self, coords) -> float:
        """Current restraint energy (for verbose stats / finalize)."""
        if not self.spec.is_active():
            return 0.0
        self._ensure(coords.device, coords.dtype)
        with torch.no_grad():
            active = coords[..., self._active_idx, :]
            e = torch_energy.total_energy(active, self._prepared)
            if self._vdw is not None:
                prot_pos = coords[..., self._vdw["prot_global"], :]
                e = e + self._vdw_energy(active, prot_pos)
            return float(e)

    def dynamic_vdw_energy(self, coords) -> float:
        """The dynamic ligand-protein VdW term alone (>= 0); for finalize stats.

        Computed directly (not as energy - static_total) so the reported value is
        exact and non-negative, with no float32/float64 cancellation error.
        """
        if not self.spec.is_active():
            return 0.0
        # _ensure builds self._vdw (via _setup_vdw); call it BEFORE the _vdw guard so a
        # fresh optimizer reports the true VdW (matches energy()) instead of 0.0, which
        # would read as a false "VdW satisfied" in the verbose finalize log.
        self._ensure(coords.device, coords.dtype)
        if self._vdw is None:
            return 0.0
        with torch.no_grad():
            active = coords[..., self._active_idx, :]
            prot_pos = coords[..., self._vdw["prot_global"], :]
            return float(self._vdw_energy(active, prot_pos))
