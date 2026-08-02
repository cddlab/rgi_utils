"""GPU restraint optimizer for PyTorch tools (boltz, protenix).

Minimizes the restraint energy on active-site coordinates using autograd for
gradients. ``method`` selects the solver: ``"CG"`` (default) -> a nonlinear
conjugate-gradient solver (Polak-Ribiere+ with a backtracking Armijo line search),
matching the jax backend's pure-jax CG (a port of this solver); ``"l-bfgs"`` ->
``torch.optim.LBFGS`` (strong-Wolfe). Operates in-place on the coordinate tensor
and stays on whatever device the coordinates live on, so ``gpu: true`` runs
entirely on GPU.

The fixed-background VdW term (``spec.vdw_config``) is handled here rather than in
the static energy layer: the ligand atoms come from the optimised ``active`` set
while the background atoms (protein / DNA/RNA / non-restrained ligand) are a *fixed
background* read from the full coordinate tensor. The clash penalty is recomputed
every closure call, so it tracks the moving ligand (only the ligand is pushed; the
background is held fixed).
"""

from __future__ import annotations

import logging
import math
import os

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
        self._vdw = None  # dict of device tensors for the fixed-background VdW term
        self._active_vdw = None  # dynamic active-active polymer neighbour metadata
        # custom restraints -> torch closures, built lazily in _ensure UNDER
        # inference_mode(False) and on the coords' device, so the baked selection-index
        # tensors are normal + on-device (an inference tensor used in the autograd gather
        # can't be saved for backward -- boltz/Lightning run under inference_mode).
        self._custom_terms = None
        # per-optimizer torch.compile'd energy+grad INCLUDING the custom closures (the
        # module-global gpu_cg energy can't see them). None=unbuilt, False=disabled/failed.
        self._custom_cvg = None

    def _custom_energy(self, active, sigma, step):
        """Per-entry gated sum of the custom-restraint closure energies at ``active``
        (local coords). Returns ``None`` when no custom term is active so the caller can
        skip adding it. Gate: the active sigma window (``stop_sigma <= sigma <=
        start_sigma``) AND the active step window (``start_step <= step <= stop_step``);
        a restraint uses one or the other (mutually exclusive). Python-float compares,
        since the eager CG has python-float sigma/step."""
        if not self._custom_terms:
            return None
        total = None
        for _name, start, stop, start_step, stop_step, closure in self._custom_terms:
            if sigma is not None and not (sigma <= start and sigma >= stop):
                continue
            if step is not None and not (start_step <= step <= stop_step):
                continue
            e = closure(active)
            total = e if total is None else total + e
        return total

    def _get_custom_cvg(self):
        """The compiled ``grad_and_value`` of ``total_energy + sum(gate_i * closure_i)``,
        built ONCE per optimizer (the custom closures are spec-specific, so this can't
        reuse gpu_cg's module-global artifact). The per-entry sigma gates are passed as a
        tensor argument (``gates``) — NOT a python-float — so one artifact serves every
        noise level without a dynamo recompile. ``None`` -> compile disabled / failed
        (caller runs eager). The AST/closures are static, so dynamo traces them to a fixed
        graph; ``fullgraph=False`` tolerates any residual break."""
        if self._custom_cvg is False:
            return None
        if self._custom_cvg is not None:
            return self._custom_cvg
        if os.environ.get("RGI_DISABLE_COMPILE", "") not in ("", "0", "false"):
            self._custom_cvg = False
            return None
        closures = [c for *_meta, c in self._custom_terms]

        def energy(a, prepared, gates):
            e = torch_energy.total_energy(a, prepared, sigma=None)
            for i in range(len(closures)):
                e = e + gates[i] * closures[i](a)
            return e

        try:
            self._custom_cvg = torch.compile(
                torch.func.grad_and_value(energy, argnums=0), fullgraph=False
            )
        except Exception as exc:
            logger.warning(
                "torch.compile of the custom GPU energy failed (%s); eager", exc
            )
            self._custom_cvg = False
            return None
        return self._custom_cvg

    def _minimize_custom_gpu(self, active, sigma, step, mi) -> bool:
        """GPU CG with the torch.compile'd custom-inclusive energy. Returns False (caller
        falls back to the eager CG) when compile is unavailable or the artifact fails."""
        cvg = self._get_custom_cvg()
        if cvg is None:
            return False
        from rgi_utils.optim._torch_cg_gpu import _cg_minimize_torch

        prepared_g = self._gated_prepared(sigma, step)
        # per-custom gate: active sigma window AND active step window (one or the other).
        gates = torch.tensor(
            [
                1.0
                if (sigma is None or (sigma <= s and sigma >= st))
                and (step is None or (sstep <= step <= estep))
                else 0.0
                for _n, s, st, sstep, estep, _c in self._custom_terms
            ],
            dtype=active.dtype,
            device=active.device,
        )
        try:
            opt = _cg_minimize_torch(
                lambda x: cvg(x, prepared_g, gates), active.detach(), mi
            )
            with torch.no_grad():
                active.copy_(opt)
            return True
        except (
            Exception
        ) as exc:  # this artifact's runtime failure -> eager, permanently
            logger.warning(
                "custom GPU CG (compiled) failed at runtime (%s); eager", exc
            )
            self._custom_cvg = False
            return False

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
                "%d custom restraint(s): on CUDA they run inside a per-optimizer "
                "torch.compile'd energy+grad (eager on CPU / on a compile fallback)",
                len(self._custom_terms),
            )

    def _gated_prepared(self, sigma, step=None):
        """Stable pre-gated ``prepared`` for the compiled GPU CG, cached by the discrete
        GATE STATE — the conformer gate plus the per-restraint gate of every per-entry
        term (distance, rmsd, group_angle, group_dihedral — ``PER_ENTRY_KEYS``). Each gate is the
        active sigma window (``stop_sigma <= sigma <= start_sigma``) AND the active step
        window (``start_step <= step <= stop_step``) — a restraint uses one or the other
        (mutually exclusive at config), the unused axis always-on so the AND is correct.
        The gate is folded into the masks here so the compiled energy is called with
        ``sigma=None``; scalar leaves (e.g. ``conf_start_sigma``) are DROPPED because a
        python-float in the compiled energy's pytree makes dynamo guard on its value and
        recompile per distinct value. Stable object identity per gate state lets
        ``torch.compile`` reuse its artifact; sigma decreases / step increases
        monotonically so each gate flips at most once -> a few states -> a few compiles,
        then reuse. distance is now a per-entry term (in ``PER_ENTRY_KEYS``), so it is gated
        and folded here like rmsd/group. The conformer-gated key set (``CONF_KEYS``) and the
        per-entry-gated set (``PER_ENTRY_KEYS``) both come from ``TERM_DEFS``, so adding a term
        can't silently leave it ungated on the compiled path."""
        p = self._prepared
        # conformer gate: active sigma window (conf_stop <= sigma <= conf_start) AND
        # active step window (conf_start_step <= step <= conf_stop_step).
        cg_on = sigma is None or (
            (sigma <= float(self.spec.conf_start_sigma))
            and (sigma >= float(self.spec.conf_stop_sigma))
        )
        if step is not None:
            cg_on = cg_on and (
                step >= float(self.spec.conf_start_step)
                and step <= float(self.spec.conf_stop_step)
            )
        cg_key = 1.0 if (sigma is None and step is None) else float(cg_on)
        # per-restraint on/off for every per-entry term present (one tiny sync each),
        # over the active sigma window AND step window.
        gates: dict[str, tuple] = {}
        if sigma is not None or step is not None:
            for gk in PER_ENTRY_KEYS:
                if gk in p:
                    pe = p[gk]
                    on = None
                    if sigma is not None:
                        on = (sigma <= pe["start_sigma"]) & (sigma >= pe["stop_sigma"])
                    if step is not None:
                        step_on = (step >= pe["start_step"]) & (step <= pe["stop_step"])
                        on = step_on if on is None else (on & step_on)
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
        else:
            self._vdw = {
                "lig_local": torch.as_tensor(
                    vc.ligand_local, dtype=torch.long, device=device
                ),
                "lig_r": torch.as_tensor(vc.ligand_radii, dtype=dtype, device=device),
                "bg_global": torch.as_tensor(
                    vc.background_global, dtype=torch.long, device=device
                ),
                "bg_r": torch.as_tensor(
                    vc.background_radii, dtype=dtype, device=device
                ),
                "weight": torch.as_tensor(float(vc.weight), dtype=dtype, device=device),
                "scale": torch.as_tensor(float(vc.scale), dtype=dtype, device=device),
            }

        ac = getattr(self.spec, "active_vdw_config", None)
        if ac is None or ac.weight <= 0:
            self._active_vdw = None
        else:
            self._active_vdw = {
                "radii": torch.as_tensor(ac.radii, dtype=dtype, device=device),
                "polymer_mask": torch.as_tensor(
                    ac.polymer_mask, dtype=torch.bool, device=device
                ),
                "excluded_codes": torch.as_tensor(
                    ac.excluded_codes, dtype=torch.long, device=device
                ),
                "weight": torch.as_tensor(float(ac.weight), dtype=dtype, device=device),
                "scale": torch.as_tensor(float(ac.scale), dtype=dtype, device=device),
                "dmax": torch.as_tensor(float(ac.dmax), dtype=dtype, device=device),
                "max_neighbors": int(ac.max_neighbors),
            }

    def _vdw_energy(self, active, bg_pos):
        """Fixed-background VdW repulsion (delegates to the pure ``_vdw_pair_energy`` so the
        all-pairs ``weight*sum(clamp(d - scale*(r_i+r_j), max=0)**2)`` maths lives once —
        the compiled GPU energy folds in the same function). ``active`` (..., n_active, 3)
        is the optimised tensor; ``bg_pos`` (..., n_bg, 3) the fixed background."""
        from rgi_utils.optim._torch_cg_gpu import _vdw_pair_energy

        v = self._vdw
        return _vdw_pair_energy(
            active,
            bg_pos,
            v["lig_local"],
            v["lig_r"],
            v["bg_r"],
            v["scale"],
            v["weight"],
        )

    def minimize(self, coords, sigma=None, step=None, start_sigma=None, max_iter=None):
        """Optimize ``coords`` (..., n_atom, 3) in-place. Each restraint is gated on its
        active sigma window AND its active step window (``step`` = diffusion step index)
        inside the energy; the whole step is skipped only when ``sigma`` exceeds every
        restraint's start_sigma (a step-windowed restraint keeps start_sigma=+inf, so it
        is never whole-step-skipped — its step gate handles activation)."""
        if not self.spec.is_active():
            return coords
        # sigma is the per-step scalar noise level. Coerce to a python float: the skip
        # test below and the GPU pre-gate (which builds the rmsd-gate tuple from it) both
        # assume a scalar; a stray multi-element tensor would otherwise fail GPU-only.
        if sigma is not None:
            sigma = float(sigma)
        # step is the diffusion step index (int); coerce so the python-float gate compares
        # in the eager CG / GPU pre-gate behave (a tensor step would break the GPU path).
        if step is not None:
            step = int(step)
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
        # VdW is a conformer restraint -> gated by the conformer window: active sigma
        # window (conf_stop <= sigma <= conf_start) AND active step window
        # (conf_start_step <= step <= conf_stop_step). NOT a TERM_DEFS entry, so this gate is
        # maintained by hand (the PER_ENTRY/CONF_KEYS safety net does not cover it).
        vdw_active = (
            self._vdw is not None
            and (
                sigma is None
                or (
                    sigma <= float(self.spec.conf_start_sigma)
                    and sigma >= float(self.spec.conf_stop_sigma)
                )
            )
            and (
                step is None
                or (
                    step >= float(self.spec.conf_start_step)
                    and step <= float(self.spec.conf_stop_step)
                )
            )
        )
        active_vdw_active = (
            self._active_vdw is not None
            and (
                sigma is None
                or (
                    sigma <= float(self.spec.conf_start_sigma)
                    and sigma >= float(self.spec.conf_stop_sigma)
                )
            )
            and (
                step is None
                or (
                    step >= float(self.spec.conf_start_step)
                    and step <= float(self.spec.conf_stop_step)
                )
            )
        )

        has_builtin = self.spec.has_conformer() or self.spec.has_per_entry()
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

            # Distance + conformer (bond/angle/chiral/cistrans/vdw) + RMSD + group-centroid
            # angle/dihedral restraints all minimise ONE objective in the gradient solver
            # (total_energy sums every active term). Distance is an autodiff CG term too: its
            # energy rescales the centroid gradient (reduced-mass scale) so each group
            # translates rigidly with the minimal-displacement split — no closed-form shift.
            if has_builtin or has_custom:
                active = active.detach().clone()
                active.requires_grad_(True)
                bg_pos = None
                if vdw_active:
                    bg_pos = torch.empty(
                        coords[..., self._vdw["bg_global"], :].shape,
                        dtype=work_dtype,
                        device=coords.device,
                    )
                    bg_pos.copy_(coords[..., self._vdw["bg_global"], :])
                active_vdw = None
                if active_vdw_active:
                    from rgi_utils.optim._torch_cg_gpu import build_active_vdw_pairs

                    av = self._active_vdw
                    neighbours, pair_factor = build_active_vdw_pairs(
                        active,
                        av["radii"],
                        av["polymer_mask"],
                        av["excluded_codes"],
                        av["dmax"],
                        av["max_neighbors"],
                    )
                    active_vdw = (neighbours, pair_factor)

                def energy_fn():
                    e = torch_energy.total_energy(active, prepared, sigma, step)
                    if bg_pos is not None:
                        e = e + self._vdw_energy(active, bg_pos)
                    if active_vdw is not None:
                        from rgi_utils.optim._torch_cg_gpu import active_vdw_pair_energy

                        av = self._active_vdw
                        e = e + active_vdw_pair_energy(
                            active,
                            active_vdw[0],
                            active_vdw[1],
                            av["radii"],
                            av["scale"],
                            av["weight"],
                        )
                    ce = self._custom_energy(active, sigma, step)
                    if ce is not None:
                        e = e + ce
                    return e

                if (
                    self._is_cg()
                    and active.is_cuda
                    and has_custom
                    and bg_pos is None
                    and active_vdw is None
                ):
                    # GPU + custom (no dynamic fixed-background VdW): a per-optimizer
                    # torch.compile'd energy that INCLUDES the custom closures (gpu_cg's
                    # module-global energy can't see them). Falls back to the eager CG if
                    # compile is disabled / fails. (custom + dynamic VdW stays eager below.)
                    if not self._minimize_custom_gpu(active, sigma, step, mi):
                        self._minimize_cg(active, energy_fn, mi)
                elif self._is_cg() and active.is_cuda and not has_custom:
                    # GPU: the same early-exit CG as the CPU path, but with an
                    # inductor-fused (NOT CUDA-graph) torch.compile'd energy+grad so the
                    # launch-bound eval stops dominating (the eager _minimize_cg below is
                    # launch/sync-bound on GPU). CPU keeps _minimize_cg (already optimal).
                    from rgi_utils.optim._torch_cg_gpu import gpu_cg

                    # _gated_prepared pre-folds the noise gate into the masks (so the
                    # compiled energy takes sigma=None and carries NO python-float scalar
                    # leaf, which would make dynamo recompile per value) and caches a
                    # stable dict per gate state for compile reuse. dynamic fixed-background
                    # VdW is folded in via gpu_cg's `vdw` tuple (bg_pos + radii/consts).
                    vdw = None
                    if bg_pos is not None:
                        v = self._vdw
                        vdw = (
                            bg_pos,
                            v["lig_local"],
                            v["lig_r"],
                            v["bg_r"],
                            v["scale"],
                            v["weight"],
                        )
                    active_vdw_args = None
                    if active_vdw is not None:
                        av = self._active_vdw
                        active_vdw_args = (
                            active_vdw[0],
                            active_vdw[1],
                            av["radii"],
                            av["scale"],
                            av["weight"],
                        )
                    prepared_g = self._gated_prepared(sigma, step)
                    opt = gpu_cg(
                        prepared_g,
                        active.detach(),
                        mi,
                        vdw=vdw,
                        active_vdw=active_vdw_args,
                    )
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
                        # all terms gated off -> no-op (see _minimize_cg's value_grad)
                        if e.requires_grad:
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
            if not e.requires_grad:
                # The objective is a constant w.r.t. `active` (nothing to optimise): return
                # a zero gradient so the CG converges on the first iteration, leaving the
                # coords untouched. Mirrors the jax backend, whose `jnp.where` gate yields a
                # 0 gradient rather than crashing `backward()` with "does not require grad".
                # This arises only via the CUSTOM closure path: a gated-off custom term is
                # DROPPED (`_custom_energy` returns None), unlike the array terms whose gate
                # is a multiplicative 0 inside `total_energy` (graph stays connected, grad
                # 0). So it needs a custom-only spec (or custom + closed-form distance, which
                # never feeds the CG) whose step window is currently closed — the sigma
                # whole-step skip does not fire there (a step-windowed term keeps
                # start_sigma=+inf). Regression: tests/test_custom.py::test_custom_gate_*.
                return e.detach(), torch.zeros_like(active)
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
                bg_pos = coords[..., self._vdw["bg_global"], :]
                e = e + self._vdw_energy(active, bg_pos)
            return float(e)

    def dynamic_vdw_energy(self, coords) -> float:
        """The dynamic fixed-background VdW term alone (>= 0); for finalize stats.

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
            bg_pos = coords[..., self._vdw["bg_global"], :]
            return float(self._vdw_energy(active, bg_pos))
