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

_EPS = 1e-12  # vdw distance-floor guard (the CG solver uses the shared EPS)


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
        self._device = device

    def _gated_prepared(self, cg, sigma):
        """Stable pre-gated ``prepared`` for the compiled GPU CG, cached by the discrete
        GATE STATE — the conformer gate ``cg`` (0/1) plus the per-restraint rmsd gate
        (``sigma <= start_sigma``). Stable object identity per state lets ``torch.compile``
        reuse its artifact (a fresh dict of fresh tensors each step would re-trace). sigma
        decreases monotonically, so each restraint's gate flips at most once -> only a
        few states -> a few compiles, then reuse. distance is excluded (closed-form)."""
        p = self._prepared
        cg_key = 1.0 if cg >= 0.5 else 0.0
        if "rmsd" in p and sigma is not None:
            # one tiny sync per step; tuple of per-restraint on/off (sigma<=start_sigma)
            rgate = tuple(bool(b) for b in (sigma <= p["rmsd"]["start_sigma"]).tolist())
        else:
            rgate = None
        key = (cg_key, rgate)
        cache = self._prepared_g
        if key not in cache:
            pg = {}
            for k, v in p.items():
                if not isinstance(v, dict):
                    pg[k] = v
                elif k in ("bond", "angle", "chiral", "dihedral", "vdw"):
                    pg[k] = {**v, "mask": v["mask"] * cg_key}
                elif k == "rmsd" and rgate is not None:
                    rg = torch.tensor(
                        rgate, dtype=v["mask"].dtype, device=v["mask"].device
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
            "weight": float(vc.weight),
            "scale": float(vc.scale),
        }

    def _vdw_energy(self, active, prot_pos):
        """Ligand-protein VdW repulsion. ``active`` (..., n_active, 3) is the
        optimised tensor; ``prot_pos`` (..., n_prot, 3) is the fixed background.
        All-pairs penalty ``weight * sum(clamp(d - scale*(r_i+r_j), max=0)**2)``;
        non-clashing pairs contribute zero gradient, so this equals a radius-
        limited contact sum without needing a neighbour search."""
        v = self._vdw
        lig = active[..., v["lig_local"], :]  # (..., n_lig, 3)
        diff = (
            lig[..., :, None, :] - prot_pos[..., None, :, :]
        )  # (..., n_lig, n_prot, 3)
        dist = torch.sqrt(torch.sum(diff**2, dim=-1) + _EPS)  # (..., n_lig, n_prot)
        r_min = v["scale"] * (
            v["lig_r"][:, None] + v["prot_r"][None, :]
        )  # (n_lig, n_prot)
        delta = torch.clamp(dist - r_min, max=0.0)
        return v["weight"] * torch.sum(delta**2)

    def minimize(self, coords, sigma=None, start_sigma=None, max_iter=None):
        """Optimize ``coords`` (..., n_atom, 3) in-place. Each restraint is gated
        on ``sigma <= its start_sigma`` inside the energy; the whole step is
        skipped only when ``sigma`` exceeds every restraint's start_sigma."""
        if not self.spec.is_active():
            return coords
        if sigma is not None and sigma > self.spec.max_start_sigma():
            return coords
        self._ensure(coords.device, coords.dtype)
        mi = max_iter if max_iter is not None else self.max_iter
        # VdW is a conformer restraint -> gated by conf_start_sigma
        vdw_active = self._vdw is not None and (
            sigma is None or sigma <= float(self.spec.conf_start_sigma)
        )

        has_dist = self.spec.has_distance()
        has_conf = self.spec.has_conformer()
        has_rmsd = self.spec.has_rmsd()
        prepared = self._prepared

        # boltz / Lightning run prediction under torch.inference_mode, where leaf
        # tensors cannot require grad. Re-enable autograd and copy the active sites
        # into a normal tensor we can mutate / attach a graph to.
        with torch.inference_mode(False), torch.enable_grad():
            active = torch.empty_like(coords[..., self._active_idx, :])
            active.copy_(coords[..., self._active_idx, :])

            # 1) Distance restraints: closed-form rigid COM translation (no solver, no
            #    autograd) -- a COM-distance restraint is a 1-DOF problem, so iterating
            #    a per-atom CG over it is wasteful. Gated per-restraint inside the shift.
            if has_dist:
                with torch.no_grad():
                    active = apply_distance_shift_torch(
                        active, prepared["distance"], sigma
                    )

            # 2) Conformer (bond/angle/chiral/dihedral/vdw) + RMSD restraints: gradient
            #    solver on the non-distance energy (distance is already applied above;
            #    total_energy(include_distance=False) covers conformer AND RMSD). Skipped
            #    entirely for a distance-only run.
            if has_conf or has_rmsd:
                active = active.detach().clone()
                active.requires_grad_(True)
                prot_pos = None
                if vdw_active:
                    prot_pos = torch.empty_like(coords[..., self._vdw["prot_global"], :])
                    prot_pos.copy_(coords[..., self._vdw["prot_global"], :])

                def energy_fn():
                    e = torch_energy.total_energy(
                        active, prepared, sigma, include_distance=False
                    )
                    if prot_pos is not None:
                        e = e + self._vdw_energy(active, prot_pos)
                    return e

                if self._is_cg() and active.is_cuda:
                    # GPU: the same early-exit CG as the CPU path, but with a
                    # torch.compile'd (CUDA-graph) energy+grad so the launch-bound eval
                    # stops dominating (the eager _minimize_cg below is launch/sync-bound
                    # on GPU). CPU keeps the early-exit _minimize_cg (already optimal).
                    from rgi_utils.optim._torch_cg_gpu import gpu_cg

                    # Pre-gate the masks so the compiled energy takes NO python-float
                    # sigma (which would force a dynamo recompile every diffusion step):
                    # fold the noise gate into the masks, then call total_energy with
                    # sigma=None. Conformer terms share conf_start_sigma; rmsd has its own
                    # per-restraint start_sigma; distance is excluded (closed-form above).
                    cg = (
                        1.0
                        if sigma is None
                        else float(sigma <= float(self.spec.conf_start_sigma))
                    )
                    # dynamic ligand-protein VdW (if any) is not part of the compiled
                    # energy -> its presence routes gpu_cg to the eager functional CG.
                    vdw_fn = (
                        (lambda a: self._vdw_energy(a, prot_pos))
                        if prot_pos is not None
                        else None
                    )
                    prepared_g = self._gated_prepared(cg, sigma)
                    opt = gpu_cg(prepared_g, active.detach(), mi, vdw_fn=vdw_fn)
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
            logger.warning("restraint step produced non-finite coords; skipping update")
            return coords
        # back in the ambient (inference) context: in-place write is allowed
        coords[..., self._active_idx, :] = new_active
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
