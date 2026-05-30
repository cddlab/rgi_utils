"""GPU restraint optimizer for PyTorch tools (boltz, protenix).

Minimizes the restraint energy on active-site coordinates with
``torch.optim.LBFGS`` (strong-Wolfe line search), using autograd for gradients.
Operates in-place on the coordinate tensor and stays on whatever device the
coordinates live on, so ``gpu: true`` runs entirely on GPU.

The ligand-protein VdW term (``spec.vdw_config``) is handled here rather than in
the static energy layer: the ligand atoms come from the optimised ``active`` set
while the protein atoms are a *fixed background* read from the full coordinate
tensor. The clash penalty is recomputed every closure call, so it tracks the
moving ligand (only the ligand is pushed; the protein is held fixed).
"""

from __future__ import annotations

import logging

import torch

from rgi_utils.energy import torch_energy

logger = logging.getLogger(__name__)

_EPS = 1e-12


class TorchRestraintOptimizer:
    def __init__(self, spec, max_iter: int = 100, method: str = "l-bfgs"):
        self.spec = spec
        self.max_iter = max_iter
        self.method = method
        self._prepared = None
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
            self._setup_vdw(device, dtype)
        self._device = device

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
        """Optimize ``coords`` (..., n_atom, 3) in-place. No-op above start_sigma."""
        if not self.spec.is_active():
            return coords
        if sigma is not None and start_sigma is not None and sigma > start_sigma:
            return coords
        self._ensure(coords.device, coords.dtype)
        mi = max_iter if max_iter is not None else self.max_iter

        # boltz / Lightning run prediction under torch.inference_mode, where leaf
        # tensors cannot require grad. Re-enable autograd and copy the slices into
        # normal leaf tensors so LBFGS can build a graph. Only active sites are
        # optimised; the protein VdW background is a fixed copy.
        with torch.inference_mode(False), torch.enable_grad():
            active = torch.empty_like(coords[..., self._active_idx, :])
            active.copy_(coords[..., self._active_idx, :])
            active.requires_grad_(True)
            prot_pos = None
            if self._vdw is not None:
                prot_pos = torch.empty_like(coords[..., self._vdw["prot_global"], :])
                prot_pos.copy_(coords[..., self._vdw["prot_global"], :])
            opt = torch.optim.LBFGS(
                [active], max_iter=mi, line_search_fn="strong_wolfe"
            )
            prepared = self._prepared

            def closure():
                opt.zero_grad()
                e = torch_energy.total_energy(active, prepared)
                if prot_pos is not None:
                    e = e + self._vdw_energy(active, prot_pos)
                e.backward()
                return e

            opt.step(closure)
            new_active = active.detach().clone()

        # back in the ambient (inference) context: in-place write is allowed
        coords[..., self._active_idx, :] = new_active
        return coords

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
