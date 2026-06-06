"""Sync-free GPU conjugate-gradient for the torch backend.

The eager CPU-style CG (`torch_optim._minimize_cg`) is **sync-bound on GPU**: every
iteration reads device scalars to host (`.item()`/`.tolist()`/`float()`, the line-search
Armijo `if`, the convergence check), and each read drains the GPU queue, killing overlap
— so the tiny restraint kernels never pipeline. The jax backend doesn't suffer because
its whole CG runs inside one `lax.while_loop` (on-device control flow, XLA-fused).

This module is the torch analogue. Two ideas remove the stalls without the redundant work
a naive fixed-unroll port would add:

  * **functional gradient** via ``torch.func.grad_and_value`` (like jax ``value_and_grad``)
    — no autograd-tape rebuild per call, and a clean ``torch.compile`` target later.
  * **vmap'd backtracking line search**: the ``max_ls`` step candidates
    ``[1, BACKTRACK, BACKTRACK^2, ...]`` are evaluated in ONE ``torch.func.vmap(vg)`` call
    and the largest Armijo-satisfying step is selected on-device — so there is no per-trial
    host sync AND no wasted sequential re-evaluation (the GPU runs the candidates in
    parallel). The outer loop is sync-light: it reads a single ``done`` scalar only once
    every ``_CHECK_EVERY`` iterations to allow an early break.

Algorithm (Polak-Ribiere+, restart on non-descent, degenerate-iteration guard, on-device
non-finite guard) is identical to ``jax_optim._cg_minimize`` and ``torch_optim._minimize_cg``
— the shared constants come from ``_cg_config`` so the convergence contract cannot drift.
Used only for CUDA coords (CPU keeps the early-exit ``_minimize_cg``).
"""

from __future__ import annotations

import torch

from rgi_utils.optim._cg_config import (
    ARMIJO_C1,
    BACKTRACK,
    EPS,
    FTOL,
    GG_FLOOR,
    GTOL,
    MAX_LS,
)

# read the on-device convergence flag to host only every this many iterations (so an
# easy problem still breaks early, but most iterations stay sync-free)
_CHECK_EVERY = 8


def _cg_minimize_torch(vg, x0, max_iter, max_ls=MAX_LS, gtol=GTOL, ftol=FTOL):
    """Sync-free nonlinear CG. ``vg(x) -> (grad, value)`` is
    ``torch.func.grad_and_value`` of the restraint energy; ``x0`` the active-site coords
    ``(..., n, 3)``. Returns the optimized coords (keeps ``x0`` if the result is
    non-finite). No per-iteration host reads except one ``done`` check every
    ``_CHECK_EVERY`` iterations."""
    vmap_vg = torch.func.vmap(vg)
    # backtracking step ladder [1, BACKTRACK, BACKTRACK^2, ...], shaped to broadcast
    # against x for the candidate construction (leading vmap dim + x's own dims).
    powers = torch.arange(max_ls, device=x0.device, dtype=x0.dtype)
    steps = BACKTRACK**powers  # (max_ls,)
    steps_b = steps.reshape((max_ls,) + (1,) * x0.dim())  # (max_ls, 1, ..., 1)

    x = x0
    g, f = vg(x)
    gg = (g * g).sum()
    d = -g

    for it in range(max_iter):
        bad = (~torch.isfinite(gg)) | (gg <= GG_FLOOR)
        d = torch.where((d * g).sum() >= 0.0, -g, d)  # restart if not a descent dir
        d = torch.where(bad, torch.zeros_like(d), d)  # degenerate iter -> no move
        slope = (d * g).sum()

        # vmap line search: evaluate all step candidates at once, pick the first
        # (largest) one satisfying Armijo -- equivalent to sequential backtracking but
        # with a single batched grad eval and zero host syncs.
        cand = x.unsqueeze(0) + steps_b * d.unsqueeze(0)  # (max_ls, ..., n, 3)
        gts, fts = vmap_vg(cand)  # (max_ls, ..., n, 3), (max_ls,)
        ok = fts <= f + ARMIJO_C1 * steps * slope  # (max_ls,) Armijo per candidate
        accepted = ok.any()
        first = torch.argmax(ok.to(torch.int8))  # index of first True (0 if none)
        xt = torch.where(accepted, cand[first], x)
        ft = torch.where(accepted, fts[first], f)
        gt = torch.where(accepted, gts[first], g)

        conv = (gt.abs().max() < gtol) | ((ft - f).abs() < ftol * (1.0 + f.abs()))
        beta = torch.clamp((gt * (gt - g)).sum() / (gg + EPS), min=0.0)  # PR+
        use = accepted & (~bad)
        x = torch.where(use, xt, x)
        f = torch.where(use, ft, f)
        ng = torch.where(use, gt, g)
        d = torch.where(use, -gt + beta * d, d)
        gg = torch.where(use, (gt * gt).sum(), gg)
        g = ng
        done = bad | (~accepted) | conv  # 0-dim bool; read to host only periodically
        if (it + 1) % _CHECK_EVERY == 0 and bool(done):
            break

    # keep the input coords if the solver diverged to non-finite (on-device, like jax)
    return torch.where(torch.isfinite(x).all(), x, x0)


def gpu_cg(energy_of, x0, max_iter):
    """Run the sync-free CG on CUDA coords. ``energy_of(a) -> scalar`` is the pure
    restraint energy (conformer + RMSD + optional ligand-protein VdW); ``x0`` the active
    coords. Returns the optimized coords tensor."""
    vg = torch.func.grad_and_value(energy_of)
    return _cg_minimize_torch(vg, x0, max_iter)
