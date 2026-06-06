"""GPU conjugate-gradient for the torch backend: same correct early-exit CG as the CPU
path, but with a ``torch.compile``-d (inductor-fused) energy+grad so it stops losing to CPU.

Why this shape. A single restraint energy+grad eval is **launch-bound** on GPU: it is
dozens of tiny kernels whose launch latency dwarfs their compute (a 690-atom RMSD Kabsch,
or ~200 conformer terms), so the eager CG (`torch_optim._minimize_cg`) ran ~1.6-2.8x
slower than torch-on-CPU. A naive "sync-free" eager rewrite does NOT help: a vmap'd
fixed-width line search either over-computes (large width -> compute-bound RMSD loses) or
cannot reach the fine backtracking steps a stiff term needs (small width -> the chiral
term silently fails to converge).

The torch analogue that is BOTH fast AND correct:
  * keep the proven SEQUENTIAL early-exit line search (reaches arbitrarily fine steps, so
    stiff chiral converges exactly like the CPU/jax path) -- it has host syncs, but they
    are cheap once the eval itself is cheap;
  * make the eval cheap: ``torch.func.grad_and_value(_energy)`` wrapped ONCE in
    ``torch.compile`` so inductor FUSES the dozens of tiny fwd+bwd kernels into far fewer
    launches -- which is what kills the launch-bound cost (measured: conformer x0.32 /
    combined x0.49 vs torch-CPU; rmsd ~parity). It is compiled a single time at module
    scope and reused: the changing data (``prepared`` masks, with the noise gate
    pre-folded in by the caller so there is no python-float ``sigma``) is passed as an
    ARGUMENT, not a closure, so dynamo guards on shapes only and the artifact is reused
    across diffusion steps and structures of the same shape.

NB: the default (inductor) compile mode is deliberate -- ``mode="reduce-overhead"``
(CUDA graphs) is ~5x SLOWER here because the CG feeds a freshly-allocated trial tensor
every line-search step, which makes the CUDA-graph tree re-record each call; plain
inductor fusion has no such static-input requirement.

Used only for CUDA coords; CPU keeps ``_minimize_cg``. Shared ``_cg_config`` constants
keep the convergence contract identical to CPU/jax. Any compile failure, or a dynamic
ligand-protein VdW term (which is not part of the compiled energy), degrades to the eager
functional CG -- still the correct early-exit algorithm.
"""

from __future__ import annotations

import logging
import math
import os

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

logger = logging.getLogger(__name__)

# set RGI_DISABLE_COMPILE=1 to force the eager functional CG (debugging / unsupported env)
_COMPILE_DISABLED = os.environ.get("RGI_DISABLE_COMPILE", "") not in ("", "0", "false")
_compile_failed = False  # flips True permanently on any compile/runtime failure
_CVG = None  # the single compiled grad_and_value(_energy), built lazily


def _energy(a, prepared):
    """Pure pre-gated restraint energy (conformer + RMSD; distance is closed-form and
    excluded). ``prepared`` carries the noise gate folded into its masks, so there is no
    ``sigma`` argument -- which is what lets the compiled graph be reused across steps."""
    return torch_energy.total_energy(a, prepared, sigma=None, include_distance=False)


def _get_cvg():
    """The compiled ``grad_and_value(_energy)`` (built once, reused). ``None`` if compile
    is disabled or has failed -> caller uses the eager functional grad."""
    global _CVG, _compile_failed
    if _COMPILE_DISABLED or _compile_failed:
        return None
    if _CVG is None:
        try:
            _CVG = torch.compile(
                torch.func.grad_and_value(_energy, argnums=0),
                fullgraph=False,
            )
        except Exception as exc:
            logger.warning("torch.compile of the GPU CG energy failed (%s); eager", exc)
            _compile_failed = True
            return None
    return _CVG


def _cg_minimize_torch(vg, x0, max_iter, max_ls=MAX_LS, gtol=GTOL, ftol=FTOL):
    """Sequential nonlinear CG (Polak-Ribiere+, backtracking Armijo line search, restart
    on non-descent) — the exact algorithm of ``torch_optim._minimize_cg`` but functional:
    ``vg(x) -> (grad, value)``. ``x0`` is the active-site coords. Returns the optimized
    coords (keeps ``x0`` if non-finite). The early-exit line search reaches arbitrarily
    fine steps, so stiff terms (chiral) converge as on CPU; the host scalar reads are
    cheap because each ``vg`` is a fused compiled call."""
    x = x0
    g, e = vg(x)
    f = float(e)
    if float(g.abs().max()) < gtol:
        return x
    d = -g
    gg = torch.sum(g * g)
    for _ in range(max_iter):
        gg_v, dg_v = torch.stack((gg, torch.sum(d * g))).tolist()
        if not math.isfinite(gg_v) or gg_v <= GG_FLOOR:
            break
        if dg_v >= 0.0:  # not a descent direction -> restart (rare)
            d = g.neg()
            dg_v = float(torch.sum(d * g))
        slope = dg_v
        xbase = x
        step, accepted = 1.0, False
        gt = g
        for _ in range(max_ls):
            xt = xbase + step * d
            gt, e = vg(xt)
            f_new = float(e)  # Armijo needs the value
            if f_new <= f + ARMIJO_C1 * step * slope:
                accepted = True
                break
            step *= BACKTRACK
        if not accepted:  # line search exhausted -> converged / stuck
            x = xbase
            break
        x = xt
        gmax_v, pr_num_v = torch.stack(
            (gt.abs().max(), torch.sum(gt * (gt - g)))
        ).tolist()
        converged = gmax_v < gtol or abs(f_new - f) < ftol * (1.0 + abs(f))
        beta = max(0.0, pr_num_v / (gg_v + EPS))  # PR+ (gg_v = this iter's gg)
        d = gt.neg() + beta * d
        f, g, gg = f_new, gt, torch.sum(gt * gt)
        if converged:
            break
    if not torch.isfinite(x).all():
        return x0
    return x


def gpu_cg(prepared, x0, max_iter, vdw_fn=None, force_eager=False):
    """Run the CG on CUDA coords. ``prepared`` is the (stable, pre-gated) backend energy
    dict; ``x0`` the active coords; ``vdw_fn`` the optional dynamic ligand-protein VdW
    term (a pure fn of ``a``). Uses the single compiled grad_and_value when there is no
    dynamic VdW and ``force_eager`` is False (set it when ``prepared`` is not a stable
    cached object, e.g. per-step rmsd gating, to avoid a recompile every call); otherwise
    (or on any compile/runtime failure) the eager functional CG. Returns optimized coords."""
    global _compile_failed
    # compile only for CUDA coords (the non-GPU tests exercise the eager functional CG,
    # which is the same correct algorithm)
    cvg = _get_cvg() if (vdw_fn is None and x0.is_cuda and not force_eager) else None
    if cvg is not None:
        try:
            return _cg_minimize_torch(lambda x: cvg(x, prepared), x0, max_iter)
        except Exception as exc:  # compiled/runtime failure -> eager, permanently
            logger.warning("GPU CG (compiled) failed at runtime (%s); eager", exc)
            _compile_failed = True

    def e_of(a):
        e = _energy(a, prepared)
        if vdw_fn is not None:
            e = e + vdw_fn(a)
        return e

    return _cg_minimize_torch(torch.func.grad_and_value(e_of), x0, max_iter)
