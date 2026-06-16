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

The dynamic ligand-protein VdW term (default boltz/protenix conformer) is folded into a
second compiled energy (``_energy_vdw``) so that path is JIT-compiled too. Used only for
CUDA coords; CPU keeps ``_minimize_cg``. Shared ``_cg_config`` constants keep the
convergence contract identical to CPU/jax. Any compile failure degrades to the eager
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
# per-artifact (keyed by with_vdw): one compiled energy failing must NOT disable the other
# independent one for the rest of the batch.
_compile_failed = {False: False, True: False}
_CVG = None  # compiled grad_and_value(_energy), built lazily
_CVG_VDW = None  # compiled grad_and_value(_energy_vdw) (dynamic ligand-protein VdW)

# Defense-in-depth: the compiled energy is module-global and reused across all structures
# in a process, so an unforeseen value-specialized leaf must not silently trip dynamo's
# recompile cap and drop to PERMANENT eager. Raise the limits once (best-effort across
# torch versions). The real fix is keeping python-float scalars OUT of the compiled pytree.
try:
    import torch._dynamo as _dynamo

    for _attr in (
        "recompile_limit",
        "accumulated_recompile_limit",
        "cache_size_limit",
        "accumulated_cache_size_limit",
    ):
        if hasattr(_dynamo.config, _attr):
            setattr(_dynamo.config, _attr, max(int(getattr(_dynamo.config, _attr)), 64))
except Exception:
    pass


def _energy(a, prepared):
    """Pure pre-gated restraint energy (conformer + RMSD; distance is closed-form and
    excluded). ``prepared`` carries the noise gate folded into its masks, so there is no
    ``sigma`` argument -- which is what lets the compiled graph be reused across steps."""
    return torch_energy.total_energy(a, prepared, sigma=None, include_distance=False)


def _vdw_pair_energy(active, prot_pos, lig_local, lig_r, prot_r, scale, weight):
    """Dynamic ligand-protein VdW repulsion — the canonical impl (the optimizer's
    ``_vdw_energy`` method delegates here), a pure fn so it can also live inside the
    compiled energy: the moving ligand atoms (``active[lig_local]``) vs the FIXED protein
    background ``prot_pos``. All-pairs ``weight * sum(clamp(d - scale*(r_i+r_j), max=0)^2)``
    (zero gradient beyond contact, so it equals a radius-limited contact sum)."""
    lig = active[..., lig_local, :]
    diff = lig[..., :, None, :] - prot_pos[..., None, :, :]
    dist = torch.sqrt(torch.sum(diff**2, dim=-1) + EPS)
    r_min = scale * (lig_r[:, None] + prot_r[None, :])
    delta = torch.clamp(dist - r_min, max=0.0)
    return weight * torch.sum(delta**2)


def _energy_vdw(a, prepared, prot_pos, lig_local, lig_r, prot_r, scale, weight):
    """``_energy`` + the dynamic ligand-protein VdW term, as one compiled energy so the
    default boltz/protenix conformer (which uses the dynamic VdW) is JIT-compiled too."""
    return _energy(a, prepared) + _vdw_pair_energy(
        a, prot_pos, lig_local, lig_r, prot_r, scale, weight
    )


def _get_cvg(with_vdw=False):
    """The compiled ``grad_and_value`` of the energy (vdw-augmented when ``with_vdw``),
    built once and reused. ``None`` if compile is disabled or has failed -> eager."""
    global _CVG, _CVG_VDW
    if _COMPILE_DISABLED or _compile_failed[with_vdw]:
        return None
    try:
        if with_vdw:
            if _CVG_VDW is None:
                _CVG_VDW = torch.compile(
                    torch.func.grad_and_value(_energy_vdw, argnums=0), fullgraph=False
                )
            return _CVG_VDW
        if _CVG is None:
            _CVG = torch.compile(
                torch.func.grad_and_value(_energy, argnums=0), fullgraph=False
            )
        return _CVG
    except Exception as exc:
        logger.warning("torch.compile of the GPU CG energy failed (%s); eager", exc)
        _compile_failed[with_vdw] = True
        return None


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


def gpu_cg(prepared, x0, max_iter, vdw=None):
    """Run the CG on CUDA coords. ``prepared`` is the (stable, pre-gated) energy dict;
    ``x0`` the active coords; ``vdw`` an optional tuple ``(prot_pos, lig_local, lig_r,
    prot_r, scale, weight)`` folding the dynamic ligand-protein VdW term into the compiled
    energy. The energy+grad is inductor-compiled for CUDA coords (the non-GPU tests
    exercise the eager functional CG, the same correct algorithm); any compile/runtime
    failure degrades permanently to eager. Returns optimized coords."""
    with_vdw = vdw is not None
    cvg = _get_cvg(with_vdw=with_vdw) if x0.is_cuda else None
    if cvg is not None:
        vg = (
            (lambda x: cvg(x, prepared, *vdw))
            if with_vdw
            else (lambda x: cvg(x, prepared))
        )
        try:
            return _cg_minimize_torch(vg, x0, max_iter)
        except (
            Exception
        ) as exc:  # this artifact's runtime failure -> eager, permanently
            logger.warning("GPU CG (compiled) failed at runtime (%s); eager", exc)
            _compile_failed[with_vdw] = True

    # eager fallback (CPU coords, compile disabled, or compiled artifact failed): the same
    # correct early-exit CG on the same (vdw-augmented) energy used by the compiled path.
    base = _energy_vdw if with_vdw else _energy
    extra = vdw if with_vdw else ()

    def e_of(a):
        return base(a, prepared, *extra)

    return _cg_minimize_torch(torch.func.grad_and_value(e_of), x0, max_iter)
