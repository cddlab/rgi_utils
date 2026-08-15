"""Shared tuning constants for the torch / jax nonlinear-CG solvers.

``torch_optim._minimize_cg`` (eager, CPU), ``_torch_cg_gpu._cg_minimize_torch``
(functional, the CUDA path) and ``jax_optim._cg_minimize`` are a deliberate
same-algorithm trio (Polak-Ribiere+, backtracking Armijo line search, restart on
non-descent). Their loop bodies legitimately differ (a python ``for`` loop vs a
``lax.while_loop``), but the convergence contract must NOT drift — so the constants
live here once and all three solvers import them as defaults. This is a leaf module (no
imports) to avoid an optim package import cycle.
"""

from __future__ import annotations

MAX_LS = 20  # max backtracking line-search trials per iteration
GTOL = 1e-7  # converged when max|grad| < GTOL
FTOL = 1e-9  # converged when |f_new - f| < FTOL * (1 + |f|)
ARMIJO_C1 = 1e-4  # Armijo sufficient-decrease coefficient
BACKTRACK = 0.5  # line-search step shrink factor
GG_FLOOR = 1e-20  # gg <= GG_FLOOR (or non-finite) -> degenerate iteration, stop
EPS = 1e-12  # denominator guard in the Polak-Ribiere+ beta

# --- line-search warm start -------------------------------------------------------------
# The trial step is CARRIED across CG iterations rather than reset to 1.0 each time:
#
#     step_trial = min(LS_STEP_MAX, max(step_prev, LS_STEP_MIN) * LS_STEP_GROW)
#
# where step_prev is the previous iteration's ACCEPTED step, initialised to LS_STEP_MAX. It is
# not reset on the non-descent restart branch nor on a degenerate iteration.
#
# Why: measured on a 1762-atom polymer conformer, the accepted step sits at 0.5**5 = 0.03125
# for essentially every iteration, so restarting at 1.0 re-descends the same five halvings and
# burns ~6 energy+grad evaluations per iteration where ~2 suffice.
#
# LS_STEP_GROW gives exactly one backtrack of headroom, so a step that shrank in a stiff
# region climbs back exponentially once the landscape softens.
#
# LS_STEP_MIN is not cosmetic. A cold start guarantees every accepted step is at least
# BACKTRACK**MAX_LS; a warm start has no such floor, and a_k = min(1, 2*a_{k-1}) * 2**-j_k
# ratchets down geometrically whenever j >= 2 is sustained. It would then terminate by
# UNDERFLOWING to a no-op step — delta ~ 0, so Armijo passes trivially, |f_new - f| == 0, and
# FTOL reports convergence — silently turning slow progress into an early stop. The floor
# restores the cold start's reachability guarantee. The measured steady state is j ~ 1.5 and
# j == 1 is the fixed point, so this is a tail guard, not the operating point.
#
# Exactness: from 1.0 the only operations are *0.5, *2.0, min(., 1.0) and max(., 2**-20), so
# every trial step stays an exact power of two. The warm start therefore adds NO new
# cross-backend float divergence — the three step sequences are bit-identical whenever the
# Armijo accept decisions agree.
LS_STEP_MAX = 1.0  # trial-step ceiling, and the first iteration's trial step
LS_STEP_GROW = 1.0 / BACKTRACK  # one backtrack of headroom per iteration
LS_STEP_MIN = BACKTRACK**MAX_LS  # floor on the CARRIED step (see above)
