"""Shared tuning constants for the torch / jax nonlinear-CG solvers.

``torch_optim._minimize_cg`` and ``jax_optim._cg_minimize`` are a deliberate
same-algorithm pair (Polak-Ribiere+, backtracking Armijo line search, restart on
non-descent). Their loop bodies legitimately differ (a python ``for`` loop vs a
``lax.while_loop``), but the convergence contract must NOT drift — so the constants
live here once and both solvers import them as defaults. This is a leaf module (no
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
