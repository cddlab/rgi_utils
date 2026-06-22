"""Compile a ``CustomSpec`` into a backend closure ``(active_coords) -> scalar``.

The selection LOCAL indices are converted to the backend's int arrays ONCE and captured
(so for jax they are static constants, lax.scan-safe); only ``active_coords`` is the live
input. ``weight`` is folded in. The optimizer multiplies the result by a per-entry sigma
gate (``stop_sigma <= sigma <= start_sigma``), which it computes from ``sigma``.
"""

from __future__ import annotations

from rgi_utils.custom.backends import get_ops
from rgi_utils.custom.context import RestraintContext
from rgi_utils.custom.dsl import eval_formula


def build_closure(cspec, ops):
    """``(active_coords) -> scalar`` for one CustomSpec on the given backend ``ops``."""
    sel = {k: ops.asint(v) for k, v in cspec.selections.items()}
    weight = cspec.weight
    # ``ops.sum`` reduces over ALL leading/batch dims to a scalar (like the built-in
    # energies), so the energy a batch/multiplicity tensor produces stays a scalar that
    # autograd / jax.value_and_grad can differentiate.
    if cspec.kind == "formula":
        ast = cspec.ast

        def energy(coords):
            return weight * ops.sum(
                eval_formula(ast, RestraintContext(ops, sel, coords))
            )
    else:
        fn = cspec.fn

        def energy(coords):
            return weight * ops.sum(fn(RestraintContext(ops, sel, coords)))

    return energy


def build_terms(custom_specs, backend: str, device=None):
    """For an optimizer on ``backend``: a list of ``(name, start_sigma, stop_sigma,
    closure)`` — one per custom restraint. ``closure(active_coords) -> scalar`` (weight
    folded); the optimizer applies the sigma gate. ``device`` (torch only) places the
    baked selection-index tensors on the coords' device."""
    ops = get_ops(backend, device)
    return [
        (c.name, c.start_sigma, c.stop_sigma, build_closure(c, ops))
        for c in custom_specs
    ]
