"""Compile a ``CustomSpec`` into a backend closure ``(active_coords) -> scalar``.

The selection LOCAL indices are converted to the backend's int arrays ONCE and captured
(so for jax they are static constants, lax.scan-safe); only ``active_coords`` is the live
input. ``weight`` is folded in. The optimizer multiplies the result by a per-entry sigma
gate (``stop_sigma <= sigma <= start_sigma``), which it computes from ``sigma``.
"""

from __future__ import annotations

from rgi_utils.custom import vocabulary as V
from rgi_utils.custom.backends import get_ops
from rgi_utils.custom.context import RestraintContext
from rgi_utils.custom.dsl import eval_formula


def _rigid_centroid(ops, block):
    """A group centroid whose VALUE is the plain mean but whose GRADIENT is N×, so the group
    translates as a rigid body under CG at ``weight: 1`` (the built-in group terms' trick, kept
    for ref-anchored built-in distance/angle/dihedral — a plain centroid's ``1/N`` per-atom
    gradient would barely move a large prediction group). ``c_eff = detach(c) + N*(c - detach(c))``
    is value-identical to ``c`` but scales its gradient by N. Internal to the ref_geom closure —
    deliberately NOT a DSL primitive (exposing it would break the strict numpy-FD custom check)."""
    n = block.shape[-2]  # static group size (known at trace time -> lax.scan-safe)
    c = ops.mean_atoms(block)
    cd = ops.stop_gradient(c)
    return cd + float(n) * (c - cd)


def _geom_value(ops, geom, cblocks):
    """The distance / angle / dihedral of group centroids, reusing the vocabulary primitives
    (each centroid is wrapped as a ``(..., 1, 3)`` pseudo-block, so ``mean_atoms`` returns it
    unchanged) — so parity with the DSL geometry is structural."""
    if geom == "distance":
        return V.distance(ops, cblocks[0], cblocks[1])
    if geom == "angle":
        return V.angle(ops, cblocks[0], cblocks[1], cblocks[2])
    if geom == "dihedral":
        return V.dihedral(ops, cblocks[0], cblocks[1], cblocks[2], cblocks[3])
    raise ValueError(f"ref_geom: unknown geom {geom!r}")


def _ref_geom_penalty(ops, geom, tcode, val, t1, t2):
    """Apply the shared penalty (DIST_TYPE_CODES: 0=harmonic, 1=flat-bottomed, 2=lower,
    3=upper) to the measured ``val``. A harmonic DIHEDRAL wraps its deviation to [-pi, pi]
    (periodicity), matching the built-in group_dihedral term; angle/distance do not wrap."""
    if tcode == 0:  # harmonic
        dev = val - t1
        if geom == "dihedral":
            dev = V.wrap(ops, dev)
        return dev * dev
    if tcode == 1:  # flat-bottomed (both bounds)
        return V.flat_bottomed(ops, val, t1, t2)
    if tcode == 2:  # flat-bottomed1 (lower bound only)
        return V.flat_bottomed1(ops, val, t1)
    return V.flat_bottomed2(ops, val, t2)  # flat-bottomed2 (upper bound only)


def build_closure(cspec, ops):
    """``(active_coords) -> scalar`` for one CustomSpec on the given backend ``ops``."""
    sel = {k: ops.asint(v) for k, v in cspec.selections.items()}
    # rmsd primitive references: (matched-target LOCAL index array -> backend int; raw numpy
    # ref coords kept as a constant). The index array is baked ONCE (like the selections, so
    # jax stays lax.scan-safe); the ref coords are converted to the LIVE coords' dtype/device
    # at eval time via ops.const_like in the ctx (a captured numpy const is a jax constant, so
    # still scan-safe, and it can't dtype-mismatch). Empty unless the restraint uses rmsd().
    refs = {
        k: (ops.asint(idx), ref_coords) for k, (idx, ref_coords) in cspec.refs.items()
    }
    # ref() primitive: bake the fit-anchor LOCAL index array ONCE (like the selections, so jax
    # stays lax.scan-safe); the fit-ref + ref-group coord blocks stay numpy constants (captured
    # -> jax constants; converted to the live coords' dtype/device at eval via const_like).
    ref_fits = {
        r: (ops.asint(idx), fit_ref) for r, (idx, fit_ref) in cspec.ref_fits.items()
    }
    ref_blocks = dict(cspec.ref_blocks)
    weight = cspec.weight
    # ``ops.sum`` reduces over ALL leading/batch dims to a scalar (like the built-in
    # energies), so the energy a batch/multiplicity tensor produces stays a scalar that
    # autograd / jax.value_and_grad can differentiate.
    if cspec.kind == "formula":
        ast = cspec.ast

        def energy(coords):
            return weight * ops.sum(
                eval_formula(
                    ast,
                    RestraintContext(ops, sel, coords, refs, ref_fits, ref_blocks),
                )
            )
    elif cspec.kind == "ref_geom":
        # ref-anchored built-in distance/angle/dihedral: prediction groups use a RIGID centroid
        # (weight=1 moves a large group as a body); reference groups are fitted-and-fixed blocks
        # via ctx.ref (the whole fit transform is stop-gradient'd). Bake each group descriptor.
        baked = [
            ("pred", ops.asint(payload)) if kind == "pred" else ("ref", payload)
            for kind, payload in cspec.groups
        ]
        geom, tcode = cspec.geom, cspec.geom_type_code
        t1, t2 = cspec.target1, cspec.target2

        def energy(coords):
            ctx = RestraintContext(ops, sel, coords, refs, ref_fits, ref_blocks)
            cblocks = []
            for kind, payload in baked:
                if kind == "pred":
                    c = _rigid_centroid(ops, ops.gather(coords, payload))
                else:  # ("ref", (sel_string, ref_name)) -> fitted, fixed block
                    c = ops.mean_atoms(ctx.ref(payload[0], payload[1]))
                cblocks.append(c[..., None, :])  # (..., 1, 3) pseudo-block
            val = _geom_value(ops, geom, cblocks)
            return weight * ops.sum(_ref_geom_penalty(ops, geom, tcode, val, t1, t2))
    else:
        fn = cspec.fn

        def energy(coords):
            return weight * ops.sum(
                fn(RestraintContext(ops, sel, coords, refs, ref_fits, ref_blocks))
            )

    return energy


def build_terms(custom_specs, backend: str, device=None):
    """For an optimizer on ``backend``: a list of ``(name, start_sigma, stop_sigma,
    start_step, stop_step, closure)`` — one per custom restraint.
    ``closure(active_coords) -> scalar`` (weight folded); the optimizer applies the gate,
    which is the active sigma window AND the active step window (a restraint uses one or
    the other — mutually exclusive at config time; the unused axis is always-on).
    ``device`` (torch only) places the baked selection-index tensors on the coords' device.
    """
    ops = get_ops(backend, device)
    return [
        (
            c.name,
            c.start_sigma,
            c.stop_sigma,
            c.start_step,
            c.stop_step,
            build_closure(c, ops),
        )
        for c in custom_specs
    ]
