"""Compile ``CustomSpec`` objects into backend energy closures."""

from __future__ import annotations

from rgi_utils.custom import vocabulary as V
from rgi_utils.custom.backends import get_ops
from rgi_utils.custom.context import RestraintContext
from rgi_utils.custom.dsl import eval_formula


def _rigid_centroid(ops, block):
    """Return a centroid with an N-times rescaled gradient."""
    n_atoms = block.shape[-2]
    centroid = ops.mean_atoms(block)
    detached = ops.stop_gradient(centroid)
    return detached + float(n_atoms) * (centroid - detached)


def _geom_value(ops, geom, blocks):
    if geom == "distance":
        return V.distance(ops, blocks[0], blocks[1])
    if geom == "angle":
        return V.angle(ops, blocks[0], blocks[1], blocks[2])
    if geom == "dihedral":
        return V.dihedral(ops, blocks[0], blocks[1], blocks[2], blocks[3])
    raise ValueError(f"ref_geom: unknown geom {geom!r}")


def _ref_geom_penalty(ops, geom, type_code, value, target1, target2):
    if type_code == 0:
        deviation = value - target1
        if geom == "dihedral":
            deviation = V.wrap(ops, deviation)
        return deviation * deviation
    if type_code == 1:
        return V.flat_bottomed(ops, value, target1, target2)
    if type_code == 2:
        return V.flat_bottomed1(ops, value, target1)
    return V.flat_bottomed2(ops, value, target2)


def build_closure(spec, ops):
    """Build ``(active_coords) -> scalar`` for one custom spec."""
    selections = {key: ops.asint(value) for key, value in spec.selections.items()}
    refs = {
        key: (ops.asint(indices), ref_coords)
        for key, (indices, ref_coords) in spec.refs.items()
    }
    selection_refs = dict(spec.selection_refs)
    ref_fits = {
        ref_name: (ops.asint(indices), fit_ref)
        for ref_name, (indices, fit_ref) in spec.ref_fits.items()
    }
    ref_blocks = dict(spec.ref_blocks)
    move_free = dict(spec.move_free)
    weight = spec.weight

    def make_context(coords):
        return RestraintContext(
            ops,
            selections,
            coords,
            refs,
            selection_refs,
            ref_fits,
            ref_blocks,
            move_free=move_free,
        )

    if spec.kind == "formula":
        ast = spec.ast

        def energy(coords):
            return weight * ops.sum(eval_formula(ast, make_context(coords)))

    elif spec.kind == "ref_geom":
        baked_groups = [
            ("pred", (ops.asint(payload[0]), payload[1]))
            if kind == "pred"
            else ("ref", payload)
            for kind, payload in spec.groups
        ]
        geom = spec.geom
        type_code = spec.geom_type_code
        target1 = spec.target1
        target2 = spec.target2

        def energy(coords):
            context = make_context(coords)
            centroid_blocks = []
            for kind, payload in baked_groups:
                if kind == "pred":
                    indices, free = payload
                    centroid = _rigid_centroid(ops, ops.gather(coords, indices))
                    if not free:
                        centroid = ops.stop_gradient(centroid)
                else:
                    centroid = ops.mean_atoms(
                        context._reference_coords(payload[0], payload[1])
                    )
                centroid_blocks.append(centroid[..., None, :])
            value = _geom_value(ops, geom, centroid_blocks)
            penalty = _ref_geom_penalty(ops, geom, type_code, value, target1, target2)
            return weight * ops.sum(penalty)

    else:
        fn = spec.fn

        def energy(coords):
            return weight * ops.sum(fn(make_context(coords)))

    return energy


def build_terms(custom_specs, backend: str, device=None):
    """Build gated custom energy term tuples for one backend."""
    ops = get_ops(backend, device)
    return [
        (
            spec.name,
            spec.start_sigma,
            spec.stop_sigma,
            spec.start_step,
            spec.stop_step,
            build_closure(spec, ops),
        )
        for spec in custom_specs
    ]
