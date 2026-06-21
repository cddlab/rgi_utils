"""Backend-agnostic restraint-term schema shared by the numpy / torch / jax energy
modules — the single source of truth for:

  * which ``RestraintSpec`` fields each restraint type carries, and their int/float
    kind (``pack_spec``);
  * which leaf energy function each term calls, with which args and which noise gate
    (``term_energies``).

Adding a 7th restraint type means editing the two tables here, not the six
total_energy/energy_breakdown/prepare_spec bodies that used to carry hand-unrolled
copies of this dispatch. This module imports no array framework: callers pass their
own converters / leaf functions, so importing it stays numpy-only.
"""

from __future__ import annotations

from rgi_utils import registry

# (prepared_key, spec_attr, [(field_name, kind), ...]); kind "i"=int array, "f"=float.
# This is the BUILT-IN half; custom (registry) restraints are spliced in by
# ``spec_schema()`` below, with spec_attr=None meaning "read spec.registered[key]".
_SPEC_SCHEMA = [
    (
        "bond",
        "bond",
        [
            ("idx", "i"),
            ("r0", "f"),
            ("slack", "f"),
            ("weight", "f"),
            ("half", "f"),
            ("mask", "f"),
        ],
    ),
    (
        "angle",
        "angle",
        [("idx", "i"), ("th0", "f"), ("slack", "f"), ("weight", "f"), ("mask", "f")],
    ),
    (
        "chiral",
        "chiral",
        [("idx", "i"), ("vol0", "f"), ("slack", "f"), ("weight", "f"), ("mask", "f")],
    ),
    # planarity (improper) of sp2 double-bond centres: same fields as chiral (it reuses
    # chiral_energy), only target vol0 ~ 0 differs. See _TERMS / spec.ImproperArrays.
    (
        "improper",
        "improper",
        [("idx", "i"), ("vol0", "f"), ("slack", "f"), ("weight", "f"), ("mask", "f")],
    ),
    (
        "cistrans",
        "cistrans",
        [("idx", "i"), ("phi0", "f"), ("slack", "f"), ("weight", "f"), ("mask", "f")],
    ),
    ("vdw", "vdw", [("idx", "i"), ("r_min", "f"), ("weight", "f"), ("mask", "f")]),
    (
        "distance",
        "distance",
        [
            ("grp1_idx", "i"),
            ("grp2_idx", "i"),
            ("grp1_mask", "f"),
            ("grp2_mask", "f"),
            ("target1", "f"),
            ("target2", "f"),
            ("dist_type", "i"),
            ("move_mode", "i"),
            ("mask", "f"),
            ("start_sigma", "f"),
            ("stop_sigma", "f"),
        ],
    ),
    (
        "rmsd",
        "rmsd",
        [
            ("fit_idx", "i"),
            ("fit_mask", "f"),
            ("fit_ref", "f"),
            ("calc_idx", "i"),
            ("calc_mask", "f"),
            ("calc_ref", "f"),
            ("target_rmsd", "f"),
            ("weight", "f"),
            ("mask", "f"),
            ("start_sigma", "f"),
            ("stop_sigma", "f"),
        ],
    ),
    # group-centroid angle (3 groups, vertex = group 2) / dihedral (4 groups, axis =
    # group2-group3): each group is a padded (n, max_grp) index list + {0,1} mask (same
    # layout as distance). target1/target2/geom_type mirror distance's four
    # types; move_free ((n,n_groups) {0,1}) drives the energy's detach-select (unlike
    # distance, where move lives in the closed-form shift).
    # start_sigma/stop_sigma are per-restraint.
    (
        "group_angle",
        "group_angle",
        [
            ("grp1_idx", "i"),
            ("grp2_idx", "i"),
            ("grp3_idx", "i"),
            ("grp1_mask", "f"),
            ("grp2_mask", "f"),
            ("grp3_mask", "f"),
            ("target1", "f"),
            ("target2", "f"),
            ("geom_type", "i"),
            ("move_free", "f"),
            ("weight", "f"),
            ("mask", "f"),
            ("start_sigma", "f"),
            ("stop_sigma", "f"),
        ],
    ),
    (
        "group_dihedral",
        "group_dihedral",
        [
            ("grp1_idx", "i"),
            ("grp2_idx", "i"),
            ("grp3_idx", "i"),
            ("grp4_idx", "i"),
            ("grp1_mask", "f"),
            ("grp2_mask", "f"),
            ("grp3_mask", "f"),
            ("grp4_mask", "f"),
            ("target1", "f"),
            ("target2", "f"),
            ("geom_type", "i"),
            ("move_free", "f"),
            ("weight", "f"),
            ("mask", "f"),
            ("start_sigma", "f"),
            ("stop_sigma", "f"),
        ],
    ),
]


def spec_schema():
    """The built-in spec schema plus one entry per registered restraint. A registered
    entry is ``(name, None, fields)``: ``spec_attr=None`` tells ``pack_spec`` to read
    ``spec.registered[name]`` instead of a named attribute. Built fresh each call so a
    restraint registered after import is picked up (the registry is tiny)."""
    schema = list(_SPEC_SCHEMA)
    for rt in registry.iter_registered():
        schema.append((rt.name, None, list(rt.spec_schema)))
    return schema


def pack_spec(spec, to_int, to_float):
    """Convert a ``RestraintSpec`` into a backend's prepared dict via the two given
    converters (numpy array -> backend array). Only sub-arrays with a non-empty mask
    are included, matching the original per-backend ``prepare_spec`` gating. Custom
    (registry) restraints live in ``spec.registered[name]`` (spec_attr=None)."""
    conv = {"i": to_int, "f": to_float}
    prepared = {
        "conf_start_sigma": float(getattr(spec, "conf_start_sigma", -1.0)),
        "conf_stop_sigma": float(getattr(spec, "conf_stop_sigma", -1.0)),
    }
    registered = getattr(spec, "registered", None) or {}
    for key, attr, fields in spec_schema():
        arr = registered.get(key) if attr is None else getattr(spec, attr, None)
        if arr is None or not (
            arr.mask.sum() > 0
        ):  # same as old `> 0` include, NaN-safe
            continue
        prepared[key] = {f: conv[kind](getattr(arr, f)) for f, kind in fields}
    return prepared


# (prepared_key, leaf_fn_name, [arg_field, ...], gate). Every leaf fn takes
# ``(positions, *args, mask)`` with mask LAST. gate:
#   "conf" -> conformer term, masked by the shared conformer gate ``cg``;
#   "dist" -> distance, per-restraint sigma gate, ONLY when include_distance;
#   "rmsd" -> RMSD, per-restraint sigma gate, ALWAYS (the CG solver calls
#             total_energy(include_distance=False) and must still see the RMSD term);
#   "group" -> group-centroid angle/dihedral, per-restraint sigma gate, ALWAYS (CG-solved
#              like rmsd). Any gate other than conf/dist gets per-entry gating + is
#              always summed in the solver; such keys are collected in PER_ENTRY_KEYS.
_TERMS = [
    ("bond", "bond_energy", ["idx", "r0", "slack", "weight", "half"], "conf"),
    ("angle", "angle_energy", ["idx", "th0", "slack", "weight"], "conf"),
    ("chiral", "chiral_energy", ["idx", "vol0", "slack", "weight"], "conf"),
    # planarity improper REUSES the chiral signed-volume leaf fn (target vol0 ~ 0); its
    # own prepared key keeps it a separate, independently-weighted/reported term.
    ("improper", "chiral_energy", ["idx", "vol0", "slack", "weight"], "conf"),
    ("cistrans", "cistrans_energy", ["idx", "phi0", "slack", "weight"], "conf"),
    ("vdw", "vdw_energy", ["idx", "r_min", "weight"], "conf"),
    (
        "distance",
        "distance_energy",
        [
            "grp1_idx",
            "grp2_idx",
            "grp1_mask",
            "grp2_mask",
            "target1",
            "target2",
            "dist_type",
        ],
        "dist",
    ),
    (
        "rmsd",
        "rmsd_energy",
        [
            "fit_idx",
            "fit_mask",
            "fit_ref",
            "calc_idx",
            "calc_mask",
            "calc_ref",
            "target_rmsd",
            "weight",
        ],
        "rmsd",
    ),
    (
        "group_angle",
        "group_angle_energy",
        [
            "grp1_idx",
            "grp2_idx",
            "grp3_idx",
            "grp1_mask",
            "grp2_mask",
            "grp3_mask",
            "target1",
            "target2",
            "geom_type",
            "move_free",
            "weight",
        ],
        "group",
    ),
    (
        "group_dihedral",
        "group_dihedral_energy",
        [
            "grp1_idx",
            "grp2_idx",
            "grp3_idx",
            "grp4_idx",
            "grp1_mask",
            "grp2_mask",
            "grp3_mask",
            "grp4_mask",
            "target1",
            "target2",
            "geom_type",
            "move_free",
            "weight",
        ],
        "group",
    ),
]


def terms():
    """The built-in dispatch table plus one row per registered restraint:
    ``(name, name, term_args, gate)`` — the leaf fn is looked up by ``name`` in the
    merged fns dict (see ``leaf_fns_for``). Registered restraints carry a per-entry gate
    (the registry rejects ``conf``/``dist``), so each is summed by the solver and
    sigma-gated per entry, exactly like rmsd / group_*."""
    table = list(_TERMS)
    for rt in registry.iter_registered():
        table.append((rt.name, rt.name, list(rt.term_args), rt.gate))
    return table


_LEAF_FNS_CACHE: dict = {}


def leaf_fns_for(backend, base):
    """Merge a backend's built-in ``_LEAF_FNS`` (``base``) with the registered restraints'
    leaf functions for ``backend`` (resolved lazily from dotted paths). This merged dict is
    what ``term_energies`` indexes by ``fn_name``.

    MEMOIZED by registry generation: ``total_energy`` is the energy the torch GPU CG
    compiles, so the importlib + dict-merge must NOT run on every call (it would risk a
    dynamo graph break on the proven built-in path). After warm-up this is a single dict
    lookup returning a STABLE object; it rebuilds only when the registry actually changes.
    Returns ``base`` unchanged when nothing is registered for ``backend``."""
    key = (backend, id(base), registry.generation())
    cached = _LEAF_FNS_CACHE.get(key)
    if cached is not None:
        return cached
    reg = registry.registered_leaf_fns(backend)
    merged = base if not reg else {**base, **reg}
    _LEAF_FNS_CACHE[key] = merged
    return merged


def term_energies(fns, prepared, positions, cg, sigma_gate, include_distance):
    """Return ``{key: energy}`` for every active restraint term, with the right noise
    gate folded into each term's mask. Shared by all three backends' total_energy and
    energy_breakdown — only ``fns`` (the backend leaf functions, merged with any
    registered ones via ``leaf_fns_for``), ``cg`` (the conformer gate multiplier) and
    ``sigma_gate`` (a per-restraint ``(start_sigma, stop_sigma, mask) -> gated_mask``
    callable; ``stop_sigma`` may be ``None`` for a term with no lower bound) vary per
    backend / noise level.

    ``cg``/``sigma_gate`` are precomputed by the caller so the backend-specific casts
    (numpy bool, torch ``.to``, jax tracer-safe ``.astype``) stay out of this loop.
    """
    out = {}
    for key, fn_name, fields, gate in terms():
        if key not in prepared:
            continue
        if gate == "dist" and not include_distance:
            continue
        p = prepared[key]
        # rmsd carries a lower noise bound (stop_sigma -> released for sigma below it);
        # distance has none (p.get -> None == no lower bound).
        mask = (
            p["mask"] * cg
            if gate == "conf"
            else sigma_gate(p["start_sigma"], p.get("stop_sigma"), p["mask"])
        )
        out[key] = fns[fn_name](positions, *[p[f] for f in fields], mask)
    return out


# the float keys energy_breakdown reports, in display order (all start at 0.0)
BREAKDOWN_KEYS = (
    "bond",
    "angle",
    "chiral",
    "improper",
    "cistrans",
    "vdw",
    "distance",
    "rmsd",
    "group_angle",
    "group_dihedral",
)

# the conformer-gated term keys (gate == "conf"): the single source of truth for which
# terms the conformer noise gate (cg) applies to. The torch GPU pre-gate path imports this
# instead of re-listing the keys, so adding a 7th conformer term to _TERMS can't silently
# leave it ungated on the compiled path.
CONF_KEYS = frozenset(key for key, _fn, _fields, gate in _TERMS if gate == "conf")

# per-restraint-gated term keys (gate not "conf"/"dist"): each carries its own
# start_sigma/stop_sigma and is ALWAYS summed in the solver's energy
# (include_distance=False). The torch GPU pre-gate (torch_optim._gated_prepared) folds
# each one's gate into its mask and keys its cache on every gate state, so adding a
# per-entry term to _TERMS can't silently leave it ungated on the compiled GPU path.
# distance is excluded (closed-form, applied separately); conformer is in CONF_KEYS.
PER_ENTRY_KEYS = frozenset(
    key for key, _fn, _fields, gate in _TERMS if gate not in ("conf", "dist")
)


# Registry-aware accessors. Consumers that must see custom restraints (energy_breakdown
# and the torch GPU pre-gate) call THESE rather than the frozen constants above, so a
# restraint registered after import is reflected — and, crucially, a registered per-entry
# term can't silently go ungated on the compiled GPU path (the CLAUDE.md footgun).
def breakdown_keys():
    """``BREAKDOWN_KEYS`` plus each registered restraint's name (so energy_breakdown
    reports custom terms too)."""
    return BREAKDOWN_KEYS + tuple(rt.name for rt in registry.iter_registered())


def conf_keys():
    """Conformer-gated keys. Registered restraints are never conformer-gated (the
    registry rejects the ``conf`` gate), so this equals the built-in ``CONF_KEYS``; the
    accessor exists for symmetry with ``per_entry_keys``."""
    return CONF_KEYS


def per_entry_keys():
    """Per-entry-gated keys: ``PER_ENTRY_KEYS`` plus every registered restraint (all of
    which are per-entry, CG-solved). The torch GPU pre-gate folds each of these into its
    mask and keys its compile cache on their gate state."""
    return PER_ENTRY_KEYS | {rt.name for rt in registry.iter_registered()}
