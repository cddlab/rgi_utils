"""Single registry for array-backed restraint terms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TermDef:
    """Packing and dispatch metadata for one restraint term."""

    key: str
    spec_attr: str
    fields: tuple[tuple[str, str], ...]
    leaf_fn: str
    args: tuple[str, ...]
    gate: str


_WINDOW_FIELDS = (
    ("start_sigma", "f"),
    ("stop_sigma", "f"),
    ("start_step", "f"),
    ("stop_step", "f"),
)
_GROUP_TARGET_FIELDS = (
    ("target1", "f"),
    ("target2", "f"),
    ("geom_type", "i"),
    ("move_free", "f"),
    ("weight", "f"),
    ("mask", "f"),
) + _WINDOW_FIELDS
_TORSION_GROUP_FIELDS = (
    ("grp1_idx", "i"),
    ("grp2_idx", "i"),
    ("grp3_idx", "i"),
    ("grp4_idx", "i"),
    ("grp1_mask", "f"),
    ("grp2_mask", "f"),
    ("grp3_mask", "f"),
    ("grp4_mask", "f"),
) + _GROUP_TARGET_FIELDS
_TORSION_ARGS = (
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
)

TERM_DEFS = (
    TermDef(
        "bond",
        "bond",
        (
            ("idx", "i"),
            ("r0", "f"),
            ("slack", "f"),
            ("weight", "f"),
            ("half", "f"),
            ("mask", "f"),
        ),
        "bond_energy",
        ("idx", "r0", "slack", "weight", "half"),
        "conf",
    ),
    TermDef(
        "angle",
        "angle",
        (
            ("idx", "i"),
            ("th0", "f"),
            ("slack", "f"),
            ("weight", "f"),
            ("mask", "f"),
        ),
        "angle_energy",
        ("idx", "th0", "slack", "weight"),
        "conf",
    ),
    TermDef(
        "chiral",
        "chiral",
        (
            ("idx", "i"),
            ("vol0", "f"),
            ("slack", "f"),
            ("weight", "f"),
            ("mask", "f"),
        ),
        "chiral_energy",
        ("idx", "vol0", "slack", "weight"),
        "conf",
    ),
    TermDef(
        "plane",
        "plane",
        (
            ("idx", "i"),
            ("grp_mask", "f"),
            ("slack", "f"),
            ("weight", "f"),
            ("mask", "f"),
        ),
        "plane_energy",
        ("idx", "grp_mask", "slack", "weight"),
        "conf",
    ),
    TermDef(
        "cistrans",
        "cistrans",
        (
            ("idx", "i"),
            ("phi0", "f"),
            ("slack", "f"),
            ("weight", "f"),
            ("mask", "f"),
        ),
        "cistrans_energy",
        ("idx", "phi0", "slack", "weight"),
        "conf",
    ),
    TermDef(
        "vdw",
        "vdw",
        (("idx", "i"), ("r_min", "f"), ("weight", "f"), ("mask", "f")),
        "vdw_energy",
        ("idx", "r_min", "weight"),
        "conf",
    ),
    TermDef(
        "distance",
        "distance",
        (
            ("grp1_idx", "i"),
            ("grp2_idx", "i"),
            ("grp1_mask", "f"),
            ("grp2_mask", "f"),
            ("target1", "f"),
            ("target2", "f"),
            ("dist_type", "i"),
            ("move_mode", "i"),
            ("weight", "f"),
            ("mask", "f"),
        )
        + _WINDOW_FIELDS,
        "distance_energy",
        (
            "grp1_idx",
            "grp2_idx",
            "grp1_mask",
            "grp2_mask",
            "target1",
            "target2",
            "dist_type",
            "move_mode",
            "weight",
        ),
        "entry",
    ),
    TermDef(
        "rmsd",
        "rmsd",
        (
            ("fit_idx", "i"),
            ("fit_mask", "f"),
            ("fit_ref", "f"),
            ("calc_idx", "i"),
            ("calc_mask", "f"),
            ("calc_ref", "f"),
            ("target1", "f"),
            ("target2", "f"),
            ("geom_type", "i"),
            ("weight", "f"),
            ("mask", "f"),
        )
        + _WINDOW_FIELDS,
        "rmsd_energy",
        (
            "fit_idx",
            "fit_mask",
            "fit_ref",
            "calc_idx",
            "calc_mask",
            "calc_ref",
            "target1",
            "target2",
            "geom_type",
            "weight",
        ),
        "entry",
    ),
    TermDef(
        "group_angle",
        "group_angle",
        (
            ("grp1_idx", "i"),
            ("grp2_idx", "i"),
            ("grp3_idx", "i"),
            ("grp1_mask", "f"),
            ("grp2_mask", "f"),
            ("grp3_mask", "f"),
        )
        + _GROUP_TARGET_FIELDS,
        "group_angle_energy",
        (
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
        ),
        "entry",
    ),
    TermDef(
        "group_dihedral",
        "group_dihedral",
        _TORSION_GROUP_FIELDS,
        "group_dihedral_energy",
        _TORSION_ARGS,
        "entry",
    ),
    TermDef(
        "group_improper",
        "group_improper",
        _TORSION_GROUP_FIELDS,
        "group_improper_energy",
        _TORSION_ARGS,
        "entry",
    ),
    TermDef(
        "group_plane",
        "group_plane",
        (
            ("idx", "i"),
            ("grp_mask", "f"),
            ("free", "f"),
            ("target1", "f"),
            ("target2", "f"),
            ("geom_type", "i"),
            ("weight", "f"),
            ("mask", "f"),
        )
        + _WINDOW_FIELDS,
        "group_plane_energy",
        (
            "idx",
            "grp_mask",
            "free",
            "target1",
            "target2",
            "geom_type",
            "weight",
        ),
        "entry",
    ),
)

TERM_BY_KEY = {term.key: term for term in TERM_DEFS}

# Compatibility views for code that used the previous two private tables.
_SPEC_SCHEMA = [(term.key, term.spec_attr, list(term.fields)) for term in TERM_DEFS]
_TERMS = [(term.key, term.leaf_fn, list(term.args), term.gate) for term in TERM_DEFS]


def iter_spec_terms(spec, keys=None):
    """Yield active ``(TermDef, array)`` pairs from a RestraintSpec."""
    selected = None if keys is None else frozenset(keys)
    for term in TERM_DEFS:
        if selected is not None and term.key not in selected:
            continue
        array = getattr(spec, term.spec_attr, None)
        if array is not None and array.mask.sum() > 0:
            yield term, array


def pack_spec(spec, to_int, to_float):
    """Convert a RestraintSpec into one backend's prepared dictionary."""
    converters = {"i": to_int, "f": to_float}
    prepared = {
        "conf_start_sigma": float(getattr(spec, "conf_start_sigma", -1.0)),
        "conf_stop_sigma": float(getattr(spec, "conf_stop_sigma", -1.0)),
        "conf_start_step": float(getattr(spec, "conf_start_step", float("-inf"))),
        "conf_stop_step": float(getattr(spec, "conf_stop_step", float("inf"))),
    }
    for term, array in iter_spec_terms(spec):
        prepared[term.key] = {
            name: converters[kind](getattr(array, name)) for name, kind in term.fields
        }
    return prepared


def term_energies(leaf_fns, prepared, positions, conformer_gate, entry_gate):
    """Evaluate every prepared term with its registered gate and leaf function."""
    output = {}
    for term in TERM_DEFS:
        if term.key not in prepared:
            continue
        params = prepared[term.key]
        mask = (
            params["mask"] * conformer_gate
            if term.gate == "conf"
            else entry_gate(
                params["start_sigma"],
                params.get("stop_sigma"),
                params.get("start_step"),
                params.get("stop_step"),
                params["mask"],
            )
        )
        output[term.key] = leaf_fns[term.leaf_fn](
            positions, *[params[name] for name in term.args], mask
        )
    return output


# Display order is stable for diagnostic callers.
BREAKDOWN_KEYS = (
    "bond",
    "angle",
    "chiral",
    "plane",
    "cistrans",
    "vdw",
    "distance",
    "rmsd",
    "group_angle",
    "group_dihedral",
    "group_plane",
    "group_improper",
)
CONF_KEYS = frozenset(term.key for term in TERM_DEFS if term.gate == "conf")
PER_ENTRY_KEYS = frozenset(term.key for term in TERM_DEFS if term.gate != "conf")
