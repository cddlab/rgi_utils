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

# (prepared_key, spec_attr, [(field_name, kind), ...]); kind "i"=int array, "f"=float.
_SPEC_SCHEMA = [
    ("bond", "bond",
     [("idx", "i"), ("r0", "f"), ("slack", "f"), ("weight", "f"), ("half", "f"),
      ("mask", "f")]),
    ("angle", "angle",
     [("idx", "i"), ("th0", "f"), ("slack", "f"), ("weight", "f"), ("mask", "f")]),
    ("chiral", "chiral",
     [("idx", "i"), ("vol0", "f"), ("slack", "f"), ("weight", "f"), ("mask", "f")]),
    ("dihedral", "dihedral",
     [("idx", "i"), ("phi0", "f"), ("slack", "f"), ("weight", "f"), ("mask", "f")]),
    ("vdw", "vdw",
     [("idx", "i"), ("r_min", "f"), ("weight", "f"), ("mask", "f")]),
    ("distance", "distance",
     [("grp1_idx", "i"), ("grp2_idx", "i"), ("grp1_mask", "f"), ("grp2_mask", "f"),
      ("target1", "f"), ("target2", "f"), ("dist_type", "i"), ("mask", "f"),
      ("start_sigma", "f")]),
    ("rmsd", "rmsd",
     [("fit_idx", "i"), ("fit_mask", "f"), ("fit_ref", "f"), ("calc_idx", "i"),
      ("calc_mask", "f"), ("calc_ref", "f"), ("target_rmsd", "f"), ("weight", "f"),
      ("mask", "f"), ("start_sigma", "f")]),
]


def pack_spec(spec, to_int, to_float):
    """Convert a ``RestraintSpec`` into a backend's prepared dict via the two given
    converters (numpy array -> backend array). Only sub-arrays with a non-empty mask
    are included, matching the original per-backend ``prepare_spec`` gating."""
    conv = {"i": to_int, "f": to_float}
    prepared = {"conf_start_sigma": float(getattr(spec, "conf_start_sigma", -1.0))}
    for key, attr, fields in _SPEC_SCHEMA:
        arr = getattr(spec, attr, None)
        if arr is None or arr.mask.sum() <= 0:
            continue
        prepared[key] = {f: conv[kind](getattr(arr, f)) for f, kind in fields}
    return prepared


# (prepared_key, leaf_fn_name, [arg_field, ...], gate). Every leaf fn takes
# ``(positions, *args, mask)`` with mask LAST. gate:
#   "conf" -> conformer term, masked by the shared conformer gate ``cg``;
#   "dist" -> distance, per-restraint sigma gate, ONLY when include_distance;
#   "rmsd" -> RMSD, per-restraint sigma gate, ALWAYS (the CG solver calls
#             total_energy(include_distance=False) and must still see the RMSD term).
_TERMS = [
    ("bond", "bond_energy", ["idx", "r0", "slack", "weight", "half"], "conf"),
    ("angle", "angle_energy", ["idx", "th0", "slack", "weight"], "conf"),
    ("chiral", "chiral_energy", ["idx", "vol0", "slack", "weight"], "conf"),
    ("dihedral", "dihedral_energy", ["idx", "phi0", "slack", "weight"], "conf"),
    ("vdw", "vdw_energy", ["idx", "r_min", "weight"], "conf"),
    ("distance", "distance_energy",
     ["grp1_idx", "grp2_idx", "grp1_mask", "grp2_mask", "target1", "target2",
      "dist_type"], "dist"),
    ("rmsd", "rmsd_energy",
     ["fit_idx", "fit_mask", "fit_ref", "calc_idx", "calc_mask", "calc_ref",
      "target_rmsd", "weight"], "rmsd"),
]


def term_energies(fns, prepared, positions, cg, sigma_gate, include_distance):
    """Return ``{key: energy}`` for every active restraint term, with the right noise
    gate folded into each term's mask. Shared by all three backends' total_energy and
    energy_breakdown — only ``fns`` (the backend leaf functions), ``cg`` (the
    conformer gate multiplier) and ``sigma_gate`` (a per-restraint
    ``(start_sigma, mask) -> gated_mask`` callable) vary per backend / noise level.

    ``cg``/``sigma_gate`` are precomputed by the caller so the backend-specific casts
    (numpy bool, torch ``.to``, jax tracer-safe ``.astype``) stay out of this loop.
    """
    out = {}
    for key, fn_name, fields, gate in _TERMS:
        if key not in prepared:
            continue
        if gate == "dist" and not include_distance:
            continue
        p = prepared[key]
        mask = (
            p["mask"] * cg if gate == "conf" else sigma_gate(p["start_sigma"], p["mask"])
        )
        out[key] = fns[fn_name](positions, *[p[f] for f in fields], mask)
    return out


# the float keys energy_breakdown reports, in display order (all start at 0.0)
BREAKDOWN_KEYS = ("bond", "angle", "chiral", "dihedral", "vdw", "distance", "rmsd")
