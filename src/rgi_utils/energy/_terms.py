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
    # planarity of sp2 double-bond centres: same fields as chiral (it reuses
    # chiral_energy), only target vol0 ~ 0 differs. See _TERMS / spec.PlanarityArrays.
    (
        "planarity",
        "planarity",
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
            ("weight", "f"),
            ("mask", "f"),
            ("start_sigma", "f"),
            ("stop_sigma", "f"),
            ("start_step", "f"),
            ("stop_step", "f"),
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
            ("target1", "f"),
            ("target2", "f"),
            ("geom_type", "i"),
            ("weight", "f"),
            ("mask", "f"),
            ("start_sigma", "f"),
            ("stop_sigma", "f"),
            ("start_step", "f"),
            ("stop_step", "f"),
        ],
    ),
    # group-centroid angle (3 groups, vertex = group 2) / dihedral (4 groups, axis =
    # group2-group3): each group is a padded (n, max_grp) index list + {0,1} mask (same
    # layout as distance). target1/target2/geom_type mirror distance's four
    # types; move_free ((n,n_groups) {0,1}) drives the energy's detach-select — like
    # distance's move_mode, both now CG-solved in the energy leaf.
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
            ("start_step", "f"),
            ("stop_step", "f"),
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
            ("start_step", "f"),
            ("stop_step", "f"),
        ],
    ),
]


def pack_spec(spec, to_int, to_float):
    """Convert a ``RestraintSpec`` into a backend's prepared dict via the two given
    converters (numpy array -> backend array). Only sub-arrays with a non-empty mask
    are included, matching the original per-backend ``prepare_spec`` gating."""
    conv = {"i": to_int, "f": to_float}
    prepared = {
        "conf_start_sigma": float(getattr(spec, "conf_start_sigma", -1.0)),
        "conf_stop_sigma": float(getattr(spec, "conf_stop_sigma", -1.0)),
        # shared conformer STEP window (the alternative gate axis); -inf/+inf = always.
        "conf_start_step": float(getattr(spec, "conf_start_step", float("-inf"))),
        "conf_stop_step": float(getattr(spec, "conf_stop_step", float("inf"))),
    }
    for key, attr, fields in _SPEC_SCHEMA:
        arr = getattr(spec, attr, None)
        if arr is None or not (
            arr.mask.sum() > 0
        ):  # same as old `> 0` include, NaN-safe
            continue
        prepared[key] = {f: conv[kind](getattr(arr, f)) for f, kind in fields}
    return prepared


# (prepared_key, leaf_fn_name, [arg_field, ...], gate). Every leaf fn takes
# ``(positions, *args, mask)`` with mask LAST. gate:
#   "conf" -> conformer term, masked by the shared conformer gate ``cg``;
#   "distance" -> centroid distance, per-restraint sigma/step gate, ALWAYS summed in the
#                 CG solver (distance is now an autodiff CG term: its energy rescales the
#                 centroid gradient so the group translates rigidly — see
#                 ``*_energy.distance_energy`` + ``_move_centroid``);
#   "rmsd" -> RMSD, per-restraint sigma gate, ALWAYS;
#   "group" -> group-centroid angle/dihedral, per-restraint sigma gate, ALWAYS (CG-solved
#              like rmsd). Any gate other than conf gets per-entry gating + is always
#              summed in the solver; such keys are collected in PER_ENTRY_KEYS.
_TERMS = [
    ("bond", "bond_energy", ["idx", "r0", "slack", "weight", "half"], "conf"),
    ("angle", "angle_energy", ["idx", "th0", "slack", "weight"], "conf"),
    ("chiral", "chiral_energy", ["idx", "vol0", "slack", "weight"], "conf"),
    # planarity REUSES the chiral signed-volume leaf fn (target vol0 ~ 0); its
    # own prepared key keeps it a separate, independently-weighted/reported term.
    ("planarity", "chiral_energy", ["idx", "vol0", "slack", "weight"], "conf"),
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
            "move_mode",
            "weight",
        ],
        "distance",
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
            "target1",
            "target2",
            "geom_type",
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


def term_energies(fns, prepared, positions, cg, sigma_gate):
    """Return ``{key: energy}`` for every active restraint term, with the right gate
    folded into each term's mask. Shared by all three backends' total_energy and
    energy_breakdown — only ``fns`` (the backend leaf functions), ``cg`` (the
    conformer gate multiplier) and ``sigma_gate`` (a per-restraint
    ``(start_sigma, stop_sigma, start_step, stop_step, mask) -> gated_mask`` callable;
    any bound may be ``None`` = no bound on that side) vary per backend / noise level.
    The gate is the AND of the sigma window AND the step window — a restraint uses one or
    the other (mutually exclusive at config time); the unused axis is always-on by default.

    ``cg``/``sigma_gate`` are precomputed by the caller so the backend-specific casts
    (numpy bool, torch ``.to``, jax tracer-safe ``.astype``) stay out of this loop.
    """
    out = {}
    for key, fn_name, fields, gate in _TERMS:
        if key not in prepared:
            continue
        p = prepared[key]
        # Per-entry terms (distance/rmsd/group) carry both a sigma window (start_sigma,
        # stop_sigma) and a step window (start_step, stop_step); the gate ANDs them.
        # Conformer terms instead use the shared conformer gate ``cg`` (which already
        # folds the conf sigma+step windows). ``p.get`` keeps a hand-built dict lacking
        # the step keys working (treated as no step bound).
        mask = (
            p["mask"] * cg
            if gate == "conf"
            else sigma_gate(
                p["start_sigma"],
                p.get("stop_sigma"),
                p.get("start_step"),
                p.get("stop_step"),
                p["mask"],
            )
        )
        out[key] = fns[fn_name](positions, *[p[f] for f in fields], mask)
    return out


# the float keys energy_breakdown reports, in display order (all start at 0.0)
BREAKDOWN_KEYS = (
    "bond",
    "angle",
    "chiral",
    "planarity",
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

# per-restraint-gated term keys (gate != "conf"): each carries its own
# start_sigma/stop_sigma (and start_step/stop_step) and is ALWAYS summed in the solver's
# energy. The torch GPU pre-gate (torch_optim._gated_prepared) folds each one's gate into
# its mask and keys its cache on every gate state, so adding a per-entry term to _TERMS
# can't silently leave it ungated on the compiled GPU path. distance is now per-entry
# (autodiff CG term, no longer closed-form); conformer is in CONF_KEYS.
PER_ENTRY_KEYS = frozenset(key for key, _fn, _fields, gate in _TERMS if gate != "conf")
