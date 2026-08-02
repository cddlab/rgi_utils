"""Parse + resolve standalone best-fit-plane restraints (``plane_restraints_config``).

This is the selection-driven form of the ``plane`` energy term that until now existed
only inside the conformer restraint (where groups are *perceived* from an RDKit mol —
aromatic rings and non-ring sp2 groups — or supplied by the monomer library / polymer
links). Here the user names the atoms with the selection DSL, so any atom group in the
structure (protein, DNA/RNA, ligand) can be held planar.

The measured quantity is the group's out-of-plane RMS deviation from its own best-fit
plane (Angstrom, target 0 = planar), shaped by the same four restraint types as the
distance / angle / dihedral restraints: ``harmonic`` / ``flat-bottomed`` /
``flat-bottomed1`` / ``flat-bottomed2`` with ``target_plane`` / ``target_plane1`` /
``target_plane2``. Unlike those, the type block is **optional**: the target of a plane
restraint is essentially always 0, so an omitted block means ``harmonic`` with
``target_plane: 0.0``. (This is also what lets the base-pair coplanarity macro express
its ``coplanar_slack`` — 0 becomes a pure harmonic, a positive slack becomes
``flat-bottomed2``.)

Several groups in ONE entry are **pooled into a single best-fit plane** — that is the
"keep these two bases coplanar" idiom (the base-pair macro's coplanarity group). Give
separate entries for separate planes. Targets are Angstrom only, so there is no ``unit``
key.

``move`` selects which groups the solver may move; atoms of the other groups still
contribute their position to the plane fit but are stop-gradient'd, so the CG holds them
fixed FOR THIS restraint (another restraint sharing those atoms still moves them — the
same semantics as the distance / angle / dihedral ``move``). The DEFAULT is **every group
free**, unlike the angle/dihedral default (arms free, anchor pinned): a plane has no
anchor group to pin.

Each entry is gated on EITHER a sigma window (``start_sigma`` / ``stop_sigma``) OR a step
window (``start_step`` / ``stop_step``) — mutually exclusive, enforced by the shared
``apply_window_params`` -> ``check_window_exclusive``.

A reference-anchored plane (``refN and <selection>`` + an entry-local ``refs`` map) is a
DIFFERENT energy — the prediction atoms are pulled onto a plane fitted from the external
structure — so ``config.py`` routes those entries to ``ref_geom_restr_data.RefGeomData``
(a ``ref_geom`` custom closure) instead of this class.
"""

from __future__ import annotations

import logging

from rgi_utils._config_util import (
    apply_window_params,
    parse_geom_type,
    parse_move_indices,
    warn_unknown_keys,
)
from rgi_utils.atom_context import FrameworkAdapter
from rgi_utils.group_geom_restr_data import resolve_group_sites

logger = logging.getLogger(__name__)

# Highest ``atom_selectionN`` a plane entry may carry. Matches the range probed by
# ``ref_geom_restr_data.is_ref_anchored`` so a reference group can sit in any slot.
MAX_PLANE_GROUPS = 4

# A best-fit plane needs at least 3 atoms to define a normal at all (3 are trivially
# coplanar, so the restraint is a no-op then — but an explicit user selection of 3 is not
# an error, unlike the conformer term's perceived groups, which require >= 4 because a
# trivially-planar group exerts no force and would only pad the arrays).
_MIN_PLANE_ATOMS = 3

_KNOWN_PLANE_KEYS = {
    "weight",
    "start_sigma",
    "stop_sigma",
    "start_step",
    "stop_step",
    "move",
    "harmonic",
    "flat-bottomed",
    "flat-bottomed1",
    "flat-bottomed2",
} | {f"atom_selection{i}" for i in range(1, MAX_PLANE_GROUPS + 1)}


def count_plane_groups(
    config: dict, label: str = "plane_restraints_config entry"
) -> int:
    """Number of ``atom_selection1..N`` keys a plane entry carries.

    The count must start at 1 and be CONTIGUOUS: an entry with ``atom_selection1`` and
    ``atom_selection3`` is a typo (the missing slot would be silently dropped, and for a
    reference-anchored entry it would shift which group the ``move`` indices name), so it
    raises. Shared with ``config.py``, which needs the count BEFORE constructing
    ``RefGeomData("plane", n_groups=...)`` — that keeps ``RefGeomData.n_groups`` a plain
    int instead of a variable-count sentinel every one of its five uses would have to
    guard.
    """
    present = [
        i
        for i in range(1, MAX_PLANE_GROUPS + 1)
        if isinstance(config.get(f"atom_selection{i}"), str)
        and config[f"atom_selection{i}"].strip()
    ]
    if not present:
        raise ValueError(f"{label}: atom_selection1 is required and must be a string")
    expected = list(range(1, len(present) + 1))
    if present != expected:
        raise ValueError(
            f"{label}: atom_selection keys must be numbered contiguously from 1 "
            f"(got {present}, expected {expected})"
        )
    return len(present)


class PlaneRestraintData:
    """One standalone best-fit-plane restraint entry (1..4 groups pooled into one plane).

    ``target1``/``target2`` are Angstrom out-of-plane RMS bounds; ``geom_type`` is one of
    the four distance-style types (defaulting to ``harmonic`` toward 0 when the entry
    carries no type block). ``start_sigma`` defaults to None here;
    ``RestraintsConfig.from_dict`` turns an omitted value into +inf.
    """

    atom_selections: list  # the entry's group selection strings, in order
    target1: float  # Angstrom
    target2: float  # Angstrom
    geom_type: str | None  # harmonic / flat-bottomed / flat-bottomed1 / flat-bottomed2
    move_free: tuple  # per-group "free to move" bools (the rest are pinned)
    weight: float
    target_sites: (
        list  # per-group lists of global atom indices (filled by resolve_sites)
    )
    run_restr: bool
    start_sigma: float
    stop_sigma: float
    start_step: float  # step-window lower bound (-inf = always); XOR the sigma window
    stop_step: float  # step-window upper bound (+inf = always)

    def __init__(self):
        self.atom_selections = []
        self.target1 = None
        self.target2 = None
        self.geom_type = None
        self.move_free = None  # set by set_config (default: every group free)
        self.weight = 1.0
        self.target_sites = None
        self.run_restr = None
        self.start_sigma = None  # from_dict defaults None -> +inf (every step)
        self.stop_sigma = -1.0  # never released (active down to sigma=0)
        self.start_step = float(
            "-inf"
        )  # step-window (omitted -> always); XOR sigma win
        self.stop_step = float("inf")

    def set_config(self, config: dict):
        label = "plane_restraints_config entry"
        warn_unknown_keys(config, _KNOWN_PLANE_KEYS, label, logger)
        n_groups = count_plane_groups(config, label)
        self.atom_selections = [
            config[f"atom_selection{i}"] for i in range(1, n_groups + 1)
        ]
        # weight + the mutually-exclusive sigma/step gate windows (shared parse).
        apply_window_params(self, config, label)
        # move: default = every group free (a plane has no anchor group to pin, unlike
        # the angle vertex / dihedral axis).
        idx = parse_move_indices(config.get("move"), n_groups)
        self.move_free = (
            tuple(True for _ in range(n_groups))
            if idx is None
            else tuple((g + 1) in idx for g in range(n_groups))
        )
        # Targets are Angstrom (no `unit` key), so conv is plain float. An omitted type
        # block means harmonic toward 0 — see the module docstring.
        self.geom_type, self.target1, self.target2 = parse_geom_type(
            config, "target_plane", float
        )
        if self.geom_type is None:
            self.geom_type, self.target1, self.target2 = "harmonic", 0.0, 0.0
        self.run_restr = True

    def resolve_sites(self, adapter: FrameworkAdapter) -> None:
        if not self.run_restr:
            return
        self.target_sites = resolve_group_sites(adapter, self.atom_selections)
        n_atoms = sum(len(s) for s in self.target_sites)
        if n_atoms < _MIN_PLANE_ATOMS:
            raise ValueError(
                f"plane restraint pooled only {n_atoms} atom(s); a best-fit plane needs "
                f"at least {_MIN_PLANE_ATOMS}"
            )
        logger.info(
            "plane restraint resolved: %d group(s) %s = %d atoms",
            len(self.target_sites),
            "/".join(str(len(s)) for s in self.target_sites),
            n_atoms,
        )

    def iter_global_sites(self):
        """Yield every resolved global coordinate index used by this restraint."""
        for group in self.target_sites or ():
            yield from group

    def is_valid(self) -> bool:
        return self.run_restr
