"""Backend-agnostic restraint specification.

A ``RestraintSpec`` holds every restraint as padded NumPy arrays so that the
numpy / torch / jax energy backends can all consume the same data. All atom
indices stored here are *local* indices into ``active_sites`` (the subset of
atoms that participate in any restraint), not global padded atom indices.

The featurizer (``featurizer.py``) builds a ``RestraintSpec`` from global atom
indices; backends convert the NumPy arrays into their own tensor type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Distance restraint type codes (shared across numpy / torch / jax backends).
# These mirror the string types used in the YAML config.
# ---------------------------------------------------------------------------
DIST_HARMONIC = 0  # penalize (d - target1)^2 everywhere
DIST_FLAT_BOTTOMED = 1  # penalize d < target1 or d > target2
DIST_LOWER_BOUND = 2  # "flat-bottomed1": penalize d < target1 only
DIST_UPPER_BOUND = 3  # "flat-bottomed2": penalize d > target2 only

# Config string type -> integer code. target_distance maps to target1.
DIST_TYPE_CODES = {
    "harmonic": DIST_HARMONIC,
    "flat-bottomed": DIST_FLAT_BOTTOMED,
    "flat-bottomed1": DIST_LOWER_BOUND,
    "flat-bottomed2": DIST_UPPER_BOUND,
}

# Which group(s) the distance restraint's centroid gradient moves (the `move` config key).
# The minimal-displacement split ("both") rescales both group centroids by the reduced mass
# so they move with ratio N2:N1; move-1/move-2 pin the OTHER group (stop-gradient) so the
# full shift lands on group1 / group2 (e.g. pull only a ligand toward a fixed pocket). The
# code equals the config value: 1 -> group1, 2 -> group2 (atom_selection1 / atom_selection2);
# "both" -> 0. Consumed by ``energy.*_energy.distance_energy`` (the autodiff CG term).
MOVE_BOTH = 0
MOVE_GROUP1 = 1
MOVE_GROUP2 = 2


@dataclass
class BondArrays:
    """Flat-bottomed bond-length restraints (padded).

    idx are local indices into active_sites. A bond is penalized when the
    distance leaves [r0 - slack, r0 + slack]; if ``half`` is set only stretching
    beyond r0 + slack is penalized (used for inter-chain link bonds).
    """

    idx: np.ndarray  # (n_bond, 2) int
    r0: np.ndarray  # (n_bond,) float
    slack: np.ndarray  # (n_bond,)
    weight: np.ndarray  # (n_bond,)
    half: np.ndarray  # (n_bond,) float {0,1}: 1 = penalize stretch only
    mask: np.ndarray  # (n_bond,) float {0,1}: 1 = valid, 0 = padding


@dataclass
class AngleArrays:
    """Flat-bottomed bond-angle restraints (padded). Vertex is column 1."""

    idx: np.ndarray  # (n_angle, 3) int
    th0: np.ndarray  # (n_angle,) radians
    slack: np.ndarray  # (n_angle,)
    weight: np.ndarray  # (n_angle,)
    mask: np.ndarray  # (n_angle,)


@dataclass
class ChiralArrays:
    """Chiral-volume restraints (padded). Center atom is column 0."""

    idx: np.ndarray  # (n_chiral, 4) int
    vol0: np.ndarray  # (n_chiral,) reference scalar triple product
    slack: np.ndarray  # (n_chiral,)
    weight: np.ndarray  # (n_chiral,)
    mask: np.ndarray  # (n_chiral,)


@dataclass
class PlaneArrays:
    """Best-fit-plane restraints over arbitrary-size planar atom GROUPS (padded).

    Servalcat-style: each row is a set of atoms the reference conformer holds
    coplanar (an aromatic / conjugated ring, or a non-ring sp2 functional group =
    an sp2 centre with its heavy neighbours). The energy is the group's out-of-plane
    RMS deviation from its own best-fit plane (target 0 = planar), so it flattens
    aromatic rings the old per-centre signed-volume term could not (a ring CH has
    only 2 heavy neighbours in the H-removed mol). Variable group size ``A`` is
    encoded purely by ``grp_mask`` (padding columns zeroed), the same layout as
    ``DistanceArrays``' groups. The backend ``plane_energy`` leaf builds each group's
    masked centroid + covariance, takes the smallest-eigenvalue plane normal
    (stop-gradient, like ``rmsd_energy``'s Kabsch rotation) and penalises the residual.
    """

    idx: np.ndarray  # (n_plane, max_atoms) int local indices
    grp_mask: np.ndarray  # (n_plane, max_atoms) float {0,1}: 1 = real atom, 0 = padding
    slack: np.ndarray  # (n_plane,) Angstrom flat-bottom tolerance (0 = pure harmonic)
    weight: np.ndarray  # (n_plane,)
    mask: np.ndarray  # (n_plane,) float {0,1}: 1 = valid restraint, 0 = padding


@dataclass
class CisTransArrays:
    """Flat-bottomed cis/trans (E/Z) torsion restraints (padded). Atom order i-j-k-l;
    the rotatable bond axis is the j-k pair (columns 1-2).

    Holds each acyclic double bond at its input E/Z geometry: the target
    ``phi0`` is the reference-conformer torsion, so the bond keeps its input
    cis/trans configuration. The energy is periodicity-safe (the deviation from
    ``phi0`` is wrapped to [-pi, pi] before the flat-bottomed square penalty).
    """

    idx: np.ndarray  # (n_cistrans, 4) int
    phi0: np.ndarray  # (n_cistrans,) target torsion in radians
    slack: np.ndarray  # (n_cistrans,) radians
    weight: np.ndarray  # (n_cistrans,)
    mask: np.ndarray  # (n_cistrans,)


@dataclass
class VdwArrays:
    """VdW repulsion for non-bonded pairs (lower-bound only).

    The torch backend recomputes ``idx``/``r_min`` every step via a radius
    search; this struct carries the static fallback / jax form.
    """

    idx: np.ndarray  # (n_vdw, 2) int
    r_min: np.ndarray  # (n_vdw,)
    weight: np.ndarray  # (n_vdw,)
    mask: np.ndarray  # (n_vdw,)


@dataclass
class VdwConfig:
    """Dynamic fixed-background VdW repulsion for the torch + jax optimizers.

    The fixed-background half of the ``intermolecular`` VdW category. Ligand atoms
    move (they live in ``active_sites``, addressed by ``ligand_local``); the
    background atoms (every non-padding atom not optimised — protein, DNA/RNA, any
    non-restrained ligand) are read from the full coordinate tensor via
    ``background_global`` and held *fixed*. Each optimization step recomputes the clash
    penalty against the moving ligand, so only the ligand is pushed out of contacts,
    keeping the optimised variable set limited to ``active_sites``. The penalty is
    ``clamp(d - scale*(r_i+r_j), max=0)**2`` summed over pairs within ``dmax`` —
    identical maths to ``vdw_energy``, only the pair list is dynamic.
    """

    weight: float
    ligand_local: np.ndarray  # (n_lig,) local indices into active_sites (moving)
    ligand_radii: np.ndarray  # (n_lig,) VdW radius per ligand atom
    background_global: np.ndarray  # (n_bg,) global atom indices (fixed background)
    background_radii: np.ndarray  # (n_bg,) VdW radius per background atom
    scale: float = 0.75
    dmax: float = 5.0


@dataclass
class ActiveVdwConfig:
    """Dynamic active-active VdW neighbours involving restrained polymer atoms.

    A fixed-width nearest-neighbour list is rebuilt from the current coordinates once
    per diffusion step and held fixed during CG.  This keeps each energy evaluation
    O(N*K), while 1-2/1-3 covalent pairs are removed through ``excluded_codes``.
    """

    weight: float
    radii: np.ndarray  # (n_active,) VdW radius for every active atom
    polymer_mask: np.ndarray  # (n_active,) bool; pair needs at least one True endpoint
    excluded_codes: np.ndarray  # sorted canonical pair codes min(i,j)*N+max(i,j)
    scale: float = 0.75
    dmax: float = 5.0
    max_neighbors: int = 32


# Largest positive value an int32 can hold; the JAX pair-code encoding lives in int32.
_ACTIVE_VDW_INT32_MAX = 2**31 - 1


def check_active_vdw_int32_safe(n_active: int) -> None:
    """Guard the active-active VdW pair-code encoding against int32 overflow (JAX).

    Pair codes are ``min(i, j) * n_active + max(i, j)``. The JAX optimizer builds them
    (and the sorted ``excluded_codes``) as int32 because JAX runs with x64 disabled by
    default, so forcing int64 there is silently downcast back. Once ``n_active**2``
    exceeds the int32 range the codes wrap negative, which breaks the sortedness
    ``jnp.searchsorted`` relies on and silently corrupts the covalent 1-2/1-3 exclusion.
    Fail loudly instead. The torch optimizer uses int64 and is unaffected, so this bound
    only limits the restrained-polymer size under JAX/AF3. (Conservative: it trips at
    ``n_active**2 > 2**31 - 1``, a hair below the exact ``n**2 - n - 1`` maximum code.)
    """
    if n_active * n_active > _ACTIVE_VDW_INT32_MAX:
        raise ValueError(
            "active-active VdW (polymer conformer) supports at most ~46340 active "
            f"atoms under the JAX int32 pair-code encoding; got n_active={n_active}. "
            "Reduce the restrained polymer selection to run this under AF3/JAX (the "
            "torch tools are unaffected)."
        )


@dataclass
class DistanceArrays:
    """Centroid distance restraints between two atom groups (padded)."""

    grp1_idx: np.ndarray  # (n_dist, max_grp) int local indices
    grp2_idx: np.ndarray  # (n_dist, max_grp) int
    grp1_mask: np.ndarray  # (n_dist, max_grp) float {0,1}
    grp2_mask: np.ndarray  # (n_dist, max_grp)
    target1: np.ndarray  # (n_dist,) lower / harmonic target
    target2: np.ndarray  # (n_dist,) upper target
    dist_type: np.ndarray  # (n_dist,) int code (see DIST_* above)
    move_mode: np.ndarray  # (n_dist,) int 0=both / 1=grp1 only / 2=grp2 only (MOVE_*)
    # (n_dist,) relative strength: at full CG convergence a no-op for a single / disjoint
    # restraint (delta -> 0), only re-weights over-constrained coupled restraints
    weight: np.ndarray
    mask: np.ndarray  # (n_dist,)
    start_sigma: np.ndarray  # (n_dist,) per-restraint; active when sigma<=start_sigma
    stop_sigma: np.ndarray  # (n_dist,) released when sigma<stop_sigma (-1=never)
    # per-restraint STEP window (the alternative gate axis; ANDed with the sigma window):
    # active when start_step <= step <= stop_step. -inf/+inf = always (default).
    start_step: np.ndarray  # (n_dist,) step-window lower bound
    stop_step: np.ndarray  # (n_dist,) step-window upper bound


@dataclass
class RmsdArrays:
    """Kabsch-superposed RMSD restraints to a fixed reference (padded).

    The optimal rotation is computed from the FIT atoms (``fit_*``) and the RMSD is
    measured over the CALC atoms (``calc_*``); both reference groups are paired
    atom-for-atom to the moving target. When fit==calc this is the plain superposed
    RMSD. Energy ``weight * delta**2`` where ``delta`` is the distance-style flat-bottom
    deviation of the RMSD value (harmonic / flat-bottomed / flat-bottomed1 /
    flat-bottomed2), active in the noise window ``stop_sigma <= sigma <= start_sigma``.
    Optimised by the CG solver, so the fit+calc
    target atoms join active_sites. The rotation is recomputed (and
    treated as constant) each evaluation — see ``energy.*_energy.rmsd_energy``. All
    ``*_idx`` are local indices into active_sites.
    """

    fit_idx: np.ndarray  # (n_rmsd, max_fit) int local indices (superposition atoms)
    fit_mask: np.ndarray  # (n_rmsd, max_fit) float {0,1}
    fit_ref: np.ndarray  # (n_rmsd, max_fit, 3) reference fit coords (padded, constant)
    calc_idx: np.ndarray  # (n_rmsd, max_calc) int local indices (measured atoms)
    calc_mask: np.ndarray  # (n_rmsd, max_calc) float {0,1}
    calc_ref: np.ndarray  # (n_rmsd, max_calc, 3) reference calc coords (padded, const)
    target1: np.ndarray  # (n_rmsd,) lower / harmonic target RMSD (Angstrom)
    target2: np.ndarray  # (n_rmsd,) upper target RMSD (Angstrom; 0 if unused)
    geom_type: np.ndarray  # (n_rmsd,) int code (DIST_*: 0=harmonic .. 3=flat-bottomed2)
    weight: np.ndarray  # (n_rmsd,)
    start_sigma: np.ndarray  # (n_rmsd,) per-restraint; active when sigma<=start_sigma
    # per-restraint LOWER noise bound: the restraint is RELEASED for sigma < stop_sigma,
    # so the model's final low-sigma denoising steps re-idealise geometry the restraint
    # would otherwise hold distorted (e.g. the peptide bond between a restrained residue
    # and a free unmodeled tail). -1 = never released (active down to sigma=0 = old
    # behaviour). The active window is stop_sigma <= sigma <= start_sigma.
    stop_sigma: np.ndarray  # (n_rmsd,) per-restraint; active when sigma>=stop_sigma
    # per-restraint STEP window (ANDed with the sigma window); -inf/+inf = always.
    start_step: np.ndarray  # (n_rmsd,) step-window lower bound
    stop_step: np.ndarray  # (n_rmsd,) step-window upper bound
    mask: np.ndarray  # (n_rmsd,) float {0,1}: 1 = valid, 0 = padding


@dataclass
class GroupAngleArrays:
    """Centroid angle restraints between three atom groups (padded).

    The angle is formed by the three groups' centroids with group 2 as the
    vertex (``centroid1 - centroid2 - centroid3``), mirroring ``AngleArrays`` (column-1 vertex) but
    on group centroids. The penalty mirrors the distance restraint's four ``geom_type``
    codes (harmonic / flat-bottomed / lower / upper) on the angle value, with
    ``target1``/``target2`` the bound(s) in radians. ``move_free`` is a per-group mask
    (1 = free, 0 = pinned): a pinned group's centroid is stop-gradient'd so the CG holds it
    fixed for this term. Optimised by the CG solver (NOT closed-form like
    distance), so the group atoms join active_sites; the centroid-only energy gradient is
    uniform across each free group, so the solver translates it rigidly. Active window
    ``stop_sigma <= sigma <= start_sigma``. All ``*_idx`` are local indices into
    active_sites; padding columns are neutralised by ``grp*_mask``.
    """

    grp1_idx: np.ndarray  # (n_grp_angle, max_grp) int local indices
    grp2_idx: np.ndarray  # (n_grp_angle, max_grp) int (vertex group)
    grp3_idx: np.ndarray  # (n_grp_angle, max_grp) int
    grp1_mask: np.ndarray  # (n_grp_angle, max_grp) float {0,1}
    grp2_mask: np.ndarray  # (n_grp_angle, max_grp)
    grp3_mask: np.ndarray  # (n_grp_angle, max_grp)
    target1: np.ndarray  # (n_grp_angle,) radians; harmonic target / flat-bottomed lower
    target2: np.ndarray  # (n_grp_angle,) radians; flat-bottomed upper (0 if unused)
    geom_type: np.ndarray  # (n_grp_angle,) int code (DIST_* : 0=harmonic..3=upper)
    move_free: np.ndarray  # (n_grp_angle, 3) {0,1}: 1 = group free to move
    weight: np.ndarray  # (n_grp_angle,)
    mask: np.ndarray  # (n_grp_angle,) float {0,1}: 1 = valid restraint, 0 = padding
    start_sigma: np.ndarray  # (n_grp_angle,) per-restraint; active when sigma<=start
    stop_sigma: np.ndarray  # (n_grp_angle,) released when sigma<stop_sigma (-1=never)
    # per-restraint STEP window (ANDed with the sigma window); -inf/+inf = always.
    start_step: np.ndarray  # (n_grp_angle,) step-window lower bound
    stop_step: np.ndarray  # (n_grp_angle,) step-window upper bound


@dataclass
class GroupDihedralArrays:
    """Centroid dihedral restraints between four atom groups (padded).

    The dihedral is formed by the four groups' centroids with the ``centroid2-centroid3``
    line as the rotatable axis (``centroid1-centroid2-centroid3-centroid4``), mirroring ``CisTransArrays``
    (i-j-k-l) but on group centroids. The penalty mirrors the distance restraint's four
    ``geom_type`` codes with ``target1``/``target2`` in radians. The ``harmonic`` code
    is periodicity-safe (the deviation ``phi - target1`` is wrapped to [-pi, pi] before
    the square, so +179 deg and -179 deg read as 2 deg apart); the flat-bottomed / lower
    / upper codes use the raw angle with ``target1 < target2`` enforced, so a window
    cannot straddle +-180 deg (use ``harmonic`` for a target near +-180). ``move_free``
    is a per-group mask as in ``GroupAngleArrays`` (1 = free, 0 = pinned via
    stop-gradient). CG-solved (rigid per free group); active window
    ``stop_sigma <= sigma <= start_sigma``. All ``*_idx`` are local indices into
    active_sites; padding columns are neutralised by ``grp*_mask``.
    """

    grp1_idx: np.ndarray  # (n_grp_dih, max_grp) int local indices
    grp2_idx: np.ndarray  # (n_grp_dih, max_grp) int (axis atom 1)
    grp3_idx: np.ndarray  # (n_grp_dih, max_grp) int (axis atom 2)
    grp4_idx: np.ndarray  # (n_grp_dih, max_grp) int
    grp1_mask: np.ndarray  # (n_grp_dih, max_grp) float {0,1}
    grp2_mask: np.ndarray  # (n_grp_dih, max_grp)
    grp3_mask: np.ndarray  # (n_grp_dih, max_grp)
    grp4_mask: np.ndarray  # (n_grp_dih, max_grp)
    target1: np.ndarray  # (n_grp_dih,) radians; harmonic target / flat-bottomed lower
    target2: np.ndarray  # (n_grp_dih,) radians; flat-bottomed upper (0 if unused)
    geom_type: np.ndarray  # (n_grp_dih,) int code (DIST_* : 0=harmonic..3=upper)
    move_free: np.ndarray  # (n_grp_dih, 4) {0,1}: 1 = group free to move
    weight: np.ndarray  # (n_grp_dih,)
    mask: np.ndarray  # (n_grp_dih,) float {0,1}: 1 = valid restraint, 0 = padding
    start_sigma: np.ndarray  # (n_grp_dih,) per-restraint; active when sigma<=start
    stop_sigma: np.ndarray  # (n_grp_dih,) released when sigma<stop_sigma (-1=never)
    # per-restraint STEP window (ANDed with the sigma window); -inf/+inf = always.
    start_step: np.ndarray  # (n_grp_dih,) step-window lower bound
    stop_step: np.ndarray  # (n_grp_dih,) step-window upper bound


@dataclass
class GroupImproperArrays(GroupDihedralArrays):
    """Centroid improper restraints between four ordered atom groups.

    The array layout and measured signed torsion are identical to
    :class:`GroupDihedralArrays`; the separate type preserves restraint identity in
    configuration, dispatch, gating, and diagnostics.
    """


@dataclass
class GroupPlaneArrays:
    """Standalone best-fit-plane restraints over selection-resolved atom groups (padded).

    The measured quantity is identical to ``PlaneArrays`` (the group's out-of-plane RMS
    deviation from its own best-fit plane, smallest-eigenvalue normal, stop-gradient), but
    this is the ``plane_restraints_config`` term rather than a conformer sub-term, so:

      * the penalty mirrors the distance restraint's four ``geom_type`` codes with
        ``target1``/``target2`` in ANGSTROM (``PlaneArrays`` has only a one-sided
        ``slack``), and
      * the gate is PER ENTRY (``stop_sigma <= sigma <= start_sigma`` ANDed with the step
        window) instead of the shared conformer gate.

    One entry may pool SEVERAL selection groups into one plane (a shared best-fit plane,
    e.g. two stacked nucleobases), so ``free`` — the ``move`` mask — is per-ATOM here
    rather than per-group as in ``GroupAngleArrays``: the number of groups varies per
    entry, and the energy only ever sees the pooled atom list. A pinned atom keeps its
    value in the plane fit but is stop-gradient'd, so the CG does not move it for this
    restraint. ``idx`` holds local indices into active_sites; padding columns are
    neutralised by ``grp_mask``.
    """

    idx: np.ndarray  # (n_grp_plane, max_atoms) int local indices
    grp_mask: np.ndarray  # (n_grp_plane, max_atoms) float {0,1}: 1 = real, 0 = padding
    free: np.ndarray  # (n_grp_plane, max_atoms) {0,1}: 1 = atom free to move
    target1: (
        np.ndarray
    )  # (n_grp_plane,) Angstrom; harmonic target / flat-bottomed lower
    target2: np.ndarray  # (n_grp_plane,) Angstrom; flat-bottomed upper (0 if unused)
    geom_type: np.ndarray  # (n_grp_plane,) int code (DIST_* : 0=harmonic..3=upper)
    weight: np.ndarray  # (n_grp_plane,)
    mask: np.ndarray  # (n_grp_plane,) float {0,1}: 1 = valid restraint, 0 = padding
    start_sigma: np.ndarray  # (n_grp_plane,) per-restraint; active when sigma<=start
    stop_sigma: np.ndarray  # (n_grp_plane,) released when sigma<stop_sigma (-1=never)
    # per-restraint STEP window (ANDed with the sigma window); -inf/+inf = always.
    start_step: np.ndarray  # (n_grp_plane,) step-window lower bound
    stop_step: np.ndarray  # (n_grp_plane,) step-window upper bound


@dataclass
class RestraintSpec:
    """Backend-agnostic restraint definition.

    All idx fields in the sub-arrays are *local* indices into ``active_sites``.
    Optimization runs only on ``active_sites`` and scatters results back.
    """

    n_active: int
    active_sites: np.ndarray  # (n_active,) global flat atom indices
    bond: BondArrays | None = None
    angle: AngleArrays | None = None
    chiral: ChiralArrays | None = None
    # best-fit-plane restraints over planar atom groups (aromatic/conjugated rings +
    # non-ring sp2 groups); out-of-plane RMS deviation, target 0 (see PlaneArrays).
    plane: PlaneArrays | None = None
    cistrans: CisTransArrays | None = None
    vdw: VdwArrays | None = None
    vdw_config: VdwConfig | None = None
    active_vdw_config: ActiveVdwConfig | None = None
    distance: DistanceArrays | None = None
    rmsd: RmsdArrays | None = None
    # centroid angle/dihedral restraints between atom GROUPS (distinct from the conformer
    # angle/cistrans above, which act on single ligand atoms). Each carries its own
    # per-restraint start_sigma/stop_sigma like distance/rmsd.
    group_angle: GroupAngleArrays | None = None
    group_dihedral: GroupDihedralArrays | None = None
    group_improper: GroupImproperArrays | None = None
    # standalone best-fit-plane restraints over selection-resolved groups
    # (plane_restraints_config) — same measured quantity as `plane` above but with the four
    # distance-style types and a per-entry gate (see GroupPlaneArrays).
    group_plane: GroupPlaneArrays | None = None
    # one start_sigma for ALL conformer (bond/angle/chiral/plane/cistrans/vdw)
    # restraints; each distance restraint carries its own in DistanceArrays.start_sigma.
    # NOTE: this internal spec-field default stays -1.0 (= conformer OFF) on purpose,
    # unlike the user-facing build_spec/config defaults (+inf = active every step):
    # build_spec ALWAYS sets this field, so the default only applies to a hand-built spec
    # that omits it, where failing OFF is the conservative choice.
    conf_start_sigma: float = -1.0
    # shared conformer LOWER bound (mirrors conf_start_sigma): conformer terms are
    # released for sigma < conf_stop_sigma. -1 (default) = never released (off).
    conf_stop_sigma: float = -1.0
    # shared conformer STEP window (the alternative gate axis; ANDed with the sigma
    # window). Active for conf_start_step <= step <= conf_stop_step; -inf/+inf = always.
    conf_start_step: float = float("-inf")
    conf_stop_step: float = float("inf")
    # custom restraints (rgi_utils.custom): a list of CustomSpec (backend-agnostic — local
    # index arrays + a config-formula AST or a code energy fn + weight/sigmas), each
    # compiled to a closure and added to the CG objective by the optimizers. NOT numpy
    # arrays (Python AST/fn live here), so this is a separate field, not part of the
    # array-based terms above. Empty when no custom restraint is configured.
    custom: list = field(default_factory=list)

    def has_conformer(self) -> bool:
        """True if any conformer (bond/angle/chiral/plane/cistrans/vdw) restraint
        exists."""
        if self.vdw_config is not None and self.vdw_config.weight > 0:
            return True
        if self.active_vdw_config is not None and self.active_vdw_config.weight > 0:
            return True
        return any(
            arr is not None and arr.mask.sum() > 0
            for arr in (
                self.bond,
                self.angle,
                self.chiral,
                self.plane,
                self.cistrans,
                self.vdw,
            )
        )

    def has_distance(self) -> bool:
        """True if any distance restraint exists."""
        return self.distance is not None and self.distance.mask.sum() > 0

    def has_rmsd(self) -> bool:
        """True if any RMSD restraint exists. Like conformer, RMSD is optimised by
        the CG solver (not closed-form), so the solver must run when this is True."""
        return self.rmsd is not None and self.rmsd.mask.sum() > 0

    def has_group_angle(self) -> bool:
        """True if any group-centroid angle restraint exists. CG-solved (not closed-form),
        so the solver must run when this is True."""
        return self.group_angle is not None and self.group_angle.mask.sum() > 0

    def has_group_dihedral(self) -> bool:
        """True if any group-centroid dihedral restraint exists. CG-solved (not
        closed-form), so the solver must run when this is True."""
        return self.group_dihedral is not None and self.group_dihedral.mask.sum() > 0

    def has_group_improper(self) -> bool:
        """True if any group-centroid improper restraint exists."""
        return self.group_improper is not None and self.group_improper.mask.sum() > 0

    def has_group_plane(self) -> bool:
        """True if any standalone best-fit-plane restraint exists. CG-solved (not
        closed-form), so the solver must run when this is True. Distinct from the
        conformer ``plane`` term, which ``has_conformer()`` covers."""
        return self.group_plane is not None and self.group_plane.mask.sum() > 0

    def has_custom(self) -> bool:
        """True if any custom restraint (rgi_utils.custom) is configured. Custom restraints
        are CG-solved closures added to the objective, so the solver must run when True."""
        return len(self.custom) > 0

    def is_active(self) -> bool:
        """True if there is any work to do."""
        return self.n_active > 0 and (
            self.has_conformer()
            or self.has_distance()
            or self.has_rmsd()
            or self.has_group_angle()
            or self.has_group_dihedral()
            or self.has_group_plane()
            or self.has_group_improper()
            or self.has_custom()
        )

    def max_start_sigma(self) -> float:
        """Largest start_sigma over all active restraints. The optimizer can skip
        a step entirely when the noise level exceeds this (nothing is active yet)."""
        vals = []
        if self.has_conformer():
            vals.append(float(self.conf_start_sigma))
        if self.has_distance() and self.distance is not None:
            vals.append(float(np.max(self.distance.start_sigma)))
        if self.has_rmsd() and self.rmsd is not None:
            vals.append(float(np.max(self.rmsd.start_sigma)))
        if self.has_group_angle() and self.group_angle is not None:
            vals.append(float(np.max(self.group_angle.start_sigma)))
        if self.has_group_dihedral() and self.group_dihedral is not None:
            vals.append(float(np.max(self.group_dihedral.start_sigma)))
        if self.has_group_improper() and self.group_improper is not None:
            vals.append(float(np.max(self.group_improper.start_sigma)))
        if self.has_group_plane() and self.group_plane is not None:
            vals.append(float(np.max(self.group_plane.start_sigma)))
        for c in self.custom:
            vals.append(float(c.start_sigma))
        return max(vals) if vals else -1.0
