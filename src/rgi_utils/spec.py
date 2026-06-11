"""Backend-agnostic restraint specification.

A ``RestraintSpec`` holds every restraint as padded NumPy arrays so that the
numpy / torch / jax energy backends can all consume the same data. All atom
indices stored here are *local* indices into ``active_sites`` (the subset of
atoms that participate in any restraint), not global padded atom indices.

The featurizer (``featurizer.py``) builds a ``RestraintSpec`` from global atom
indices; backends convert the NumPy arrays into their own tensor type.
"""

from __future__ import annotations

from dataclasses import dataclass

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
class DihedralArrays:
    """Flat-bottomed dihedral (torsion) restraints (padded). Atom order i-j-k-l;
    the rotatable bond axis is the j-k pair (columns 1-2).

    Used for cis/trans (E/Z) preservation of acyclic double bonds: the target
    ``phi0`` is the reference-conformer dihedral, so the bond is held at its
    input E/Z geometry. The energy is periodicity-safe (the deviation from
    ``phi0`` is wrapped to [-pi, pi] before the flat-bottomed square penalty).
    """

    idx: np.ndarray  # (n_dihedral, 4) int
    phi0: np.ndarray  # (n_dihedral,) target dihedral in radians
    slack: np.ndarray  # (n_dihedral,) radians
    weight: np.ndarray  # (n_dihedral,)
    mask: np.ndarray  # (n_dihedral,)


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
    """Dynamic ligand-protein VdW repulsion for the torch optimizer.

    Ligand atoms move (they live in ``active_sites``, addressed by
    ``ligand_local``); protein atoms are a *fixed background* read from the full
    coordinate tensor via ``protein_global``. Each optimization step recomputes
    the ligand-protein clash penalty against the moving ligand, so only the
    ligand is pushed out of contacts (protein is held fixed). This matches
    boltz's ``ligand_only`` VdW behaviour while keeping the optimised variable
    set limited to ``active_sites``. The penalty is ``clamp(d - scale*(r_i+r_j),
    max=0)**2`` summed over pairs within ``dmax`` — identical maths to
    ``vdw_energy``, only the pair list is dynamic.
    """

    weight: float
    ligand_local: np.ndarray  # (n_lig,) local indices into active_sites (moving)
    ligand_radii: np.ndarray  # (n_lig,) VdW radius per ligand atom
    protein_global: np.ndarray  # (n_prot,) global atom indices (fixed background)
    protein_radii: np.ndarray  # (n_prot,) VdW radius per protein atom
    scale: float = 0.75
    dmax: float = 5.0


@dataclass
class DistanceArrays:
    """COM distance restraints between two atom groups (padded)."""

    grp1_idx: np.ndarray  # (n_dist, max_grp) int local indices
    grp2_idx: np.ndarray  # (n_dist, max_grp) int
    grp1_mask: np.ndarray  # (n_dist, max_grp) float {0,1}
    grp2_mask: np.ndarray  # (n_dist, max_grp)
    target1: np.ndarray  # (n_dist,) lower / harmonic target
    target2: np.ndarray  # (n_dist,) upper target
    dist_type: np.ndarray  # (n_dist,) int code (see DIST_* above)
    mask: np.ndarray  # (n_dist,)
    start_sigma: np.ndarray  # (n_dist,) per-restraint; active when sigma<=start_sigma
    stop_sigma: np.ndarray  # (n_dist,) released when sigma<stop_sigma (-1=never)


@dataclass
class RmsdArrays:
    """Kabsch-superposed RMSD restraints to a fixed reference (padded).

    The optimal rotation is computed from the FIT atoms (``fit_*``) and the RMSD is
    measured over the CALC atoms (``calc_*``); both reference groups are paired
    atom-for-atom to the moving target. When fit==calc this is the plain superposed
    RMSD. Energy ``weight * (rmsd - target_rmsd)**2``, active in the noise window
    ``stop_sigma <= sigma <= start_sigma``. Optimised by the CG solver, so the fit+calc
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
    target_rmsd: np.ndarray  # (n_rmsd,) target RMSD value
    weight: np.ndarray  # (n_rmsd,)
    start_sigma: np.ndarray  # (n_rmsd,) per-restraint; active when sigma<=start_sigma
    # per-restraint LOWER noise bound: the restraint is RELEASED for sigma < stop_sigma,
    # so the model's final low-sigma denoising steps re-idealise geometry the restraint
    # would otherwise hold distorted (e.g. the peptide bond between a restrained residue
    # and a free unmodeled tail). -1 = never released (active down to sigma=0 = old
    # behaviour). The active window is stop_sigma <= sigma <= start_sigma.
    stop_sigma: np.ndarray  # (n_rmsd,) per-restraint; active when sigma>=stop_sigma
    mask: np.ndarray  # (n_rmsd,) float {0,1}: 1 = valid, 0 = padding


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
    dihedral: DihedralArrays | None = None
    vdw: VdwArrays | None = None
    vdw_config: VdwConfig | None = None
    distance: DistanceArrays | None = None
    rmsd: RmsdArrays | None = None
    # one start_sigma for ALL conformer (bond/angle/chiral/vdw) restraints; each
    # distance restraint carries its own in DistanceArrays.start_sigma.
    conf_start_sigma: float = -1.0
    # shared conformer LOWER bound (mirrors conf_start_sigma): conformer terms are
    # released for sigma < conf_stop_sigma. -1 (default) = never released (off).
    conf_stop_sigma: float = -1.0

    def has_conformer(self) -> bool:
        """True if any conformer (bond/angle/chiral/dihedral/vdw) restraint exists."""
        if self.vdw_config is not None and self.vdw_config.weight > 0:
            return True
        return any(
            arr is not None and arr.mask.sum() > 0
            for arr in (self.bond, self.angle, self.chiral, self.dihedral, self.vdw)
        )

    def has_distance(self) -> bool:
        """True if any distance restraint exists."""
        return self.distance is not None and self.distance.mask.sum() > 0

    def has_rmsd(self) -> bool:
        """True if any RMSD restraint exists. Like conformer, RMSD is optimised by
        the CG solver (not closed-form), so the solver must run when this is True."""
        return self.rmsd is not None and self.rmsd.mask.sum() > 0

    def is_active(self) -> bool:
        """True if there is any work to do."""
        return self.n_active > 0 and (
            self.has_conformer() or self.has_distance() or self.has_rmsd()
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
        return max(vals) if vals else -1.0
