"""Reference-anchored built-in geometry restraints (distance / angle / dihedral).

A ``distance_restraints_config`` / ``angle_restraints_config`` / ``dihedral_restraints_config``
entry that carries REFERENCE keys — ``ref_pdb`` / ``ref_cif``, and/or any ``atom_selectionN_ref``
— is *reference-anchored*: the reference structure is Kabsch-fit onto the prediction
(``atom_selection_target_fit`` on the prediction, ``atom_selection_ref_fit`` on the reference),
then the measured distance / angle / dihedral is taken between PREDICTION group centroids
(``atom_selectionN``) and FITTED-REFERENCE group centroids (``atom_selectionN_ref``: atoms
selected ON THE REFERENCE, placed into the prediction frame by the fit and held fixed).

Such an entry is routed here (NOT to ``DistanceData`` / ``AngleRestraintData`` /
``DihedralRestraintData``) by ``config.from_dict`` and compiled to a ``kind="ref_geom"``
``CustomSpec`` closure. This reuses the custom-restraint closure path (so NO 3-backend energy
layer change is needed — the ref fit needs a per-eval Kabsch the array energy does not do) while
KEEPING the built-in group terms' rigid-centroid prediction-group motion: a prediction group uses
the ``_move_centroid`` N×-rescale (via ``closure._rigid_centroid``), so ``weight: 1`` moves a
large group as a rigid body — a plain centroid's ``1/N`` per-atom gradient would barely move it.

The whole fit transform is stop-gradient'd (like the built-in RMSD term), so a reference group is
a fixed landmark that tracks the anchor but exerts no force on it; the restraint pulls only the
prediction group(s). Grad parity is therefore torch-vs-jax, not numpy-FD.
"""

from __future__ import annotations

import logging
import math

from rgi_utils._config_util import (
    apply_window_params,
    coerce_bool,
    parse_geom_type,
    warn_unknown_keys,
)
from rgi_utils._moltype import polymer_type
from rgi_utils.atom_context import FrameworkAdapter, candidate_dict
from rgi_utils.custom.data import CustomSpec, _select_ref_coords
from rgi_utils.pdb_ref import read_cif_atoms, read_pdb_atoms
from rgi_utils.rmsd_restr_data import build_resid_map, pair_target_to_ref
from rgi_utils.selection import AtomSelector
from rgi_utils.spec import DIST_TYPE_CODES

logger = logging.getLogger(__name__)

# (geom -> n_groups, base target key). distance = 2 groups (native Angstrom), angle = 3 groups /
# dihedral = 4 groups (targets in degrees by default, stored in radians).
_GEOM_SPEC = {
    "distance": (2, "target_distance"),
    "angle": (3, "target_angle"),
    "dihedral": (4, "target_dihedral"),
}

# reference name used internally (one reference per ref_geom entry).
_REF = "__ref__"


def is_ref_anchored(entry: dict) -> bool:
    """True if a distance/angle/dihedral config entry uses a reference (so it is routed here)."""
    if "ref_pdb" in entry or "ref_cif" in entry:
        return True
    return any(f"atom_selection{i}_ref" in entry for i in range(1, 5))


class RefGeomData:
    """One reference-anchored distance / angle / dihedral entry. Duck-types ``CustomData``
    (``run_restr`` / ``resolve_sites`` / ``iter_global_sites`` / ``build_spec``) so it flows
    through the same custom-restraint path in ``combined.setup`` / ``featurizer.build_spec``."""

    def __init__(self, geom: str) -> None:
        if geom not in _GEOM_SPEC:
            raise ValueError(f"ref_geom: unknown geom {geom!r}")
        self.geom = geom
        self.n_groups, self._base = _GEOM_SPEC[geom]
        self.name = f"ref_{geom}"
        self.run_restr = False
        self.weight = 1.0
        self.start_sigma: float | None = None
        self.stop_sigma = -1.0
        self.start_step = float("-inf")
        self.stop_step = float("inf")
        # reference + fit
        self.ref_cif = None
        self.ref_path = None
        self.sel_ref_fit = None
        self.sel_target_fit = None
        self.pairing = "align"
        self.best_effort = True
        # penalty
        self.geom_type: str | None = None
        self.target1 = 0.0
        self.target2 = 0.0
        # per-group source: (kind, payload) — ("pred", selection) | ("ref", selection)
        self._group_sel: list[tuple[str, str]] = []
        # resolved
        self._pred_globals: dict[
            int, list[int]
        ] = {}  # group index -> global atom indices
        self._ref_group_coords: dict[
            str, "object"
        ] = {}  # ref selection -> (k,3) coords
        self._fit: tuple | None = None  # (fit_target_globals, fit_ref_coords) | None

    def _known_keys(self) -> set:
        keys = {
            "ref_pdb",
            "ref_cif",
            "atom_selection_ref_fit",
            "atom_selection_target_fit",
            "pairing",
            "best_effort",
            "weight",
            "start_sigma",
            "stop_sigma",
            "start_step",
            "stop_step",
            "harmonic",
            "flat-bottomed",
            "flat-bottomed1",
            "flat-bottomed2",
        }
        if self.geom != "distance":
            keys.add("unit")
        else:
            keys.add("calc_method")
        for i in range(1, self.n_groups + 1):
            keys.add(f"atom_selection{i}")
            keys.add(f"atom_selection{i}_ref")
        return keys

    def set_config(self, config: dict) -> None:
        label = f"{self.geom}_restraints_config entry (reference-anchored)"
        warn_unknown_keys(config, self._known_keys(), label, logger)
        # a reference-anchored group is always FIXED, so `move` (which selects the free groups)
        # is meaningless here — prediction groups are all free (rigid), reference groups fixed.
        if "move" in config:
            raise ValueError(
                f"{label}: 'move' is not supported on a reference-anchored restraint "
                "(prediction groups are all free/rigid, reference groups all fixed). Use the "
                "custom_restraints_config DSL for finer per-group control."
            )
        # reference structure: ref_pdb XOR ref_cif.
        ref_pdb = config.get("ref_pdb")
        self.ref_cif = config.get("ref_cif")
        if (ref_pdb is None) == (self.ref_cif is None):
            raise ValueError(f"{label}: give exactly one of ref_pdb / ref_cif")
        self.ref_path = ref_pdb if ref_pdb is not None else self.ref_cif
        # Kabsch-fit anchor (both optional: omit BOTH to use the reference in its own frame).
        self.sel_ref_fit = config.get("atom_selection_ref_fit")
        self.sel_target_fit = config.get("atom_selection_target_fit")
        self.pairing = config.get("pairing") or "align"
        if self.pairing not in ("identity", "align"):
            raise ValueError(f"{label}: pairing must be 'identity' or 'align'")
        self.best_effort = coerce_bool(config.get("best_effort"), True)
        # weight + the sigma/step gate windows (shared parse).
        apply_window_params(self, config, label)
        # per-group source: exactly one of atom_selectionI / atom_selectionI_ref per index.
        self._group_sel = []
        n_ref = 0
        for i in range(1, self.n_groups + 1):
            pred = config.get(f"atom_selection{i}")
            ref = config.get(f"atom_selection{i}_ref")
            if (pred is None) == (ref is None):
                raise ValueError(
                    f"{label}: group {i} needs exactly one of atom_selection{i} "
                    f"(prediction) or atom_selection{i}_ref (reference)"
                )
            if ref is not None:
                self._group_sel.append(("ref", ref))
                n_ref += 1
            else:
                self._group_sel.append(("pred", pred))
        if n_ref == 0:
            raise ValueError(
                f"{label}: no atom_selectionN_ref group — this is not reference-anchored "
                "(use the plain distance/angle/dihedral restraint instead)"
            )
        if n_ref == self.n_groups:
            raise ValueError(
                f"{label}: every group is a reference selection, so the measured quantity is "
                "constant (zero gradient) — at least one group must be a prediction selection"
            )
        # penalty type + target(s): degrees -> radians for angle/dihedral (unit key), native
        # Angstrom for distance. Reuses the shared parse so the four types stay in lockstep.
        if self.geom == "distance":
            conv = float
        else:
            unit = config.get("unit", "degrees")
            if unit not in ("degrees", "radians"):
                raise ValueError(f"{label}: unit must be 'degrees' or 'radians'")
            conv = float if unit == "radians" else (lambda x: math.radians(float(x)))
        self.geom_type, self.target1, self.target2 = parse_geom_type(
            config, self._base, conv
        )
        if self.geom_type is None:
            raise ValueError(
                f"{label}: needs a restraint-type block (harmonic / flat-bottomed / "
                f"flat-bottomed1 / flat-bottomed2) with {self._base}(1/2)"
            )
        self.run_restr = True

    def resolve_sites(self, adapter: FrameworkAdapter) -> None:
        if not self.run_restr:
            return
        atoms = list(adapter.iter_atoms())
        # prediction groups: resolve each selection to global atom indices.
        self._pred_globals = {}
        for i, (kind, sel) in enumerate(self._group_sel, start=1):
            if kind != "pred":
                continue
            selector = AtomSelector(sel)
            gi = [a.index for a in atoms if selector.eval(candidate_dict(a))]
            if not gi:
                raise ValueError(
                    f"{self.name}: atom_selection{i} matched no prediction atoms: {sel!r}"
                )
            self._pred_globals[i] = gi
        # reference: load + (optionally) align.
        reader = read_cif_atoms if self.ref_cif is not None else read_pdb_atoms
        ref_atoms = reader(self.ref_path)
        has_polymer = any(
            polymer_type(a.mol_type, a.resname) is not None for a in atoms
        )
        align = self.pairing == "align" and has_polymer
        resid_map = build_resid_map(atoms, ref_atoms, self.ref_path) if align else None
        # reference groups: resolve each selection ON THE REFERENCE to its coords.
        self._ref_group_coords = {}
        for kind, sel in self._group_sel:
            if kind != "ref":
                continue
            self._ref_group_coords[sel] = _select_ref_coords(
                ref_atoms, sel, f"{self.name} reference group", self.ref_path
            )
        # fit anchor (optional): pair the prediction anchor to the reference anchor.
        self._fit = None
        if self.sel_target_fit is not None or self.sel_ref_fit is not None:
            fit_target_globals, fit_ref_coords = pair_target_to_ref(
                atoms,
                ref_atoms,
                self.sel_target_fit,
                self.sel_ref_fit,
                f"{self.name} fit",
                ref_path=self.ref_path,
                best_effort=self.best_effort,
                align=align,
                resid_map=resid_map,
            )
            if len(fit_target_globals) < 3:
                raise ValueError(
                    f"{self.name}: the Kabsch fit paired only {len(fit_target_globals)} "
                    "anchor atom(s); it needs at least 3 non-collinear atoms (widen "
                    "atom_selection_target_fit / atom_selection_ref_fit)"
                )
            self._fit = (fit_target_globals, fit_ref_coords)
        logger.info(
            "%s resolved: %d prediction group(s), %d reference group(s), fit=%s",
            self.name,
            len(self._pred_globals),
            len(self._ref_group_coords),
            "yes" if self._fit is not None else "own-frame",
        )

    def iter_global_sites(self):
        out: list[int] = []
        for gi in self._pred_globals.values():
            out.extend(int(x) for x in gi)
        if self._fit is not None:  # fit anchor prediction atoms must be gatherable
            out.extend(int(x) for x in self._fit[0])
        return out

    def build_spec(self, g2l: dict[int, int]) -> CustomSpec:
        import numpy as np

        groups = []
        for i, (kind, sel) in enumerate(self._group_sel, start=1):
            if kind == "pred":
                local = np.array(
                    [g2l[int(x)] for x in self._pred_globals[i]], dtype=np.int64
                )
                groups.append(("pred", local))
            else:  # reference group -> keyed by (selection, _REF) into ref_blocks
                groups.append(("ref", (sel, _REF)))
        ref_fits = {}
        if self._fit is not None:
            fit_target_globals, fit_ref_coords = self._fit
            ref_fits[_REF] = (
                np.array([g2l[int(x)] for x in fit_target_globals], dtype=np.int64),
                fit_ref_coords,
            )
        ref_blocks = {
            (sel, _REF): coords for sel, coords in self._ref_group_coords.items()
        }
        return CustomSpec(
            name=self.name,
            selections={},
            kind="ref_geom",
            ast=None,
            fn=None,
            weight=float(self.weight),
            start_sigma=float(self.start_sigma)
            if self.start_sigma is not None
            else float("inf"),
            stop_sigma=float(self.stop_sigma),
            start_step=float(self.start_step),
            stop_step=float(self.stop_step),
            ref_fits=ref_fits,
            ref_blocks=ref_blocks,
            geom=self.geom,
            groups=groups,
            geom_type_code=DIST_TYPE_CODES[self.geom_type],
            target1=float(self.target1),
            target2=float(self.target2),
        )

    def is_valid(self) -> bool:
        return self.run_restr
