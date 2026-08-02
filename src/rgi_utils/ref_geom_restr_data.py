"""Built-in distance, angle, dihedral, and plane restraints with reference groups.

Each group keeps the normal ``atom_selectionN`` key. A prediction group contains
the ordinary atom-selection DSL; a reference group starts with
``refN and <selection DSL>`` and names an entry in the restraint's shared
``refs`` map. Each reference is independently fitted into the prediction frame.

These entries compile to ``kind="ref_geom"`` ``CustomSpec`` closures. Prediction
groups retain the built-in rigid-centroid gradient scaling, while reference
groups and their fit transforms are fixed with stop-gradient.
``move`` applies a static free/pinned mask to prediction groups; references can never move.

``plane`` differs from the other three geoms in two ways. It takes a VARIABLE number of
groups (1..4, so ``n_groups`` comes from the caller — see ``_GEOM_SPEC``), and it does not
reduce its groups to centroids: the reference group's atoms define a fixed best-fit plane
and the measured quantity is the RMS distance of the prediction atoms from that plane (the
closure's ``plane`` branch). Without a reference, a plane restraint is an ordinary array
term instead — ``plane_restr_data.PlaneRestraintData``.
"""

from __future__ import annotations

import logging
import math

from rgi_utils._config_util import (
    apply_window_params,
    parse_geom_type,
    parse_move_indices,
    warn_unknown_keys,
)
from rgi_utils._moltype import polymer_type
from rgi_utils.atom_context import FrameworkAdapter, candidate_dict
from rgi_utils.custom.data import CustomSpec, _select_ref_coords
from rgi_utils.pdb_ref import read_cif_atoms, read_pdb_atoms
from rgi_utils.ref_config import (
    parse_ref_defs,
    split_ref_selection,
    validate_ref_usage,
)
from rgi_utils.rmsd_restr_data import build_resid_map, pair_target_to_ref
from rgi_utils.selection import AtomSelector
from rgi_utils.spec import DIST_TYPE_CODES

logger = logging.getLogger(__name__)

_GEOM_SPEC = {
    "distance": (2, "target_distance"),
    "angle": (3, "target_angle"),
    "dihedral": (4, "target_dihedral"),
    "improper": (4, "target_improper"),
    # plane takes 1..4 groups, so its group count is NOT fixed by the geometry: `None`
    # means "the caller supplies n_groups" (config.py counts the entry's contiguous
    # atom_selectionN keys via plane_restr_data.count_plane_groups). Keeping n_groups a
    # plain int from __init__ onward is deliberate — it is read in five places that all
    # rely on "a missing atom_selectionN raises".
    "plane": (None, "target_plane"),
}

# A plane needs >= 3 atoms to define a normal at all.
_MIN_PLANE_REF_ATOMS = 3


def is_ref_anchored(entry: dict) -> bool:
    """Return whether a built-in geometry entry uses the named-reference syntax."""
    if "refs" in entry:
        return True
    for i in range(1, 5):
        selection = entry.get(f"atom_selection{i}")
        if not isinstance(selection, str):
            continue
        try:
            if split_ref_selection(selection, f"atom_selection{i}") is not None:
                return True
        except ValueError:
            # Route malformed ref-like prefixes here so the shared parser emits
            # its targeted config error instead of an ordinary DSL parse error.
            return True
    return False


class RefGeomData:
    """One distance, angle, or dihedral entry containing reference groups."""

    def __init__(self, geom: str, n_groups: int | None = None) -> None:
        if geom not in _GEOM_SPEC:
            raise ValueError(f"ref_geom: unknown geom {geom!r}")
        self.geom = geom
        default_groups, self._base = _GEOM_SPEC[geom]
        if default_groups is None and n_groups is None:
            raise ValueError(
                f"ref_geom: geom {geom!r} takes a variable number of groups, so "
                "n_groups must be given by the caller"
            )
        self.n_groups = default_groups if n_groups is None else int(n_groups)
        self.name = f"ref_{geom}"
        self.run_restr = False
        self.weight = 1.0
        self.start_sigma: float | None = None
        self.stop_sigma = -1.0
        self.start_step = float("-inf")
        self.stop_step = float("inf")
        self.ref_defs: dict[str, dict] = {}
        self.geom_type: str | None = None
        self.target1 = 0.0
        self.target2 = 0.0
        self.move_free: tuple[bool, ...] = ()
        # ("pred", selection) | ("ref", (reference_selection, ref_name))
        self._group_sel: list[tuple[str, object]] = []
        self._pred_globals: dict[int, list[int]] = {}
        self._ref_group_coords: dict[tuple[str, str], object] = {}
        self._fits: dict[str, tuple | None] = {}

    def _known_keys(self) -> set[str]:
        keys = {
            "refs",
            "move",
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
        if self.geom == "distance":
            keys.add("calc_method")
        elif self.geom != "plane":
            keys.add("unit")  # plane targets are Angstrom only, like distance
        for i in range(1, self.n_groups + 1):
            keys.add(f"atom_selection{i}")
        return keys

    def set_config(self, config: dict) -> None:
        label = f"{self.geom}_restraints_config entry"
        warn_unknown_keys(config, self._known_keys(), label, logger)
        if (
            self.geom == "distance"
            and config.get("calc_method", "unfixed-absolute") != "unfixed-absolute"
        ):
            raise ValueError("calc_method must be unfixed-absolute")

        self.ref_defs = parse_ref_defs(config.get("refs", {}), label)
        self._group_sel = []
        used_refs: set[str] = set()
        n_prediction = 0
        for i in range(1, self.n_groups + 1):
            key = f"atom_selection{i}"
            selection = config.get(key)
            if not isinstance(selection, str) or not selection.strip():
                raise ValueError(f"{label}: {key} is required and must be a string")
            parsed = split_ref_selection(selection, f"{label} {key}")
            if parsed is None:
                self._group_sel.append(("pred", selection))
                n_prediction += 1
            else:
                ref_name, ref_selection = parsed
                self._group_sel.append(("ref", (ref_selection, ref_name)))
                used_refs.add(ref_name)

        if not used_refs:
            raise ValueError(
                f"{label}: 'refs' is present but no atom_selectionN starts with "
                "'refN and <atom selection>'"
            )
        validate_ref_usage(
            self.ref_defs,
            used_refs,
            label,
            max_refs=self.n_groups - 1,
        )
        if n_prediction == 0:
            raise ValueError(
                f"{label}: every group is a reference selection, so the measured "
                "quantity is constant; at least one group must select prediction atoms"
            )

        prediction_groups = {
            i
            for i, (kind, _payload) in enumerate(self._group_sel, start=1)
            if kind == "pred"
        }
        move = config.get("move")
        if move is None or (
            isinstance(move, str) and move.strip().lower() in ("all", "both")
        ):
            selected_groups = prediction_groups
        else:
            selected_groups = parse_move_indices(move, self.n_groups)
            selected_refs = selected_groups - prediction_groups
            if selected_refs:
                raise ValueError(
                    f"{label}: move selects reference group(s) {sorted(selected_refs)}; "
                    "reference groups are fixed, so select prediction group indices only"
                )
        self.move_free = tuple(
            i in selected_groups for i in range(1, self.n_groups + 1)
        )

        apply_window_params(self, config, label)
        if self.geom in ("distance", "plane"):
            conv = float  # native Angstrom
        else:
            unit = str(config.get("unit", "degrees")).strip().lower()
            if unit not in ("degrees", "radians"):
                raise ValueError(f"{label}: unit must be 'degrees' or 'radians'")
            conv = float if unit == "radians" else (lambda x: math.radians(float(x)))
        self.geom_type, self.target1, self.target2 = parse_geom_type(
            config, self._base, conv
        )
        if self.geom_type is None:
            # plane's target is essentially always 0 (planar), so its type block is
            # optional and defaults to harmonic toward 0 — mirroring the array-path
            # PlaneRestraintData. Every other geom needs an explicit target.
            if self.geom == "plane":
                self.geom_type, self.target1, self.target2 = "harmonic", 0.0, 0.0
            else:
                raise ValueError(
                    f"{label}: needs a restraint-type block (harmonic / flat-bottomed / "
                    f"flat-bottomed1 / flat-bottomed2) with {self._base}(1/2)"
                )
        self.run_restr = True

    def resolve_sites(self, adapter: FrameworkAdapter) -> None:
        if not self.run_restr:
            return
        atoms = list(adapter.iter_atoms())
        has_polymer = any(
            polymer_type(a.mol_type, a.resname) is not None for a in atoms
        )

        self._pred_globals = {}
        for i, (kind, payload) in enumerate(self._group_sel, start=1):
            if kind != "pred":
                continue
            selection = str(payload)
            selector = AtomSelector(selection)
            global_indices = [
                a.index for a in atoms if selector.eval(candidate_dict(a))
            ]
            if not global_indices:
                raise ValueError(
                    f"{self.name}: atom_selection{i} matched no prediction atoms: "
                    f"{selection!r}"
                )
            self._pred_globals[i] = global_indices

        used_refs = {payload[1] for kind, payload in self._group_sel if kind == "ref"}
        cache: dict[str, tuple] = {}
        for ref_name in used_refs:
            ref_def = self.ref_defs[ref_name]
            reader = (
                read_cif_atoms if ref_def["ref_cif"] is not None else read_pdb_atoms
            )
            ref_atoms = reader(ref_def["ref_path"])
            align = ref_def["pairing"] == "align" and has_polymer
            resid_map = (
                build_resid_map(atoms, ref_atoms, ref_def["ref_path"])
                if align
                else None
            )
            cache[ref_name] = (ref_atoms, align, resid_map)

        self._ref_group_coords = {}
        for kind, payload in self._group_sel:
            if kind != "ref":
                continue
            selection, ref_name = payload
            ref_atoms, _align, _resid_map = cache[ref_name]
            key = (selection, ref_name)
            if key not in self._ref_group_coords:
                self._ref_group_coords[key] = _select_ref_coords(
                    ref_atoms,
                    selection,
                    f"{self.name} reference group {ref_name}",
                    self.ref_defs[ref_name]["ref_path"],
                )
            # A ref-anchored PLANE defines its plane from the reference atoms alone, so a
            # 1- or 2-atom reference selection leaves the normal undefined (the energy
            # would be finite but meaningless). Raise, mirroring the fit's >= 3 anchor
            # check below — the other geoms only need a centroid, so any count works.
            if (
                self.geom == "plane"
                and len(self._ref_group_coords[key]) < _MIN_PLANE_REF_ATOMS
            ):
                raise ValueError(
                    f"{self.name}: reference group {ref_name!r} selected only "
                    f"{len(self._ref_group_coords[key])} atom(s); a best-fit plane needs "
                    f"at least {_MIN_PLANE_REF_ATOMS} ({selection!r})"
                )

        self._fits = {}
        for ref_name in used_refs:
            ref_def = self.ref_defs[ref_name]
            ref_atoms, align, resid_map = cache[ref_name]
            if ref_def["sel_target_fit"] is None and ref_def["sel_ref_fit"] is None:
                self._fits[ref_name] = None
                continue
            fit_target_globals, fit_ref_coords = pair_target_to_ref(
                atoms,
                ref_atoms,
                ref_def["sel_target_fit"],
                ref_def["sel_ref_fit"],
                f"{self.name} fit->{ref_name}",
                ref_path=ref_def["ref_path"],
                best_effort=ref_def["best_effort"],
                align=align,
                resid_map=resid_map,
            )
            if len(fit_target_globals) < 3:
                raise ValueError(
                    f"{self.name}: fit for {ref_name!r} paired only "
                    f"{len(fit_target_globals)} anchor atom(s); Kabsch needs at least "
                    "3 non-collinear atoms"
                )
            self._fits[ref_name] = (fit_target_globals, fit_ref_coords)

        logger.info(
            "%s resolved: %d prediction group(s), %d reference group(s), %d fitted ref(s)",
            self.name,
            len(self._pred_globals),
            len(self._ref_group_coords),
            sum(fit is not None for fit in self._fits.values()),
        )

    def iter_global_sites(self):
        out: list[int] = []
        for global_indices in self._pred_globals.values():
            out.extend(int(x) for x in global_indices)
        for fit in self._fits.values():
            if fit is not None:
                out.extend(int(x) for x in fit[0])
        return out

    def build_spec(self, g2l: dict[int, int]) -> CustomSpec:
        import numpy as np

        groups = []
        for i, (kind, payload) in enumerate(self._group_sel, start=1):
            if kind == "pred":
                local = np.array(
                    [g2l[int(x)] for x in self._pred_globals[i]], dtype=np.int64
                )
                groups.append(("pred", (local, self.move_free[i - 1])))
            else:
                groups.append(("ref", payload))

        ref_fits = {}
        for ref_name, fit in self._fits.items():
            if fit is None:
                continue
            fit_target_globals, fit_ref_coords = fit
            ref_fits[ref_name] = (
                np.array([g2l[int(x)] for x in fit_target_globals], dtype=np.int64),
                fit_ref_coords,
            )
        return CustomSpec(
            name=self.name,
            selections={},
            kind="ref_geom",
            ast=None,
            fn=None,
            weight=float(self.weight),
            start_sigma=(
                float(self.start_sigma)
                if self.start_sigma is not None
                else float("inf")
            ),
            stop_sigma=float(self.stop_sigma),
            start_step=float(self.start_step),
            stop_step=float(self.stop_step),
            ref_fits=ref_fits,
            ref_blocks=dict(self._ref_group_coords),
            geom=self.geom,
            groups=groups,
            geom_type_code=DIST_TYPE_CODES[self.geom_type],
            target1=float(self.target1),
            target2=float(self.target2),
        )

    def is_valid(self) -> bool:
        return self.run_restr
