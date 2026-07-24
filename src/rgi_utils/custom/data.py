"""Parse and resolve one custom restraint entry."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rgi_utils._config_util import apply_window_params, warn_unknown_keys
from rgi_utils._moltype import polymer_type
from rgi_utils.atom_context import candidate_dict
from rgi_utils.custom.context import ResolveContext
from rgi_utils.custom.dsl import eval_formula, parse_formula
from rgi_utils.custom.registry import get_custom_fn
from rgi_utils.group_geom_restr_data import _resolve_group_sites
from rgi_utils.pdb_ref import read_cif_atoms, read_pdb_atoms
from rgi_utils.ref_config import (
    is_ref_name,
    parse_ref_defs,
    split_ref_selection,
    validate_ref_usage,
)
from rgi_utils.rmsd_restr_data import build_resid_map, pair_target_to_ref
from rgi_utils.selection import AtomSelector

logger = logging.getLogger(__name__)


def _select_ref_coords(ref_atoms, sel_string, tag, ref_path):
    """Return coordinates selected on one external reference structure."""
    selector = AtomSelector(sel_string)
    coords = [
        (atom.x, atom.y, atom.z)
        for atom in ref_atoms
        if selector.eval(candidate_dict(atom, resname_attr="res_name"))
    ]
    if not coords:
        raise ValueError(
            f"{tag}: reference selection matched no atoms: {sel_string!r} "
            f"in {ref_path!r}"
        )
    return np.asarray(coords, dtype=np.float64)


_KNOWN_CUSTOM_KEYS = {
    "name",
    "energy",
    "use",
    "fn",
    "selections",
    "refs",
    "move",
    "weight",
    "start_sigma",
    "stop_sigma",
    "start_step",
    "stop_step",
}


@dataclass
class CustomSpec:
    """Backend-agnostic resolved custom restraint."""

    name: str
    selections: dict[str, np.ndarray]
    kind: str
    ast: Any
    fn: Any
    weight: float
    start_sigma: float
    stop_sigma: float
    start_step: float
    stop_step: float
    # rmsd(A, B): (prediction selection id, reference selection id) ->
    # (matched prediction local indices, row-aligned constant ref coordinates)
    refs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    # Reference-backed selection id -> (selection evaluated on ref, ref name).
    selection_refs: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Prediction selection id -> whether this term may move it.
    move_free: dict[str, bool] = field(default_factory=dict)
    # One optional prediction-frame fit per reference and the constant reference
    # blocks consumed by reference-backed selections.
    ref_fits: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    ref_blocks: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    # Built-in geometry routed through the custom closure path.
    geom: str = ""
    groups: list = field(default_factory=list)
    geom_type_code: int = 0
    target1: float = 0.0
    target2: float = 0.0


class CustomData:
    """Parse one entry and resolve prediction/reference selections."""

    def __init__(self) -> None:
        self.name = "custom"
        self.kind: str | None = None
        self.ast: Any = None
        self.fn: Any = None
        self.selections: dict[str, str] = {}
        self.weight = 1.0
        self.start_sigma: float | None = None
        self.stop_sigma = -1.0
        self.start_step = float("-inf")
        self.stop_step = float("inf")
        self.run_restr = False
        self.ref_defs: dict[str, dict] = {}
        self.move = None
        self._identifiers: list[str] = []
        self._selection_sources: dict[str, tuple[str, object]] = {}
        self._global: dict[str, list[int]] = {}
        self._selection_refs: dict[str, tuple[str, str]] = {}
        self._move_free: dict[str, bool] = {}
        self._ref_pairs: list[tuple[str, str]] = []
        self._ref_resolved: dict[tuple[str, str], tuple[list[int], np.ndarray]] = {}
        self._ref_group_resolved: dict[tuple[str, str], np.ndarray] = {}
        self._ref_fit_resolved: dict[str, tuple[list[int], np.ndarray] | None] = {}

    def set_config(self, config: dict) -> None:
        warn_unknown_keys(
            config, _KNOWN_CUSTOM_KEYS, "custom_restraints_config entry", logger
        )
        self.name = str(config.get("name", "custom"))
        raw_selections = config.get("selections", {}) or {}
        if not isinstance(raw_selections, dict):
            raise ValueError(
                "custom_restraints_config entry: 'selections' must be a mapping"
            )
        self.selections = {str(k): str(v) for k, v in raw_selections.items()}
        reserved_aliases = sorted(k for k in self.selections if is_ref_name(k))
        if reserved_aliases:
            raise ValueError(
                "custom_restraints_config entry: selection alias name(s) "
                f"{reserved_aliases} are reserved for refs map keys"
            )
        self.ref_defs = parse_ref_defs(
            config.get("refs", {}), "custom_restraints_config entry"
        )
        self.move = config.get("move")
        apply_window_params(self, config, "custom_restraints_config entry")

        sources = [
            key for key in ("energy", "use", "fn") if config.get(key) is not None
        ]
        if len(sources) != 1:
            raise ValueError(
                "custom_restraints_config entry needs exactly one of 'energy' "
                f"(formula), 'use' (registered name), or 'fn' (callable); got {sources}"
            )
        source = sources[0]
        if source == "energy":
            self.kind = "formula"
            self.ast = parse_formula(str(config["energy"]))
        elif source == "use":
            fn = get_custom_fn(str(config["use"]))
            if fn is None:
                raise ValueError(
                    f"custom restraint 'use': no function registered as "
                    f"{config['use']!r} (register it with @custom_restraint)"
                )
            self.kind, self.fn = "fn", fn
        else:
            if not callable(config["fn"]):
                raise ValueError("custom_restraints_config 'fn' must be a callable")
            self.kind, self.fn = "fn", config["fn"]
        self.run_restr = True

    def _evaluate_resolve(self, context: ResolveContext) -> None:
        if self.kind == "formula":
            eval_formula(self.ast, context)
        else:
            self.fn(context)

    def _classify_selections(self, identifiers: list[str]) -> set[str]:
        self._selection_sources = {}
        self._selection_refs = {}
        used_refs: set[str] = set()
        for identifier in identifiers:
            selection = self.selections.get(identifier, identifier)
            parsed = split_ref_selection(
                selection,
                f"custom restraint {self.name!r} selection {identifier!r}",
            )
            if parsed is None:
                self._selection_sources[identifier] = ("pred", selection)
                continue
            ref_name, ref_selection = parsed
            payload = (ref_selection, ref_name)
            self._selection_sources[identifier] = ("ref", payload)
            self._selection_refs[identifier] = payload
            used_refs.add(ref_name)
        validate_ref_usage(
            self.ref_defs,
            used_refs,
            f"custom restraint {self.name!r}",
        )
        return used_refs

    def _resolve_move(self, prediction_ids: list[str]) -> None:
        prediction_set = set(prediction_ids)
        move = self.move
        if move is None or (
            isinstance(move, str) and move.strip().lower() in ("all", "both")
        ):
            selected = prediction_set
        elif isinstance(move, str):
            if move in self._identifiers:
                selected = {move}
            else:
                selected = {item.strip() for item in move.split(",") if item.strip()}
        elif isinstance(move, (list, tuple, set)):
            selected = set(move)
        else:
            raise ValueError(
                f"custom restraint {self.name!r}: move must be 'all', a selection "
                f"name, or a list of selection names; got {move!r}"
            )

        if any(not isinstance(identifier, str) for identifier in selected):
            raise ValueError(
                f"custom restraint {self.name!r}: move selection names must be strings; "
                f"got {move!r}"
            )

        unknown = selected - set(self._identifiers)
        if not selected or unknown:
            raise ValueError(
                f"custom restraint {self.name!r}: move contains unknown selection "
                f"name(s) {sorted(unknown) if unknown else []}; available names are "
                f"{self._identifiers}"
            )
        selected_refs = selected - prediction_set
        if selected_refs:
            raise ValueError(
                f"custom restraint {self.name!r}: move selects reference-backed "
                f"selection(s) {sorted(selected_refs)}; reference selections are fixed"
            )
        self._move_free = {
            identifier: identifier in selected for identifier in prediction_ids
        }

    def _load_refs(self, atoms, used_refs: set[str]) -> dict[str, tuple]:
        has_polymer = any(
            polymer_type(atom.mol_type, atom.resname) is not None for atom in atoms
        )
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
        return cache

    def resolve_sites(self, adapter) -> None:
        if not self.run_restr:
            return
        context = ResolveContext()
        self._evaluate_resolve(context)
        self._identifiers = list(context.selections)
        if not self._identifiers:
            raise ValueError(
                f"custom restraint {self.name!r} references no atom selection"
            )

        used_refs = self._classify_selections(self._identifiers)
        prediction_ids = [
            identifier
            for identifier, (kind, _payload) in self._selection_sources.items()
            if kind == "pred"
        ]
        if not prediction_ids:
            raise ValueError(
                f"custom restraint {self.name!r} measures only reference atoms; "
                "at least one selection must use prediction atoms"
            )
        self._resolve_move(prediction_ids)

        prediction_strings = [
            self._selection_sources[identifier][1] for identifier in prediction_ids
        ]
        prediction_sites = _resolve_group_sites(adapter, prediction_strings)
        self._global = dict(zip(prediction_ids, prediction_sites))

        atoms = list(adapter.iter_atoms())
        cache = self._load_refs(atoms, used_refs)
        self._ref_group_resolved = {}
        for _identifier, (selection, ref_name) in self._selection_refs.items():
            key = (selection, ref_name)
            if key in self._ref_group_resolved:
                continue
            ref_atoms, _align, _resid_map = cache[ref_name]
            self._ref_group_resolved[key] = _select_ref_coords(
                ref_atoms,
                selection,
                f"custom restraint {self.name!r} reference group {ref_name}",
                self.ref_defs[ref_name]["ref_path"],
            )

        self._ref_fit_resolved = {}
        for ref_name in used_refs:
            ref_def = self.ref_defs[ref_name]
            ref_atoms, align, resid_map = cache[ref_name]
            if ref_def["sel_target_fit"] is None and ref_def["sel_ref_fit"] is None:
                self._ref_fit_resolved[ref_name] = None
                continue
            fit_target_globals, fit_ref_coords = pair_target_to_ref(
                atoms,
                ref_atoms,
                ref_def["sel_target_fit"],
                ref_def["sel_ref_fit"],
                f"custom {self.name!r} fit->{ref_name}",
                ref_path=ref_def["ref_path"],
                best_effort=ref_def["best_effort"],
                align=align,
                resid_map=resid_map,
            )
            if len(fit_target_globals) < 3:
                raise ValueError(
                    f"custom restraint {self.name!r}: fit for {ref_name!r} "
                    f"paired only {len(fit_target_globals)} anchor atom(s); "
                    "Kabsch needs at least 3 non-collinear atoms"
                )
            self._ref_fit_resolved[ref_name] = (
                fit_target_globals,
                fit_ref_coords,
            )

        self._resolve_rmsd_pairs(context.rmsd_pairs, atoms, cache)
        for first, second in context.kabsch_pairs:
            n_first = self._selection_size(first)
            n_second = self._selection_size(second)
            if n_first != n_second:
                raise ValueError(
                    f"custom restraint {self.name!r}: kabsch({first}, {second}) "
                    f"needs equal atom counts, got {n_first} and {n_second}"
                )

        logger.info(
            "custom restraint (%s) resolved: %d prediction selection(s), "
            "%d reference selection(s)",
            self.name,
            len(self._global),
            len(self._selection_refs),
        )

    def _selection_size(self, identifier: str) -> int:
        source = self._selection_sources[identifier]
        if source[0] == "pred":
            return len(self._global[identifier])
        selection, ref_name = source[1]
        return len(self._ref_group_resolved[(selection, ref_name)])

    def _resolve_rmsd_pairs(self, pairs, atoms, cache) -> None:
        self._ref_pairs = list(pairs)
        self._ref_resolved = {}
        for target_id, ref_id in self._ref_pairs:
            target_source = self._selection_sources[target_id]
            ref_source = self._selection_sources[ref_id]
            if target_source[0] != "pred" or ref_source[0] != "ref":
                raise ValueError(
                    f"custom restraint {self.name!r}: rmsd(A, B) requires A to "
                    "select prediction atoms and B to use 'refN and <selection>'"
                )
            target_selection = target_source[1]
            ref_selection, ref_name = ref_source[1]
            ref_def = self.ref_defs[ref_name]
            ref_atoms, align, resid_map = cache[ref_name]
            target_globals, ref_coords = pair_target_to_ref(
                atoms,
                ref_atoms,
                target_selection,
                ref_selection,
                f"custom {self.name!r} rmsd({target_id}, {ref_id})",
                ref_path=ref_def["ref_path"],
                best_effort=ref_def["best_effort"],
                align=align,
                resid_map=resid_map,
            )
            self._ref_resolved[(target_id, ref_id)] = (
                target_globals,
                ref_coords,
            )

    def iter_global_sites(self):
        out: list[int] = []
        for sites in self._global.values():
            out.extend(int(index) for index in sites)
        for target_globals, _ref_coords in self._ref_resolved.values():
            out.extend(int(index) for index in target_globals)
        for fit in self._ref_fit_resolved.values():
            if fit is not None:
                out.extend(int(index) for index in fit[0])
        return out

    def build_spec(self, g2l: dict[int, int]) -> CustomSpec:
        local = {
            identifier: np.array([g2l[int(index)] for index in sites], dtype=np.int64)
            for identifier, sites in self._global.items()
        }
        refs = {
            key: (
                np.array(
                    [g2l[int(index)] for index in target_globals],
                    dtype=np.int64,
                ),
                ref_coords,
            )
            for key, (target_globals, ref_coords) in self._ref_resolved.items()
        }
        ref_fits = {}
        for ref_name, fit in self._ref_fit_resolved.items():
            if fit is None:
                continue
            fit_target_globals, fit_ref_coords = fit
            ref_fits[ref_name] = (
                np.array(
                    [g2l[int(index)] for index in fit_target_globals],
                    dtype=np.int64,
                ),
                fit_ref_coords,
            )
        return CustomSpec(
            name=self.name,
            selections=local,
            kind=self.kind,
            ast=self.ast,
            fn=self.fn,
            weight=float(self.weight),
            start_sigma=(
                float(self.start_sigma)
                if self.start_sigma is not None
                else float("inf")
            ),
            stop_sigma=float(self.stop_sigma),
            start_step=float(self.start_step),
            stop_step=float(self.stop_step),
            refs=refs,
            selection_refs=dict(self._selection_refs),
            move_free=dict(self._move_free),
            ref_fits=ref_fits,
            ref_blocks=dict(self._ref_group_resolved),
        )
