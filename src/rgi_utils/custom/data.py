"""One ``custom_restraints_config`` entry: parse + per-structure selection resolution.

A custom restraint is a backend-agnostic energy ``energy(ctx) -> scalar`` given as either
a config ``energy`` formula string, a registered function referenced by ``use``, or a
callable passed directly via ``fn``. ``CustomData`` parses the entry and, at setup,
resolves the selections the energy touches to atoms via a *resolve pass* (run the energy
with a ``ResolveContext`` that records selection identifiers and returns shaped dummies).
The featurizer turns the resolved global indices into ``CustomSpec`` (local indices +
the AST/fn + weight/sigmas) which the optimizers compile to a per-backend closure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rgi_utils._config_util import (
    apply_window_params,
    coerce_bool,
    warn_unknown_keys,
)
from rgi_utils._moltype import polymer_type
from rgi_utils.atom_context import candidate_dict
from rgi_utils.custom.context import ResolveContext
from rgi_utils.custom.dsl import eval_formula, parse_formula
from rgi_utils.custom.registry import get_custom_fn
from rgi_utils.group_geom_restr_data import _resolve_group_sites
from rgi_utils.pdb_ref import read_cif_atoms, read_pdb_atoms
from rgi_utils.rmsd_restr_data import build_resid_map, pair_target_to_ref
from rgi_utils.selection import AtomSelector

logger = logging.getLogger(__name__)


def _select_ref_coords(ref_atoms, sel_string, tag, ref_path):
    """``(k, 3)`` coordinates of the reference atoms matched by ``sel_string`` (the atom-
    selection DSL evaluated ON THE REFERENCE, so the same vocabulary picks ref atoms as it does
    prediction atoms). Raises loudly if nothing matches (a ref() over an empty group is a
    silent no-op otherwise)."""
    sr = AtomSelector(sel_string)
    coords = [
        (a.x, a.y, a.z)
        for a in ref_atoms
        if sr.eval(candidate_dict(a, resname_attr="res_name"))
    ]
    if not coords:
        raise ValueError(
            f"{tag}: reference selection matched no atoms: {sel_string!r} in {ref_path!r}"
        )
    return np.asarray(coords, dtype=np.float64)


_KNOWN_CUSTOM_KEYS = {
    "name",
    "energy",
    "use",
    "fn",
    "selections",
    "refs",
    "weight",
    "start_sigma",
    "stop_sigma",
    "start_step",
    "stop_step",
}
# one reference definition inside a custom entry's 'refs' map. ``atom_selection_ref`` is the
# rmsd() calc selection; ``atom_selection_{ref,target}_fit`` are the ref() Kabsch-fit anchor
# selections (reference side / prediction side). A ref may carry both (used by rmsd() and ref()).
_KNOWN_REF_KEYS = {
    "ref_pdb",
    "ref_cif",
    "atom_selection_ref",
    "atom_selection_ref_fit",
    "atom_selection_target_fit",
    "pairing",
    "best_effort",
}


@dataclass
class CustomSpec:
    """Backend-agnostic resolved custom restraint (stored in ``RestraintSpec.custom``).

    ``selections`` maps each identifier to its LOCAL indices (into active_sites). ``kind``
    is ``"formula"`` (evaluate ``ast``) or ``"fn"`` (call ``fn``). ``refs`` maps a
    ``(selection_identifier, ref_name)`` pair to ``(target_local_idx, ref_coords)`` for the
    ``rmsd`` primitive: ``target_local_idx`` are the LOCAL indices of the MATCHED subset of
    that selection's atoms (a best-effort pairing can drop unmatched atoms), and
    ``ref_coords`` is the ``(m, 3)`` reference block row-aligned to them — so
    ``rmsd(A, ref)`` gathers exactly that subset and compares to the reference. Empty unless
    the restraint uses ``rmsd(A, ref)``. Optimizers build a backend closure
    ``(active_coords) -> scalar`` from this (see ``custom.closure``)."""

    name: str
    selections: dict[str, np.ndarray]
    kind: str
    ast: Any
    fn: Any
    weight: float
    start_sigma: float
    stop_sigma: float
    start_step: float  # step-window lower bound (-inf = always); XOR the sigma window
    stop_step: float  # step-window upper bound (+inf = always)
    refs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    # reference-anchored geometry (the ref() primitive), empty unless it is used:
    #   ref_fits[ref_name] = (fit_target_local_idx, fit_ref_coords) — the prediction anchor's
    #     LOCAL indices (into active_sites) + the constant reference anchor block, for the
    #     per-eval Kabsch fit of the reference onto the prediction. A ref with no fit selections
    #     has no entry here (its ref() blocks are used in the reference's own frame).
    #   ref_blocks[(reference_selection, ref_name)] = (k, 3) constant block of atoms selected
    #     ON THE REFERENCE (reference frame), placed into the prediction frame at eval time.
    ref_fits: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    ref_blocks: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    # ref-anchored BUILT-IN geometry (kind == "ref_geom"): a distance/angle/dihedral over group
    # centroids where some groups are prediction (RIGID centroid, so weight=1 moves a large group
    # as a body) and some are fitted reference groups (fixed). Routed here from the distance /
    # angle / dihedral config sections when they carry reference keys, so the closure path is
    # reused (no 3-backend energy change) while the prediction-group rigid-translation of the
    # built-in group terms is preserved. Empty for the formula / fn kinds.
    geom: str = ""  # "distance" | "angle" | "dihedral"
    groups: list = field(
        default_factory=list
    )  # [("pred", local_idx_ndarray) | ("ref", (sel, ref_name))]
    geom_type_code: int = 0  # spec.DIST_TYPE_CODES (0=harmonic .. 3=flat-bottomed2)
    target1: float = 0.0  # radians for angle/dihedral, Angstrom for distance
    target2: float = 0.0


class CustomData:
    """Parse one entry + resolve its selections to global atom indices."""

    def __init__(self) -> None:
        self.name: str = "custom"
        self.kind: str | None = None  # "formula" | "fn"
        self.ast: Any = None
        self.fn: Any = None
        self.selections: dict[str, str] = {}  # config name -> selection string
        self.weight: float = 1.0
        self.start_sigma: float | None = None
        self.stop_sigma: float = -1.0
        # step-window (XOR the sigma window); omitted -> -inf/+inf = always.
        self.start_step: float = float("-inf")
        self.stop_step: float = float("inf")
        self.run_restr: bool = False
        self._identifiers: list[str] = []
        self._global: dict[str, list[int]] = {}
        # external references for the rmsd() primitive: {ref_name -> parsed ref def};
        # {(sel, ref) -> (matched target globals, ref coords)} filled at resolve.
        self.ref_defs: dict[str, dict] = {}
        self._ref_pairs: list[tuple[str, str]] = []
        self._ref_resolved: dict[tuple[str, str], tuple[list[int], np.ndarray]] = {}
        # reference-anchored geometry (the ref() primitive), filled at resolve:
        #   _ref_group_resolved[(sel, ref)] = (k,3) ref-frame coords of that ref selection
        #   _ref_fit_resolved[ref] = (fit_target_globals, fit_ref_coords) | None (no fit)
        self._ref_group_pairs: list[tuple[str, str]] = []
        self._ref_group_resolved: dict[tuple[str, str], np.ndarray] = {}
        self._ref_fit_resolved: dict[str, tuple[list[int], np.ndarray] | None] = {}

    def set_config(self, config: dict) -> None:
        warn_unknown_keys(
            config, _KNOWN_CUSTOM_KEYS, "custom_restraints_config entry", logger
        )
        self.name = str(config.get("name", "custom"))
        self.selections = dict(config.get("selections", {}) or {})
        self.ref_defs = self._parse_refs(config.get("refs", {}) or {})
        # weight + the sigma/step gate windows: one shared parse (so the null/zero handling
        # can't drift across distance/rmsd/angle/dihedral/custom). The windows default to
        # always-on (set in __init__); start_sigma None -> +inf is filled by from_dict.
        apply_window_params(self, config, "custom_restraints_config entry")

        sources = [k for k in ("energy", "use", "fn") if config.get(k) is not None]
        if len(sources) != 1:
            raise ValueError(
                "custom_restraints_config entry needs exactly one of 'energy' (formula), "
                f"'use' (registered name), or 'fn' (callable); got {sources}"
            )
        src = sources[0]
        if src == "energy":
            self.kind = "formula"
            self.ast = parse_formula(str(config["energy"]))
        elif src == "use":
            fn = get_custom_fn(str(config["use"]))
            if fn is None:
                raise ValueError(
                    f"custom restraint 'use': no function registered as "
                    f"{config['use']!r} (register it with @custom_restraint)"
                )
            self.kind, self.fn = "fn", fn
        else:  # direct callable
            if not callable(config["fn"]):
                raise ValueError("custom_restraints_config 'fn' must be a callable")
            self.kind, self.fn = "fn", config["fn"]
        self.run_restr = True

    def _parse_refs(self, refs: dict) -> dict:
        """Parse the entry's ``refs`` map ({ref_name -> {ref_pdb|ref_cif, atom_selection_ref,
        pairing, best_effort}}) used by the ``rmsd()`` primitive. Mirrors the built-in
        ``rmsd_restraints_config`` reference keys so the two behave the same."""
        out: dict[str, dict] = {}
        for rname, rcfg in refs.items():
            rcfg = dict(rcfg or {})
            warn_unknown_keys(rcfg, _KNOWN_REF_KEYS, f"custom refs[{rname!r}]", logger)
            ref_pdb = rcfg.get("ref_pdb")
            ref_cif = rcfg.get("ref_cif")
            if (ref_pdb is None) == (ref_cif is None):
                raise ValueError(
                    f"custom refs[{rname!r}]: give exactly one of ref_pdb / ref_cif"
                )
            pairing = rcfg.get("pairing") or "align"
            if pairing not in ("identity", "align"):
                raise ValueError(
                    f"custom refs[{rname!r}]: pairing must be 'identity' or 'align', "
                    f"got {pairing!r}"
                )
            out[str(rname)] = {
                "ref_cif": ref_cif,
                "ref_path": ref_pdb if ref_pdb is not None else ref_cif,
                "sel_ref": rcfg.get("atom_selection_ref"),
                # ref() Kabsch-fit anchor: prediction side (target) + reference side. Both None
                # -> ref() uses the reference in its own frame (no superposition).
                "sel_ref_fit": rcfg.get("atom_selection_ref_fit"),
                "sel_target_fit": rcfg.get("atom_selection_target_fit"),
                "pairing": pairing,
                "best_effort": coerce_bool(rcfg.get("best_effort"), True),
            }
        return out

    def _evaluate_resolve(self, rc: ResolveContext) -> None:
        if self.kind == "formula":
            eval_formula(self.ast, rc)
        else:
            self.fn(rc)

    def resolve_sites(self, adapter) -> None:
        if not self.run_restr:
            return
        # resolve pass: run the energy ONCE with a recording ctx to collect the selection
        # identifiers it touches. The DSL (`energy=`) path is safe because eval_formula
        # evaluates every Call arg eagerly (even where(c,a,b) touches all three). A `fn`/
        # `use` CODE restraint is arbitrary Python, so DATA-DEPENDENT BRANCHING resolves only
        # the branch the fixed resolve-dummies take; a different branch on real coords then
        # hits an UNRESOLVED selection and RestraintContext._idx raises a clear error. Rule
        # for code restraints: reference every selection unconditionally (outside branches).
        rc = ResolveContext()
        self._evaluate_resolve(rc)
        self._identifiers = list(rc.selections)
        if not self._identifiers:
            if rc.ref_groups:
                # every measured group is a ref(...) -> a constant (the fitted reference is
                # stop-gradient'd), so the restraint has zero gradient and can't move anything.
                raise ValueError(
                    f"custom restraint {self.name!r} measures only reference atoms (every "
                    "group is ref(...)) — at least one group must be a PREDICTION selection, "
                    "else the energy is constant with zero gradient"
                )
            raise ValueError(
                f"custom restraint {self.name!r} references no atom selection "
                "(use distance/angle/.../centroid/rg over named or string selections)"
            )
        # a config name maps via 'selections'; a raw string is its own selection
        sel_strings = [self.selections.get(i, i) for i in self._identifiers]
        resolved = _resolve_group_sites(adapter, sel_strings)
        self._global = dict(zip(self._identifiers, resolved))
        logger.info(
            "custom restraint (%s) resolved: %s",
            self.name,
            ", ".join(f"{i}={len(s)}" for i, s in self._global.items()),
        )
        # kabsch(A, B) needs a positional 1:1 atom correspondence -> |A| == |B|. Raise loudly
        # at setup (the runtime matmul would otherwise fail with a cryptic shape error).
        for a, b in rc.kabsch_pairs:
            na, nb = len(self._global[a]), len(self._global[b])
            if na != nb:
                raise ValueError(
                    f"custom restraint {self.name!r}: kabsch({a}, {b}) needs equal atom "
                    f"counts (positional correspondence), got |{a}|={na} != |{b}|={nb}"
                )
        self._resolve_refs(adapter, rc.refs)
        self._resolve_ref_groups(adapter, rc.ref_groups)

    def _resolve_refs(self, adapter, ref_pairs) -> None:
        """For each ``rmsd(sel, ref)`` call recorded in the resolve pass, load the reference
        and pair it to that selection's atoms (reusing the built-in RMSD ``pair_target_to_ref``
        / ``build_resid_map``), storing the MATCHED subset of target globals + the ref coords
        row-aligned to them. The ref is aligned to a subset of the selection's atoms, so the
        alignment holds even under best-effort skipping (unlike relying on selection order)."""
        self._ref_pairs = list(ref_pairs)
        self._ref_resolved = {}
        if not self._ref_pairs:
            return
        atoms = list(adapter.iter_atoms())
        has_polymer = any(
            polymer_type(a.mol_type, a.resname) is not None for a in atoms
        )
        cache: dict[str, tuple] = {}  # ref_name -> (ref_atoms, align, resid_map)
        for sel_name, ref_name in self._ref_pairs:
            rdef = self.ref_defs.get(ref_name)
            if rdef is None:
                raise ValueError(
                    f"custom restraint {self.name!r}: rmsd() uses reference {ref_name!r} "
                    "which is not defined in the entry's 'refs' map"
                )
            if ref_name not in cache:
                reader = (
                    read_cif_atoms if rdef["ref_cif"] is not None else read_pdb_atoms
                )
                ref_atoms = reader(rdef["ref_path"])
                align = rdef["pairing"] == "align" and has_polymer
                resid_map = (
                    build_resid_map(atoms, ref_atoms, rdef["ref_path"])
                    if align
                    else None
                )
                cache[ref_name] = (ref_atoms, align, resid_map)
            ref_atoms, align, resid_map = cache[ref_name]
            sel_string = self.selections.get(sel_name, sel_name)
            tgt_globals, ref_coords = pair_target_to_ref(
                atoms,
                ref_atoms,
                sel_string,
                rdef["sel_ref"],
                f"custom {self.name}:{sel_name}->{ref_name}",
                ref_path=rdef["ref_path"],
                best_effort=rdef["best_effort"],
                align=align,
                resid_map=resid_map,
            )
            self._ref_resolved[(sel_name, ref_name)] = (tgt_globals, ref_coords)
            logger.info(
                "custom restraint (%s) rmsd(%s -> %s): %d atoms paired to ref %s",
                self.name,
                sel_name,
                ref_name,
                len(tgt_globals),
                rdef["ref_path"],
            )

    def _resolve_ref_groups(self, adapter, ref_group_pairs) -> None:
        """For each ``ref(sel, r)`` call recorded in the resolve pass: (1) resolve ``sel`` on the
        REFERENCE structure to its ``(k,3)`` coords, and (2) resolve each used reference's Kabsch
        FIT once (the prediction anchor ``atom_selection_target_fit`` paired to the reference
        anchor ``atom_selection_ref_fit`` via the shared ``pair_target_to_ref``). The fit is
        shared by every ``ref(sel, r)`` with the same ``r``; a ref with no fit selections is used
        in its own frame (``_ref_fit_resolved[r] = None``)."""
        self._ref_group_pairs = list(ref_group_pairs)
        self._ref_group_resolved = {}
        self._ref_fit_resolved = {}
        if not self._ref_group_pairs:
            return
        atoms = list(adapter.iter_atoms())
        has_polymer = any(
            polymer_type(a.mol_type, a.resname) is not None for a in atoms
        )
        cache: dict[str, tuple] = {}  # ref_name -> (ref_atoms, align, resid_map)
        used_refs: list[str] = []
        for sel_name, ref_name in self._ref_group_pairs:
            rdef = self.ref_defs.get(ref_name)
            if rdef is None:
                raise ValueError(
                    f"custom restraint {self.name!r}: ref() uses reference {ref_name!r} "
                    "which is not defined in the entry's 'refs' map"
                )
            if ref_name not in cache:
                reader = (
                    read_cif_atoms if rdef["ref_cif"] is not None else read_pdb_atoms
                )
                ref_atoms = reader(rdef["ref_path"])
                align = rdef["pairing"] == "align" and has_polymer
                resid_map = (
                    build_resid_map(atoms, ref_atoms, rdef["ref_path"])
                    if align
                    else None
                )
                cache[ref_name] = (ref_atoms, align, resid_map)
                used_refs.append(ref_name)
            ref_atoms, align, resid_map = cache[ref_name]
            # ref() selection is a REFERENCE selection (a config name maps via 'selections' to
            # its string, then it is evaluated on the reference atoms, not the prediction).
            sel_string = self.selections.get(sel_name, sel_name)
            self._ref_group_resolved[(sel_name, ref_name)] = _select_ref_coords(
                ref_atoms,
                sel_string,
                f"custom {self.name!r} ref({sel_name}->{ref_name})",
                rdef["ref_path"],
            )
        # resolve each used reference's fit ONCE (shared across its ref(sel, r) calls).
        for ref_name in used_refs:
            rdef = self.ref_defs[ref_name]
            ref_atoms, align, resid_map = cache[ref_name]
            if rdef["sel_target_fit"] is None and rdef["sel_ref_fit"] is None:
                self._ref_fit_resolved[ref_name] = None  # no fit -> reference own frame
                continue
            fit_target_globals, fit_ref_coords = pair_target_to_ref(
                atoms,
                ref_atoms,
                rdef["sel_target_fit"],
                rdef["sel_ref_fit"],
                f"custom {self.name!r} fit->{ref_name}",
                ref_path=rdef["ref_path"],
                best_effort=rdef["best_effort"],
                align=align,
                resid_map=resid_map,
            )
            if len(fit_target_globals) < 3:
                raise ValueError(
                    f"custom restraint {self.name!r}: ref fit for {ref_name!r} paired only "
                    f"{len(fit_target_globals)} anchor atom(s); the Kabsch superposition needs "
                    "at least 3 non-collinear atoms (widen atom_selection_target_fit / "
                    "atom_selection_ref_fit)"
                )
            self._ref_fit_resolved[ref_name] = (fit_target_globals, fit_ref_coords)
            logger.info(
                "custom restraint (%s) ref fit -> %s: %d anchor atoms paired to ref %s",
                self.name,
                ref_name,
                len(fit_target_globals),
                rdef["ref_path"],
            )

    def iter_global_sites(self):
        out: list[int] = []
        for s in self._global.values():
            out.extend(int(x) for x in s)
        # matched rmsd target atoms are a subset of their selection, but include them
        # explicitly so they are guaranteed in active_sites (hence resolvable via g2l).
        for tgt_globals, _ref in self._ref_resolved.values():
            out.extend(int(x) for x in tgt_globals)
        # ref() fit anchors are prediction atoms the Kabsch fit reads at eval time -> they must
        # be in active_sites so the closure can gather them (by their local index).
        for fit in self._ref_fit_resolved.values():
            if fit is not None:
                out.extend(int(x) for x in fit[0])
        return out

    def build_spec(self, g2l: dict[int, int]) -> CustomSpec:
        """Resolved global indices -> CustomSpec with LOCAL index arrays."""
        local = {
            i: np.array([g2l[int(x)] for x in sites], dtype=np.int64)
            for i, sites in self._global.items()
        }
        refs = {
            key: (
                np.array([g2l[int(x)] for x in tgt_globals], dtype=np.int64),
                ref_coords,
            )
            for key, (tgt_globals, ref_coords) in self._ref_resolved.items()
        }
        # ref() primitive: fit-anchor prediction globals -> LOCAL indices; ref-group coords
        # kept as numpy constants (converted to the live dtype/device at eval via const_like).
        ref_fits = {}
        for rname, fit in self._ref_fit_resolved.items():
            if fit is None:
                continue
            fit_target_globals, fit_ref_coords = fit
            ref_fits[rname] = (
                np.array([g2l[int(x)] for x in fit_target_globals], dtype=np.int64),
                fit_ref_coords,
            )
        ref_blocks = dict(self._ref_group_resolved)
        return CustomSpec(
            name=self.name,
            selections=local,
            kind=self.kind,
            ast=self.ast,
            fn=self.fn,
            weight=float(self.weight),
            start_sigma=float(self.start_sigma)
            if self.start_sigma is not None
            else float("inf"),
            stop_sigma=float(self.stop_sigma),
            start_step=float(self.start_step),
            stop_step=float(self.stop_step),
            refs=refs,
            ref_fits=ref_fits,
            ref_blocks=ref_blocks,
        )
