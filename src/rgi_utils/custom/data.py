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
from dataclasses import dataclass
from typing import Any

import numpy as np

from rgi_utils._config_util import apply_window_params, warn_unknown_keys
from rgi_utils.custom.context import ResolveContext
from rgi_utils.custom.dsl import eval_formula, parse_formula
from rgi_utils.custom.registry import get_custom_fn
from rgi_utils.group_geom_restr_data import _resolve_group_sites

logger = logging.getLogger(__name__)

_KNOWN_CUSTOM_KEYS = {
    "name",
    "energy",
    "use",
    "fn",
    "selections",
    "weight",
    "start_sigma",
    "stop_sigma",
    "start_step",
    "stop_step",
}


@dataclass
class CustomSpec:
    """Backend-agnostic resolved custom restraint (stored in ``RestraintSpec.custom``).

    ``selections`` maps each identifier to its LOCAL indices (into active_sites). ``kind``
    is ``"formula"`` (evaluate ``ast``) or ``"fn"`` (call ``fn``). Optimizers build a
    backend closure ``(active_coords) -> scalar`` from this (see ``custom.closure``)."""

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

    def set_config(self, config: dict) -> None:
        warn_unknown_keys(
            config, _KNOWN_CUSTOM_KEYS, "custom_restraints_config entry", logger
        )
        self.name = str(config.get("name", "custom"))
        self.selections = dict(config.get("selections", {}) or {})
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

    def iter_global_sites(self):
        out: list[int] = []
        for s in self._global.values():
            out.extend(int(x) for x in s)
        return out

    def build_spec(self, g2l: dict[int, int]) -> CustomSpec:
        """Resolved global indices -> CustomSpec with LOCAL index arrays."""
        local = {
            i: np.array([g2l[int(x)] for x in sites], dtype=np.int64)
            for i, sites in self._global.items()
        }
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
        )
