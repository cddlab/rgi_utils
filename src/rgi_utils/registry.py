"""Custom-restraint registry — the public extension point for adding new restraint
types to the RGI engine without editing the engine's per-type code.

The engine ships five first-class restraints (conformer / distance / angle / dihedral
/ RMSD), each hand-wired across spec / featurizer / config / energy / optim. Those five
are **frozen** as the validated reference implementation. *New* restraint types go
through this registry instead: a caller describes a restraint once as a
``RestraintType`` and calls :func:`register_restraint`; the engine's generic branches
(``_terms`` dispatch, ``featurizer`` build loop, ``config`` parse loop, ``optim`` solver
condition) then pick it up with no further edits.

Two authoring patterns share this one mechanism:

* **Code-level** (pattern A): write a ``RestraintType`` with your own data class +
  per-backend leaf energy functions. Full freedom; works on every backend.
* **Config-only** (pattern B): the engine ships one built-in ``RestraintType`` named
  ``custom`` whose data class *interprets a declarative config* (a vocabulary of
  geometric measures + standard penalty forms). Users add an original restraint by
  editing the config alone. Its leaf functions are engine code — so backend parity is
  satisfied for pattern B for free.

Design constraints honoured here:

* **numpy-only import.** This module imports numpy and the stdlib only — never torch,
  jax, ``spec`` or ``energy._terms`` (which imports *this* module, so importing it back
  would be circular). Per-backend leaf functions may therefore be given as ``"module:func"``
  dotted-path strings that are imported **lazily**, only when that backend actually runs,
  keeping top-level ``import rgi_utils`` torch/jax-free.
* **Process-global, like a class registry.** Restraint *types* are definitions, not
  per-structure state, so they live in a process-global table (the analogue of a pytest
  or RDKit plugin registry). Per-structure state stays in the instance-scoped
  ``CombinedRestraints``; the registry never holds a structure's coordinates or config.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

# Names the five frozen built-in restraints already occupy in the energy dispatch
# (``_terms._SPEC_SCHEMA`` / ``_TERMS`` prepared keys) plus the two reserved gate
# labels. A registered restraint may not reuse any of them. Hard-coded here (rather than
# imported from ``_terms``) to keep this module free of the reverse import cycle; the set
# is stable because the built-ins are frozen.
# Backends a custom restraint must supply a leaf energy for (numpy = parity reference,
# torch = 5 tools, jax = AF3). Enforced at registration to keep parity structural.
REQUIRED_BACKENDS = ("numpy", "torch", "jax")

_RESERVED_NAMES = frozenset(
    {
        "bond",
        "angle",
        "chiral",
        "improper",
        "cistrans",
        "vdw",
        "distance",
        "rmsd",
        "group_angle",
        "group_dihedral",
        # reserved gate labels (see RestraintType.gate)
        "conf",
        "dist",
    }
)

# Built-in top-level config keys a registered restraint's ``config_section`` may not
# reuse (mirrors config.py ``_KNOWN_TOP_LEVEL`` + ``start_sigma``). Hard-coded here to
# avoid importing config.py (which imports this module); the built-in surface is stable.
_RESERVED_SECTIONS = frozenset(
    {
        "verbose",
        "gpu",
        "backend",
        "method",
        "max_iter",
        "start_sigma",
        "conformer_restraints_config",
        "distance_restraints_config",
        "rmsd_restraints_config",
        "angle_restraints_config",
        "dihedral_restraints_config",
    }
)


@dataclass
class RestraintType:
    """Descriptor bundling everything the engine needs to support one restraint type.

    A registered restraint flows through the same three-layer pipeline as the built-ins:

    1. **config -> data objects.** For each entry under ``config_section`` the config
       layer instantiates ``data_class()`` and calls ``.set_config(entry_dict)``. The
       data object owns per-entry parsing (selections, target, form, weight, gate) and
       per-structure site resolution.
    2. **data objects -> spec arrays.** After ``CombinedRestraints.setup`` calls each
       object's ``.resolve_sites(adapter)``, the featurizer unions every referenced atom
       into ``active_sites`` and calls ``data_builder(items, g2l)`` to produce one
       padded arrays-dataclass (local indices via the ``g2l`` global->local map). The
       result is stored in ``spec.registered[name]``.
    3. **spec arrays -> energy.** ``spec_schema`` / ``term_args`` register the arrays and
       the leaf-call signature in the shared ``_terms`` dispatch, and ``leaf_fns`` supply
       the differentiable energy per backend. The dispatch handles masking + the
       per-restraint sigma gate identically across numpy / torch / jax.

    Contract required of ``data_class`` instances (mirrors the built-in ``*RestraintData``
    classes, e.g. ``group_geom_restr_data.AngleRestraintData``):

    * ``set_config(entry: dict) -> None`` — parse one config entry; set ``run_restr``.
    * ``resolve_sites(adapter) -> None`` — resolve selections to *global* atom indices.
    * ``run_restr: bool`` — True iff this entry is complete and should be built.
    * ``start_sigma: float | None`` / ``stop_sigma: float`` — per-entry noise gate
      (``start_sigma=None`` means "active at every step"; ``stop_sigma=-1`` means "never
      released"), used for the empty-window warning.
    * ``iter_global_sites() -> Iterable[int]`` — every global atom index this entry
      references, so the featurizer can union them into ``active_sites`` generically
      (it cannot guess which attributes hold the sites).

    ``data_builder(items: list, g2l: dict[int, int]) -> arrays`` returns an
    arrays-dataclass whose fields match ``spec_schema``; every ``*_idx`` field must hold
    *local* indices (``g2l[global_index]``) and the dataclass must carry a ``mask`` field
    (1 = valid row, 0 = padding) plus ``start_sigma`` / ``stop_sigma`` arrays (the
    per-entry gate the dispatch reads).
    """

    name: str
    """Unique key — used as the spec.registered key, the energy prepared-key, the
    ``energy_breakdown`` label and the leaf-fn lookup name. May not be a reserved name."""

    config_section: str
    """Top-level config key carrying this restraint's list of entries, e.g.
    ``"custom_restraints_config"``. Added to the config top-level whitelist on register
    (an unknown top-level section otherwise RAISES)."""

    data_class: type
    """Per-entry config/resolve class (see the contract above)."""

    data_builder: Callable[[list, dict], Any]
    """``(resolved_items, g2l) -> arrays-dataclass`` matching ``spec_schema``."""

    spec_schema: tuple
    """``((field_name, kind), ...)`` with kind ``"i"`` (int array) or ``"f"`` (float),
    one entry per field of the arrays-dataclass — the same shape as one
    ``_terms._SPEC_SCHEMA`` field list. Must include ``mask``, ``start_sigma``,
    ``stop_sigma``."""

    term_args: tuple
    """``(field_name, ...)`` passed positionally to the leaf fn, in order; the dispatch
    appends the gated ``mask`` as the final argument (so the leaf signature is
    ``(positions, *term_args_values, mask)``)."""

    leaf_fns: dict
    """``{backend_name: fn_or_path}`` for ``"numpy"`` / ``"torch"`` / ``"jax"``. Each
    value is either a direct callable or a ``"module:function"`` dotted-path string
    imported lazily (only when that backend runs), so registering a restraint never
    forces a torch/jax import. A missing backend means "unavailable there": the restraint
    raises if that backend is requested rather than running ungated."""

    gate: str = "registered"
    """Noise-gate class. Registered restraints are per-entry, CG-solved terms (each entry
    carries its own ``start_sigma`` / ``stop_sigma`` and is always summed by the solver),
    so any label other than the reserved ``"conf"`` / ``"dist"`` is accepted and lands the
    restraint in ``_terms.per_entry_keys()``. The closed-form (``"dist"``) and shared
    conformer-gate (``"conf"``) paths are reserved for built-ins."""

    n_groups: int = 1
    """Number of atom-selection groups each entry defines (for padding helpers)."""

    metadata: dict = field(default_factory=dict)
    """Optional free-form metadata (unused by the engine)."""


# Process-global table, insertion-ordered so the dispatch tables build deterministically.
_REGISTERED: dict[str, RestraintType] = {}

# Generation counter bumped on every registry mutation, so consumers (``_terms.leaf_fns_for``)
# can memoize the merged leaf-fn dict and only rebuild when the registry actually changes —
# keeping importlib + dict-build OUT of the per-``total_energy`` hot path (it is the energy
# the torch GPU CG compiles). ``_RESOLVED`` caches dotted-path imports across calls.
_GENERATION = 0
_RESOLVED: dict[tuple, Callable] = {}


def generation() -> int:
    """Monotonic registry version; changes whenever a restraint is (un)registered."""
    return _GENERATION


def _bump() -> None:
    global _GENERATION
    _GENERATION += 1
    _RESOLVED.clear()


def register_restraint(rt: RestraintType) -> None:
    """Register a custom restraint type. Raises on a name/section collision or an
    invalid gate so a mistake fails loudly at registration rather than silently
    producing a no-op restraint later."""
    if not isinstance(rt, RestraintType):
        raise TypeError(f"register_restraint expects a RestraintType, got {type(rt)!r}")
    if not rt.name or not rt.name.isidentifier():
        raise ValueError(f"restraint name must be a valid identifier, got {rt.name!r}")
    if rt.name in _RESERVED_NAMES:
        raise ValueError(
            f"restraint name {rt.name!r} collides with a built-in term; pick another"
        )
    if rt.gate in ("conf", "dist"):
        raise ValueError(
            f"gate {rt.gate!r} is reserved for built-ins; registered restraints are "
            "per-entry CG terms — use the default gate (or any other label)"
        )
    # Parity is a hard invariant of this engine, so a custom restraint must supply a leaf
    # energy for EVERY backend (numpy reference + torch + jax/AF3). They may be lazy
    # "module:func" dotted paths, so declaring all three costs no eager torch/jax import.
    # This makes the "all-backends" guarantee structural: every configured restraint has a
    # leaf fn for whatever backend runs, so the dispatch can never silently skip it.
    missing = [b for b in REQUIRED_BACKENDS if b not in rt.leaf_fns]
    if missing:
        raise ValueError(
            f"restraint {rt.name!r} leaf_fns must cover all backends "
            f"{REQUIRED_BACKENDS}; missing {missing}. Custom restraints run on every "
            "tool, so all three are required (give lazy 'module:func' paths if needed)."
        )
    existing = _REGISTERED.get(rt.name)
    if existing is not None and existing is not rt:
        raise ValueError(
            f"a different restraint named {rt.name!r} is already registered; "
            "call clear_registered() first if you mean to replace it"
        )
    if rt.config_section in _RESERVED_SECTIONS:
        raise ValueError(
            f"config_section {rt.config_section!r} is a built-in top-level config key; "
            "pick a distinct section (e.g. '<name>_restraints_config')"
        )
    for other in _REGISTERED.values():
        if other is not rt and other.config_section == rt.config_section:
            raise ValueError(
                f"config_section {rt.config_section!r} already used by restraint "
                f"{other.name!r}"
            )
    _REGISTERED[rt.name] = rt
    _bump()


def unregister_restraint(name: str) -> None:
    """Remove a single registered restraint by name (no-op if absent)."""
    if _REGISTERED.pop(name, None) is not None:
        _bump()


def clear_registered() -> None:
    """Drop every registered restraint. Primarily for test isolation."""
    _REGISTERED.clear()
    _bump()


def iter_registered() -> list[RestraintType]:
    """Registered restraints in registration order (a fresh list each call, so callers
    may build dispatch tables from it without mutating the registry)."""
    return list(_REGISTERED.values())


def get_registered(name: str) -> RestraintType | None:
    """Look up a registered restraint by name, or ``None``."""
    return _REGISTERED.get(name)


def resolve_leaf_fn(rt: RestraintType, backend: str) -> Callable | None:
    """Resolve ``rt``'s leaf energy function for ``backend``, importing a dotted-path
    ``"module:function"`` string lazily. Returns ``None`` if the restraint declares no
    function for that backend."""
    fn = rt.leaf_fns.get(backend)
    if fn is None:
        return None
    if callable(fn):
        return fn
    # dotted path "module:function" -> import lazily, then CACHE the resolved callable so a
    # repeated lookup (every total_energy call) never re-imports. Cache cleared on _bump().
    cache_key = (rt.name, backend)
    cached = _RESOLVED.get(cache_key)
    if cached is not None:
        return cached
    if isinstance(fn, str):
        module_name, sep, attr = fn.partition(":")
        if not sep:
            raise ValueError(
                f"leaf_fns[{backend!r}] dotted path {fn!r} must be 'module:function'"
            )
        resolved = getattr(importlib.import_module(module_name), attr)
        _RESOLVED[cache_key] = resolved
        return resolved
    raise TypeError(
        f"leaf_fns[{backend!r}] must be a callable or 'module:function' string, "
        f"got {type(fn)!r}"
    )


def registered_leaf_fns(backend: str) -> dict[str, Callable]:
    """``{name: leaf_fn}`` for every registered restraint that supplies ``backend``.
    Merged into a backend's built-in ``_LEAF_FNS`` at call time so the shared
    ``term_energies`` dispatch finds registered leaf functions by name."""
    out: dict[str, Callable] = {}
    for rt in _REGISTERED.values():
        fn = resolve_leaf_fn(rt, backend)
        if fn is not None:
            out[rt.name] = fn
    return out


def iter_global_sites(item: Any) -> Iterable[int]:
    """Best-effort accessor for a data object's referenced global atom indices: prefers
    an explicit ``iter_global_sites()`` method, else falls back to concatenating
    ``target_sites1..n`` attributes (the built-in group restraints' convention)."""
    method = getattr(item, "iter_global_sites", None)
    if callable(method):
        return method()
    sites: list[int] = []
    i = 1
    while True:
        attr = getattr(item, f"target_sites{i}", None)
        if attr is None:
            break
        sites.extend(int(s) for s in attr)
        i += 1
    return sites
