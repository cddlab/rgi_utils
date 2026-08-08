"""Shared helpers for parsing the ``restraints_config`` dict.

Kept dependency-free (no numpy/torch/jax) and import-cycle-free: both ``config.py``
and the per-restraint ``*_restr_data.py`` modules import from here, so this module
must never import them back.
"""

from __future__ import annotations

import logging
import math
from numbers import Integral, Real
from typing import Iterable

_TRUE_STRINGS = ("1", "true", "yes", "on")


VDW_SCALE_DEFAULT = 0.92
VDW_MAX_ATOM_STEP_DEFAULT = 0.1
VDW_NEIGHBOR_REBUILD_INTERVAL_DEFAULT = 10


def validate_vdw_config(conformer_config: dict | None) -> None:
    """Validate the nested conformer ``vdw`` block without importing array libraries."""
    cfg = conformer_config or {}
    if "vdw" not in cfg:
        return
    raw = cfg.get("vdw")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("conformer vdw must be a mapping")

    known = {
        "weight",
        "mode",
        "scale",
        "dmax",
        "max_neighbors",
        "max_atom_step",
        "neighbor_rebuild_interval",
    }
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"conformer vdw: unknown key(s) {sorted(unknown)}. "
            f"Known keys: {sorted(known)}"
        )

    mode = raw.get("mode", "both")
    if mode == "ligand_protein":
        raise ValueError(
            "conformer vdw mode 'ligand_protein' was renamed to 'intermolecular'"
        )
    if mode not in ("intramolecular", "intermolecular", "both"):
        raise ValueError(
            "conformer vdw mode must be 'intramolecular', 'intermolecular', or "
            f"'both', got {mode!r}"
        )

    def finite_real(key: str, default, *, positive: bool = False) -> float:
        value = raw.get(key, default)
        if value is None and key == "weight":
            return 0.0
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"conformer vdw {key} must be a finite number")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"conformer vdw {key} must be finite")
        if positive and parsed <= 0.0:
            raise ValueError(f"conformer vdw {key} must be > 0")
        return parsed

    finite_real("weight", 1.0)
    finite_real("scale", VDW_SCALE_DEFAULT, positive=True)
    finite_real("dmax", 5.0, positive=True)
    finite_real("max_atom_step", VDW_MAX_ATOM_STEP_DEFAULT, positive=True)

    for key, default in (
        ("max_neighbors", 32),
        ("neighbor_rebuild_interval", VDW_NEIGHBOR_REBUILD_INTERVAL_DEFAULT),
    ):
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"conformer vdw {key} must be an integer >= 1")
        if int(value) < 1:
            raise ValueError(f"conformer vdw {key} must be >= 1")


def coerce_bool(value, default: bool = False) -> bool:
    """Coerce a config value to ``bool``, treating quoted strings correctly.

    A quoted ``"false"`` / ``"no"`` / ``"off"`` / ``"0"`` is falsey — plain
    ``bool("false")`` would be ``True`` (any non-empty string is truthy), which is the
    trap this avoids. ``None`` -> ``default``; numeric ``0`` -> ``False``, non-zero ->
    ``True``.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in _TRUE_STRINGS


def warn_unknown_keys(
    config: dict, known: Iterable[str], label: str, logger: logging.Logger
) -> None:
    """Warn (don't raise) for any key in ``config`` not present in ``known``.

    A typo'd or misplaced entry key (e.g. a bare ``atom_selection`` on an RMSD entry,
    or a ``weight`` on a distance entry) is otherwise silently dropped — the value
    never applies and nothing flags it. Entry blocks are varied (type sub-dicts such as
    ``harmonic`` / ``flat-bottomed`` are themselves keys), so this is a soft warning,
    not a hard reject like the top-level section whitelist in ``config.py``.
    """
    unknown = set(config) - set(known)
    if unknown:
        logger.warning(
            "%s: ignoring unknown config key(s) %s (known keys: %s)",
            label,
            sorted(unknown),
            sorted(known),
        )


# the two mutually-exclusive gate windows a restraint entry may carry
_SIGMA_WINDOW_KEYS = ("start_sigma", "stop_sigma")
_STEP_WINDOW_KEYS = ("start_step", "stop_step")


def check_window_exclusive(config: dict, label: str = "restraint entry") -> None:
    """Raise if an entry mixes the sigma-window and the step-window gate keys.

    A restraint is gated on EITHER the noise level (``start_sigma`` / ``stop_sigma``) OR
    the diffusion step index (``start_step`` / ``stop_step``) — the two are mutually
    exclusive. Mixing them is a config error, not a silent precedence rule,
    so this raises rather than warning. Shared by every restraint type so the rule (and
    its message) can't diverge across distance / rmsd / angle / dihedral / conformer /
    custom.
    """
    has_sigma = any(k in config for k in _SIGMA_WINDOW_KEYS)
    has_step = any(k in config for k in _STEP_WINDOW_KEYS)
    if has_sigma and has_step:
        raise ValueError(
            f"{label}: choose either the sigma-window (start_sigma/stop_sigma) OR the "
            f"step-window (start_step/stop_step), not both — they are mutually exclusive "
            f"gates (noise level vs diffusion step index)."
        )


def apply_window_params(obj, config: dict, label: str) -> None:
    """Parse the shared ``weight`` + gate-window keys from a restraint entry onto ``obj``.

    The distance / rmsd / angle / dihedral entry classes all parse the identical block of
    ``weight`` plus the two mutually-exclusive gate windows (``start_sigma``/``stop_sigma``
    vs ``start_step``/``stop_step``). Centralising it here keeps the (subtle) null/zero
    handling from drifting between them — the bug class this guards is exactly the kind of
    per-type divergence that made rmsd's ``stop_sigma`` default differ in wording.

    Contract:
      * ``check_window_exclusive`` runs first (``label`` keys its error message to the
        calling restraint type, so the message stays specific).
      * ``weight`` is normalised: omitted / ``None`` -> 1.0; an explicit ``0`` stays 0.0 (a
        zero-weight no-op restraint), so truthiness (``or 1.0``) must NOT be used. (The
        three classes pre-init weight 1.0 EXCEPT rmsd, whose field default is ``None`` —
        this normalisation is what unifies them.)
      * the four window keys are OVERRIDDEN only when present (a ``null`` is treated as
        omitted, not a crash); ``obj`` MUST pre-initialise each window default itself
        (``start_sigma`` None, ``stop_sigma`` -1, ``start_step`` -inf, ``stop_step`` +inf)
        — the helper unifies only the parse, not the default storage.
    """
    check_window_exclusive(config, label)
    _w = config.get("weight")
    obj.weight = 1.0 if _w is None else float(_w)
    for key in _SIGMA_WINDOW_KEYS + _STEP_WINDOW_KEYS:
        value = config.get(key)
        if value is not None:
            setattr(obj, key, float(value))


def parse_move_indices(mv, n_groups: int) -> set[int] | None:
    """Parse a ``move`` config value into a set of 1-based group indices, or ``None``.

    The single source of truth for the ``move`` key's *vocabulary* — shared by the
    2-group distance restraint (``distance_restr_data``) and the 3/4-group angle/dihedral
    restraints (``group_geom_restr_data``) so the accepted spellings can't drift apart.
    Only the vocabulary is shared; each caller maps the returned index set onto its own
    representation (distance -> a 0/1/2 ``move_mode`` enum; angle/dihedral -> a per-group
    free mask).

    Accepts:
      * ``None`` (key omitted) -> returns ``None`` so the caller applies its own default;
      * ``"all"`` / ``"both"`` -> every group (``{1, .., n_groups}``);
      * a single int ``k`` (or its string form ``"k"``) -> ``{k}``;
      * a list / tuple / comma- or space-separated string of 1-based indices
        (``[1, 3]`` / ``"1,3"`` / ``"1 3"``) -> exactly those indices.

    Empty / out-of-range / non-integer values RAISE — a silent fallback would move the
    wrong groups. (``"both"`` is the distance restraint's 2-group word, kept as a synonym
    of ``"all"`` for both callers.)
    """
    if mv is None:
        return None
    if isinstance(mv, str):
        s = mv.strip().lower()
        if s in ("all", "both"):
            return set(range(1, n_groups + 1))
        items = [p for p in s.replace(",", " ").split() if p]
    elif isinstance(mv, (list, tuple)):
        items = list(mv)
    else:
        items = [mv]  # a bare int / float
    try:
        idx = {int(x) for x in items}
    except (TypeError, ValueError):
        raise ValueError(
            f"'move' must be 'all'/'both' or group indices 1..{n_groups} (got {mv!r})"
        )
    if not idx or any(k < 1 or k > n_groups for k in idx):
        raise ValueError(f"'move' indices must be within 1..{n_groups} (got {mv!r})")
    return idx


def parse_geom_type(config: dict, base: str, conv):
    """Parse the distance-style restraint-type block shared by angle/dihedral and RMSD.

    Exactly one of the four mutually-exclusive type keys may be present: ``harmonic``
    (carries ``base``), ``flat-bottomed`` (``base+"1"`` and ``base+"2"``),
    ``flat-bottomed1`` (``base+"1"`` only — lower bound), ``flat-bottomed2``
    (``base+"2"`` only — upper bound). ``base`` is e.g. ``"target_angle"`` /
    ``"target_dihedral"`` / ``"target_rmsd"``. ``conv`` is applied to each target value —
    ``math.radians`` for the angular restraints (config DEGREES -> stored RADIANS) or
    ``float`` for native-unit restraints (distance / RMSD Angstroms). Returns
    ``(geom_type_str, target1, target2)`` (the unused bound -> 0.0), or
    ``(None, None, None)`` if no type block is present. The flat-bottomed ``t1 < t2``
    ordering check runs on the RAW (pre-conv) values — it is monotone under ``conv``.
    Mirrors ``DistanceData``'s own type parse so the four types stay in lockstep.
    """
    if "harmonic" in config:
        t = config["harmonic"].get(base)
        if t is None:
            raise ValueError(f"harmonic needs {base}")
        return "harmonic", conv(t), 0.0
    if "flat-bottomed" in config:
        t1 = config["flat-bottomed"].get(f"{base}1")
        t2 = config["flat-bottomed"].get(f"{base}2")
        if t1 is None or t2 is None:
            raise ValueError(f"flat-bottomed needs {base}1 and {base}2")
        t1, t2 = float(t1), float(t2)
        if t1 >= t2:
            raise ValueError(f"{base}1 must be smaller than {base}2")
        return "flat-bottomed", conv(t1), conv(t2)
    if "flat-bottomed1" in config:
        t1 = config["flat-bottomed1"].get(f"{base}1")
        if t1 is None:
            raise ValueError(f"flat-bottomed1 needs {base}1")
        return "flat-bottomed1", conv(t1), 0.0
    if "flat-bottomed2" in config:
        t2 = config["flat-bottomed2"].get(f"{base}2")
        if t2 is None:
            raise ValueError(f"flat-bottomed2 needs {base}2")
        return "flat-bottomed2", 0.0, conv(t2)
    return None, None, None
