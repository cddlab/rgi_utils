"""Shared helpers for parsing the ``restraints_config`` dict.

Kept dependency-free (no numpy/torch/jax) and import-cycle-free: both ``config.py``
and the per-restraint ``*_restr_data.py`` modules import from here, so this module
must never import them back.
"""

from __future__ import annotations

import logging
from typing import Iterable

_TRUE_STRINGS = ("1", "true", "yes", "on")


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
