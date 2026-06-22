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


# the two mutually-exclusive gate windows a restraint entry may carry
_SIGMA_WINDOW_KEYS = ("start_sigma", "stop_sigma")
_STEP_WINDOW_KEYS = ("start_step", "stop_step")


def check_window_exclusive(config: dict, label: str = "restraint entry") -> None:
    """Raise if an entry mixes the sigma-window and the step-window gate keys.

    A restraint is gated on EITHER the noise level (``start_sigma`` / ``stop_sigma``) OR
    the diffusion step index (``start_step`` / ``stop_step``) — the two are mutually
    exclusive (排他選択). Mixing them is a config error, not a silent precedence rule,
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
