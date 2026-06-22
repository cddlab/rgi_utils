"""Lightweight named-function registry for code-level custom restraints.

``@custom_restraint("name")`` registers a reusable ``energy(ctx) -> scalar`` function so a
config entry can reference it by ``{use: "name"}``. This is the *reuse* path; the
*throwaway* path is passing a callable directly (``CombinedRestraints.add_custom(fn)`` or a
config entry ``{"fn": my_energy}``) with no registration at all.

Process-global (function definitions, like a plugin registry); per-structure state lives in
the instance-scoped ``CombinedRestraints``.
"""

from __future__ import annotations

from collections.abc import Callable

_CUSTOM_FNS: dict[str, Callable] = {}


def custom_restraint(name: str):
    """Decorator registering ``energy(ctx) -> scalar`` under ``name`` for config reference."""

    def deco(fn: Callable) -> Callable:
        if not isinstance(name, str) or not name:
            raise ValueError("custom_restraint(name): name must be a non-empty string")
        if not callable(fn):
            raise TypeError("custom_restraint expects a function energy(ctx) -> scalar")
        _CUSTOM_FNS[name] = fn
        return fn

    return deco


def get_custom_fn(name: str) -> Callable | None:
    """The registered function for ``name``, or ``None``."""
    return _CUSTOM_FNS.get(name)


def clear_custom_fns() -> None:
    """Drop all registered functions (test isolation)."""
    _CUSTOM_FNS.clear()
