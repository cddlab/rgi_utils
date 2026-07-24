"""Custom restraints — define an ORIGINAL restraint and run it, two ways:

* **config (expression DSL)**: a ``custom_restraints_config`` entry with an ``energy``
  formula string over a shared geometry+math+penalty vocabulary and named selections,
  e.g. ``energy: "(distance(A,B) - distance(C,D))**2"``. Reference-backed
  selections use ``refN and <selection>`` and work with the same geometry vocabulary.
* **code (ctx function)**: a function ``energy(ctx) -> scalar`` using the same vocabulary
  via ``ctx`` — passed directly (``CombinedRestraints.add_custom(fn)`` / config
  ``{"fn": ...}``) or registered with ``@custom_restraint("name")`` and referenced from
  config by ``{"use": "name"}``.

Both compile to one backend-agnostic energy that runs on numpy / torch / jax. This package
imports numpy only (torch/jax are pulled lazily per backend), so ``import rgi_utils`` stays
torch/jax-free.
"""

from __future__ import annotations

from rgi_utils.custom.data import CustomData, CustomSpec
from rgi_utils.custom.registry import custom_restraint, get_custom_fn

__all__ = ["custom_restraint", "get_custom_fn", "CustomData", "CustomSpec"]
