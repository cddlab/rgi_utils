"""Restraint-Guided Inference utilities.

The top-level package imports only numpy-level modules so that ``import rgi_utils``
works without torch or jax installed. Heavy backends (torch/jax) are imported
lazily by ``combined.py`` / the ``energy`` and ``optim`` subpackages.
"""

import logging

from rgi_utils.atom_context import (
    AtomRecord,
    ConformerAdapter,
    FrameworkAdapter,
    LigandConf,
)
from rgi_utils.combined import CombinedRestraints
from rgi_utils.registry import (
    RestraintType,
    register_restraint,
    unregister_restraint,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Register the built-in config-only "custom" restraint (pattern B). Imported here so it
# is available out of the box; custom_restraint is numpy-only (its per-backend leaf fns
# are lazy dotted paths), so this keeps `import rgi_utils` torch/jax-free.
from rgi_utils import custom_restraint as _custom_restraint  # noqa: E402

_custom_restraint.register()

__all__ = [
    "AtomRecord",
    "FrameworkAdapter",
    "ConformerAdapter",
    "LigandConf",
    "CombinedRestraints",
    # public extension API (pattern A: register your own restraint type)
    "RestraintType",
    "register_restraint",
    "unregister_restraint",
]
