from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable


@dataclass
class AtomRecord:
    """Framework-agnostic representation of a single atom for selection."""

    chain: str  # chain name (e.g. "A")
    resid: int  # 1-based residue index (token index + 1)
    index: int  # global padded atom index


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Protocol that any structure prediction framework must implement to provide atom
    records."""

    def iter_atoms(self) -> Iterator[AtomRecord]:
        """Iterate over all non-padded atoms in the structure."""
        ...
