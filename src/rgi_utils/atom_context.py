from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from rdkit import Chem


@dataclass
class AtomRecord:
    """Framework-agnostic representation of a single atom for selection."""

    chain: str  # chain name (e.g. "A")
    resid: int  # 1-based residue/token ordinal WITHIN the chain (resets per chain)
    index: int  # global padded atom index


@dataclass
class LigandConf:
    """One ligand's ideal conformer, used to build conformer restraints.

    ``global_indices[i]`` is the global padded atom index of RDKit mol atom ``i``,
    so bond/angle/chiral restraints derived from the mol can be expressed directly
    in the framework's atom space. Multiple ligands are handled by supplying one
    ``LigandConf`` each — their disjoint ``global_indices`` prevent any site
    collision.
    """

    mol: "Chem.Mol"  # H-removed RDKit mol
    conf_coords: "np.ndarray"  # (n_lig_atoms, 3) ideal conformer coordinates
    global_indices: "np.ndarray"  # (n_lig_atoms,) global padded atom index per mol atom
    invert_chirality: bool = False
    atom_names: list[str] | None = None  # optional filter (e.g. CCD atom names)
    # per-ligand opt-in: when False this ligand is skipped entirely (no bond/
    # angle/chiral/VdW restraints). Adapters set it from the tool's per-ligand
    # ``conformer_restraints`` input flag; default True keeps existing callers
    # (which already pass only restrained ligands) unchanged.
    conformer_restraints: bool = True


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Minimal protocol: iterate non-padded atoms for distance-restraint selection."""

    def iter_atoms(self) -> Iterator[AtomRecord]:
        """Iterate over all non-padded atoms in the structure."""
        ...


@runtime_checkable
class ConformerAdapter(Protocol):
    """Optional protocol for conformer (bond/angle/chiral) and VdW restraints.

    A framework only needs to implement this if it uses conformer/VdW restraints.
    ``CombinedRestraints`` checks for it at runtime, so distance-only tools can
    implement just ``FrameworkAdapter``.
    """

    def num_atoms(self) -> int:
        """Total number of (padded) atoms — the global flat coordinate length."""
        ...

    def get_elements(self) -> "np.ndarray":
        """(num_atoms,) atomic numbers; 0 marks padding. Used for VdW radii."""
        ...

    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        """Yield one LigandConf per ligand that should get conformer restraints."""
        ...
