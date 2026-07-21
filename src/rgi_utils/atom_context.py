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
    name: str | None = None  # atom name (e.g. "CA"); enables identity-based RMSD
    # pairing (chain, resid, name) instead of selection-order. None = unavailable
    # (RMSD then falls back to selection-order pairing).
    mol_type: str | None = None  # NORMALIZED molecule type: "protein"/"dna"/"rna"/
    # "ligand", or None (water/unknown). Set by each adapter from its framework's
    # entity/molecule-type enum, normalized to these strings — raw enum ints DIFFER
    # across tools (boltz/esm DNA=1,RNA=2 vs chai/openfold RNA=1,DNA=2), so the
    # string is the only safe cross-tool currency. Powers the protein/dna/rna
    # selectors; None never matches them.
    resname: str | None = None  # 3-letter residue/CCD code (e.g. "ALA"); OPTIONAL,
    # enables the PyMOL-align-like (pairing="align") RMSD correspondence. None =
    # unavailable (that adapter has not been plumbed; align pairing then errors loudly).
    # Per-chain opt-in propagated from the sequence input. Polymer conformer geometry
    # is built only for records whose chain sets this flag.
    conformer_restraints: bool = False


def candidate_dict(record, resname_attr: str = "resname") -> dict:
    """Build the selector-input dict for one atom record (the single source of truth).

    The atom-selection DSL (``AtomSelector.eval`` / ``.matches``) consumes a six-key
    dict; distance / group / RMSD selection all need exactly the same keys. Centralising
    the construction here keeps a new selectable field from being added to some call
    sites but not others (a divergence that would make one config select different atoms
    in different restraints). ``record`` may be an ``AtomRecord`` (resname attr
    ``resname``) or a ``pdb_ref.PdbAtom`` (attr ``res_name``) — pass ``resname_attr`` for
    the latter; the other five keys share attribute names across both.
    """
    return {
        "chain": record.chain,
        "resid": record.resid,
        "index": record.index,
        "name": record.name,
        "mol_type": record.mol_type,
        "resname": getattr(record, resname_attr),
    }


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
    # Per-chain opt-in: conformer restraints are built only when this is True. Every
    # tool gets the flag from the ligand sequence input (chai uses its chain sidecar).
    conformer_restraints: bool = False


def decode_atom_name(codes) -> str | None:
    """Decode ``ord(c)-32`` char codes to an atom name, or None when empty.

    ``codes`` is a 1-D iterable of ints, one per character, encoding ``ord(char)-32``
    (0 = padding). Several frameworks store atom names this way and their adapters
    decode them identically — boltz (after a one-hot argmax), esmfold2 and AF3 read
    the codes directly — so the decode lives here as the single cross-tool kernel.
    """
    nm = "".join(chr(int(c) + 32) for c in codes if int(c) != 0).strip()
    return nm or None


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

    def get_reference_positions(self) -> "np.ndarray":
        """(num_atoms, 3) residue-local ideal coordinates for polymer geometry."""
        ...

    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        """Yield one LigandConf per ligand that should get conformer restraints."""
        ...
