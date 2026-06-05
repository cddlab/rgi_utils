"""Minimal, dependency-free PDB reader for RMSD reference structures.

rgi_utils core stays numpy-only (no gemmi/biotite at import), so the RMSD
restraint parses ``ATOM``/``HETATM`` records directly. We expose just what the
atom-selection DSL (``selection.py``) needs:

- ``chain``  — chain id (column 22)
- ``resid``  — **per-chain 1-based residue ORDINAL** (resets at each chain), matching
  ``AtomRecord.resid`` (NOT the author ``resSeq``), so one ``atom_selection`` string
  selects the same residues in the reference PDB and in the diffusion structure.
- ``index``  — 0-based row in the parsed atom list (for ``index ...`` selections).
- ``element`` + ``x/y/z`` coordinates.

PDB ATOM/HETATM columns are fixed-width (PDB 3.30): record(1-6), serial(7-11),
name(13-16), altLoc(17), resName(18-20), chainID(22), resSeq(23-26), iCode(27),
x(31-38), y(39-46), z(47-54), element(77-78) — here in 0-based Python slices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rgi_utils.selection import AtomSelector


@dataclass
class PdbAtom:
    chain: str
    resid: int  # per-chain 1-based ordinal (resets per chain)
    index: int  # 0-based row in the parsed atom list
    name: str  # atom name (cols 13-16), for identity-based RMSD pairing
    element: str
    x: float
    y: float
    z: float


def read_pdb_atoms(path: str) -> list[PdbAtom]:
    """Parse ATOM/HETATM records from ``path``. Raises ValueError on a missing /
    unreadable file or one with no atoms (the RMSD restraint must fail loudly)."""
    try:
        with open(path) as fh:
            lines = fh.readlines()
    except OSError as exc:
        raise ValueError(f"rmsd ref_pdb could not be read: {path!r} ({exc})") from exc

    atoms: list[PdbAtom] = []
    chain_ord: dict[str, int] = {}
    last_key: dict[str, tuple | None] = {}
    idx = 0
    for ln in lines:
        rec = ln[0:6]
        if rec not in ("ATOM  ", "HETATM"):
            continue
        # Keep only the primary alternate location so an atom is not double-counted.
        altloc = ln[16] if len(ln) > 16 else " "
        if altloc not in (" ", "A"):
            continue
        chain = (ln[21].strip() if len(ln) > 21 else "") or " "
        res_seq = ln[22:26].strip()
        icode = ln[26] if len(ln) > 26 else " "
        name = ln[12:16].strip() if len(ln) > 15 else ""
        try:
            x = float(ln[30:38])
            y = float(ln[38:46])
            z = float(ln[46:54])
        except ValueError:
            continue  # malformed coordinate columns -> skip the record
        element = ln[76:78].strip() if len(ln) >= 78 else ""
        # per-chain 1-based residue ordinal: bump on a new (resSeq, iCode) group
        key = (res_seq, icode)
        if chain not in chain_ord:
            chain_ord[chain] = 0
            last_key[chain] = None
        if last_key[chain] != key:
            chain_ord[chain] += 1
            last_key[chain] = key
        atoms.append(
            PdbAtom(
                chain=chain,
                resid=chain_ord[chain],
                index=idx,
                name=name,
                element=element,
                x=x,
                y=y,
                z=z,
            )
        )
        idx += 1

    if not atoms:
        raise ValueError(f"rmsd ref_pdb has no ATOM/HETATM records: {path!r}")
    return atoms


def select_ref_atoms(path: str, selection: str) -> list[PdbAtom]:
    """Return the reference ``PdbAtom`` records matching ``selection`` (file order).

    ``selection`` uses the same DSL as distance restraints (chain / resid / index),
    evaluated against each parsed atom's ``{chain, resid, index}`` via
    ``AtomSelector.matches``. Each record carries its atom ``name`` so the RMSD
    restraint can pair against the diffusion structure by identity (chain, resid,
    name) rather than by selection order.
    """
    atoms = read_pdb_atoms(path)
    sel = AtomSelector(selection)
    return [
        a
        for a in atoms
        if sel.matches({"chain": a.chain, "resid": a.resid, "index": a.index})
    ]


def select_ref_coords(path: str, selection: str) -> np.ndarray:
    """``(n_sel, 3)`` coords of the matched reference atoms, in file order."""
    sel = select_ref_atoms(path, selection)
    return np.asarray([(a.x, a.y, a.z) for a in sel], dtype=np.float64).reshape(-1, 3)
