"""Shared biotite-``AtomArray`` extraction for the protenix + openfold3 adapters.

Both tools expose the SAME biotite ``AtomArray`` type (per-atom element/chain + a
``BondList``), so ``get_elements`` and the ligand-conformer loop are identical up to
a few framework knobs: which annotation marks a ligand atom, which chain-id field
names the chain, where the conformer geometry comes from, and the default
``conformer_restraints`` opt-in when the per-ligand annotation is absent. Those knobs
are parameters here so the two adapters stay thin and can't drift apart.

Like the adapters, this module does NOT import biotite — the ``AtomArray`` is
duck-typed (``.element`` / ``.bonds`` / ``.get_annotation_categories`` / named
annotations), so ``import rgi_utils`` stays numpy-only.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

from rgi_utils._mol_build import atomic_number as _atomic_number
from rgi_utils._mol_build import build_ligand_mol as _build_ligand_mol
from rgi_utils.atom_context import LigandConf


def biotite_get_elements(atom_array, n_atom: int) -> np.ndarray:
    """(n_atom,) atomic numbers from a biotite AtomArray; padding atoms (beyond the
    AtomArray length) are 0."""
    elements = np.zeros(int(n_atom), dtype=np.int64)
    if atom_array is not None:
        syms = np.asarray(atom_array.element)
        n = min(len(atom_array), int(n_atom))
        for i in range(n):
            elements[i] = _atomic_number(syms[i])
    return elements


def biotite_ligand_confs(
    atom_array,
    *,
    ligand_mask: np.ndarray,
    chain_attr: str,
    coords_all: np.ndarray,
    conf_rest_default: bool,
    post_build: Callable | None = None,
) -> Iterator[LigandConf]:
    """Yield one ``LigandConf`` per ligand chain of a biotite ``AtomArray``.

    Shared core of the protenix + openfold3 adapters; the framework-specific knobs
    are parameters:
      - ``ligand_mask`` (n_atom,) bool: which atoms are ligand atoms (protenix uses
        biotite ``hetero``; openfold uses ``molecule_type_id == LIGAND``).
      - ``chain_attr``: AtomArray annotation naming the chain (protenix
        ``label_asym_id``; openfold ``chain_id``).
      - ``coords_all`` (n_atom, 3): conformer-geometry source aligned to the atom
        order (protenix ``atom_array.coord``; openfold ``ref_pos``).
      - ``conf_rest_default``: per-ligand opt-in used when the AtomArray has no
        ``conformer_restraints`` annotation (protenix + openfold both False -> the
        per-ligand flag is required; each tool sets the annotation from its input).
      - ``post_build(chain_id, mol, coords, idxs, elements_all, bonds_local) ->
        (mol, coords)``: optional hook to replace the target geometry per ligand
        (protenix rebuilds a stereo-correct SMILES ideal conformer); None = identity.

    A monatomic ion (1 atom, 0 bonds) still yields a ``LigandConf`` (no bond/angle/
    chiral terms, but it joins fixed-background VdW), matching boltz.
    """
    elements_all = np.asarray(atom_array.element)
    coords_all = np.asarray(coords_all, dtype=np.float64)
    chains = np.asarray(getattr(atom_array, chain_attr))
    ligand_mask = np.asarray(ligand_mask, dtype=bool)
    # per-ligand conformer_restraints opt-in (annotation); absent -> conf_rest_default.
    conf_rest_annot = None
    if "conformer_restraints" in atom_array.get_annotation_categories():
        conf_rest_annot = np.asarray(atom_array.conformer_restraints, dtype=bool)
    # bonds may be absent (a structure whose only hetero atoms are monatomic ions has
    # no BondList); treat as no bonds rather than bailing so the ions still surface as
    # LigandConf for fixed-background VdW (matching boltz).
    bond_arr = (
        atom_array.bonds.as_array()  # (n_bond, 3): i, j, order
        if getattr(atom_array, "bonds", None) is not None
        else np.empty((0, 3), dtype=np.int64)
    )

    for chain_id in np.unique(chains[ligand_mask]):
        idxs = np.where((chains == chain_id) & ligand_mask)[0]
        g2l = {int(g): li for li, g in enumerate(idxs)}
        bonds_local = [
            (g2l[int(i)], g2l[int(j)], int(o))
            for i, j, o in bond_arr
            if int(i) in g2l and int(j) in g2l
        ]
        coords = coords_all[idxs]
        mol = _build_ligand_mol(elements_all[idxs], coords, bonds_local)
        if post_build is not None:
            mol, coords = post_build(
                chain_id, mol, coords, idxs, elements_all, bonds_local
            )
        conf_rest = conf_rest_default
        if conf_rest_annot is not None:
            conf_rest = bool(conf_rest_annot[idxs].any())
        yield LigandConf(
            mol=mol,
            conf_coords=coords,
            global_indices=idxs.astype(np.int64),
            conformer_restraints=conf_rest,
        )
