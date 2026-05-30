"""Adapter from a Protenix feature dict to the rgi_utils adapter protocols.

Protenix exposes a biotite ``AtomArray`` (with per-atom chain/res/element and a
``BondList``) plus the padded coordinate length ``N_atom``. We stash the
``atom_array`` into ``input_feature_dict`` at feature-build time; this adapter
reads it to provide:
  - FrameworkAdapter.iter_atoms        (distance restraint selection)
  - ConformerAdapter.num_atoms / get_elements / iter_ligand_confs

The ligand RDKit mol is rebuilt from the atom_array subset (elements + bonds +
coords), so mol atom ``i`` corresponds to ``global_indices[i]`` by construction —
no atom-order ambiguity. Multiple ligand chains yield disjoint global indices.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from rgi_utils.atom_context import AtomRecord, LigandConf

logger = logging.getLogger(__name__)


def _atomic_number(symbol: str) -> int:
    from rdkit.Chem import GetPeriodicTable

    try:
        return int(GetPeriodicTable().GetAtomicNumber(str(symbol).capitalize()))
    except Exception:
        return 0


def _build_ligand_mol(elements, coords, bonds_local):
    """Build an RDKit mol from a ligand's atoms (element symbols), 3D coords and
    local bond tuples ``(i, j, order)``. Chirality is assigned from 3D."""
    from rdkit import Chem

    rw = Chem.RWMol()
    conf = Chem.Conformer(len(elements))
    for i, sym in enumerate(elements):
        rw.AddAtom(Chem.Atom(str(sym).capitalize()))
        conf.SetAtomPosition(
            i, (float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2]))
        )
    order_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    for li, lj, order in bonds_local:
        rw.AddBond(int(li), int(lj), order_map.get(int(order), Chem.BondType.SINGLE))
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # geometry-only restraints don't need a clean valence model
        pass
    try:
        Chem.AssignStereochemistryFrom3D(mol)  # chiral tags for chiral restraints
    except Exception:
        pass
    return mol


class ProtenixAdapter:
    def __init__(self, input_feature_dict: dict) -> None:
        self.feats = input_feature_dict
        self.atom_array = input_feature_dict.get("atom_array")
        # padded atom count = coordinate length in the diffusion loop
        self._n_atom = int(input_feature_dict["atom_to_token_idx"].shape[-1])

    # --- FrameworkAdapter -----------------------------------------------------
    def iter_atoms(self) -> Iterator[AtomRecord]:
        aa = self.atom_array
        if aa is None:
            return
        chains = np.asarray(aa.label_asym_id)
        resids = np.asarray(aa.res_id)
        for i in range(len(aa)):
            yield AtomRecord(chain=str(chains[i]), resid=int(resids[i]), index=int(i))

    # --- ConformerAdapter -----------------------------------------------------
    def num_atoms(self) -> int:
        return self._n_atom

    def get_elements(self) -> np.ndarray:
        """(N_atom,) atomic numbers; padding atoms (beyond the atom_array) are 0."""
        elements = np.zeros(self._n_atom, dtype=np.int64)
        aa = self.atom_array
        if aa is not None:
            syms = np.asarray(aa.element)
            n = min(len(aa), self._n_atom)
            for i in range(n):
                elements[i] = _atomic_number(syms[i])
        return elements

    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        aa = self.atom_array
        if aa is None or getattr(aa, "bonds", None) is None:
            return
        hetero = np.asarray(aa.hetero, dtype=bool)
        asym = np.asarray(aa.label_asym_id)
        coords_all = np.asarray(aa.coord, dtype=np.float64)
        elements_all = np.asarray(aa.element)
        bond_arr = aa.bonds.as_array()  # (n_bond, 3): i, j, order

        for chain_id in np.unique(asym[hetero]):
            idxs = np.where((asym == chain_id) & hetero)[0]
            if len(idxs) < 2:  # lone ion: no conformer geometry to restrain
                continue
            g2l = {int(g): li for li, g in enumerate(idxs)}
            bonds_local = [
                (g2l[int(i)], g2l[int(j)], int(o))
                for i, j, o in bond_arr
                if int(i) in g2l and int(j) in g2l
            ]
            coords = coords_all[idxs]
            mol = _build_ligand_mol(elements_all[idxs], coords, bonds_local)
            yield LigandConf(
                mol=mol,
                conf_coords=coords,
                global_indices=idxs.astype(np.int64),
            )
