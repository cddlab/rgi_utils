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

from rgi_utils._biotite_adapter import biotite_get_elements, biotite_ligand_confs
from rgi_utils._mol_build import build_ligand_mol as _build_ligand_mol
from rgi_utils.atom_context import AtomRecord, LigandConf

logger = logging.getLogger(__name__)


# protenix biotite AtomArray.mol_type is already a normalized string
# ("protein"/"rna"/"dna"/"ligand"); pass the polymer/ligand values through and map
# anything else (water/empty/unknown) to None. mol_type is entity-derived, so a MODIFIED
# residue in a protein chain reads "protein" -> forwarding it powers protein/dna/rna +
# backbone/sidechain selectors and RMSD align pairing for modified residues.
_VALID_MOLTYPES = {"protein", "dna", "rna", "ligand"}


def _norm_moltype(v) -> "str | None":
    s = str(v).strip().lower()
    return s if s in _VALID_MOLTYPES else None


class ProtenixAdapter:
    def __init__(self, input_feature_dict: dict) -> None:
        self.feats = input_feature_dict
        self.atom_array = input_feature_dict.get("atom_array")
        # {label_asym_id -> SMILES} for SMILES ligands (set by json_to_feature); used to
        # build a stereo-correct ideal conformer as the restraint target.
        self._smiles_by_chain = input_feature_dict.get("smiles_by_chain", {}) or {}
        # padded atom count = coordinate length in the diffusion loop
        self._n_atom = int(input_feature_dict["atom_to_token_idx"].shape[-1])

    # --- FrameworkAdapter -----------------------------------------------------
    def iter_atoms(self) -> Iterator[AtomRecord]:
        aa = self.atom_array
        if aa is None:
            return
        chains = np.asarray(aa.label_asym_id)
        resids = np.asarray(aa.res_id)
        hetero = np.asarray(aa.hetero, dtype=bool)
        names = np.asarray(aa.atom_name) if hasattr(aa, "atom_name") else None
        resnames = np.asarray(aa.res_name) if hasattr(aa, "res_name") else None
        mtypes = np.asarray(aa.mol_type) if hasattr(aa, "mol_type") else None
        categories = aa.get_annotation_categories()
        conf_restraints = (
            np.asarray(aa.conformer_restraints, dtype=bool)
            if "conformer_restraints" in categories
            else None
        )
        # Per-chain 1-based residue/token ordinal, matching boltz/AF3 so one
        # selection string means the same atom in every tool. protenix tokenizes a
        # ligand per atom but sets res_id=1 for ALL atoms of a single-CCD ligand,
        # so res_id can't be trusted directly: count polymer residues by res_id
        # group, but give each hetero (ligand) atom its own ordinal -> 1..N.
        chain_resmap: dict[str, dict[int, int]] = {}
        chain_counter: dict[str, int] = {}
        for i in range(len(aa)):
            ch = str(chains[i])
            if bool(hetero[i]):
                chain_counter[ch] = chain_counter.get(ch, 0) + 1
                ordinal = chain_counter[ch]
            else:
                rid = int(resids[i])
                seen = chain_resmap.setdefault(ch, {})
                if rid not in seen:
                    chain_counter[ch] = chain_counter.get(ch, 0) + 1
                    seen[rid] = chain_counter[ch]
                ordinal = seen[rid]
            nm = str(names[i]).strip() if names is not None else None
            rnm = str(resnames[i]).strip() if resnames is not None else None
            mt = _norm_moltype(mtypes[i]) if mtypes is not None else None
            yield AtomRecord(
                chain=ch,
                resid=ordinal,
                index=int(i),
                name=nm,
                resname=rnm,
                mol_type=mt,
                conformer_restraints=(
                    False if conf_restraints is None else bool(conf_restraints[i])
                ),
            )

    # --- ConformerAdapter -----------------------------------------------------
    def num_atoms(self) -> int:
        return self._n_atom

    def get_elements(self) -> np.ndarray:
        """(N_atom,) atomic numbers; padding atoms (beyond the atom_array) are 0."""
        return biotite_get_elements(self.atom_array, self._n_atom)

    def _feature_numpy(self, name: str) -> np.ndarray:
        value = self.feats[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        while value.ndim > (2 if name == "ref_pos" else 1):
            value = value[0]
        return value

    def get_reference_positions(self) -> np.ndarray:
        return self._feature_numpy("ref_pos").astype(np.float64)

    def get_reference_space_uid(self) -> np.ndarray:
        return self._feature_numpy("ref_space_uid").astype(np.int64)

    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        aa = self.atom_array
        if aa is None:
            return

        def _post_build(chain_id, mol, coords, idxs, elements_all, bonds_local):
            # SMILES ligand: replace the target with a stereo-correct ETKDG ideal
            # conformer (atom_array.coord may be the wrong isomer, e.g. maleate predicted
            # trans -> the restraint would converge to trans). The atom_array atom order
            # equals the SMILES mol's RDKit order (json_parser builds it from
            # mol.GetAtoms()), so the ideal coords line up with idxs.
            smiles = self._smiles_by_chain.get(str(chain_id))
            if smiles is None:
                return mol, coords
            from rdkit import Chem

            from rgi_utils._mol_build import generate_ideal_conformer

            smol = Chem.MolFromSmiles(smiles)  # carries the SMILES stereo
            # target_mol=mol fixes the atom order to atom_array (global_indices) order.
            ideal = (
                generate_ideal_conformer(smol, target_mol=mol)
                if smol is not None
                else None
            )
            if ideal is not None and len(ideal) == len(idxs):
                return _build_ligand_mol(elements_all[idxs], ideal, bonds_local), ideal
            logger.warning(
                "protenix chain %s: SMILES ETKDG failed; using model coords "
                "as the restraint target",
                chain_id,
            )
            return mol, coords

        # protenix marks ligand atoms with biotite hetero; chains are label_asym_id;
        # the conformer target is atom_array.coord (overridden per-SMILES-ligand in
        # _post_build); conformer_restraints defaults OFF when the annotation is absent.
        yield from biotite_ligand_confs(
            aa,
            ligand_mask=np.asarray(aa.hetero, dtype=bool),
            chain_attr="label_asym_id",
            coords_all=np.asarray(aa.coord, dtype=np.float64),
            conf_rest_default=False,
            post_build=_post_build,
        )
