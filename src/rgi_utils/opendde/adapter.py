"""Adapter from an OpenDDE feature dict to the rgi_utils protocols.

OpenDDE keeps its inference ``AtomArray`` alongside the tensor features.  The
array provides atom metadata and bonds, while the tensor features provide the
reference conformer and the atom-to-token mapping used by the diffusion model.
The adapter is framework-free: torch tensors are converted by duck typing, so
importing it does not import OpenDDE or torch.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from rgi_utils._biotite_adapter import biotite_get_elements, biotite_ligand_confs
from rgi_utils._mol_build import align_stereo_mol
from rgi_utils._mol_build import build_ligand_mol as _build_ligand_mol
from rgi_utils.atom_context import AtomRecord, LigandConf

logger = logging.getLogger(__name__)

_VALID_MOLTYPES = {"protein", "dna", "rna", "ligand"}


def _norm_moltype(value) -> str | None:
    mol_type = str(value).strip().lower()
    return mol_type if mol_type in _VALID_MOLTYPES else None


class OpenDDEAdapter:
    """Expose one OpenDDE structure to ``CombinedRestraints``.

    ``input_feature_dict`` must contain the tool-side ``atom_array`` plus the
    ordinary OpenDDE reference and token-mapping features.  After structural
    token expansion, ``residue_level_atom_to_token_idx`` is preferred because it
    preserves the cross-tool residue/token convention.
    """

    def __init__(self, input_feature_dict: dict) -> None:
        self.feats = input_feature_dict
        self.atom_array = input_feature_dict.get("atom_array")
        self._smiles_by_chain = input_feature_dict.get("smiles_by_chain", {}) or {}
        token_key = (
            "residue_level_atom_to_token_idx"
            if "residue_level_atom_to_token_idx" in input_feature_dict
            else "atom_to_token_idx"
        )
        self._token_key = token_key
        self._n_atom = int(self._feature_numpy(token_key, ndim=1).shape[0])

    def _feature_numpy(self, name: str, *, ndim: int) -> np.ndarray:
        value = self.feats[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        while value.ndim > ndim:
            value = value[0]
        return value

    # --- FrameworkAdapter -------------------------------------------------
    def iter_atoms(self) -> Iterator[AtomRecord]:
        aa = self.atom_array
        if aa is None:
            return

        chains = np.asarray(aa.label_asym_id)
        token_idx = self._feature_numpy(self._token_key, ndim=1)
        names = np.asarray(aa.atom_name) if hasattr(aa, "atom_name") else None
        resnames = np.asarray(aa.res_name) if hasattr(aa, "res_name") else None
        mol_types = np.asarray(aa.mol_type) if hasattr(aa, "mol_type") else None
        categories = aa.get_annotation_categories()
        conf_restraints = (
            np.asarray(aa.conformer_restraints, dtype=bool)
            if "conformer_restraints" in categories
            else None
        )

        chain_tokens: dict[str, dict[int, int]] = {}
        for i in range(len(aa)):
            chain = str(chains[i])
            token = int(token_idx[i])
            seen = chain_tokens.setdefault(chain, {})
            if token not in seen:
                seen[token] = len(seen) + 1
            yield AtomRecord(
                chain=chain,
                resid=seen[token],
                index=i,
                name=str(names[i]).strip() if names is not None else None,
                resname=(str(resnames[i]).strip() if resnames is not None else None),
                mol_type=(
                    _norm_moltype(mol_types[i]) if mol_types is not None else None
                ),
                conformer_restraints=(
                    False if conf_restraints is None else bool(conf_restraints[i])
                ),
            )

    # --- ConformerAdapter -------------------------------------------------
    def num_atoms(self) -> int:
        return self._n_atom

    def get_elements(self) -> np.ndarray:
        return biotite_get_elements(self.atom_array, self._n_atom)

    def get_reference_positions(self) -> np.ndarray:
        return self._feature_numpy("ref_pos", ndim=2).astype(np.float64)

    def get_reference_space_uid(self) -> np.ndarray:
        return self._feature_numpy("ref_space_uid", ndim=1).astype(np.int64)

    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        aa = self.atom_array
        if aa is None:
            return

        def _post_build(chain_id, mol, coords, idxs, elements_all, bonds_local):
            smiles = self._smiles_by_chain.get(str(chain_id))
            if smiles is None:
                return mol, coords

            from rdkit import Chem

            from rgi_utils._mol_build import generate_ideal_conformer

            source_mol = Chem.MolFromSmiles(smiles)
            stereo_mol = (
                align_stereo_mol(source_mol, mol) if source_mol is not None else None
            )
            categories = aa.get_annotation_categories()
            stereo_required = "conformer_restraints" in categories and bool(
                np.asarray(aa.conformer_restraints, dtype=bool)[idxs].any()
            )
            if stereo_mol is None:
                if stereo_required:
                    raise ValueError(
                        f"OpenDDE chain {chain_id}: cannot map source SMILES "
                        "stereochemistry to the model atom order"
                    )
                return mol, coords, None
            ideal = generate_ideal_conformer(stereo_mol)
            if ideal is not None and len(ideal) == len(idxs):
                return (
                    _build_ligand_mol(elements_all[idxs], ideal, bonds_local),
                    ideal,
                    stereo_mol,
                )
            logger.warning(
                "OpenDDE chain %s: SMILES ETKDG failed; using ref_pos as the "
                "restraint target",
                chain_id,
            )
            return mol, coords, stereo_mol

        mol_types = np.asarray(aa.mol_type)
        yield from biotite_ligand_confs(
            aa,
            ligand_mask=np.char.lower(mol_types.astype(str)) == "ligand",
            chain_attr="label_asym_id",
            coords_all=self.get_reference_positions(),
            conf_rest_default=False,
            post_build=_post_build,
        )
