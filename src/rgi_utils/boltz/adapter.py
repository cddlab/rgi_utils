from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import torch

from rgi_utils.atom_context import AtomRecord, LigandConf

logger = logging.getLogger(__name__)


class BoltzFeatsAdapter:
    """Adapter from a boltz feats dict to the rgi_utils adapter protocols.

    Implements:
      - FrameworkAdapter.iter_atoms      (distance restraint selection)
      - ConformerAdapter.num_atoms / get_elements / iter_ligand_confs
        (conformer + VdW restraints)

    All tensors are read at batch index 0 (boltz replicates structure across the
    multiplicity dimension, so batch 0 is representative).
    """

    def __init__(self, feats: dict) -> None:
        self.feats = feats
        self._asym_id_atom = None
        self._atom_to_token = None

    def _per_atom(self):
        """Cache per-atom asym_id (chain) and the atom->token map for batch 0."""
        if self._asym_id_atom is None:
            feats = self.feats
            asym_id_token = feats["asym_id"]
            atom_to_token = feats["atom_to_token"]
            asym_id_atom = (
                torch.bmm(atom_to_token.float(), asym_id_token.unsqueeze(-1).float())
                .squeeze(-1)
                .long()
            )
            self._asym_id_atom = asym_id_atom[0]
            self._atom_to_token = atom_to_token[0]
        return self._asym_id_atom, self._atom_to_token

    # --- FrameworkAdapter -----------------------------------------------------
    def iter_atoms(self) -> Iterator[AtomRecord]:
        """Yield AtomRecord for every atom (for distance restraint selection).

        ``resid`` is the 1-based residue/token ordinal WITHIN the chain (it resets
        to 1 at each chain), so a selection like "chain B and resid 5" means
        residue 5 of chain B in every framework (consistent with protenix's
        per-chain res_id and AF3) — not a cumulative global token index.
        """
        asym_id_atom_b0, atom_to_token_b0 = self._per_atom()
        record = self.feats["record"]
        for chain in record[0].chains:
            chain_id = chain.chain_id
            chain_sites = torch.where(asym_id_atom_b0 == chain_id)[0].tolist()
            toks = [
                int(torch.argmax(atom_to_token_b0[gidx, :]).item())
                for gidx in chain_sites
            ]
            # rank this chain's tokens -> per-chain 1-based ordinal
            tok2resid = {t: i + 1 for i, t in enumerate(sorted(set(toks)))}
            for gidx, t in zip(chain_sites, toks):
                yield AtomRecord(
                    chain=chain.chain_name,
                    resid=tok2resid[t],
                    index=int(gidx),
                )

    # --- ConformerAdapter -----------------------------------------------------
    def num_atoms(self) -> int:
        return int(self.feats["atom_pad_mask"][0].shape[0])

    def get_elements(self) -> np.ndarray:
        elem = self.feats["ref_element"]
        elem0 = elem[0] if elem.dim() > 1 else elem
        if elem0.dim() > 1:  # one-hot (natoms, n_elem) -> atomic number
            elem0 = elem0.argmax(dim=-1)
        return elem0.detach().cpu().numpy()

    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        """Yield one LigandConf per non-polymer (ligand) chain (conformer restr).

        Uses ``feats['ligand_mols']`` (asym_id -> CCD RDKit mol, populated by
        boltz inferencev2). The structure's atoms for a ligand follow the CCD
        mol's heavy-atom order, so ``RemoveHs(mol)`` is aligned to the chain's
        atom sites; an element-order check guards against any mismatch.
        """
        feats = self.feats
        ligand_mols = feats.get("ligand_mols")
        if isinstance(ligand_mols, (list, tuple)):  # collated -> per-batch list
            ligand_mols = ligand_mols[0] if ligand_mols else None
        if not ligand_mols:
            return
        from rdkit import Chem

        asym_id_atom_b0, _ = self._per_atom()
        elements = np.asarray(self.get_elements())
        record = feats["record"]

        for chain in record[0].chains:
            mol = ligand_mols.get(chain.chain_id)
            if mol is None:
                continue
            try:
                mol = Chem.RemoveHs(mol)
            except Exception as exc:  # keep the run alive; just skip this ligand
                logger.warning(
                    "ligand %s: RemoveHs failed (%s); skip", chain.chain_name, exc
                )
                continue
            if mol.GetNumConformers() == 0:
                logger.warning(
                    "ligand %s: mol has no conformer; skip", chain.chain_name
                )
                continue

            chain_sites = torch.where(asym_id_atom_b0 == chain.chain_id)[0]
            global_indices = chain_sites.detach().cpu().numpy()
            n_mol = mol.GetNumAtoms()
            if n_mol != len(global_indices):
                logger.warning(
                    "ligand %s (id=%s): mol heavy %d != sites %d; skip conformer",
                    chain.chain_name,
                    chain.chain_id,
                    n_mol,
                    len(global_indices),
                )
                continue

            # element order must match, else mol atom i != structure atom i
            mol_elems = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
            site_elems = elements[global_indices]
            if not np.array_equal(mol_elems, site_elems):
                logger.warning(
                    "ligand %s: element order mismatch (mol=%s sites=%s); skipping",
                    chain.chain_name,
                    mol_elems.tolist(),
                    site_elems.tolist(),
                )
                continue

            conf_coords = np.asarray(mol.GetConformer().GetPositions())
            yield LigandConf(
                mol=mol,
                conf_coords=conf_coords,
                global_indices=global_indices,
            )
