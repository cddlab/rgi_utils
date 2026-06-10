from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import torch

from rgi_utils.atom_context import AtomRecord, LigandConf

logger = logging.getLogger(__name__)

# boltz const.chain_types order: PROTEIN=0, DNA=1, RNA=2, NONPOLYMER=3. NOTE the
# DNA/RNA order is the OPPOSITE of chai/openfold (RNA=1, DNA=2) — normalize to the
# shared string so one `dna`/`rna` selector means the same atom in every tool.
_BOLTZ_MOLTYPE = {0: "protein", 1: "dna", 2: "rna", 3: "ligand"}


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
        self._pad0 = None  # per-atom real-atom mask (True=real), batch 0

    def _per_atom(self):
        """Cache per-atom asym_id (chain), the atom->token map and the real-atom
        pad mask for batch 0."""
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
            # boltz pads the atom dim to a multiple of the window size; padding atoms
            # get asym_id=0 and an all-zero atom_to_token row, so without this mask
            # they would be emitted as chain-0 / residue-ordinal-1 and corrupt any
            # selection touching them (and inflate a chain-0 ligand's atom count).
            self._pad0 = self.feats["atom_pad_mask"][0].bool()
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
        name_of = self._atom_name_lookup()
        # per-token molecule type (batch 0) -> normalized string; absent -> None
        mt_tok = self.feats.get("mol_type")
        if mt_tok is not None:
            mt0 = (mt_tok[0] if mt_tok.dim() > 1 else mt_tok).detach().cpu().numpy()
        else:
            mt0 = None
        for chain in record[0].chains:
            chain_id = chain.chain_id
            # exclude padding atoms (else they surface as chain-0 / resid 1)
            chain_sites = torch.where((asym_id_atom_b0 == chain_id) & self._pad0)[
                0
            ].tolist()
            toks = [
                int(torch.argmax(atom_to_token_b0[gidx, :]).item())
                for gidx in chain_sites
            ]
            # rank this chain's tokens -> per-chain 1-based ordinal. boltz emits a
            # chain's atoms in ascending token order, so sorted-token-index equals
            # first-appearance order, matching the other adapters' resid convention.
            tok2resid = {t: i + 1 for i, t in enumerate(sorted(set(toks)))}
            for gidx, t in zip(chain_sites, toks):
                yield AtomRecord(
                    chain=chain.chain_name,
                    resid=tok2resid[t],
                    index=int(gidx),
                    name=name_of(int(gidx)),
                    mol_type=(None if mt0 is None else _BOLTZ_MOLTYPE.get(int(mt0[t]))),
                )

    def _atom_name_lookup(self):
        """Return ``gidx -> atom name`` (or all-None if unavailable).

        Names come from ``feats['ref_atom_name_chars']``, which boltz builds in the
        same token-driven loop as ``ref_pos`` — so it is index-aligned with the
        coordinate tensor handed to ``minimize`` (no structure-order remap). It holds
        ord(c)-32 char codes, one-hot encoded to ``(n_atom, 4, 64)``; decode each
        atom's 4 codes via ``argmax(-1)`` then ``chr(code+32)`` for non-zero codes
        (the same encoding the esm/AF3 adapters decode). RMSD identity pairing keys on
        these names; without them it falls back to selection-order pairing."""
        ranc = self.feats.get("ref_atom_name_chars")
        if ranc is None:
            logger.warning(
                "boltz feats has no 'ref_atom_name_chars'; RMSD identity pairing "
                "will fall back to selection-order pairing"
            )
            return lambda gidx: None
        try:
            arr = ranc[0]  # drop the batch dim
            if arr.dim() == 3:  # one-hot (n_atom, 4, 64) -> codes (n_atom, 4)
                arr = arr.argmax(dim=-1)
            codes = arr.detach().cpu().numpy().astype(np.int64)  # (n_atom, 4)
        except Exception as exc:  # unexpected shape/dtype: surface it, don't hide it
            logger.warning(
                "boltz 'ref_atom_name_chars' decode failed (%s); RMSD identity "
                "pairing falls back to selection-order pairing",
                exc,
            )
            return lambda gidx: None

        def f(gidx):
            if gidx < 0 or gidx >= codes.shape[0]:
                return None
            nm = "".join(chr(int(c) + 32) for c in codes[gidx] if int(c) != 0).strip()
            return nm or None

        return f

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
        mol's heavy-atom order, so ``RemoveAllHs(mol)`` is aligned to the chain's
        atom sites; an element-order check guards against any mismatch. (Use
        RemoveAllHs, not RemoveHs: a CCD mol may carry an explicit N-H that
        RemoveHs keeps, inflating the atom count past the heavy-only sites.)
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
        # per-ligand conformer_restraints flag, stored per-atom (the ch_rest input
        # flag -> ref_conformer_restraint); a ligand chain's atoms share one value.
        # Absent (older feats) -> default on.
        ref_conf_restr = feats.get("ref_conformer_restraint")
        rcr0 = None
        if ref_conf_restr is not None:
            rcr0 = ref_conf_restr[0] if ref_conf_restr.dim() > 1 else ref_conf_restr

        for chain in record[0].chains:
            mol = ligand_mols.get(chain.chain_id)
            if mol is None:
                continue
            try:
                # RemoveAllHs, NOT RemoveHs: boltz CCD mols can carry an explicit
                # H (e.g. an N-H drawn in the component) that RemoveHs PRESERVES, so
                # GetNumAtoms() would then exceed the heavy-only structure sites and
                # the count guard below would silently skip the whole conformer
                # (n_active=0). RemoveAllHs strips every H, restoring heavy==sites.
                mol = Chem.RemoveAllHs(mol)
            except Exception as exc:  # keep the run alive; just skip this ligand
                logger.warning(
                    "ligand %s: RemoveAllHs failed (%s); skip", chain.chain_name, exc
                )
                continue
            if mol.GetNumConformers() == 0:
                logger.warning(
                    "ligand %s: mol has no conformer; skip", chain.chain_name
                )
                continue

            chain_sites = torch.where((asym_id_atom_b0 == chain.chain_id) & self._pad0)[
                0
            ]
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
            # per-ligand opt-in (ch_rest input flag); absent -> off (no conformer)
            conf_restr = False
            if rcr0 is not None and len(chain_sites) > 0:
                conf_restr = bool(rcr0[chain_sites].any().item())
            yield LigandConf(
                mol=mol,
                conf_coords=conf_coords,
                global_indices=global_indices,
                conformer_restraints=conf_restr,
            )
