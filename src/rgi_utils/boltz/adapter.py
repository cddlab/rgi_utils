from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

# boltz adapter imports torch ON PURPOSE (the other five rgi_utils adapters are
# framework-free): boltz feats arrive as native torch tensors, read at batch 0 here.
# It is a lazily-imported submodule, so top-level `import rgi_utils` still needs only
# numpy (the framework-free invariant's actual intent).
import torch

from rgi_utils._moltype import MOLTYPE_BY_ID
from rgi_utils.atom_context import AtomRecord, LigandConf, decode_atom_name

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

    def __init__(self, feats: dict, token_names: list | None = None) -> None:
        self.feats = feats
        # boltz const.tokens (token-id -> 3-letter CCD name), passed in by the boltz
        # caller so the adapter stays framework-free. Enables AtomRecord.resname (and
        # thus pairing="align" RMSD restraints); None -> resname unavailable.
        self.token_names = token_names
        self._asym_id_atom = None
        self._atom_to_token = None
        self._atom_token_indices = None
        self._atom_records = None
        self._pad0 = None  # per-atom real-atom mask (True=real), batch 0

    def _per_atom(self):
        """Cache per-atom asym_id (chain), the atom->token map and the real-atom
        pad mask for batch 0."""
        if self._asym_id_atom is None:
            feats = self.feats
            self._atom_to_token = feats["atom_to_token"][0]
            self._atom_token_indices = self._atom_to_token.argmax(dim=-1)
            self._asym_id_atom = feats["asym_id"][0][self._atom_token_indices].long()
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
        if self._atom_records is None:
            self._atom_records = tuple(self._build_atom_records())
        yield from self._atom_records

    def _build_atom_records(self):
        """Transfer metadata in bulk once per structure, without per-atom CUDA syncs."""
        asym_id_atom_b0, _ = self._per_atom()
        atom_chains = asym_id_atom_b0.detach().cpu().numpy()
        token_indices = self._atom_token_indices.detach().cpu().numpy()
        real_atoms = self._pad0.detach().cpu().numpy()
        record = self.feats["record"]
        name_of = self._atom_name_lookup()
        # per-token molecule type (batch 0) -> normalized string; absent -> None
        mt_tok = self.feats.get("mol_type")
        if mt_tok is not None:
            mt0 = (mt_tok[0] if mt_tok.dim() > 1 else mt_tok).detach().cpu().numpy()
        else:
            mt0 = None
        # per-token residue 3-letter name via res_type (one-hot [B, N_tok, num_tokens])
        # + the passed-in const.tokens vocab; powers AtomRecord.resname (align RMSD).
        rt = self.feats.get("res_type")
        if rt is not None and self.token_names is not None:
            rt0 = rt[0] if rt.dim() > 2 else rt
            res_type_idx = torch.argmax(rt0, dim=-1).detach().cpu().numpy()
        else:
            res_type_idx = None
        ref_conf_restr = self.feats.get("ref_conformer_restraint")
        if ref_conf_restr is not None:
            rcr0 = ref_conf_restr[0] if ref_conf_restr.dim() > 1 else ref_conf_restr
            rcr0 = rcr0.detach().cpu().numpy()
        else:
            rcr0 = None
        for chain in record[0].chains:
            chain_id = chain.chain_id
            # exclude padding atoms (else they surface as chain-0 / resid 1)
            chain_sites = np.flatnonzero((atom_chains == chain_id) & real_atoms)
            toks = token_indices[chain_sites].tolist()
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
                    mol_type=(None if mt0 is None else MOLTYPE_BY_ID.get(int(mt0[t]))),
                    resname=(
                        None
                        if res_type_idx is None
                        else self.token_names[int(res_type_idx[t])]
                    ),
                    conformer_restraints=(False if rcr0 is None else bool(rcr0[gidx])),
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
            return decode_atom_name(codes[gidx])

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

    def get_reference_positions(self) -> np.ndarray:
        pos = self.feats["ref_pos"]
        pos0 = pos[0] if pos.dim() > 2 else pos
        return pos0.detach().float().cpu().numpy()

    def get_reference_space_uid(self) -> np.ndarray:
        uid = self.feats["ref_space_uid"]
        uid0 = uid[0] if uid.dim() > 1 else uid
        return uid0.detach().cpu().numpy()

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
        # Per-chain conformer-restraints flag stored per atom. A ligand chain's atoms
        # share one value; absent older features default off.
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
                stereo_mol=mol,
            )
