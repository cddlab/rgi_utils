"""Framework-free rgi_utils adapter for AlphaFold 3.

Mirrors the torch tools' ``rgi_utils/<tool>/adapter.py``: it imports NO alphafold3.
The in-tool shim (``alphafold3_restr`` ``model/restraints/adapter.py``) reads the
fold_input + featurised batch, resolves each ligand's RDKit mol (CCD/SMILES — the
only alphafold3-coupled step), and hands this adapter plain data:

  - ``batch``: the featurised example dict (numpy arrays) — atom layout, reference
    coords/elements/atom-names, per-token molecule-type + residue-type.
  - ``chain_id_to_asym``: ``{fold_input chain id -> 1-based asym int}``.
  - ``polymer_types``: ``residue_names.POLYMER_TYPES`` passed as data (resname lookup).
  - ``ligand_mols``: ``[(chain_id, mol, is_smiles)]`` resolved by the shim.

AF3 atom layout: positions are ``(num_tokens, max_atoms_per_token, 3)`` and
``flat_idx = token_idx * max_atoms_per_token + within_token_idx`` (a ligand atom
occupies its own token with ``within_token_idx = 0``). This adapter implements the
rgi_utils adapter protocols so ``rgi_utils.featurizer.build_spec`` builds the
``RestraintSpec`` directly:

  - FrameworkAdapter.iter_atoms      (distance restraint selection)
  - ConformerAdapter.num_atoms / get_elements / iter_ligand_confs

The flat-index / per-chain-resid mapping and the leaving-atom subset (a CCD mol may
contain atoms — e.g. glucose O1 — that AF3 tokenisation drops) live here; the
CCD-by-name mol lookup that produces those mols lives in the shim.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from rgi_utils.atom_context import AtomRecord, LigandConf, decode_atom_name

logger = logging.getLogger(__name__)


class AF3RestraintAdapter:
    """rgi_utils adapter over an AF3 featurised batch (numpy) + shim-resolved mols.

    Framework-free: all alphafold3 access (folding_input, CCD machinery) happens in
    the in-tool shim that constructs this object.
    """

    def __init__(self, batch, chain_id_to_asym, polymer_types, ligand_mols) -> None:
        self.token_asym_ids = np.asarray(batch["asym_id"])  # (num_tokens,)
        self.ref_mask = np.asarray(batch["ref_mask"])  # (num_tokens, max)
        self.ref_pos = np.asarray(batch["ref_pos"])  # (num_tokens, max, 3)
        self.ref_atom_name_chars = np.asarray(batch["ref_atom_name_chars"])
        self.ref_element = np.asarray(batch["ref_element"])  # (num_tokens, max)
        self.max_atoms_per_token = self.ref_pos.shape[1]
        # Per-token molecule-type masks -> normalized "protein"/"dna"/"rna" for the
        # selection DSL (powers the protein/dna/rna selectors). AF3 always emits these
        # in the BatchDict, so index them directly: a missing key is a real batch-shape
        # bug that should fail loudly here, not silently yield mol_type=None.
        self.is_protein = np.asarray(batch["is_protein"]).astype(bool)  # (num_tokens,)
        self.is_dna = np.asarray(batch["is_dna"]).astype(bool)
        self.is_rna = np.asarray(batch["is_rna"]).astype(bool)
        # per-token residue-type index into polymer_types -> 3-letter resname (powers
        # AtomRecord.resname -> pairing="align" RMSD restraints).
        self.aatype = np.asarray(batch["aatype"])  # (num_tokens,)
        self.polymer_types = polymer_types
        # chain.id -> asym int (1-based, fold_input chain order); resolved by the shim.
        self.chain_id_to_asym = dict(chain_id_to_asym)
        self.asym_int_to_chain = {v: k for k, v in self.chain_id_to_asym.items()}
        # [(chain_id, mol, is_smiles)] resolved by the shim (the only CCD/SMILES step).
        self.ligand_mols = list(ligand_mols)
        # The chain<->asym mapping assumes fold_input.chains order matches the batch
        # asym_id assignment (both 1-based by appearance), which holds for standard
        # inference (no cropping). Warn if the assumed asym ids aren't all present in
        # the batch, so a misalignment is visible rather than silently restraining the
        # wrong atoms (e.g. a chain dropped by structure cleaning).
        batch_asyms = {int(a) for a in np.unique(self.token_asym_ids)}
        if not set(self.asym_int_to_chain) <= batch_asyms:
            logger.warning(
                "chain<->asym mapping may be misaligned: fold_input asym ids %s are "
                "not all present in the batch (batch asym ids: %s).",
                sorted(set(self.asym_int_to_chain) - batch_asyms),
                sorted(batch_asyms),
            )

    # --- FrameworkAdapter ------------------------------------------------------
    def _token_mol_type(self, token_idx: int) -> str | None:
        """Normalized molecule type of a token from AF3's per-token masks.

        Returns "protein"/"dna"/"rna", or None for a ligand / water / unknown token, so
        the protein/dna/rna selectors EXCLUDE it (cross-tool convention: None never
        matches a molecule-type keyword)."""
        if self.is_protein[token_idx]:
            return "protein"
        if self.is_dna[token_idx]:
            return "dna"
        if self.is_rna[token_idx]:
            return "rna"
        return None

    def num_atoms(self) -> int:
        return int(self.ref_pos.shape[0] * self.max_atoms_per_token)

    def get_elements(self) -> np.ndarray:
        """(num_tokens*max,) atomic numbers; padding atoms (ref_mask 0) -> 0."""
        elem = self.ref_element.reshape(-1).astype(np.int64)
        mask = self.ref_mask.reshape(-1)
        return np.where(mask > 0, elem, 0)

    def iter_atoms(self) -> Iterator[AtomRecord]:
        """Yield AtomRecord(chain, resid, index) for every real atom.

        ``resid`` is the 1-based residue/token ordinal WITHIN the chain (it resets at
        each chain) and ``index`` is the global flat index ``token*max+within`` —
        consistent with boltz/protenix, so a selection like "chain B and resid 5" means
        residue 5 of chain B in every framework.
        """
        asym = self.token_asym_ids
        # per-chain 1-based token ordinal
        per_chain_resid = np.zeros(len(asym), dtype=int)
        counts: dict[int, int] = {}
        for ti, aint in enumerate(asym):
            aint = int(aint)
            counts[aint] = counts.get(aint, 0) + 1
            per_chain_resid[ti] = counts[aint]
        for token_idx, aint in enumerate(asym):
            chain = self.asym_int_to_chain.get(int(aint), "")
            resid = int(per_chain_resid[token_idx])
            for within in range(self.max_atoms_per_token):
                if not bool(self.ref_mask[token_idx, within]):
                    continue
                flat = int(token_idx) * self.max_atoms_per_token + within
                name = decode_atom_name(self.ref_atom_name_chars[token_idx, within])
                at = int(self.aatype[token_idx])
                resname = (
                    self.polymer_types[at]
                    if 0 <= at < len(self.polymer_types)
                    else None
                )
                yield AtomRecord(
                    chain=chain,
                    resid=resid,
                    index=flat,
                    name=name,
                    mol_type=self._token_mol_type(token_idx),
                    resname=resname,
                )

    # --- ConformerAdapter ------------------------------------------------------
    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        """Yield one LigandConf per shim-resolved ligand mol."""
        try:
            from rdkit import Chem  # pylint: disable=g-import-not-at-top
        except ImportError:
            return
        pos_flat = self.ref_pos.reshape(-1, 3)
        for chain_id, mol, is_smiles in self.ligand_mols:
            if mol is None:
                continue
            flat_indices, mol = self._ligand_flat_indices(chain_id, mol, is_smiles, Chem)
            if mol is None or len(flat_indices) == 0:
                continue
            conf_crds = pos_flat[flat_indices]  # (n_atoms, 3) reference coords
            # A SMILES mol carries no 3D geometry, so chiral tags exist only if the
            # SMILES annotated them (@/@@); the featurizer keys chiral restraints on
            # GetChiralTag. Attach the reference conformer and perceive stereo from it
            # (matching the CCD path, which assigns stereo from the ideal conformer) so
            # an unannotated SMILES stereocentre still gets chiral restraints.
            # NOTE: an ETKDG ideal-conformer target (like boltz/protenix) is NOT used
            # here because af3's ligand atoms are in TOKEN order, which differs from the
            # SMILES mol's RDKit-canonical order, and for SYMMETRIC ligands (e.g.
            # fumarate/maleate) a connectivity-only substructure match can't pick the
            # correct 1:1 atom mapping. So af3 keeps ref_pos as the dihedral target =>
            # cis/trans is only partially corrected on af3 (documented limitation).
            if mol.GetNumConformers() == 0 and mol.GetNumAtoms() == len(conf_crds):
                conf = Chem.Conformer(mol.GetNumAtoms())
                for i in range(len(conf_crds)):
                    conf.SetAtomPosition(
                        i,
                        (float(conf_crds[i, 0]), float(conf_crds[i, 1]), float(conf_crds[i, 2])),
                    )
                mol.AddConformer(conf, assignId=True)
                try:
                    Chem.AssignStereochemistryFrom3D(mol)
                except Exception:  # geometry-only restraints don't need a clean valence
                    pass
            yield LigandConf(
                mol=mol,
                conf_coords=conf_crds,
                global_indices=np.asarray(flat_indices, dtype=np.int64),
                # opted-in: ligands that set conformer_restraints=False are dropped by
                # the shim. Pass True explicitly (LigandConf defaults to False).
                conformer_restraints=True,
            )

    # --- flat-index mapping (framework-free) -----------------------------------
    def _ligand_flat_indices(self, chain_id, mol, is_smiles, Chem):
        """Return (flat_indices, mol) for a shim-resolved ligand mol.

        SMILES ligands map positionally (1 atom per token); CCD ligands map by atom
        name and drop CCD-only (leaving) atoms absent from the structure, subsetting
        the mol to the kept atoms.
        """
        if is_smiles:
            return self._smiles_flat_indices(chain_id), mol
        names = [a.GetProp("atom_name").strip() for a in mol.GetAtoms()]
        flat, kept = self._ccd_flat_indices(chain_id, names)
        # Restrain only atoms present in the structure (drop CCD-only atoms).
        if 0 < len(kept) < mol.GetNumAtoms():
            mol = self._subset_mol(mol, kept, Chem)
        return flat, mol

    def _smiles_flat_indices(self, chain_id) -> np.ndarray:
        asym_int = self.chain_id_to_asym[chain_id]
        token_indices = np.where(self.token_asym_ids == asym_int)[0]
        # ligand atom: flat_idx = token_idx * max_atoms_per_token (within = 0)
        return (token_indices * self.max_atoms_per_token).astype(np.int32)

    def _ccd_flat_indices(self, chain_id, atom_names):
        """Map CCD mol atom names to flat indices; return (flat, kept mol indices)."""
        asym_int = self.chain_id_to_asym[chain_id]
        token_indices = np.where(self.token_asym_ids == asym_int)[0]
        name_to_flat: dict[str, int] = {}
        for token_idx in token_indices:
            for within in range(self.max_atoms_per_token):
                if not bool(self.ref_mask[token_idx, within]):
                    continue
                name = decode_atom_name(self.ref_atom_name_chars[token_idx, within])
                if name:
                    name_to_flat.setdefault(
                        name, int(token_idx) * self.max_atoms_per_token + within
                    )
        kept = [i for i, nm in enumerate(atom_names) if nm in name_to_flat]
        missing = [nm for nm in atom_names if nm not in name_to_flat]
        if missing:
            logger.info(
                "chain %s: dropping %d ligand atom(s) absent from the structure: %s",
                chain_id,
                len(missing),
                missing,
            )
        flat = np.array([name_to_flat[atom_names[i]] for i in kept], dtype=np.int32)
        return flat, kept

    @staticmethod
    def _subset_mol(mol, kept, Chem):
        """Copy ``mol`` keeping only atoms ``kept`` (elements, atom_name, chiral tags,
        bonds among kept atoms and the conformer)."""
        rw = Chem.RWMol()
        old2new = {}
        for new_i, old_i in enumerate(kept):
            a = mol.GetAtomWithIdx(int(old_i))
            na = Chem.Atom(a.GetAtomicNum())
            if a.HasProp("atom_name"):
                na.SetProp("atom_name", a.GetProp("atom_name"))
            na.SetChiralTag(a.GetChiralTag())
            rw.AddAtom(na)
            old2new[int(old_i)] = new_i
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in old2new and j in old2new:
                rw.AddBond(old2new[i], old2new[j], b.GetBondType())
        out = rw.GetMol()
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            newconf = Chem.Conformer(len(kept))
            for new_i, old_i in enumerate(kept):
                newconf.SetAtomPosition(new_i, conf.GetAtomPosition(int(old_i)))
            out.AddConformer(newconf, assignId=True)
        try:
            Chem.SanitizeMol(out)
        except Exception:  # geometry-only restraints don't need a clean valence model
            pass
        return out
