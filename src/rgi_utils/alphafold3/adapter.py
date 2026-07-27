"""Framework-free rgi_utils adapter for AlphaFold 3.

Mirrors the torch tools' ``rgi_utils/<tool>/adapter.py``: it imports NO alphafold3.
The in-tool shim (``alphafold3_restr`` ``model/restraints/adapter.py``) reads the
fold_input + featurised batch, resolves each ligand's RDKit mol (CCD/SMILES — the
only alphafold3-coupled step), and hands this adapter plain data:

  - ``batch``: the featurised example dict (numpy arrays) — atom layout, reference
    coords/elements/atom-names, per-token molecule-type + residue-type.
  - ``chain_id_to_asym``: ``{fold_input chain id -> 1-based asym int}``.
  - ``polymer_residue_names``: AF3's polymer CCD-name vocabulary.
  - ``conformer_restraints_by_asym``: per-chain opt-in values.
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

# AF3's residue-type vocabulary is 20 amino acids, UNK, then the nucleotides. The
# `aatype` encoding inserts a GAP token right after UNK; a vocabulary handed over
# without it shifts every name from here on (see `_resolve_name_shift`).
_GAP_INDEX = 21
_RNA_NAMES = frozenset({"A", "G", "C", "U", "N"})
_DNA_NAMES = frozenset({"DA", "DG", "DC", "DT", "DN"})


class AF3RestraintAdapter:
    """rgi_utils adapter over an AF3 featurised batch (numpy) + shim-resolved mols.

    Framework-free: all alphafold3 access (folding_input, CCD machinery) happens in
    the in-tool shim that constructs this object.
    """

    def __init__(
        self,
        batch,
        chain_id_to_asym,
        polymer_residue_names,
        ligand_mols,
        conformer_restraints_by_asym=None,
    ) -> None:
        self.token_asym_ids = np.asarray(batch["asym_id"])  # (num_tokens,)
        self.ref_mask = np.asarray(batch["ref_mask"])  # (num_tokens, max)
        self.ref_pos = np.asarray(batch["ref_pos"])  # (num_tokens, max, 3)
        self.ref_atom_name_chars = np.asarray(batch["ref_atom_name_chars"])
        self.ref_element = np.asarray(batch["ref_element"])  # (num_tokens, max)
        ref_space_uid = batch.get("ref_space_uid")
        self.ref_space_uid = (
            None if ref_space_uid is None else np.asarray(ref_space_uid)
        )
        self.max_atoms_per_token = self.ref_pos.shape[1]
        # Per-token molecule-type masks -> normalized "protein"/"dna"/"rna" for the
        # selection DSL (powers the protein/dna/rna selectors). AF3 always emits these
        # in the BatchDict, so index them directly: a missing key is a real batch-shape
        # bug that should fail loudly here, not silently yield mol_type=None.
        self.is_protein = np.asarray(batch["is_protein"]).astype(bool)  # (num_tokens,)
        self.is_dna = np.asarray(batch["is_dna"]).astype(bool)
        self.is_rna = np.asarray(batch["is_rna"]).astype(bool)
        # Per-token residue-type index into the CCD-name vocabulary.
        self.aatype = np.asarray(batch["aatype"])  # (num_tokens,)
        self.polymer_residue_names = polymer_residue_names
        # ...but WHICH vocabulary: AF3 encodes `aatype` with the order that carries a
        # GAP token after UNK, and a caller handing over the gap-less list shifts every
        # nucleic name by one while leaving proteins (index < 21) correct. Resolve it
        # against the batch's own molecule-type flags instead of trusting either.
        self._name_shift = self._resolve_name_shift()
        # chain.id -> asym int (1-based, fold_input chain order); resolved by the shim.
        self.chain_id_to_asym = dict(chain_id_to_asym)
        self.asym_int_to_chain = {v: k for k, v in self.chain_id_to_asym.items()}
        self.conformer_restraints_by_asym = {
            int(key): bool(value)
            for key, value in (conformer_restraints_by_asym or {}).items()
        }
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

    def _resolve_name_shift(self) -> int:
        """0, or 1 when the residue-name vocabulary is missing AF3's gap token.

        AF3 fills ``aatype`` from ``POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP``
        (``… 20:UNK, 21:'-', 22:A, 23:G, 24:C, 25:U, 26:DA …``), while the plain
        ``POLYMER_TYPES`` list has no gap entry (``… 20:UNK, 21:A, 22:G …``). Indexing
        one with the other leaves proteins right (both agree below the gap) and shifts
        every NUCLEIC name by one — an adenine token reads as ``G`` and a uridine as
        ``DA``. That silently mis-identifies bases for the base-pair macro and for
        monomer-library lookups, so decide it from evidence: score both readings by how
        many nucleic tokens land on a name their own ``is_rna``/``is_dna`` flag allows.

        Only nucleic tokens can discriminate (protein indices are below the gap and read
        identically either way), so a protein-only or ligand-only batch keeps shift 0.
        """
        nucleic = np.flatnonzero(self.is_rna | self.is_dna)
        if not len(nucleic):
            return 0

        def consistent(shift: int) -> int:
            hits = 0
            for token_idx in nucleic:
                name = self._residue_name(int(token_idx), shift)
                if name is None:
                    continue
                allowed = _RNA_NAMES if self.is_rna[token_idx] else _DNA_NAMES
                hits += name in allowed
            return hits

        direct, gapless = consistent(0), consistent(1)
        if gapless <= direct:
            return 0
        logger.warning(
            "residue-name vocabulary looks gap-less: %d/%d nucleic tokens name a "
            "matching base after dropping AF3's gap token vs %d before — reading it "
            "shifted. Pass the vocabulary that indexes `aatype` (the …_WITH_UNKNOWN_AND"
            "_GAP order) to silence this.",
            gapless,
            len(nucleic),
            direct,
        )
        return 1

    def _residue_name(self, token_idx: int, shift: int | None = None) -> str | None:
        """CCD residue name of a token, or None when it is outside the vocabulary."""
        at = int(self.aatype[token_idx])
        # The gap token sits directly after UNK, so only names at or past it move.
        if shift is None:
            shift = self._name_shift
        if at >= _GAP_INDEX:
            at -= shift
        return (
            self.polymer_residue_names[at]
            if 0 <= at < len(self.polymer_residue_names)
            else None
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

    def get_reference_positions(self) -> np.ndarray:
        return self.ref_pos.reshape(-1, 3).astype(np.float64)

    def get_reference_space_uid(self) -> np.ndarray:
        if self.ref_space_uid is None:
            raise AttributeError("AF3 batch does not contain ref_space_uid")
        return self.ref_space_uid.reshape(-1).astype(np.int64)

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
                resname = self._residue_name(token_idx)
                yield AtomRecord(
                    chain=chain,
                    resid=resid,
                    index=flat,
                    name=name,
                    mol_type=self._token_mol_type(token_idx),
                    resname=resname,
                    conformer_restraints=self.conformer_restraints_by_asym.get(
                        int(aint), False
                    ),
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
            flat_indices, mol = self._ligand_flat_indices(
                chain_id, mol, is_smiles, Chem
            )
            if mol is None or len(flat_indices) == 0:
                continue
            conf_crds = pos_flat[flat_indices]  # (n_atoms, 3) reference coords
            # A SMILES mol carries no 3D geometry, so chiral tags exist only if the
            # SMILES annotated them (@/@@); the featurizer keys chiral restraints on
            # GetChiralTag. Attach the reference conformer and perceive stereo from it
            # (matching the CCD path, which assigns stereo from the ideal conformer) so
            # an unannotated SMILES stereocentre still gets chiral restraints.
            # NOTE: af3 keeps ref_pos ITSELF as the conformer/cistrans target (no ETKDG-
            # ideal substitution like boltz/protenix) because it needs none. af3 does NOT
            # canonicalize SMILES: the featurised ligand atoms are in Chem.MolFromSmiles
            # order (assign_atom_names_from_graph names them <element><counter> in that
            # order), which matches this shim's bare MolFromSmiles mol POSITIONALLY
            # (token i == mol atom i; see _smiles_flat_indices), and ref_pos is a stereo-
            # correct ETKDGv3 conformer encoding the SMILES @/@@ and /\. So bond/angle/
            # chiral AND cistrans all target the correct geometry. E2E-verified on af3:
            # SMILES fumarate (E) keeps its C=C at -180 deg and maleate (Z) at ~0 deg,
            # both with cistrans=1 — so cis/trans is fully (not partially) handled.
            if mol.GetNumConformers() == 0 and mol.GetNumAtoms() == len(conf_crds):
                conf = Chem.Conformer(mol.GetNumAtoms())
                for i in range(len(conf_crds)):
                    conf.SetAtomPosition(
                        i,
                        (
                            float(conf_crds[i, 0]),
                            float(conf_crds[i, 1]),
                            float(conf_crds[i, 2]),
                        ),
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
