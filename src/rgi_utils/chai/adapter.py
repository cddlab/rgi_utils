"""Adapter from a chai-lab ``AllAtomStructureContext`` to the rgi_utils protocols.

chai exposes ``AllAtomStructureContext`` (per-atom token index / element / ideal
reference coords + a covalent bond index pair) — the chai analogue of a biotite
AtomArray. This adapter reads it the same way the protenix adapter reads its
AtomArray: rebuild the ligand mol from the atom subset (ideal ref coords + bonds +
atomic numbers). Chain ids come from the per-token ``subchain_id`` tensorcode
(label_asym_id); the per-chain 1-based resid is computed like every other tool.

The tensorcode decode is reimplemented here (a trivial uint8->str with pad token
255) so rgi_utils never imports chai_lab — keeping the dependency direction
rgi_utils -> nothing, like the boltz/protenix adapters.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from rgi_utils._mol_build import build_ligand_mol as _build_ligand_mol
from rgi_utils._mol_build import generate_ideal_conformer as _generate_ideal_conformer
from rgi_utils.atom_context import AtomRecord, LigandConf

logger = logging.getLogger(__name__)

_LIGAND_ENTITY = 3  # chai EntityType.LIGAND
_TENSORCODE_PAD = 255  # chai TENSORCODE_PAD_TOKEN


def _decode_tensorcode(codes) -> str:
    """Inverse of chai's string_to_tensorcode: uint8 codes -> ASCII string,
    dropping the pad token (255)."""
    return "".join(chr(int(c)) for c in codes if int(c) != _TENSORCODE_PAD)


class ChaiStructureAdapter:
    """rgi_utils adapter over a chai ``AllAtomStructureContext``.

    ``num_atoms`` is the padded coordinate length in the diffusion loop
    (``atom_single_mask.shape[-1]``), passed in from the build site so the global
    flat index matches the ``atom_pos`` tensor exactly.
    """

    def __init__(self, structure_context, num_atoms: int, smiles_by_subchain=None) -> None:
        self.sc = structure_context
        self._n_atom = int(num_atoms)
        # {subchain_id -> SMILES} for ligand chains (chai drops bond ORDERS at every layer,
        # so the conformer mol is otherwise perceived all-single and can't be UFF-relaxed to
        # the aromatic-ideal target). The structure_context ligand atoms are in MolFromSmiles
        # heavy-atom order, so a fresh MolFromSmiles maps its bonds back by index.
        self._smiles_by_subchain = dict(smiles_by_subchain or {})

    def _token_chains(self) -> list[str]:
        """Per-token chain id string decoded from the subchain_id tensorcode."""
        sub = np.asarray(self.sc.subchain_id)  # (n_tokens, 4) uint8
        return [_decode_tensorcode(sub[t]) for t in range(sub.shape[0])]

    # --- FrameworkAdapter -----------------------------------------------------
    def iter_atoms(self) -> Iterator[AtomRecord]:
        sc = self.sc
        if sc is None:
            return
        atom_token = np.asarray(sc.atom_token_index)
        exists = np.asarray(sc.atom_exists_mask, dtype=bool)
        names = getattr(sc, "atom_ref_name", None)  # list[str], per-atom
        token_chain = self._token_chains()
        # per-token 3-letter residue name (sc.residue_names is a cached_property, NOT a
        # method); powers AtomRecord.resname -> pairing="align" RMSD restraints.
        resn = getattr(sc, "residue_names", None)
        # per-chain 1-based PER-TOKEN ordinal (the cross-tool convention shared by
        # boltz/protenix/esmfold2): a standard polymer residue is one token, so all
        # its atoms share an ordinal (= residue ordinal); a ligand atom and each atom
        # of a NON-standard residue is its own token, so it gets its own ordinal.
        # (Previously chai grouped non-standard residues by token_residue_index, which
        # gave them ONE ordinal and diverged from the other tools for that edge case;
        # standard residues + ligands are unaffected by this change.)
        chain_seen: dict[str, dict[int, int]] = {}
        chain_counter: dict[str, int] = {}
        for i in range(len(atom_token)):
            if not bool(exists[i]):
                continue
            tok = int(atom_token[i])
            ch = token_chain[tok]
            seen = chain_seen.setdefault(ch, {})
            if tok not in seen:
                chain_counter[ch] = chain_counter.get(ch, 0) + 1
                seen[tok] = chain_counter[ch]
            nm = str(names[i]).strip() if names is not None else None
            rnm = str(resn[tok]).strip() if resn is not None else None
            yield AtomRecord(
                chain=ch, resid=seen[tok], index=int(i), name=nm, resname=rnm
            )

    # --- ConformerAdapter -----------------------------------------------------
    def num_atoms(self) -> int:
        return self._n_atom

    def get_elements(self) -> np.ndarray:
        """(num_atoms,) atomic numbers; padding atoms are 0. chai stores
        ``atom_ref_element`` as the atomic number directly."""
        elements = np.zeros(self._n_atom, dtype=np.int64)
        sc = self.sc
        if sc is not None:
            z = np.asarray(sc.atom_ref_element).astype(np.int64)
            exists = np.asarray(sc.atom_exists_mask, dtype=bool)
            n = min(len(z), self._n_atom)
            elements[:n] = np.where(exists[:n], z[:n], 0)
        return elements

    def _mol_from_smiles(self, smiles, idxs, elements, coords):
        """Build the ligand mol from the source SMILES so it carries real bond ORDERS
        (chai drops them at every layer). chai names a SMILES ligand's atoms
        ``element+counter`` over the AddHs atom order, uppercased; we replicate that
        naming on a fresh ``MolFromSmiles`` and map its bonds to chai's atoms BY NAME --
        chai reorders ligand atoms, so a positional map is wrong (verified: by-order RMS
        came out WORSE than perceive).

        Bonds are emitted KEKULIZED (explicit single/double), NOT as order 4 (AROMATIC):
        AROMATIC bonds leave the rebuild's SanitizeMol unable to restore the implicit-H
        on an aromatic N-H (uracil/adenine) -- it picks the pyridine tautomer (n, 0 H)
        over the real [nH] -- a different mol whose uff_relax gets wrong UFF types and
        blows the bond/angle TARGETS up (ZKD: max bond 35 A -> the prediction explodes).
        Kekule lets SanitizeMol re-perceive aromaticity AND get the correct valence/H --
        mirrors protenix's json_parser bond emission (the "other tool" this matches).

        Returns None on incomplete name match -> caller falls back to perceive_bonds, so
        a naming-convention drift degrades gracefully instead of corrupting the restraint.
        """
        from collections import defaultdict

        from rdkit import Chem

        base = Chem.MolFromSmiles(smiles)
        if base is None or base.GetNumAtoms() != len(idxs):
            return None
        nbase = base.GetNumAtoms()
        cnt: dict = defaultdict(int)
        base_name: dict[int, str] = {}
        for i, atom in enumerate(Chem.AddHs(base).GetAtoms()):
            s = atom.GetSymbol()
            cnt[s] += 1
            if i < nbase:  # heavy atoms come first (AddHs appends H)
                base_name[i] = (s + str(cnt[s])).upper()
        names = getattr(self.sc, "atom_ref_name", None)
        if names is None:
            return None

        def _norm(nm: object) -> str:
            # chai atom names are <elem><counter>_<copy> (C1_1); drop the trailing
            # _<digits> copy index to match build_ligand_mol's <elem><counter> (C1).
            # else the by-name map fails and chai falls back to perception, which
            # fragments on large ligands -> incomplete restraint (n_active<natoms).
            s = str(nm).strip().upper()
            base, _, suf = s.rpartition("_")
            return base if (base and suf.isdigit()) else s

        name_to_local = {_norm(names[int(g)]): li for li, g in enumerate(idxs)}
        base_to_local = {
            bi: name_to_local[nm] for bi, nm in base_name.items() if nm in name_to_local
        }
        if len(base_to_local) != nbase:
            return None  # incomplete name match -> safe fallback
        # Kekulize a copy so aromatic bonds become explicit single/double (1/2); atom
        # indices are unchanged, so base_to_local still maps. Fall back to as-is orders
        # on a non-kekulizable (exotic-valence) mol.
        kmol = Chem.Mol(base)
        try:
            Chem.Kekulize(kmol, clearAromaticFlags=True)
        except Exception:
            kmol = base
        bonds_local = [
            (
                base_to_local[b.GetBeginAtomIdx()],
                base_to_local[b.GetEndAtomIdx()],
                int(b.GetBondTypeAsDouble()),
            )
            for b in kmol.GetBonds()
        ]
        return _build_ligand_mol(elements, coords, bonds_local)

    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        from rdkit import Chem

        sc = self.sc
        if sc is None:
            return
        atom_token = np.asarray(sc.atom_token_index)
        exists = np.asarray(sc.atom_exists_mask, dtype=bool)
        token_entity = np.asarray(sc.token_entity_type)
        ref_pos = np.asarray(sc.atom_ref_pos, dtype=np.float64)
        ref_elem = np.asarray(sc.atom_ref_element).astype(np.int64)
        token_chain = self._token_chains()
        per_atom_chain = np.array([token_chain[int(t)] for t in atom_token])
        is_lig = np.array(
            [int(token_entity[int(t)]) == _LIGAND_ENTITY for t in atom_token]
        )
        lig_mask = is_lig & exists
        # IMPORTANT: chai's ``atom_covalent_bond_indices`` holds ONLY inter-residue
        # links (glycan inter-sugar bonds + user COVALENT-constraint bonds); for a
        # normal small-molecule ligand it is EMPTY (intra-ligand connectivity lives
        # in the per-residue reference ConformerData, which is dropped during
        # tokenization). So we do NOT source the conformer mol's bonds from it —
        # instead we perceive connectivity from the reference conformer geometry
        # (atom_ref_pos) via build_ligand_mol(perceive_bonds=True). That topology is
        # self-consistent with the conformer, so the bond/angle/chiral restraints
        # keep the ligand at its ideal geometry. (Without this, conformer restraints
        # were a silent no-op on chai ligands.)
        for ch in np.unique(per_atom_chain[lig_mask]):
            idxs = np.where((per_atom_chain == ch) & lig_mask)[0]
            coords = ref_pos[idxs]
            # Prefer the source SMILES (real, Kekulized bond orders) over geometry-
            # perceived connectivity: with orders, build_ligand_mol re-perceives
            # aromaticity so the featurizer's UFF-relax can idealise the bond/angle
            # target. Fall back to perceive_bonds only if SMILES is absent / unmatched.
            smiles = self._smiles_by_subchain.get(str(ch))
            mol = None
            if smiles is not None:
                mol = self._mol_from_smiles(smiles, idxs, ref_elem[idxs], coords)
            if mol is not None:
                # Replace the target geometry with a stereo-correct ETKDG ideal conformer
                # (mirrors protenix): chai's atom_ref_pos can carry the WRONG enantiomer
                # (e.g. 6fck) and the chiral restraint then faithfully enforces it --
                # finalize chiral=0 yet the prediction's CIP disagrees with the SMILES.
                # MolFromSmiles holds the intended stereo; generate_ideal_conformer
                # reorders it to mol's atom order. Rebuild the mol on the ideal coords so
                # its chiral tags AND the featurizer's chiral-volume sign read correct.
                # Falls back to the model conformer when the embed / atom-order match
                # fails (e.g. a charged ring the rebuild can't reproduce, like an N-oxide).
                base = Chem.MolFromSmiles(smiles)
                ideal = (
                    _generate_ideal_conformer(base, target_mol=mol)
                    if base is not None
                    else None
                )
                if ideal is not None and len(ideal) == len(idxs):
                    coords = ideal
                    mol = self._mol_from_smiles(smiles, idxs, ref_elem[idxs], ideal)
                    logger.info("chai ligand %s: stereo-correct ETKDG target", ch)
                else:
                    logger.info(
                        "chai ligand %s: ETKDG target unavailable, using ref_pos", ch
                    )
            if mol is None:
                mol = _build_ligand_mol(
                    ref_elem[idxs], coords, [], perceive_bonds=True
                )
            yield LigandConf(
                mol=mol,
                conf_coords=coords,
                global_indices=idxs.astype(np.int64),
                # chai has no per-ligand conformer_restraints input flag, so opt-in is
                # governed by conformer_restraints_config presence (the build_spec gate).
                conformer_restraints=True,
            )
