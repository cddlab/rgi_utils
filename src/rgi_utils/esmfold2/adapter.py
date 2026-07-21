"""Adapter from ESMFold2 model features to the rgi_utils protocols.

ESMFold2 (the transformers fork's ``models/esmfold2``) builds per-token and
per-atom feature tensors in ``prepare_esmfold2_input`` and feeds them to
``ESMFold2Model.forward`` with a leading batch dim of 1. The pieces this adapter
needs are:

  - per token : ``asym_id``, ``mol_type`` (3 == nonpolymer/ligand)
  - per atom  : ``atom_to_token``, ``atom_attention_mask`` (real-atom mask),
                ``ref_pos`` (CCD/SMILES ideal conformer), ``ref_element`` (atomic
                number)
  - ``token_bonds`` : dense (n_tok, n_tok) connectivity. A ligand is tokenised one
                token PER ATOM, so this carries intra-ligand atom-atom bonds.

From these it provides:
  - FrameworkAdapter.iter_atoms        (distance restraint selection)
  - ConformerAdapter.num_atoms / get_elements / iter_ligand_confs
    (ligand bond/angle/chiral + intramolecular VdW)

Like the other adapters it imports no framework: tensors are duck-typed to numpy
(``.detach().cpu().numpy()`` when present) so ``import rgi_utils`` stays numpy-only.

Conformer bonds come from ``token_bonds`` (the real CCD/SMILES connectivity) rather
than perceived geometry — more reliable than the chai path. ``token_bonds`` is itself
binary (connectivity only), but the real bond ORDERS are recovered from
``ChainInfo.ligand_bond_orders`` (CCD via ``get_ligand_ccd_bonds``; SMILES via the
Kekulized 3-tuples emitted by ``tokenize_ligand_smiles``) and applied in
``iter_ligand_confs`` below. So bond/angle/chiral AND the cistrans (cis/trans) term —
which keys on ``BondType.DOUBLE`` — all work on ESMFold2 ligands (CCD and SMILES alike),
for any ligand with an acyclic, non-aromatic double bond (e.g. fumarate/maleate).
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from rgi_utils._mol_build import build_ligand_mol as _build_ligand_mol
from rgi_utils._moltype import MOLTYPE_BY_ID
from rgi_utils.atom_context import AtomRecord, LigandConf, decode_atom_name

logger = logging.getLogger(__name__)

_MOL_TYPE_NONPOLYMER = 3  # esmfold2 constants.MOL_TYPE_NONPOLYMER (ligand)


def _to_numpy(t):
    """Tensor/array -> numpy (cpu), without importing torch."""
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(t)


def _batch0(t) -> np.ndarray:
    """Drop the leading batch dim that prepare_esmfold2_input adds (``v[None]``)."""
    a = _to_numpy(t)
    return a[0] if a.ndim >= 1 else a


class ESMFold2Adapter:
    """rgi_utils adapter over the ESMFold2 forward-pass feature dict.

    ``features`` is the dict produced by ``prepare_esmfold2_input`` (each tensor
    carries a leading batch dim of 1). ``chain_infos`` is the matching list of
    ``ChainInfo`` (maps the integer ``asym_id`` back to the user chain id "A"/"B").
    ``num_atoms`` defaults to the padded atom count (``atom_to_token`` length),
    which equals the row count of the coordinate tensor handed to ``minimize``.
    """

    def __init__(
        self,
        features: dict,
        chain_infos=None,
        num_atoms: int | None = None,
        res_type_names: dict | None = None,
    ) -> None:
        self._asym = _batch0(features["asym_id"]).astype(np.int64)  # (n_tok,)
        # Precondition guard data: per-chain ordinals assume NO token padding (pad
        # tokens would consume ordinals and shift real residue/ligand resids). Keep the
        # token mask (if present) so _compute_token_ordinals can verify it is all-ones.
        _tam = features.get("token_attention_mask")
        self._token_mask = _batch0(_tam).astype(bool) if _tam is not None else None
        self._mol_type = _batch0(features["mol_type"]).astype(np.int64)  # (n_tok,)
        # per-token residue-type int + the {int -> 3-letter} vocab passed in by the
        # esm caller (adapter stays framework-free); powers AtomRecord.resname ->
        # pairing="align" RMSD. None -> resname unavailable.
        rt = features.get("res_type")
        self._res_type = _batch0(rt).astype(np.int64) if rt is not None else None
        self._res_type_names = res_type_names
        self._atom_to_token = _batch0(features["atom_to_token"]).astype(
            np.int64
        )  # (n_atom,)
        self._exists = _batch0(features["atom_attention_mask"]).astype(
            bool
        )  # (n_atom,)
        self._ref_pos = _batch0(features["ref_pos"]).astype(np.float64)  # (n_atom, 3)
        ref_uid = features.get("ref_space_uid")
        self._ref_space_uid = (
            _batch0(ref_uid).astype(np.int64) if ref_uid is not None else None
        )
        self._ref_element = _batch0(features["ref_element"]).astype(
            np.int64
        )  # (n_atom,)
        ranc = features.get("ref_atom_name_chars")  # (n_atom, 4) ord(c)-32 codes
        self._ref_atom_name_chars = (
            _batch0(ranc).astype(np.int64) if ranc is not None else None
        )
        tb = _batch0(features["token_bonds"])  # (n_tok, n_tok) or (n_tok, n_tok, 1)
        self._token_bonds = tb[..., 0] if tb.ndim == 3 else tb
        self._n_atom = (
            int(num_atoms)
            if num_atoms is not None
            else int(self._atom_to_token.shape[0])
        )
        self._asym_to_name = {
            int(c.asym_id): str(c.chain_id) for c in (chain_infos or [])
        }
        # {asym_id -> [(atom_name1, atom_name2, order), ...]} CCD ligand bonds with the
        # Kekulized bond order (prepare_input populates ChainInfo.ligand_bond_orders).
        # Used to upgrade the binary token_bonds connectivity with real bond orders so
        # build_ligand_mol can re-perceive aromaticity for the UFF-relaxed restraint
        # target. Absent (SMILES ligand / older feats) -> orders default to single.
        self._asym_to_bond_orders = {
            int(c.asym_id): list(getattr(c, "ligand_bond_orders", None) or [])
            for c in (chain_infos or [])
        }
        # {asym_id -> bool} per-chain conformer-restraints opt-in from ChainInfo.
        self._asym_to_conf_restraints = {
            int(c.asym_id): bool(getattr(c, "conformer_restraints", False))
            for c in (chain_infos or [])
        }
        self._tok_ordinal = self._compute_token_ordinals()

    def _atom_name(self, i: int) -> str | None:
        """Decoded atom name for atom row ``i`` (ref_atom_name_chars), or None."""
        if self._ref_atom_name_chars is None:
            return None
        return decode_atom_name(self._ref_atom_name_chars[i])

    def _compute_token_ordinals(self) -> dict[int, int]:
        """Per-chain 1-based ordinal for each token (resets at each chain).

        ESMFold2 tokenises a protein residue as one token and a ligand atom as one
        token, so this is the residue ordinal for polymers and a per-atom ordinal
        for ligands — matching the cross-tool ``AtomRecord.resid`` convention.

        Precondition: ``prepare_esmfold2_input`` emits NO token-level padding (every
        token is real; ``token_attention_mask`` is all-ones), so counting all tokens
        is correct. A token-padded features dict would let pad tokens consume ordinals
        and shift the real residue/ligand ordinals — guard on the token mask first if
        that precondition ever changes.
        """
        if self._token_mask is not None and not bool(self._token_mask.all()):
            raise ValueError(
                "ESMFold2Adapter: token_attention_mask has padding (not all-ones); "
                "per-chain resid ordinals would be shifted by pad tokens. Filter pad "
                "tokens before building AtomRecords."
            )
        counter: dict[int, int] = {}
        ordinal: dict[int, int] = {}
        for tok in range(len(self._asym)):
            ch = int(self._asym[tok])
            counter[ch] = counter.get(ch, 0) + 1
            ordinal[tok] = counter[ch]
        return ordinal

    # --- FrameworkAdapter -----------------------------------------------------
    def iter_atoms(self) -> Iterator[AtomRecord]:
        a2t = self._atom_to_token
        for i in range(self._n_atom):
            if not bool(self._exists[i]):
                continue  # padding atom
            tok = int(a2t[i])
            asym = int(self._asym[tok])
            chain = self._asym_to_name.get(asym, str(asym))
            rnm = None
            if self._res_type is not None and self._res_type_names is not None:
                rnm = self._res_type_names.get(int(self._res_type[tok]))
            yield AtomRecord(
                chain=chain,
                resid=int(self._tok_ordinal[tok]),
                index=int(i),
                name=self._atom_name(i),
                mol_type=MOLTYPE_BY_ID.get(int(self._mol_type[tok])),
                resname=rnm,
                conformer_restraints=self._asym_to_conf_restraints.get(asym, False),
            )

    # --- ConformerAdapter -----------------------------------------------------
    def num_atoms(self) -> int:
        return self._n_atom

    def get_elements(self) -> np.ndarray:
        """(num_atoms,) atomic numbers; padding atoms are 0."""
        elements = np.zeros(self._n_atom, dtype=np.int64)
        n = min(len(self._ref_element), self._n_atom)
        elements[:n] = np.where(self._exists[:n], self._ref_element[:n], 0)
        return elements

    def get_reference_positions(self) -> np.ndarray:
        positions = np.zeros((self._n_atom, 3), dtype=np.float64)
        n = min(len(self._ref_pos), self._n_atom)
        positions[:n] = self._ref_pos[:n]
        return positions

    def get_reference_space_uid(self) -> np.ndarray:
        if self._ref_space_uid is None:
            raise AttributeError("ESMFold2 features do not contain ref_space_uid")
        uid = np.full(self._n_atom, -1, dtype=np.int64)
        n = min(len(self._ref_space_uid), self._n_atom)
        uid[:n] = self._ref_space_uid[:n]
        return uid

    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        a2t = self._atom_to_token
        # ligand atoms = real atoms whose token is a nonpolymer
        lig_atoms = [
            i
            for i in range(self._n_atom)
            if bool(self._exists[i])
            and int(self._mol_type[int(a2t[i])]) == _MOL_TYPE_NONPOLYMER
        ]
        if not lig_atoms:
            return
        # group by ligand chain (the asym_id of each atom's token)
        by_chain: dict[int, list[int]] = {}
        for i in lig_atoms:
            by_chain.setdefault(int(self._asym[int(a2t[i])]), []).append(int(i))

        for asym, idxs in by_chain.items():
            idxs = np.array(sorted(idxs), dtype=np.int64)
            elements = self._ref_element[idxs]
            coords = self._ref_pos[idxs]
            # Connectivity comes from token_bonds (binary, reliable). Bond ORDERS, when
            # available, come from ChainInfo.ligand_bond_orders (CCD, Kekulized) matched by
            # atom name -> local index; pairs without a recorded order default to single.
            # Real orders let build_ligand_mol re-perceive aromaticity so the UFF-relaxed
            # bond/angle target keeps aromatic rings planar (a flat single-bond ring would
            # otherwise pucker to sp3). 1 token/atom -> a token pair is an atom bond.
            order_by_pair: dict[tuple[int, int], int] = {}
            name_orders = self._asym_to_bond_orders.get(int(asym))
            if name_orders:
                namemap: dict[str, int] = {}
                for li, g in enumerate(idxs):
                    nm = self._atom_name(int(g))
                    if nm is not None:
                        namemap[nm] = li
                for n1, n2, o in name_orders:
                    if n1 in namemap and n2 in namemap:
                        a, b = namemap[n1], namemap[n2]
                        order_by_pair[(min(a, b), max(a, b))] = int(o)
            toks = [int(a2t[int(g)]) for g in idxs]
            bonds_local = [
                (li, lj, order_by_pair.get((li, lj), 1))
                for li in range(len(idxs))
                for lj in range(li + 1, len(idxs))
                if self._token_bonds[toks[li], toks[lj]] > 0
            ]
            mol = _build_ligand_mol(elements, coords, bonds_local)
            yield LigandConf(
                mol=mol,
                conf_coords=coords,
                global_indices=idxs,
                # Per-chain opt-in from ChainInfo; absent defaults to False.
                conformer_restraints=self._asym_to_conf_restraints.get(
                    int(asym), False
                ),
            )
