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
than perceived geometry — more reliable than the chai path. ``token_bonds`` is
binary, so every bond is order 1: bond/angle/chiral apply, but the dihedral
(cis/trans) term — which keys on ``BondType.DOUBLE`` — finds nothing on ESMFold2
ligands (dihedrals=0), the same limitation as chai.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from rgi_utils._mol_build import build_ligand_mol as _build_ligand_mol
from rgi_utils.atom_context import AtomRecord, LigandConf

logger = logging.getLogger(__name__)

_MOL_TYPE_NONPOLYMER = 3  # esmfold2 constants.MOL_TYPE_NONPOLYMER (ligand)
# esmfold2 constants: PROTEIN=0, DNA=1, RNA=2, NONPOLYMER=3 (same order as boltz;
# OPPOSITE of chai/openfold RNA=1/DNA=2). Normalize to the shared string.
_ESM_MOLTYPE = {0: "protein", 1: "dna", 2: "rna", 3: "ligand"}


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
        self, features: dict, chain_infos=None, num_atoms: int | None = None
    ) -> None:
        self._asym = _batch0(features["asym_id"]).astype(np.int64)  # (n_tok,)
        self._mol_type = _batch0(features["mol_type"]).astype(np.int64)  # (n_tok,)
        self._atom_to_token = _batch0(features["atom_to_token"]).astype(
            np.int64
        )  # (n_atom,)
        self._exists = _batch0(features["atom_attention_mask"]).astype(
            bool
        )  # (n_atom,)
        self._ref_pos = _batch0(features["ref_pos"]).astype(np.float64)  # (n_atom, 3)
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
        self._tok_ordinal = self._compute_token_ordinals()

    def _atom_name(self, i: int) -> str | None:
        """Decoded atom name for atom row ``i`` (ref_atom_name_chars), or None."""
        if self._ref_atom_name_chars is None:
            return None
        return (
            "".join(
                chr(int(x) + 32) for x in self._ref_atom_name_chars[i] if int(x) != 0
            ).strip()
            or None
        )

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
            yield AtomRecord(
                chain=chain,
                resid=int(self._tok_ordinal[tok]),
                index=int(i),
                name=self._atom_name(i),
                mol_type=_ESM_MOLTYPE.get(int(self._mol_type[tok])),
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
                # ESMFold2 has no per-ligand conformer_restraints input flag, so opt-in
                # is governed by conformer_restraints_config presence (build_spec gate).
                conformer_restraints=True,
            )
