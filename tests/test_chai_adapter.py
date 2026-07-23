"""ChaiStructureAdapter regression tests (the adapter had NO test coverage).

The chai adapter is the largest of the six and carries a mol-type table that is a KNOWN
cross-tool hazard: chai's EntityType is PROTEIN=0/RNA=1/DNA=2/LIGAND=3 -- RNA BEFORE DNA --
so `_MOLTYPE_BY_ID_CHAI` maps 1->rna / 2->dna, the OPPOSITE of the shared MOLTYPE_BY_ID
(DNA=1/RNA=2) used by boltz/esm. A silent swap here would send `dna`/`rna`/backbone
selectors and RMSD align pairing to the wrong atoms. These tests pin that table plus the
per-chain 1-based resid ordinal, both driven purely from a hand-built fake structure
context (no chai_lab import).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rgi_utils.chai.adapter import ChaiStructureAdapter


def _tensorcode(chain: str, width: int = 4) -> list[int]:
    """chai's string_to_tensorcode: ASCII codes padded to `width` with 255."""
    codes = [ord(ch) for ch in chain]
    return codes + [255] * (width - len(codes))


def _fake_context(token_chains, token_entity, token_resname, atom_names):
    """Minimal chai AllAtomStructureContext stand-in: one atom per token, in order."""
    n = len(token_chains)
    return SimpleNamespace(
        atom_token_index=np.arange(n, dtype=np.int64),
        atom_exists_mask=np.ones(n, dtype=bool),
        atom_ref_name=list(atom_names),
        subchain_id=np.array([_tensorcode(c) for c in token_chains], dtype=np.uint8),
        residue_names=list(token_resname),
        token_entity_type=np.array(token_entity, dtype=np.int64),
    )


def test_chai_moltype_table_rna_before_dna():
    """entity 1 -> 'rna' and entity 2 -> 'dna' (chai's RNA-before-DNA ordering), NOT the
    swapped shared table. This is the exact divergence the adapter comment warns about."""
    sc = _fake_context(
        token_chains=["A", "A", "B", "C", "D"],
        token_entity=[0, 0, 1, 2, 3],  # protein, protein, rna, dna, ligand
        token_resname=["ALA", "ALA", "A", "DA", "LIG"],
        atom_names=["CA", "CA", "P", "P", "C1"],
    )
    ad = ChaiStructureAdapter(sc, num_atoms=5)
    recs = list(ad.iter_atoms())
    assert [r.mol_type for r in recs] == ["protein", "protein", "rna", "dna", "ligand"]


def test_chai_resid_resets_per_chain():
    """resid is a per-chain 1-based PER-TOKEN ordinal that resets at each chain: the two
    protein tokens in chain A are residues 1 and 2; each other chain restarts at 1."""
    sc = _fake_context(
        token_chains=["A", "A", "B", "C", "D"],
        token_entity=[0, 0, 1, 2, 3],
        token_resname=["ALA", "ALA", "A", "DA", "LIG"],
        atom_names=["CA", "CA", "P", "P", "C1"],
    )
    ad = ChaiStructureAdapter(sc, num_atoms=5)
    recs = list(ad.iter_atoms())
    assert [r.chain for r in recs] == ["A", "A", "B", "C", "D"]
    assert [r.resid for r in recs] == [1, 2, 1, 1, 1]
    assert [r.index for r in recs] == [0, 1, 2, 3, 4]


def test_chai_moltype_none_when_entity_absent():
    """No token_entity_type -> mol_type is None (the non-polymer / unknown fallback), so
    backbone/sidechain/protein selectors simply don't match rather than mis-classify."""
    sc = _fake_context(
        token_chains=["A", "A"],
        token_entity=[0, 0],
        token_resname=["ALA", "ALA"],
        atom_names=["CA", "CB"],
    )
    sc.token_entity_type = None  # entity typing unavailable
    ad = ChaiStructureAdapter(sc, num_atoms=2)
    recs = list(ad.iter_atoms())
    assert [r.mol_type for r in recs] == [None, None]


def test_chai_padding_atoms_skipped():
    """Atoms whose atom_exists_mask is False (diffusion padding) are not yielded."""
    sc = _fake_context(
        token_chains=["A", "A", "B"],
        token_entity=[0, 0, 3],
        token_resname=["ALA", "ALA", "LIG"],
        atom_names=["CA", "CB", "C1"],
    )
    sc.atom_exists_mask = np.array([True, False, True])  # middle atom is padding
    ad = ChaiStructureAdapter(sc, num_atoms=3)
    recs = list(ad.iter_atoms())
    assert [r.index for r in recs] == [0, 2]  # padded atom 1 dropped
