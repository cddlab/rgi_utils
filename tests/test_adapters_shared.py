"""Tests for the consolidated adapter helpers (no framework / no GPU):

- ``decode_atom_name`` (boltz/esm/AF3 shared ord(c)-32 kernel)
- ``MOLTYPE_BY_ID`` (boltz/esm shared enum)
- ``rgi_utils.alphafold3.adapter.AF3RestraintAdapter`` (framework-free, fed plain
  data by the in-tool shim) — verifies it imports NO alphafold3 and that
  iter_atoms / iter_ligand_confs (SMILES positional + CCD by-name leaving-atom drop)
  produce the expected records.
- ``rgi_utils._biotite_adapter`` (protenix/openfold shared core) over a duck-typed
  fake AtomArray, covering both tools' parameterisation.
"""

from __future__ import annotations

import sys

import numpy as np
from rdkit import Chem

from rgi_utils._biotite_adapter import biotite_get_elements, biotite_ligand_confs
from rgi_utils._moltype import MOLTYPE_BY_ID
from rgi_utils.alphafold3.adapter import AF3RestraintAdapter
from rgi_utils.atom_context import decode_atom_name


# --- shared helpers --------------------------------------------------------------
def test_decode_atom_name():
    # ord(c) - 32 encoding: 'C'->35, 'A'->33, 'B'->34, '1'->17
    assert decode_atom_name([35, 33, 0, 0]) == "CA"
    assert decode_atom_name([35, 34]) == "CB"
    assert decode_atom_name([35, 17]) == "C1"
    assert decode_atom_name([0, 0, 0, 0]) is None  # all-padding -> None
    assert decode_atom_name([]) is None


def test_moltype_by_id():
    # boltz/esm enum order: PROTEIN=0 DNA=1 RNA=2 NONPOLYMER=3
    assert MOLTYPE_BY_ID == {0: "protein", 1: "dna", 2: "rna", 3: "ligand"}


# --- AF3 framework-free adapter --------------------------------------------------
def _enc(name: str, width: int = 4) -> np.ndarray:
    a = np.zeros(width, dtype=np.int64)
    for i, ch in enumerate(name):
        a[i] = ord(ch) - 32
    return a


def _af3_batch():
    """2 tokens, max_atoms_per_token=3: token0 = protein residue (atoms CA, CB),
    token1 = a ligand atom (C1). flat_idx = token*3 + within."""
    ranc = np.zeros((2, 3, 4), dtype=np.int64)
    ranc[0, 0] = _enc("CA")
    ranc[0, 1] = _enc("CB")
    ranc[1, 0] = _enc("C1")
    return {
        "asym_id": np.array([1, 2]),
        "ref_mask": np.array([[1, 1, 0], [1, 0, 0]]),
        "ref_pos": np.array(
            [
                [[0.0, 0, 0], [1, 0, 0], [0, 0, 0]],
                [[5, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]
        ),
        "ref_atom_name_chars": ranc,
        "ref_element": np.array([[6, 6, 0], [6, 0, 0]]),
        "is_protein": np.array([True, False]),
        "is_dna": np.array([False, False]),
        "is_rna": np.array([False, False]),
        "aatype": np.array([0, 0]),
    }


_POLY = ("ALA", "ARG", "ASN")  # POLYMER_TYPES stand-in; index 0 -> "ALA"


def test_af3_adapter_imports_no_alphafold3():
    # the whole point of the split: the rgi_utils adapter must not pull in alphafold3
    assert "alphafold3" not in sys.modules


def test_af3_iter_atoms():
    ad = AF3RestraintAdapter(_af3_batch(), {"A": 1, "B": 2}, _POLY, ligand_mols=[])
    recs = list(ad.iter_atoms())
    # 2 real atoms in token0 + 1 in token1 (padding atoms skipped)
    assert [(r.chain, r.resid, r.index, r.name) for r in recs] == [
        ("A", 1, 0, "CA"),
        ("A", 1, 1, "CB"),
        ("B", 1, 3, "C1"),  # flat = token1*3 + 0
    ]
    assert [r.mol_type for r in recs] == ["protein", "protein", None]
    assert [r.resname for r in recs] == ["ALA", "ALA", "ALA"]
    assert ad.num_atoms() == 6  # 2 tokens * 3 max
    # get_elements: flat with padding (mask 0) -> 0
    assert ad.get_elements().tolist() == [6, 6, 0, 6, 0, 0]


def test_af3_iter_ligand_confs_smiles():
    """SMILES ligand: positional flat-index mapping (1 atom per token)."""
    mol = Chem.MolFromSmiles("C")  # methane, 1 heavy atom
    ad = AF3RestraintAdapter(
        _af3_batch(), {"A": 1, "B": 2}, _POLY, ligand_mols=[("B", mol, True)]
    )
    confs = list(ad.iter_ligand_confs())
    assert len(confs) == 1
    lc = confs[0]
    assert lc.global_indices.tolist() == [3]  # token1*3
    assert lc.conf_coords.shape == (1, 3)
    assert lc.conf_coords[0].tolist() == [5.0, 0.0, 0.0]  # ref_pos[token1, within0]
    assert lc.conformer_restraints is True


def test_af3_iter_ligand_confs_ccd_leaving_atom_drop():
    """CCD ligand: by-name mapping; an atom absent from the structure (C2) is dropped
    and the mol is subset to the kept atom (C1)."""
    rw = Chem.RWMol()
    for nm in ("C1", "C2"):
        a = Chem.Atom(6)
        a.SetProp("atom_name", nm)
        rw.AddAtom(a)
    rw.AddBond(0, 1, Chem.BondType.SINGLE)
    ccd_mol = rw.GetMol()
    ad = AF3RestraintAdapter(
        _af3_batch(), {"A": 1, "B": 2}, _POLY, ligand_mols=[("B", ccd_mol, False)]
    )
    confs = list(ad.iter_ligand_confs())
    assert len(confs) == 1
    lc = confs[0]
    # only C1 survives (C2 is not in the structure) -> 1 atom, flat index 3
    assert lc.global_indices.tolist() == [3]
    assert lc.mol.GetNumAtoms() == 1
    assert lc.mol.GetAtomWithIdx(0).GetProp("atom_name") == "C1"


# --- biotite shared core (protenix/openfold) -------------------------------------
class _FakeBonds:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def as_array(self):
        return self._arr


class _FakeAtomArray:
    """Duck-typed biotite AtomArray (no biotite import needed)."""

    def __init__(self, element, coord, bonds=None, annots=None, **named):
        self.element = np.asarray(element)
        self.coord = np.asarray(coord, dtype=float)
        self.bonds = _FakeBonds(bonds) if bonds is not None else None
        self._annots = list(annots or [])
        for k, v in named.items():
            setattr(self, k, np.asarray(v))

    def __len__(self):
        return len(self.element)

    def get_annotation_categories(self):
        return self._annots


def _fake_aa(**extra):
    # 3 atoms: a 2-atom ligand chain "L" (C-C bonded) + 1 protein atom chain "P".
    return _FakeAtomArray(
        element=["C", "C", "N"],
        coord=[[0, 0, 0], [1.5, 0, 0], [9, 9, 9]],
        bonds=[[0, 1, 1]],  # i, j, order
        label_asym_id=["L", "L", "P"],
        chain_id=["L", "L", "P"],
        hetero=[True, True, False],
        molecule_type_id=[3, 3, 0],  # 3 == LIGAND
        **extra,
    )


def test_biotite_get_elements():
    aa = _fake_aa()
    # n_atom=5 padded: ['C','C','N'] -> [6,6,7], padding -> 0
    assert biotite_get_elements(aa, 5).tolist() == [6, 6, 7, 0, 0]
    assert biotite_get_elements(None, 3).tolist() == [0, 0, 0]


def test_biotite_ligand_confs_default_on():
    # helper contract: honor an explicit conf_rest_default=True (no tool uses this now --
    # protenix + openfold both pass False -- but the helper must still respect it).
    aa = _fake_aa()
    confs = list(
        biotite_ligand_confs(
            aa,
            ligand_mask=aa.molecule_type_id == 3,
            chain_attr="chain_id",
            coords_all=aa.coord,
            conf_rest_default=True,  # helper honors an explicit default-on
            post_build=None,
        )
    )
    assert len(confs) == 1
    lc = confs[0]
    assert lc.global_indices.tolist() == [0, 1]
    assert lc.mol.GetNumAtoms() == 2
    assert lc.mol.GetNumBonds() == 1
    assert lc.conformer_restraints is True  # default (no annotation)


def test_biotite_ligand_confs_protenix_style_default_off():
    aa = _fake_aa()
    confs = list(
        biotite_ligand_confs(
            aa,
            ligand_mask=aa.hetero.astype(bool),
            chain_attr="label_asym_id",
            coords_all=aa.coord,
            conf_rest_default=False,  # protenix default OFF
            post_build=None,
        )
    )
    assert len(confs) == 1
    assert confs[0].conformer_restraints is False  # default off, no annotation


def test_biotite_ligand_confs_annotation_overrides_default():
    # a conformer_restraints annotation present -> used regardless of the default
    aa = _fake_aa(conformer_restraints=[True, True, False])
    aa._annots = ["conformer_restraints"]
    confs = list(
        biotite_ligand_confs(
            aa,
            ligand_mask=aa.hetero.astype(bool),
            chain_attr="label_asym_id",
            coords_all=aa.coord,
            conf_rest_default=False,
            post_build=None,
        )
    )
    assert confs[0].conformer_restraints is True  # annotation any() over [0,1]


def test_biotite_ligand_confs_post_build_hook():
    """post_build replaces the target geometry (protenix's SMILES ideal-conformer)."""
    aa = _fake_aa()
    moved = np.array([[7.0, 7, 7], [8, 8, 8]])

    def _post(chain_id, mol, coords, idxs, elements_all, bonds_local):
        return mol, moved

    confs = list(
        biotite_ligand_confs(
            aa,
            ligand_mask=aa.hetero.astype(bool),
            chain_attr="label_asym_id",
            coords_all=aa.coord,
            conf_rest_default=False,
            post_build=_post,
        )
    )
    assert np.array_equal(confs[0].conf_coords, moved)


# --- the real adapter classes through the shared core (delegation wiring) --------
# These construct ProtenixAdapter / Openfold3Adapter and call the delegating methods,
# so a runtime NameError / wrong kwarg INSIDE iter_ligand_confs/get_elements (which
# "import succeeds" can't catch) is caught here on CPU, not only at the GPU smoke.
def test_protenix_adapter_delegation():
    from rgi_utils.protenix.adapter import ProtenixAdapter

    aa = _fake_aa()  # hetero=[T,T,F] marks the 2-atom ligand chain "L"
    ad = ProtenixAdapter({"atom_array": aa, "atom_to_token_idx": np.zeros((1, 5))})
    assert ad.get_elements().tolist() == [6, 6, 7, 0, 0]
    confs = list(ad.iter_ligand_confs())
    assert len(confs) == 1
    assert confs[0].global_indices.tolist() == [0, 1]
    assert confs[0].conformer_restraints is False  # protenix default OFF


def test_openfold3_adapter_delegation():
    from rgi_utils.openfold3.adapter import Openfold3Adapter

    # molecule_type_id annotation present -> _ligand_mask uses it (3 == LIGAND)
    aa = _FakeAtomArray(
        element=["C", "C", "N"],
        coord=[[0, 0, 0], [1.5, 0, 0], [9, 9, 9]],
        bonds=[[0, 1, 1]],
        annots=["molecule_type_id"],
        chain_id=["L", "L", "P"],
        hetero=[True, True, False],
        molecule_type_id=[3, 3, 0],
    )
    ad = Openfold3Adapter(aa, num_atoms=5, ref_coords=aa.coord)
    assert ad.get_elements().tolist() == [6, 6, 7, 0, 0]
    confs = list(ad.iter_ligand_confs())
    assert len(confs) == 1
    assert confs[0].global_indices.tolist() == [0, 1]
    # no conformer_restraints annotation -> default OFF (opt-in required, like every tool)
    assert confs[0].conformer_restraints is False

    # conformer_restraints annotation present -> the flagged ligand opts in
    aa_optin = _FakeAtomArray(
        element=["C", "C", "N"],
        coord=[[0, 0, 0], [1.5, 0, 0], [9, 9, 9]],
        bonds=[[0, 1, 1]],
        annots=["molecule_type_id", "conformer_restraints"],
        chain_id=["L", "L", "P"],
        hetero=[True, True, False],
        molecule_type_id=[3, 3, 0],
        conformer_restraints=[True, True, False],
    )
    ad_optin = Openfold3Adapter(aa_optin, num_atoms=5, ref_coords=aa_optin.coord)
    confs_optin = list(ad_optin.iter_ligand_confs())
    assert confs_optin[0].conformer_restraints is True
