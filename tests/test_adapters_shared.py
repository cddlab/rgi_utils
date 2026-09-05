"""Tests for the consolidated adapter helpers (no framework / no GPU):

- ``decode_atom_name`` (boltz/esm/AF3 shared ord(c)-32 kernel)
- ``MOLTYPE_BY_ID`` (boltz/esm shared enum)
- ``rgi_utils.alphafold3.adapter.AF3RestraintAdapter`` (framework-free, fed plain
  data by the in-tool shim) — verifies it imports NO alphafold3 and that
  iter_atoms / iter_ligand_confs (SMILES positional + CCD by-name leaving-atom drop)
  produce the expected records.
- ``rgi_utils._biotite_adapter`` (protenix/openfold/OpenDDE shared core) over a
  duck-typed fake AtomArray, covering the tools' parameterisation.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rgi_utils._biotite_adapter import biotite_get_elements, biotite_ligand_confs
from rgi_utils._mol_build import _expected_stereo
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


_STEREO_SMILES = "F/C=C/Cl"


def _stereo_graph():
    mol = Chem.MolFromSmiles(_STEREO_SMILES)
    assert mol is not None
    return mol


def _assert_source_e_stereo(ligand_conf):
    assert ligand_conf.stereo_mol is not None
    _atoms, bonds = _expected_stereo(
        ligand_conf.stereo_mol,
        np.zeros((ligand_conf.stereo_mol.GetNumAtoms(), 3)),
    )
    assert list(bonds.values()) == ["E"]


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


_POLY = ("ALA", "ARG", "ASN")  # AF3 residue-name vocabulary; index 0 -> "ALA"

# The two real AF3 vocabularies. `aatype` is filled from the one WITH the gap token, so
# a caller passing the gap-less list shifts every nucleic name by one (A reads as G,
# U as DA) while proteins, which sit below the gap, stay correct.
_AA20 = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)  # fmt: skip
_NT = ("A", "G", "C", "U", "DA", "DG", "DC", "DT")
_POLY_GAPLESS = (*_AA20, "UNK", *_NT)
_POLY_WITH_GAP = (*_AA20, "UNK", "-", *_NT, "N")
_AATYPE_A, _AATYPE_U = 22, 25  # adenine / uridine in the gap-carrying encoding


def _af3_nucleotide_batch():
    """2 nucleotide tokens (one atom each), aatype in AF3's gap-carrying encoding."""
    ranc = np.zeros((2, 1, 4), dtype=np.int64)
    ranc[0, 0] = _enc("N9")
    ranc[1, 0] = _enc("N1")
    return {
        "asym_id": np.array([1, 1]),
        "ref_mask": np.array([[1], [1]]),
        "ref_pos": np.zeros((2, 1, 3)),
        "ref_atom_name_chars": ranc,
        "ref_element": np.array([[7], [7]]),
        "is_protein": np.array([False, False]),
        "is_dna": np.array([False, False]),
        "is_rna": np.array([True, True]),
        "aatype": np.array([_AATYPE_A, _AATYPE_U]),
    }


@pytest.mark.parametrize("vocabulary", [_POLY_GAPLESS, _POLY_WITH_GAP])
def test_af3_nucleic_resnames_survive_either_residue_vocabulary(vocabulary):
    # Read straight off the gap-less list these would be "G" and "DA" -- a base-pair
    # macro would then reject a real A-U pair, and a monomer-library lookup would
    # restrain a ribonucleotide with deoxy geometry. The molecule-type flags settle it.
    ad = AF3RestraintAdapter(
        _af3_nucleotide_batch(), {"A": 1}, vocabulary, ligand_mols=[]
    )
    assert [r.resname for r in ad.iter_atoms()] == ["A", "U"]


@pytest.mark.parametrize(
    "aatype,name", [(22, "A"), (23, "G"), (24, "C"), (26, "DA"), (27, "DG")]
)
@pytest.mark.parametrize("vocabulary", [_POLY_GAPLESS, _POLY_WITH_GAP])
def test_af3_vocabulary_resolution_does_not_depend_on_base_composition(
    aatype, name, vocabulary
):
    batch = _af3_nucleotide_batch()
    batch["aatype"][:] = aatype
    if name.startswith("D"):
        batch["is_rna"][:] = False
        batch["is_dna"][:] = True
    adapter = AF3RestraintAdapter(batch, {"A": 1}, vocabulary, ligand_mols=[])
    assert [record.resname for record in adapter.iter_atoms()] == [name, name]


def test_af3_subset_preserves_chemistry_properties_and_coordinate_order():
    source = Chem.MolFromSmiles("C[N+](C)(C)[13CH3].F/C=C/Cl.N[C@@H](C)C(=O)O.[Cl-]")
    source.SetProp("source", "fixture")
    source.SetProp("identifier", "00123")
    source.SetIntProp("integer", 7)
    source.SetBoolProp("boolean", True)
    source.SetDoubleProp("real", 2.5)
    for atom in source.GetAtoms():
        atom.SetProp("atom_name", f"A{atom.GetIdx()}")
    source.GetBondWithIdx(0).SetProp("source_bond", "retained")
    conformer = Chem.Conformer(source.GetNumAtoms())
    for i in range(source.GetNumAtoms()):
        conformer.SetAtomPosition(i, (float(i), 0.0, 0.0))
    source.AddConformer(conformer)
    before = Chem.MolToSmiles(source)
    kept = list(reversed(range(source.GetNumAtoms() - 1)))
    result = AF3RestraintAdapter._subset_mol(source, kept, Chem)
    assert Chem.MolToSmiles(source) == before
    assert Chem.GetFormalCharge(result) == 1
    assert result.GetProp("source") == "fixture"
    assert result.GetProp("identifier") == "00123"
    assert result.GetIntProp("integer") == 7
    assert result.GetBoolProp("boolean") is True
    assert result.GetDoubleProp("real") == 2.5
    for new, old in enumerate(kept):
        atom, original = result.GetAtomWithIdx(new), source.GetAtomWithIdx(old)
        assert atom.GetAtomicNum() == original.GetAtomicNum()
        assert atom.GetIsotope() == original.GetIsotope()
        assert atom.GetFormalCharge() == original.GetFormalCharge()
        assert atom.GetProp("atom_name") == original.GetProp("atom_name")
    np.testing.assert_array_equal(result.GetConformer().GetPositions()[:, 0], kept)
    expected = Chem.RWMol(source)
    expected.RemoveAtom(source.GetNumAtoms() - 1)
    assert Chem.MolToSmiles(result) == Chem.MolToSmiles(expected.GetMol())
    assert any(b.HasProp("source_bond") for b in result.GetBonds())


def test_af3_protein_only_batch_keeps_the_vocabulary_as_given():
    # Nothing below the gap moves, so a protein/ligand batch must not be "corrected".
    ad = AF3RestraintAdapter(_af3_batch(), {"A": 1, "B": 2}, _POLY, ligand_mols=[])
    assert ad._name_shift == 0


def test_boltz_atom_metadata_uses_bulk_reads(monkeypatch):
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from rgi_utils.boltz.adapter import BoltzFeatsAdapter

    tokens = torch.tensor([0, 0, 1, 2, 3, 4, 0])
    mapping = torch.nn.functional.one_hot(tokens, num_classes=5).float()
    feats = {
        "atom_to_token": torch.stack([mapping, mapping]),
        "asym_id": torch.tensor([[7, 7, 9, 9, 9], [99, 99, 99, 99, 99]]),
        "atom_pad_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0]] * 2),
        "mol_type": torch.tensor([[0, 0, 1, 1, 1]] * 2),
        "ref_conformer_restraint": torch.tensor([[0, 0, 0, 1, 1, 1, 0]] * 2),
        "record": [
            SimpleNamespace(
                chains=[
                    SimpleNamespace(chain_id=7, chain_name="A"),
                    SimpleNamespace(chain_id=9, chain_name="B"),
                ]
            )
        ],
    }
    calls = []
    original_item = torch.Tensor.item

    def counted_item(tensor, *args):
        calls.append(None)
        return original_item(tensor, *args)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    adapter = BoltzFeatsAdapter(feats)
    records = list(adapter.iter_atoms())
    assert list(adapter.iter_atoms()) == records
    assert [(r.chain, r.resid, r.index, r.conformer_restraints) for r in records] == [
        ("A", 1, 0, False),
        ("A", 1, 1, False),
        ("A", 2, 2, False),
        ("B", 1, 3, True),
        ("B", 2, 4, True),
        ("B", 3, 5, True),
    ]
    assert calls == []


def test_boltz_ligand_retains_source_stereo():
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from rgi_utils.boltz.adapter import BoltzFeatsAdapter

    source = Chem.AddHs(_stereo_graph())
    assert AllChem.EmbedMolecule(source, randomSeed=7) == 0
    source = Chem.RemoveHs(source)
    n_atom = source.GetNumAtoms()
    identity = torch.eye(n_atom).unsqueeze(0)
    feats = {
        "atom_to_token": identity,
        "asym_id": torch.full((1, n_atom), 7),
        "atom_pad_mask": torch.ones((1, n_atom), dtype=torch.bool),
        "ref_element": torch.tensor(
            [[a.GetAtomicNum() for a in source.GetAtoms()]], dtype=torch.long
        ),
        "ref_conformer_restraint": torch.ones((1, n_atom), dtype=torch.bool),
        "record": [
            SimpleNamespace(chains=[SimpleNamespace(chain_id=7, chain_name="B")])
        ],
        "ligand_mols": {7: source},
    }
    ligand = list(BoltzFeatsAdapter(feats).iter_ligand_confs())
    assert len(ligand) == 1
    _assert_source_e_stereo(ligand[0])


def test_af3_adapter_imports_no_alphafold3():
    # the whole point of the split: the rgi_utils adapter must not pull in alphafold3
    assert "alphafold3" not in sys.modules


def test_af3_iter_atoms():
    ad = AF3RestraintAdapter(
        _af3_batch(),
        {"A": 1, "B": 2},
        _POLY,
        ligand_mols=[],
        conformer_restraints_by_asym={1: True, 2: False},
    )
    recs = list(ad.iter_atoms())
    # 2 real atoms in token0 + 1 in token1 (padding atoms skipped)
    assert [(r.chain, r.resid, r.index, r.name) for r in recs] == [
        ("A", 1, 0, "CA"),
        ("A", 1, 1, "CB"),
        ("B", 1, 3, "C1"),  # flat = token1*3 + 0
    ]
    assert [r.mol_type for r in recs] == ["protein", "protein", None]
    assert [r.resname for r in recs] == ["ALA", "ALA", "ALA"]
    assert [r.conformer_restraints for r in recs] == [True, True, False]
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


def test_af3_smiles_ligand_retains_source_stereo():
    source = _stereo_graph()
    n_atom = source.GetNumAtoms()
    names = np.stack(
        [_enc(f"{a.GetSymbol()}{i + 1}") for i, a in enumerate(source.GetAtoms())]
    )
    batch = {
        "asym_id": np.ones(n_atom, dtype=np.int64),
        "ref_mask": np.ones((n_atom, 1), dtype=np.int64),
        "ref_pos": np.zeros((n_atom, 1, 3), dtype=np.float64),
        "ref_atom_name_chars": names[:, None, :],
        "ref_element": np.array(
            [[a.GetAtomicNum()] for a in source.GetAtoms()], dtype=np.int64
        ),
        "is_protein": np.zeros(n_atom, dtype=bool),
        "is_dna": np.zeros(n_atom, dtype=bool),
        "is_rna": np.zeros(n_atom, dtype=bool),
        "aatype": np.zeros(n_atom, dtype=np.int64),
    }
    adapter = AF3RestraintAdapter(
        batch, {"B": 1}, _POLY, ligand_mols=[("B", source, True)]
    )
    _assert_source_e_stereo(list(adapter.iter_ligand_confs())[0])


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


def test_esmfold2_adapter_retains_source_stereo():
    from types import SimpleNamespace

    from rgi_utils.esmfold2.adapter import ESMFold2Adapter

    source = _stereo_graph()
    names = ["F1", "C1", "C2", "CL1"]
    token_bonds = np.zeros((4, 4), dtype=np.float32)
    orders = []
    for bond in source.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        token_bonds[i, j] = token_bonds[j, i] = 1
        orders.append((names[i], names[j], int(bond.GetBondTypeAsDouble())))
    features = {
        "asym_id": np.ones((1, 4), dtype=np.int64),
        "token_attention_mask": np.ones((1, 4), dtype=np.int64),
        "mol_type": np.full((1, 4), 3, dtype=np.int64),
        "atom_to_token": np.arange(4, dtype=np.int64)[None],
        "atom_attention_mask": np.ones((1, 4), dtype=np.int64),
        "ref_pos": np.zeros((1, 4, 3), dtype=np.float64),
        "ref_element": np.array(
            [[atom.GetAtomicNum() for atom in source.GetAtoms()]], dtype=np.int64
        ),
        "ref_atom_name_chars": np.stack([_enc(name) for name in names])[None],
        "token_bonds": token_bonds[None],
    }
    chain = SimpleNamespace(
        asym_id=1,
        chain_id="B",
        ligand_bond_orders=orders,
        conformer_restraints=True,
        source_smiles=_STEREO_SMILES,
    )
    ligand = list(ESMFold2Adapter(features, [chain]).iter_ligand_confs())
    assert len(ligand) == 1
    _assert_source_e_stereo(ligand[0])


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


def _stereo_aa():
    source = _stereo_graph()
    return _FakeAtomArray(
        element=[atom.GetSymbol() for atom in source.GetAtoms()],
        coord=[[-1, 1, 0], [0, 0, 0], [1, 0, 0], [2, -1, 0]],
        bonds=[
            [
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                int(bond.GetBondTypeAsDouble()),
            ]
            for bond in source.GetBonds()
        ],
        annots=["molecule_type_id", "conformer_restraints"],
        label_asym_id=["L"] * 4,
        chain_id=["L"] * 4,
        hetero=[True] * 4,
        molecule_type_id=[3] * 4,
        mol_type=["ligand"] * 4,
        atom_name=["F1", "C1", "C2", "CL1"],
        res_name=["UNL"] * 4,
        conformer_restraints=[True] * 4,
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


def test_protenix_adapter_retains_source_stereo():
    from rgi_utils.protenix.adapter import ProtenixAdapter

    aa = _stereo_aa()
    adapter = ProtenixAdapter(
        {
            "atom_array": aa,
            "atom_to_token_idx": np.zeros((1, 4)),
            "smiles_by_chain": {"L": _STEREO_SMILES},
        }
    )
    _assert_source_e_stereo(list(adapter.iter_ligand_confs())[0])


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


def test_openfold3_adapter_retains_source_stereo():
    from rgi_utils.openfold3.adapter import Openfold3Adapter

    aa = _stereo_aa()
    adapter = Openfold3Adapter(
        aa,
        num_atoms=4,
        ref_coords=aa.coord,
        smiles_by_chain={"L": _STEREO_SMILES},
    )
    _assert_source_e_stereo(list(adapter.iter_ligand_confs())[0])


def test_opendde_adapter_uses_residue_level_tokens_and_ref_pos():
    from rgi_utils.opendde.adapter import OpenDDEAdapter

    aa = _FakeAtomArray(
        element=["N", "C", "C", "C"],
        coord=np.zeros((4, 3)),
        bonds=[[2, 3, 2]],
        annots=["conformer_restraints"],
        label_asym_id=["A", "A", "L", "L"],
        atom_name=["N", "CA", "C1", "C2"],
        res_name=["ALA", "ALA", "UNL", "UNL"],
        mol_type=["protein", "protein", "ligand", "ligand"],
        conformer_restraints=[False, False, True, True],
    )
    ref_pos = np.array([[0, 0, 0], [1, 0, 0], [4, 0, 0], [5.3, 0, 0]], dtype=float)
    ad = OpenDDEAdapter(
        {
            "atom_array": aa,
            # Structural tokens intentionally differ. The adapter must use the
            # residue-level mapping retained by OpenDDE before expansion.
            "atom_to_token_idx": np.array([[0, 1, 2, 3]]),
            "residue_level_atom_to_token_idx": np.array([[0, 0, 1, 2]]),
            "ref_pos": ref_pos[None],
            "ref_space_uid": np.array([[0, 0, 1, 1]]),
        }
    )

    records = list(ad.iter_atoms())
    assert [(r.chain, r.resid, r.index, r.mol_type) for r in records] == [
        ("A", 1, 0, "protein"),
        ("A", 1, 1, "protein"),
        ("L", 1, 2, "ligand"),
        ("L", 2, 3, "ligand"),
    ]
    assert ad.num_atoms() == 4
    assert ad.get_elements().tolist() == [7, 6, 6, 6]
    assert np.array_equal(ad.get_reference_positions(), ref_pos)
    assert ad.get_reference_space_uid().tolist() == [0, 0, 1, 1]

    ligand = list(ad.iter_ligand_confs())
    assert len(ligand) == 1
    assert ligand[0].global_indices.tolist() == [2, 3]
    assert np.array_equal(ligand[0].conf_coords, ref_pos[2:])
    assert ligand[0].mol.GetBondWithIdx(0).GetBondTypeAsDouble() == 2.0
    assert ligand[0].conformer_restraints is True


def test_opendde_adapter_retains_source_stereo():
    from rgi_utils.opendde.adapter import OpenDDEAdapter

    aa = _stereo_aa()
    adapter = OpenDDEAdapter(
        {
            "atom_array": aa,
            "atom_to_token_idx": np.arange(4)[None],
            "ref_pos": aa.coord[None],
            "ref_space_uid": np.zeros((1, 4), dtype=np.int64),
            "smiles_by_chain": {"L": _STEREO_SMILES},
        }
    )
    _assert_source_e_stereo(list(adapter.iter_ligand_confs())[0])


def test_opendde_adapter_imports_no_tool_or_torch():
    import importlib

    before = set(sys.modules)
    importlib.import_module("rgi_utils.opendde.adapter")
    newly_loaded = set(sys.modules) - before
    assert not any(
        name == "opendde" or name.startswith("opendde.") for name in newly_loaded
    )
    assert "torch" not in newly_loaded
