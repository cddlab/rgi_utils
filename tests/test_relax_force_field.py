"""``conformer_restraints_config.relax_force_field.ligand``: reference relaxation.

The conformer restraint measures its bond/angle/chiral/cistrans/plane targets off a
force-field-relaxed copy of the tool's cached conformer, not off the cached conformer
itself. These tests cover the force-field selection: the config surface, the deliberate
UFF-falls-back / MMFF-raises asymmetry, and the RDKit trap that MMFF typing kekulizes the
molecule it is handed.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import rgi_utils._mol_build as mol_build
from rgi_utils import CombinedRestraints
from rgi_utils._mol_build import (
    RelaxError,
    StereoGenerationError,
    _embedded_heavy_coords,
    _expected_stereo,
    _stereo_mismatch_counts,
    align_stereo_mol,
    build_ligand_mol,
    ff_relax,
    parse_relax_force_field,
    repair_stereo,
)
from rgi_utils.atom_context import AtomRecord, LigandConf
from rgi_utils.config import RestraintsConfig
from rgi_utils.featurizer import _extract_conformer, build_spec


def _embed(smiles, seed=7):
    """Heavy-atom mol + its ETKDG conformer coords, in mol atom order."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(m, randomSeed=seed) == 0
    m = Chem.RemoveHs(m)
    return m, np.asarray(m.GetConformer(0).GetPositions(), dtype=np.float64)


def _bond_snapshot(mol):
    return [(b.GetIsAromatic(), str(b.GetBondType())) for b in mol.GetBonds()]


def _stereo_mismatches(reference_mol, reference_coords, candidate_coords):
    expected = _expected_stereo(reference_mol, reference_coords)
    return expected, _stereo_mismatch_counts(reference_mol, candidate_coords, expected)


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "value,expected",
    [
        ("uff", "uff"),
        ("mmff94", "mmff94"),
        ("mmff94s", "mmff94s"),
        ("none", "none"),
        ("MMFF94s", "mmff94s"),  # case-insensitive
        ("UFF", "uff"),
    ],
)
def test_parse_relax_force_field_values(value, expected):
    assert parse_relax_force_field({"relax_force_field": {"ligand": value}}) == expected


def test_parse_relax_force_field_defaults_to_uff():
    # Omitted ligand and an explicit YAML null mean "not set" -> the existing default.
    assert parse_relax_force_field({}) == "uff"
    assert parse_relax_force_field(None) == "uff"
    assert parse_relax_force_field({"relax_force_field": {}}) == "uff"
    assert parse_relax_force_field({"relax_force_field": {"ligand": None}}) == "uff"


@pytest.mark.parametrize("spec", ["uff", None, 1, []])
def test_parse_relax_force_field_rejects_non_mapping(spec):
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_relax_force_field({"relax_force_field": spec})


def test_parse_relax_force_field_scalar_error_has_migration_hint():
    with pytest.raises(ValueError) as exc_info:
        RestraintsConfig.from_dict(
            {
                "conformer_restraints_config": {
                    "bond": {},
                    "relax_force_field": "mmff94s",
                }
            }
        )
    assert "scalar form is no longer supported" in str(exc_info.value)
    assert "relax_force_field: {ligand: 'mmff94s'}" in str(exc_info.value)


def test_parse_relax_force_field_rejects_unknown_key():
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['monomer'\]"):
        parse_relax_force_field({"relax_force_field": {"monomer": "uff"}})


def test_parse_relax_force_field_rejects_typo():
    with pytest.raises(ValueError, match="unknown value 'mmf94'"):
        parse_relax_force_field({"relax_force_field": {"ligand": "mmf94"}})


def test_config_validates_value_not_just_key():
    """A bad VALUE must raise while parsing the config.

    The conformer whitelist only checks the KEY, so without the value check a typo would
    silently run UFF on any structure where no ligand opts in (build_spec never reaches
    the force field at all).
    """
    with pytest.raises(ValueError, match="relax_force_field"):
        RestraintsConfig.from_dict(
            {
                "conformer_restraints_config": {
                    "bond": {},
                    "relax_force_field": {"ligand": "mmf94"},
                }
            }
        )
    # The valid mapping is preserved in the parsed config.
    cfg = RestraintsConfig.from_dict(
        {
            "conformer_restraints_config": {
                "bond": {},
                "relax_force_field": {"ligand": "mmff94s"},
            }
        }
    )
    assert cfg.conformer_config["relax_force_field"] == {"ligand": "mmff94s"}


# ----------------------------------------------------------------------- the switch


def test_force_fields_give_different_targets():
    """The option must actually change the geometry the targets are measured off."""
    mol, crds = _embed("Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O")
    out = {ff: ff_relax(mol, crds, ff) for ff in ("uff", "mmff94", "mmff94s")}
    assert all(c is not None and len(c) == len(crds) for c in out.values())

    # the exocyclic amine C-N: MMFF pulls the conjugated bond well below UFF (measured
    # 1.428 / 1.389 / 1.376 against a monomer-library 1.330).
    def c6_n6(c):
        return float(np.linalg.norm(c[0] - c[1]))

    assert c6_n6(out["uff"]) - c6_n6(out["mmff94"]) > 0.02
    assert c6_n6(out["mmff94"]) - c6_n6(out["mmff94s"]) > 0.005


def test_mmff94s_differs_from_mmff94_on_amide():
    """94s is the planar-nitrogen variant, so an amide is where it must diverge."""
    mol, crds = _embed("CC(=O)NC")
    a = ff_relax(mol, crds, "mmff94")
    b = ff_relax(mol, crds, "mmff94s")
    assert a is not None and b is not None
    assert not np.allclose(a, b, atol=1e-6)


def test_none_keeps_the_cached_conformer_exactly():
    """ligand: none -> targets come straight off the tool's conformer."""
    mol, crds = _embed("Nc1ncnc2n(cnc12)[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O")
    lc = LigandConf(
        mol=mol,
        conf_coords=crds,
        global_indices=np.arange(mol.GetNumAtoms()),
        conformer_restraints=True,
    )
    raw, *_ = _extract_conformer([lc], force_field="none")
    relaxed, *_ = _extract_conformer([lc], force_field="uff")

    expected = {
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx()): float(
            np.linalg.norm(crds[b.GetBeginAtomIdx()] - crds[b.GetEndAtomIdx()])
        )
        for b in mol.GetBonds()
    }
    for g0, g1, r0, _esd in raw:
        assert r0 == pytest.approx(expected[(g0, g1)], abs=1e-12)
    # ...and the relax really was doing something, so "none" is not a no-op distinction
    assert any(
        r0 != pytest.approx(expected[(g0, g1)], abs=1e-6)
        for g0, g1, r0, _esd in relaxed
    )


def test_bad_coordinate_count_keeps_uff_soft_but_mmff_explicit():
    mol, coords = _embed("CC(=O)NC")
    assert ff_relax(mol, coords[:-1], "uff") is None
    with pytest.raises(RelaxError, match="coordinate count"):
        ff_relax(mol, coords[:-1], "mmff94")


# -------------------------------------------------------- stereo-preserving retries


def test_stereo_validation_detects_ez_and_chiral_inversions():
    trans, trans_coords = _embed("OC(=O)/C=C/C(=O)O")
    cis, cis_coords = _embed(r"OC(=O)/C=C\C(=O)O")
    assert [a.GetAtomicNum() for a in trans.GetAtoms()] == [
        a.GetAtomicNum() for a in cis.GetAtoms()
    ]
    expected, mismatch = _stereo_mismatches(trans, trans_coords, cis_coords)
    assert len(expected[1]) == 1
    assert mismatch == (0, 1)

    left, left_coords = _embed("N[C@@H](C)C(=O)O")
    right, right_coords = _embed("N[C@H](C)C(=O)O")
    expected, mismatch = _stereo_mismatches(left, left_coords, right_coords)
    assert len(expected[0]) == 1
    assert mismatch == (1, 0)


@pytest.mark.parametrize("force_field", ["uff", "mmff94", "mmff94s"])
def test_stereo_inversion_retries_from_an_etkdg_seed(monkeypatch, caplog, force_field):
    trans, trans_coords = _embed("OC(=O)/C=C/C(=O)O")
    _cis, cis_coords = _embed(r"OC(=O)/C=C\C(=O)O")
    calls = 0

    def fake_relax_once(_mol, coords, _force_field):
        nonlocal calls
        calls += 1
        return cis_coords if calls == 1 else np.asarray(coords, dtype=np.float64)

    monkeypatch.setattr(mol_build, "_ff_relax_once", fake_relax_once)
    out = ff_relax(trans, trans_coords, force_field)
    expected = _expected_stereo(trans, trans_coords)
    assert out is not None
    assert _stereo_mismatch_counts(trans, out, expected) == (0, 0)
    assert calls == 2
    assert "stereo retry succeeded" in caplog.text


@pytest.mark.parametrize("force_field", ["uff", "mmff94"])
def test_all_stereo_retries_fail_with_existing_force_field_policy(
    monkeypatch, force_field
):
    trans, trans_coords = _embed("OC(=O)/C=C/C(=O)O")
    _cis, cis_coords = _embed(r"OC(=O)/C=C\C(=O)O")

    monkeypatch.setattr(
        mol_build,
        "_ff_relax_once",
        lambda _mol, _coords, _force_field: cis_coords,
    )
    if force_field == "uff":
        assert ff_relax(trans, trans_coords, force_field) is None
    else:
        with pytest.raises(RelaxError, match="could not preserve"):
            ff_relax(trans, trans_coords, force_field)


def test_axt_real_uff_ez_flip_retries_to_correct_stereo():
    smiles = (
        "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/"
        "C2=C(C)C(=O)[C@@H](O)CC2(C)C)C(C)(C)C[C@H](O)C1=O"
    )
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    coords = _embedded_heavy_coords(mol, 6)
    assert coords is not None
    expected = _expected_stereo(mol, coords)
    assert len(expected[0]) == 2
    assert len(expected[1]) == 9

    inverted = mol_build._ff_relax_once(mol, coords, "uff")
    assert inverted is not None
    assert _stereo_mismatch_counts(mol, inverted, expected) == (0, 1)

    retried = ff_relax(mol, coords, "uff")
    assert retried is not None
    assert _stereo_mismatch_counts(mol, retried, expected) == (0, 0)


def _coordinate_built_mol(graph_mol, coords):
    elements = [atom.GetSymbol() for atom in graph_mol.GetAtoms()]
    bonds = [
        (
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            int(bond.GetBondTypeAsDouble()),
        )
        for bond in graph_mol.GetBonds()
    ]
    return build_ligand_mol(elements, coords, bonds)


def test_source_stereo_mol_detects_an_already_inverted_reference(monkeypatch):
    source, source_coords = _embed("OC(=O)/C=C/C(=O)O")
    _cis, cis_coords = _embed(r"OC(=O)/C=C\C(=O)O")
    working = _coordinate_built_mol(source, cis_coords)
    expected = _expected_stereo(source, source_coords)
    assert _stereo_mismatch_counts(source, cis_coords, expected) == (0, 1)

    calls = 0

    def fake_relax_once(_mol, coords, _force_field):
        nonlocal calls
        calls += 1
        return cis_coords if calls == 1 else np.asarray(coords, dtype=np.float64)

    monkeypatch.setattr(mol_build, "_ff_relax_once", fake_relax_once)
    monkeypatch.setattr(
        mol_build,
        "_embedded_heavy_coords",
        lambda _mol, _seed: source_coords,
    )
    out = ff_relax(working, cis_coords, "uff", stereo_mol=source)
    assert _stereo_mismatch_counts(source, out, expected) == (0, 0)
    assert calls == 2


def test_source_stereo_failure_stops_after_four_regenerations(monkeypatch):
    source, source_coords = _embed("OC(=O)/C=C/C(=O)O")
    _cis, cis_coords = _embed(r"OC(=O)/C=C\C(=O)O")
    working = _coordinate_built_mol(source, cis_coords)
    relax_calls = 0
    embed_calls = 0

    def always_wrong(_mol, _coords, _force_field):
        nonlocal relax_calls
        relax_calls += 1
        return cis_coords

    def correct_embedding(_mol, _seed):
        nonlocal embed_calls
        embed_calls += 1
        return source_coords

    monkeypatch.setattr(mol_build, "_ff_relax_once", always_wrong)
    monkeypatch.setattr(mol_build, "_embedded_heavy_coords", correct_embedding)
    with pytest.raises(StereoGenerationError, match="after 4 ETKDG"):
        ff_relax(working, cis_coords, "uff", stereo_mol=source)
    assert relax_calls == 5  # one local relax plus four regenerated candidates
    assert embed_calls == 4


def test_saturated_chiral_reference_is_repaired_without_normal_relax():
    source, source_coords = _embed("F[C@](Cl)(Br)I")
    _opposite, opposite_coords = _embed("F[C@@](Cl)(Br)I")
    working = _coordinate_built_mol(source, opposite_coords)
    assert not any(
        bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.DOUBLE
        for bond in working.GetBonds()
    )
    expected = _expected_stereo(source, source_coords)
    assert _stereo_mismatch_counts(source, opposite_coords, expected) == (1, 0)

    repaired = repair_stereo(working, opposite_coords, source, force_field="none")
    assert _stereo_mismatch_counts(source, repaired, expected) == (0, 0)


def test_align_stereo_mol_preserves_labels_in_target_order():
    source = Chem.MolFromSmiles("F[C@](Cl)(Br)I")
    target = Chem.RenumberAtoms(source, [1, 4, 0, 3, 2])
    aligned = align_stereo_mol(source, target)
    assert aligned is not None
    assert [a.GetAtomicNum() for a in aligned.GetAtoms()] == [
        a.GetAtomicNum() for a in target.GetAtoms()
    ]
    source_labels, _ = _expected_stereo(source, np.zeros((source.GetNumAtoms(), 3)))
    aligned_labels, _ = _expected_stereo(aligned, np.zeros((aligned.GetNumAtoms(), 3)))
    assert len(source_labels) == len(aligned_labels) == 1
    assert next(iter(source_labels.values())) == next(iter(aligned_labels.values()))


# --------------------------------------------------------------- the two RDKit traps


def test_mmff_does_not_mutate_the_caller_mol():
    """MMFF typing KEKULIZES its argument (aromatic flags cleared, bonds -> SINGLE/DOUBLE).

    ``_extract_conformer`` perceives ``cistrans`` from ``BondType.DOUBLE`` and ``plane``
    from ring info AFTER the relax, so a leaked mutation would silently change which
    restraints get built. ``ff_relax`` works on a ``Chem.Mol`` copy to prevent it.
    """
    mol, crds = _embed("Cn1cnc2n(C)c(=O)n(C)c(=O)c12")  # caffeine
    before = _bond_snapshot(mol)
    assert any(arom for arom, _ in before), "fixture must have aromatic bonds"

    for ff in ("mmff94", "mmff94s"):
        assert ff_relax(mol, crds, ff) is not None
        assert _bond_snapshot(mol) == before, f"{ff} mutated the caller's mol"


def test_mmff_raises_on_metal_where_uff_succeeds():
    """MMFF has no metal parameters; UFF does. An explicit MMFF must not degrade quietly."""
    mol, crds = _embed("[Fe]")
    assert ff_relax(mol, crds, "uff") is not None
    with pytest.raises(RelaxError, match="no MMFF94 parameters"):
        ff_relax(mol, crds, "mmff94")


def test_mmff_raises_on_unsanitized_mol():
    """A mol whose SanitizeMol failed makes RDKit raise; UFF soft-fails, MMFF must not."""
    rw = Chem.RWMol()
    for sym in ("C", "O", "O"):
        rw.AddAtom(Chem.Atom(sym))
    rw.AddBond(0, 1, Chem.BondType.DOUBLE)
    rw.AddBond(0, 2, Chem.BondType.DOUBLE)
    mol = rw.GetMol()
    conf = Chem.Conformer(3)
    for i, p in enumerate([(0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (-1.2, 0.0, 0.0)]):
        conf.SetAtomPosition(i, p)
    mol.AddConformer(conf, assignId=True)
    crds = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)

    assert ff_relax(mol, crds, "uff") is None  # unchanged soft-fail
    with pytest.raises(RelaxError):
        ff_relax(mol, crds, "mmff94")


# ------------------------------------------------------- the relax-skipped guard


def _all_single_ligand():
    """A mol with no aromatic/double bond -- what the has_orders guard skips."""
    mol, crds = _embed("CCO")
    return LigandConf(
        mol=mol,
        conf_coords=crds,
        global_indices=np.arange(mol.GetNumAtoms()),
        conformer_restraints=True,
    )


def test_explicit_mmff_raises_when_the_relax_is_skipped():
    """An MMFF that would never run at all is exactly the invisible outcome to prevent."""
    lc = _all_single_ligand()
    with pytest.raises(ValueError, match="no aromatic or double bond"):
        _extract_conformer([lc], force_field="mmff94")


def test_uff_and_none_skip_silently():
    """uff keeps its long-standing silent skip; none never relaxes anything anyway."""
    lc = _all_single_ligand()
    for ff in ("uff", "none"):
        bonds, *_ = _extract_conformer([lc], force_field=ff)
        assert len(bonds) == lc.mol.GetNumBonds()


def test_build_spec_routes_the_config_value():
    lc = _all_single_ligand()
    with pytest.raises(ValueError, match="relax_force_field"):
        build_spec([lc], [], {"bond": {}, "relax_force_field": {"ligand": "mmff94"}})
    # the same ligand builds fine under the default
    spec = build_spec([lc], [], {"bond": {}})
    assert spec.bond is not None


class _MockAdapter:
    """Minimal FrameworkAdapter + ConformerAdapter (mirrors test_combined_restraints)."""

    def __init__(self, atoms, ligand_confs):
        self._atoms = atoms
        self._ligand_confs = ligand_confs

    def iter_atoms(self):
        yield from self._atoms

    def iter_ligand_confs(self):
        yield from self._ligand_confs

    def num_atoms(self):
        return max((a.index for a in self._atoms), default=-1) + 1


def test_relax_error_escapes_combined_setup():
    """The raise must survive the WHOLE path, not just ff_relax in isolation.

    config parse -> build_spec -> _extract_conformer -> ff_relax -> out of setup(). A
    broad `except Exception` anywhere along it would turn the deliberate raise back into
    the silent skip the explicit setting exists to rule out, so exercise it end to end
    rather than trusting that no such handler exists.
    """
    # A metal ligand WITH a double bond: UFF has parameters, MMFF does not. The double
    # bond matters -- it clears the has_orders guard, so this reaches ff_relax and tests
    # the RelaxError path rather than the earlier "relax would be skipped" raise.
    mol, crds = _embed("O=[Fe]")
    n = mol.GetNumAtoms()
    lc = LigandConf(
        mol=mol,
        conf_coords=crds,
        global_indices=np.arange(n),
        conformer_restraints=True,
    )
    adapter = _MockAdapter(
        [AtomRecord(chain="B", resid=1, index=i) for i in range(n)], [lc]
    )
    conf = {"bond": {}, "angle": {}}

    cr = CombinedRestraints()
    with pytest.raises(RelaxError, match="no MMFF94 parameters"):
        cr.setup(
            adapter,
            1,
            config={
                "conformer_restraints_config": {
                    **conf,
                    "relax_force_field": {"ligand": "mmff94"},
                }
            },
        )

    # ...and the same structure sets up fine under the default, so the raise is about the
    # force field rather than anything else in the fixture.
    CombinedRestraints().setup(adapter, 1, config={"conformer_restraints_config": conf})


def test_polymer_path_is_never_relaxed():
    """relax=False is structural: the monomer-library path must not raise or relax.

    Polymer residue conformers are all-single fragments, so if relax_force_field leaked
    into that call site an mmff config would raise on every protein.
    """
    lc = _all_single_ligand()
    bonds_a, *_ = _extract_conformer([lc], relax=False, force_field="mmff94")
    bonds_b, *_ = _extract_conformer([lc], relax=False, force_field="uff")
    assert bonds_a == bonds_b
