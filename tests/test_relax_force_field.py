"""``conformer_restraints_config.relax_force_field``: UFF / MMFF reference relaxation.

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

from rgi_utils import CombinedRestraints
from rgi_utils._mol_build import RelaxError, ff_relax, parse_relax_force_field
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
    assert parse_relax_force_field({"relax_force_field": value}) == expected


def test_parse_relax_force_field_defaults_to_uff():
    # omitted and an explicit YAML null both mean "not set" -> the long-standing default.
    assert parse_relax_force_field({}) == "uff"
    assert parse_relax_force_field(None) == "uff"
    assert parse_relax_force_field({"relax_force_field": None}) == "uff"


def test_parse_relax_force_field_rejects_typo():
    with pytest.raises(ValueError, match="unknown value 'mmf94'"):
        parse_relax_force_field({"relax_force_field": "mmf94"})


def test_config_validates_value_not_just_key():
    """A bad VALUE must raise while parsing the config.

    The conformer whitelist only checks the KEY, so without the value check a typo would
    silently run UFF on any structure where no ligand opts in (build_spec never reaches
    the force field at all).
    """
    with pytest.raises(ValueError, match="relax_force_field"):
        RestraintsConfig.from_dict(
            {"conformer_restraints_config": {"bond": {}, "relax_force_field": "mmf94"}}
        )
    # the valid values parse
    cfg = RestraintsConfig.from_dict(
        {"conformer_restraints_config": {"bond": {}, "relax_force_field": "mmff94s"}}
    )
    assert cfg.conformer_config["relax_force_field"] == "mmff94s"


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
    """relax_force_field: none -> targets come straight off the tool's conformer."""
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
    for g0, g1, r0 in raw:
        assert r0 == pytest.approx(expected[(g0, g1)], abs=1e-12)
    # ...and the relax really was doing something, so "none" is not a no-op distinction
    assert any(
        r0 != pytest.approx(expected[(g0, g1)], abs=1e-6) for g0, g1, r0 in relaxed
    )


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
        build_spec([lc], [], {"bond": {}, "relax_force_field": "mmff94"})
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
                "conformer_restraints_config": {**conf, "relax_force_field": "mmff94"}
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
