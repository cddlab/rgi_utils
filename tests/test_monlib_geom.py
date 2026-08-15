"""Polymer conformer targets taken from a CCP4 monomer library.

The library path exists because the DEFAULT targets are measured from the predictor's
per-residue reference conformer, which is approximate chemistry (AF3 ETKDG-embeds the
free CCD component). Every test below therefore gives the fixture library values that
are DELIBERATELY far from the fixture conformer's own geometry — an assertion can only
pass if the target really came from the library, never by coincidence.

The fixture is a minimal self-contained library (one component + one link + a stub
energy library) written to tmp_path, so the suite needs no CCP4 installation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rgi_utils.atom_context import AtomRecord
from rgi_utils.combined import CombinedRestraints
from rgi_utils.config import RestraintsConfig

# Library values, all far from the conformer distances/angles below.
_LIB_BOND_P_O5 = 1.777
_LIB_BOND_O5_C5 = 1.888
_LIB_ANGLE_P_O5_C5 = 111.0
# Link modifications rewrite the FREE residue's own targets once it is polymer-bonded --
# CCP4's TRANS does exactly this (DEL-OXT: C-O 1.251 -> 1.229, CA-C-O 117.191 -> 120.614;
# DEL-HN1: CA-N 1.483 -> 1.453), because a monomer entry describes the free zwitterion.
# Distinct values per side so a test can tell which mod reached which residue.
_MOD_S1_BOND_O5_C5 = 1.808
_MOD_S2_BOND_P_O5 = 1.707
_MOD_S2_ANGLE_P_O5_C5 = 104.0
_LIB_LINK_BOND = 1.666  # the built-in phosphodiester target is 1.607
_LIB_LINK_ANGLE_C3_O3_P = 122.5  # built-in 119.7
_LIB_LINK_ANGLE_O3_P_O5 = 101.5  # built-in 104.0

_COMPONENT_CIF = f"""data_comp_list
loop_
_chem_comp.id
_chem_comp.three_letter_code
_chem_comp.name
_chem_comp.group
_chem_comp.number_atoms_all
_chem_comp.number_atoms_nh
_chem_comp.desc_level
 XYZ XYZ 'fixture nucleotide' DNA/RNA 5 5 .

data_comp_XYZ
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.type_energy
_chem_comp_atom.charge
 XYZ P   P P    0
 XYZ O5' O O2   0
 XYZ C5' C CH2  0
 XYZ C3' C CH1  0
 XYZ O3' O OH1  0
 XYZ H5' H HCH2 0
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.type
_chem_comp_bond.value_dist
_chem_comp_bond.value_dist_esd
 XYZ P   O5' single {_LIB_BOND_P_O5:.3f} 0.010
 XYZ O5' C5' single {_LIB_BOND_O5_C5:.3f} 0.011
 XYZ C5' H5' single 1.089 0.020
loop_
_chem_comp_angle.comp_id
_chem_comp_angle.atom_id_1
_chem_comp_angle.atom_id_2
_chem_comp_angle.atom_id_3
_chem_comp_angle.value_angle
_chem_comp_angle.value_angle_esd
 XYZ P O5' C5' {_LIB_ANGLE_P_O5_C5:.1f} 1.5
 XYZ O5' C5' H5' 109.5 1.5
loop_
_chem_comp_plane_atom.comp_id
_chem_comp_plane_atom.plane_id
_chem_comp_plane_atom.atom_id
_chem_comp_plane_atom.dist_esd
 XYZ plan-1 P   0.020
 XYZ plan-1 O5' 0.020
 XYZ plan-1 C5' 0.020
 XYZ plan-1 C3' 0.020
 XYZ plan-1 H5' 0.020
"""

_LINK_CIF = f"""data_link_list
loop_
_chem_link.id
_chem_link.comp_id_1
_chem_link.mod_id_1
_chem_link.group_comp_1
_chem_link.comp_id_2
_chem_link.mod_id_2
_chem_link.group_comp_2
_chem_link.name
 p  .  MOD-S1  DNA/RNA  .  MOD-S2  DNA/RNA  'phosphodiester link'

data_link_p
loop_
_chem_link_bond.link_id
_chem_link_bond.atom_1_comp_id
_chem_link_bond.atom_id_1
_chem_link_bond.atom_2_comp_id
_chem_link_bond.atom_id_2
_chem_link_bond.type
_chem_link_bond.value_dist
_chem_link_bond.value_dist_esd
 p 1 O3' 2 P single {_LIB_LINK_BOND:.3f} 0.010
loop_
_chem_link_angle.link_id
_chem_link_angle.atom_1_comp_id
_chem_link_angle.atom_id_1
_chem_link_angle.atom_2_comp_id
_chem_link_angle.atom_id_2
_chem_link_angle.atom_3_comp_id
_chem_link_angle.atom_id_3
_chem_link_angle.value_angle
_chem_link_angle.value_angle_esd
 p 1 C3' 1 O3' 2 P {_LIB_LINK_ANGLE_C3_O3_P:.1f} 1.5
 p 1 O3' 2 P 2 O5' {_LIB_LINK_ANGLE_O3_P_O5:.1f} 1.5

data_mod_MOD-S1
loop_
_chem_mod_bond.mod_id
_chem_mod_bond.function
_chem_mod_bond.atom_id_1
_chem_mod_bond.atom_id_2
_chem_mod_bond.new_type
_chem_mod_bond.new_value_dist
_chem_mod_bond.new_value_dist_esd
 MOD-S1 change O5' C5' single {_MOD_S1_BOND_O5_C5:.3f} 0.011

data_mod_MOD-S2
loop_
_chem_mod_bond.mod_id
_chem_mod_bond.function
_chem_mod_bond.atom_id_1
_chem_mod_bond.atom_id_2
_chem_mod_bond.new_type
_chem_mod_bond.new_value_dist
_chem_mod_bond.new_value_dist_esd
 MOD-S2 change P O5' single {_MOD_S2_BOND_P_O5:.3f} 0.010
loop_
_chem_mod_angle.mod_id
_chem_mod_angle.function
_chem_mod_angle.atom_id_1
_chem_mod_angle.atom_id_2
_chem_mod_angle.atom_id_3
_chem_mod_angle.new_value_angle
_chem_mod_angle.new_value_angle_esd
 MOD-S2 change P O5' C5' {_MOD_S2_ANGLE_P_O5_C5:.1f} 1.5
"""

_ENER_LIB = """data_energy
loop_
_lib_atom.type
_lib_atom.weight
_lib_atom.hb_type
_lib_atom.vdw_radius
_lib_atom.vdwh_radius
_lib_atom.ion_radius
_lib_atom.element
_lib_atom.valency
_lib_atom.sp
 P    30.974 N 1.90 1.90 . P 4 0
 O    15.999 A 1.52 1.52 . O 2 0
 O2   15.999 A 1.52 1.52 . O 2 0
 OH1  15.999 B 1.52 1.52 . O 2 0
 C    12.011 N 1.70 1.70 . C 4 0
 CH1  12.011 N 1.70 1.70 . C 4 0
 CH2  12.011 N 1.70 1.70 . C 4 0
 HCH2  1.008 H 1.20 1.20 . H 1 0
"""

_NAMES = ["P", "O5'", "C5'", "C3'", "O3'"]
# Conformer geometry: P-O5' 1.60, O5'-C5' 1.30, C3'-O3' 1.26 -- every one of them well
# clear of the library values, so a passing assertion pins the source unambiguously.
_COORDS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.6, 0.0, 0.0],
        [2.8, 0.5, 0.0],
        [5.0, 1.0, 0.0],
        [6.2, 1.4, 0.0],
    ]
)


@pytest.fixture
def library_dir(tmp_path):
    """A minimal CCP4-layout monomer library: <dir>/x/XYZ.cif + list/ + ener_lib.cif."""
    (tmp_path / "x").mkdir()
    (tmp_path / "list").mkdir()
    (tmp_path / "x" / "XYZ.cif").write_text(_COMPONENT_CIF)
    (tmp_path / "list" / "mon_lib_list.cif").write_text(_LINK_CIF)
    (tmp_path / "ener_lib.cif").write_text(_ENER_LIB)
    return str(tmp_path)


class _NucleotideAdapter:
    """Two identical nucleotide residues on chain A, atoms 0-4 and 5-9."""

    def __init__(self, resname: str = "XYZ"):
        self._resname = resname
        self._positions = np.concatenate([_COORDS, _COORDS + np.array([8.0, 0.0, 0.0])])
        self._elements = np.array([15, 8, 6, 6, 8] * 2, dtype=np.int64)
        self._uid = np.repeat(np.arange(2), len(_NAMES))

    def iter_atoms(self):
        for resid in (1, 2):
            offset = (resid - 1) * len(_NAMES)
            for local, name in enumerate(_NAMES):
                yield AtomRecord(
                    chain="A",
                    resid=resid,
                    index=offset + local,
                    name=name,
                    mol_type="rna",
                    resname=self._resname,
                    conformer_restraints=True,
                )

    def get_elements(self):
        return self._elements

    def get_reference_positions(self):
        return self._positions

    def get_reference_space_uid(self):
        return self._uid


def _config(library=None, on_missing=None):
    conformer = {"bond": {}, "angle": {}, "plane": {}}
    if library is not None:
        conformer["monomer_library"] = (
            library
            if on_missing is None
            else {"path": library, "on_missing": on_missing}
        )
    return {"gpu": False, "max_iter": 100, "conformer_restraints_config": conformer}


def _setup(adapter, config):
    restr = CombinedRestraints()
    restr.setup(adapter, config=config)
    return restr.spec


def _bond_targets(spec):
    """{(lo, hi) atom pair: target}. Pairs are sorted: a bond's stored atom order
    follows whichever source built it (RDKit begin/end vs the library's atom_id_1/2)."""
    return {
        tuple(sorted((int(i), int(j)))): float(r)
        for (i, j), r, m in zip(spec.bond.idx, spec.bond.r0, spec.bond.mask)
        if m > 0
    }


def _angle_targets(spec):
    """{(i, vertex, k): radians}, keyed with the two arms sorted (the vertex is fixed
    in the middle, and the arms' order is likewise source-dependent)."""
    out = {}
    for (i, j, k), t, m in zip(spec.angle.idx, spec.angle.th0, spec.angle.mask):
        if m > 0:
            arms = sorted((int(i), int(k)))
            out[(arms[0], int(j), arms[1])] = float(t)
    return out


def test_library_bond_and_angle_targets_replace_the_reference_conformer(library_dir):
    spec = _setup(_NucleotideAdapter(), _config(library_dir))
    bonds, angles = _bond_targets(spec), _angle_targets(spec)

    # Library values, not the 1.60 / 1.30 measured from the reference conformer. Residue 1
    # is only side 1 of the link, so P-O5' keeps the free-residue value while O5'-C5' takes
    # MOD-S1's; residue 2 is only side 2, so the reverse (see the link-mod test below).
    assert bonds[(0, 1)] == pytest.approx(_LIB_BOND_P_O5)
    assert bonds[(1, 2)] == pytest.approx(_MOD_S1_BOND_O5_C5)
    assert angles[(0, 1, 2)] == pytest.approx(math.radians(_LIB_ANGLE_P_O5_C5))
    # The second residue gets the targets ITS side of the link prescribes.
    assert bonds[(5, 6)] == pytest.approx(_MOD_S2_BOND_P_O5)

    # REPLACED, not added: the conformer-derived duplicate for a covered residue is
    # dropped, so each intra-residue pair appears exactly once.
    pairs = [tuple(sorted(k)) for k in bonds]
    assert len(pairs) == len(set(pairs))
    # Hydrogens are in the library but not in the structure -> those restraints vanish
    # rather than referencing a missing atom.
    assert len(bonds) == 2 * 2 + 1  # 2 per residue + the inter-residue link


def test_link_modifications_rewrite_the_bonded_residue_targets(library_dir):
    # A monomer entry describes the FREE residue. For an amino acid that is the zwitterion
    # (-NH3+ / -COO-), and CCP4's TRANS link carries `_chem_mod` records that rewrite those
    # targets once the residue is peptide-bonded: DEL-OXT takes C-O 1.251 -> 1.229 and
    # CA-C-O 117.191 -> 120.614, DEL-HN1 takes CA-N 1.483 -> 1.453. Skipping them restrains
    # a whole chain to free-amino-acid geometry -- measured on QBP, that alone moved the
    # backbone ~0.025 A off Engh-Huber and took MolProbity's rms_bond from 0.005 to 0.014.
    #
    # Which mod applies is POSITIONAL, and the fixture's two residues pin both ends of it:
    # residue 1 is side 1 only (the chain's last residue is nobody's side 1, so a real
    # C-terminus keeps its -COO-), residue 2 is side 2 only (the N-terminus keeps -NH3+).
    spec = _setup(_NucleotideAdapter(), _config(library_dir))
    bonds, angles = _bond_targets(spec), _angle_targets(spec)

    # Residue 1: side 1 -> MOD-S1 only.
    assert bonds[(1, 2)] == pytest.approx(_MOD_S1_BOND_O5_C5)
    assert bonds[(0, 1)] == pytest.approx(_LIB_BOND_P_O5)  # MOD-S2 must NOT reach it
    assert angles[(0, 1, 2)] == pytest.approx(math.radians(_LIB_ANGLE_P_O5_C5))

    # Residue 2: side 2 -> MOD-S2 only, including its angle override.
    assert bonds[(5, 6)] == pytest.approx(_MOD_S2_BOND_P_O5)
    assert bonds[(6, 7)] == pytest.approx(_LIB_BOND_O5_C5)  # MOD-S1 must NOT reach it
    assert angles[(5, 6, 7)] == pytest.approx(math.radians(_MOD_S2_ANGLE_P_O5_C5))


def test_library_planes_replace_conformer_ring_perception(library_dir):
    # The fixture conformer has no ring, so SSSR perception yields no plane at all: any
    # plane group present must be the library's named group (minus its hydrogen).
    spec = _setup(_NucleotideAdapter(), _config(library_dir))
    assert spec.plane is not None
    groups = {
        frozenset(int(i) for i, m in zip(idx, grp) if m > 0)
        for idx, grp, active in zip(
            spec.plane.idx, spec.plane.grp_mask, spec.plane.mask
        )
        if active > 0
    }
    assert groups == {frozenset({0, 1, 2, 3}), frozenset({5, 6, 7, 8})}


def test_link_targets_come_from_the_library(library_dir):
    spec = _setup(_NucleotideAdapter(), _config(library_dir))
    bonds, angles = _bond_targets(spec), _angle_targets(spec)

    assert bonds[(4, 5)] == pytest.approx(_LIB_LINK_BOND)
    assert angles[(3, 4, 5)] == pytest.approx(math.radians(_LIB_LINK_ANGLE_C3_O3_P))
    assert angles[(4, 5, 6)] == pytest.approx(math.radians(_LIB_LINK_ANGLE_O3_P_O5))


def test_without_a_library_the_reference_conformer_still_drives_the_targets(
    library_dir,
):
    spec = _setup(_NucleotideAdapter(), _config())
    bonds = _bond_targets(spec)
    assert bonds[(0, 1)] == pytest.approx(1.6)  # measured, not 1.777
    assert bonds[(4, 5)] == pytest.approx(1.607)  # built-in phosphodiester link


def test_uncovered_residue_keeps_reference_conformer_targets(library_dir):
    # on_missing defaults to fallback: an unknown residue is restrained from its
    # conformer instead of losing its restraints or aborting the run.
    spec = _setup(_NucleotideAdapter(resname="QQQ"), _config(library_dir))
    bonds = _bond_targets(spec)
    assert bonds[(0, 1)] == pytest.approx(1.6)
    # The link is still library-driven: it is keyed on polymer type, not residue name.
    assert bonds[(4, 5)] == pytest.approx(_LIB_LINK_BOND)


def test_on_missing_error_rejects_an_uncovered_residue(library_dir):
    with pytest.raises(ValueError, match="no library entry for residue"):
        _setup(
            _NucleotideAdapter(resname="QQQ"), _config(library_dir, on_missing="error")
        )


def test_missing_library_directory_raises(tmp_path):
    # A silently empty library looks exactly like a working one (every residue quietly
    # falls back), so a bad path must fail loudly.
    with pytest.raises(ValueError, match="is not a directory"):
        _setup(_NucleotideAdapter(), _config(str(tmp_path / "nope")))


@pytest.mark.parametrize(
    "spec,message",
    [
        ({"path": "/tmp", "on_missing": "sometimes"}, "on_missing"),
        ({"path": "/tmp", "typo": 1}, "unknown key"),
        ({"on_missing": "error"}, "'path' is required"),
        (17, "must be a path string"),
    ],
)
def test_config_rejects_a_malformed_monomer_library_spec(spec, message):
    with pytest.raises(ValueError, match=message):
        RestraintsConfig.from_dict(
            {"conformer_restraints_config": {"bond": {}, "monomer_library": spec}}
        )


def test_library_target_pulls_a_distorted_bond_back(library_dir):
    # The restraint must have FORCE, not just a spec entry: when a library target
    # coincides with what the model already produces, "working" and "never built" look
    # identical. Start off-target and check the minimizer lands on the library value.
    torch = pytest.importorskip("torch")
    restr = CombinedRestraints()
    restr.setup(_NucleotideAdapter(), config=_config(library_dir))
    coords = torch.tensor(
        np.concatenate([_COORDS, _COORDS + np.array([8.0, 0.0, 0.0])]),
        dtype=torch.float64,
    )
    before = abs(
        float(torch.linalg.vector_norm(coords[1] - coords[0])) - _LIB_BOND_P_O5
    )
    restr.minimize(coords, istep=0, sigma=100.0)
    after = abs(float(torch.linalg.vector_norm(coords[1] - coords[0])) - _LIB_BOND_P_O5)
    assert before > 0.15  # the fixture really does start off-target
    assert after < before
    assert after < 0.02
