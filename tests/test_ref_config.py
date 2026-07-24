import pytest

from rgi_utils.config import RestraintsConfig
from rgi_utils.custom.dsl import parse_formula
from rgi_utils.ref_config import split_ref_selection


def _ref(path: str) -> dict:
    return {"ref_pdb": path, "pairing": "identity"}


def test_split_ref_selection_uses_reserved_prefix():
    assert split_ref_selection("ref12 and chain A and resid 3", "selection") == (
        "ref12",
        "chain A and resid 3",
    )
    assert split_ref_selection("chain A and resid 3", "selection") is None

    for malformed in ("ref1", "ref0 and chain A", "ref01 and chain A"):
        with pytest.raises(ValueError, match="malformed reference selection"):
            split_ref_selection(malformed, "selection")


def test_reference_names_are_reserved_ref_numbers():
    with pytest.raises(ValueError, match="invalid reference name"):
        RestraintsConfig.from_dict(
            {
                "distance_restraints_config": [
                    {
                        "atom_selection1": "chain A",
                        "atom_selection2": "r1 and chain B",
                        "refs": {"r1": _ref("ref.pdb")},
                        "harmonic": {"target_distance": 5.0},
                    }
                ]
            }
        )


def test_geometry_reference_count_limits():
    with pytest.raises(ValueError, match="at most 1 distinct reference"):
        RestraintsConfig.from_dict(
            {
                "distance_restraints_config": [
                    {
                        "atom_selection1": "ref1 and chain A",
                        "atom_selection2": "ref2 and chain B",
                        "refs": {
                            "ref1": _ref("ref1.pdb"),
                            "ref2": _ref("ref2.pdb"),
                        },
                        "harmonic": {"target_distance": 5.0},
                    }
                ]
            }
        )

    angle = RestraintsConfig.from_dict(
        {
            "angle_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "ref1 and chain B",
                    "atom_selection3": "ref2 and chain C",
                    "refs": {
                        "ref1": _ref("ref1.pdb"),
                        "ref2": _ref("ref2.pdb"),
                    },
                    "harmonic": {"target_angle": 90.0},
                }
            ]
        }
    )
    assert len(angle.custom_data) == 1

    dihedral = RestraintsConfig.from_dict(
        {
            "dihedral_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "ref1 and chain B",
                    "atom_selection3": "ref2 and chain C",
                    "atom_selection4": "ref3 and chain D",
                    "refs": {
                        "ref1": _ref("ref1.pdb"),
                        "ref2": _ref("ref2.pdb"),
                        "ref3": _ref("ref3.pdb"),
                    },
                    "harmonic": {"target_dihedral": 180.0},
                }
            ]
        }
    )
    assert len(dihedral.custom_data) == 1


def test_legacy_atom_selection_ref_suffix_is_removed():
    with pytest.raises(ValueError, match="distance restraints not run"):
        RestraintsConfig.from_dict(
            {
                "distance_restraints_config": [
                    {
                        "ref_pdb": "ref.pdb",
                        "atom_selection1": "chain A",
                        "atom_selection2_ref": "chain B",
                        "harmonic": {"target_distance": 5.0},
                    }
                ]
            }
        )


def test_legacy_custom_ref_call_is_not_in_the_dsl():
    with pytest.raises(ValueError, match="only calls"):
        parse_formula('distance(A, ref("chain B", ref1))')
