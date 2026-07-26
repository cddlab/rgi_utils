"""Group-centroid angle / dihedral restraint data parsing + site resolution.

Mirrors ``test_distance_data.py``: the four restraint types (harmonic / flat-bottomed /
flat-bottomed1 / flat-bottomed2) and the ``move`` key. Targets are DEGREES in the config
and stored in RADIANS on the dataclass.
"""

import math

import pytest

from rgi_utils.atom_context import AtomRecord
from rgi_utils.config import RestraintsConfig
from rgi_utils.group_geom_restr_data import AngleRestraintData, DihedralRestraintData


class MockAdapter:
    def __init__(self, atoms: list[AtomRecord]):
        self._atoms = atoms

    def iter_atoms(self):
        yield from self._atoms


_SEL3 = {
    "atom_selection1": "chain A",
    "atom_selection2": "chain B",
    "atom_selection3": "chain C",
}
_SEL4 = {**_SEL3, "atom_selection4": "chain D"}


def _angle(extra: dict) -> AngleRestraintData:
    ad = AngleRestraintData()
    ad.set_config({**_SEL3, **extra})
    return ad


def _dihedral(extra: dict) -> DihedralRestraintData:
    dd = DihedralRestraintData()
    dd.set_config({**_SEL4, **extra})
    return dd


class TestAngleTypes:
    def test_harmonic_degrees_to_radians(self):
        ad = _angle({"harmonic": {"target_angle": 90.0}})
        assert ad.geom_type == "harmonic"
        assert ad.target1 == pytest.approx(math.pi / 2)
        assert ad.run_restr is True
        assert ad.weight == pytest.approx(1.0)  # default
        assert ad.move_free == (True, False, True)  # default: arms free, vertex pinned

    def test_flat_bottomed(self):
        ad = _angle({"flat-bottomed": {"target_angle1": 80.0, "target_angle2": 100.0}})
        assert ad.geom_type == "flat-bottomed"
        assert ad.target1 == pytest.approx(math.radians(80.0))
        assert ad.target2 == pytest.approx(math.radians(100.0))

    def test_flat_bottomed1_and_2(self):
        lo = _angle({"flat-bottomed1": {"target_angle1": 70.0}})
        assert lo.geom_type == "flat-bottomed1"
        assert lo.target1 == pytest.approx(math.radians(70.0))
        hi = _angle({"flat-bottomed2": {"target_angle2": 110.0}})
        assert hi.geom_type == "flat-bottomed2"
        assert hi.target2 == pytest.approx(math.radians(110.0))

    def test_flat_bottomed_requires_t1_lt_t2(self):
        with pytest.raises(ValueError, match="must be smaller"):
            _angle({"flat-bottomed": {"target_angle1": 100.0, "target_angle2": 80.0}})

    def test_weight_override(self):
        ad = _angle({"harmonic": {"target_angle": 90.0}, "weight": 2.5})
        assert ad.weight == pytest.approx(2.5)

    def test_missing_type_block_raises(self):
        with pytest.raises(ValueError, match="type block"):
            _angle({})

    def test_missing_selection_raises(self):
        with pytest.raises(ValueError, match="atom_selection"):
            ad = AngleRestraintData()
            ad.set_config(
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "harmonic": {"target_angle": 90.0},
                }
            )


class TestAngleMove:
    def test_default_arms_free_vertex_pinned(self):
        # default move: groups 1 + 3 (arms) free, group 2 (vertex) pinned
        assert _angle({"harmonic": {"target_angle": 90.0}}).move_free == (
            True,
            False,
            True,
        )

    def test_move_single_group(self):
        ad = _angle({"harmonic": {"target_angle": 90.0}, "move": 2})
        assert ad.move_free == (False, True, False)  # only group 2 free

    def test_move_multi_group(self):
        # list and comma/space string forms both free groups 1 and 3
        for mv in ([1, 3], "1,3", "1 3"):
            ad = _angle({"harmonic": {"target_angle": 90.0}, "move": mv})
            assert ad.move_free == (True, False, True), mv

    def test_move_all_and_both_keywords(self):
        for kw in ("all", "both"):  # 'both' tolerated as a synonym for 'all'
            ad = _angle({"harmonic": {"target_angle": 90.0}, "move": kw})
            assert ad.move_free == (True, True, True)

    def test_move_out_of_range_raises(self):
        with pytest.raises(ValueError, match="move"):
            _angle(
                {"harmonic": {"target_angle": 90.0}, "move": 4}
            )  # angle has 3 groups


class TestAngleResolve:
    def test_resolve_three_groups(self):
        ad = AngleRestraintData()
        ad.set_config(
            {
                "atom_selection1": "chain A",
                "atom_selection2": "chain B",
                "atom_selection3": "chain C and resid 1",
                "harmonic": {"target_angle": 90.0},
            }
        )
        atoms = [
            AtomRecord(chain="A", resid=1, index=0),
            AtomRecord(chain="A", resid=2, index=1),
            AtomRecord(chain="B", resid=1, index=2),
            AtomRecord(chain="C", resid=1, index=3),
            AtomRecord(chain="C", resid=2, index=4),
        ]
        ad.resolve_sites(MockAdapter(atoms))
        assert ad.target_sites1 == [0, 1]
        assert ad.target_sites2 == [2]
        assert ad.target_sites3 == [3]  # resid 2 (index 4) excluded

    def test_resolve_empty_group_raises(self):
        ad = _angle({"harmonic": {"target_angle": 90.0}})
        ad.atom_selection1 = "chain Z"  # matches nothing
        atoms = [
            AtomRecord(chain="B", resid=1, index=0),
            AtomRecord(chain="C", resid=1, index=1),
        ]
        with pytest.raises(ValueError, match="matched no atoms"):
            ad.resolve_sites(MockAdapter(atoms))


class TestDihedral:
    def test_harmonic_degrees_to_radians(self):
        dd = _dihedral({"harmonic": {"target_dihedral": 180.0}})
        assert dd.geom_type == "harmonic"
        assert dd.target1 == pytest.approx(math.pi)
        assert dd.run_restr is True

    def test_flat_bottomed(self):
        dd = _dihedral(
            {"flat-bottomed": {"target_dihedral1": -30.0, "target_dihedral2": 30.0}}
        )
        assert dd.geom_type == "flat-bottomed"
        assert dd.target1 == pytest.approx(math.radians(-30.0))
        assert dd.target2 == pytest.approx(math.radians(30.0))

    def test_default_ends_free_axis_pinned(self):
        # default move: groups 1 + 4 (ends) free, axis (groups 2 + 3) pinned
        dd = _dihedral({"harmonic": {"target_dihedral": 180.0}})
        assert dd.move_free == (True, False, False, True)

    def test_move_multi_group(self):
        # the user's "move: 1,4" -> groups 1 and 4 free, 2 and 3 pinned
        dd = _dihedral({"harmonic": {"target_dihedral": 180.0}, "move": "1,4"})
        assert dd.move_free == (True, False, False, True)
        dd2 = _dihedral({"harmonic": {"target_dihedral": 180.0}, "move": [1, 4]})
        assert dd2.move_free == (True, False, False, True)

    def test_move_out_of_range_raises(self):
        with pytest.raises(ValueError, match="move"):
            _dihedral({"harmonic": {"target_dihedral": 180.0}, "move": 5})

    def test_missing_fourth_selection_raises(self):
        with pytest.raises(ValueError, match="atom_selection"):
            dd = DihedralRestraintData()
            dd.set_config({**_SEL3, "harmonic": {"target_dihedral": 180.0}})

    def test_resolve_four_groups(self):
        dd = _dihedral({"harmonic": {"target_dihedral": 180.0}})
        atoms = [
            AtomRecord(chain="A", resid=1, index=0),
            AtomRecord(chain="B", resid=1, index=1),
            AtomRecord(chain="C", resid=1, index=2),
            AtomRecord(chain="D", resid=1, index=3),
        ]
        dd.resolve_sites(MockAdapter(atoms))
        assert dd.target_sites1 == [0]
        assert dd.target_sites4 == [3]


class TestConfigParsing:
    def test_from_dict_populates_lists_and_default_start_sigma(self):
        cfg = RestraintsConfig.from_dict(
            {
                "angle_restraints_config": [
                    {**_SEL3, "harmonic": {"target_angle": 90.0}}
                ],
                "dihedral_restraints_config": [
                    {
                        **_SEL4,
                        "flat-bottomed": {
                            "target_dihedral1": -30.0,
                            "target_dihedral2": 30.0,
                        },
                        "start_sigma": 1.0,
                    }
                ],
            }
        )
        assert len(cfg.angle_data) == 1
        assert len(cfg.dihedral_data) == 1
        # omitted start_sigma -> +inf (active at every step)
        assert cfg.angle_data[0].start_sigma == float("inf")
        # explicit start_sigma is preserved
        assert cfg.dihedral_data[0].start_sigma == pytest.approx(1.0)
        assert cfg.dihedral_data[0].geom_type == "flat-bottomed"


class TestAngleUnit:
    def test_radians_unit_matches_degree_conversion(self):
        """unit: radians takes the target verbatim; the default (degrees) converts. Both
        must land on the same internal radian value for the equivalent input."""
        deg = _angle({"harmonic": {"target_angle": 90.0}})
        rad = _angle({"unit": "radians", "harmonic": {"target_angle": math.pi / 2}})
        assert deg.target1 == pytest.approx(math.pi / 2)
        assert rad.target1 == pytest.approx(math.pi / 2)
        assert rad.target1 == pytest.approx(deg.target1)

    def test_radians_unit_flat_bottomed_both_targets(self):
        """unit applies to BOTH flat-bottomed targets of the entry."""
        rad = _angle(
            {
                "unit": "radians",
                "flat-bottomed": {"target_angle1": -0.5, "target_angle2": 0.5},
            }
        )
        assert rad.target1 == pytest.approx(-0.5)
        assert rad.target2 == pytest.approx(0.5)

    def test_dihedral_radians_unit(self):
        rad = _dihedral({"unit": "radians", "harmonic": {"target_dihedral": math.pi}})
        assert rad.target1 == pytest.approx(math.pi)

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="unit"):
            _angle({"unit": "grad", "harmonic": {"target_angle": 90.0}})


class TestStepWindow:
    def test_step_window_parsed(self):
        ad = _angle(
            {"start_step": 5, "stop_step": 10, "harmonic": {"target_angle": 90.0}}
        )
        assert ad.start_step == pytest.approx(5.0)
        assert ad.stop_step == pytest.approx(10.0)

    def test_step_window_default_always_on(self):
        ad = _angle({"harmonic": {"target_angle": 90.0}})
        assert ad.start_step == float("-inf")
        assert ad.stop_step == float("inf")

    def test_sigma_and_step_windows_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive|not both"):
            _angle(
                {
                    "start_sigma": 1.0,
                    "start_step": 5,
                    "harmonic": {"target_angle": 90.0},
                }
            )

    def test_stop_sigma_with_stop_step_also_rejected(self):
        with pytest.raises(ValueError, match="mutually exclusive|not both"):
            _dihedral(
                {
                    "stop_sigma": 1.0,
                    "stop_step": 50,
                    "harmonic": {"target_dihedral": 180.0},
                }
            )
