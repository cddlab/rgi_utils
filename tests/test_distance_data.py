import pytest

from rgi_utils.atom_context import AtomRecord
from rgi_utils.distance_restr_data import DistanceData


class MockAdapter:
    def __init__(self, atoms: list[AtomRecord]):
        self._atoms = atoms

    def iter_atoms(self):
        yield from self._atoms


def _make_dd(config: dict) -> DistanceData:
    dd = DistanceData()
    dd.set_config(config)
    return dd


def _harmonic(target: float) -> DistanceData:
    return _make_dd(
        {
            "atom_selection1": "chain A",
            "atom_selection2": "chain B",
            "harmonic": {"target_distance": target},
        }
    )


def _flat_bottomed(d1: float, d2: float) -> DistanceData:
    return _make_dd(
        {
            "atom_selection1": "chain A",
            "atom_selection2": "chain B",
            "flat-bottomed": {"target_distance1": d1, "target_distance2": d2},
        }
    )


class TestDistanceDataSetConfig:
    def test_harmonic_config(self):
        dd = _harmonic(10.0)
        assert dd.distance_restraint_type == "harmonic"
        assert dd.target_distance == pytest.approx(10.0)
        assert dd.run_restr is True

    def test_flat_bottomed_config(self):
        dd = _flat_bottomed(3.0, 10.0)
        assert dd.distance_restraint_type == "flat-bottomed"
        assert dd.target_distance1 == pytest.approx(3.0)
        assert dd.target_distance2 == pytest.approx(10.0)

    def test_flat_bottomed1_config(self):
        dd = _make_dd(
            {
                "atom_selection1": "chain A",
                "atom_selection2": "chain B",
                "flat-bottomed1": {"target_distance1": 5.0},
            }
        )
        assert dd.distance_restraint_type == "flat-bottomed1"
        assert dd.target_distance1 == pytest.approx(5.0)

    def test_flat_bottomed2_config(self):
        dd = _make_dd(
            {
                "atom_selection1": "chain A",
                "atom_selection2": "chain B",
                "flat-bottomed2": {"target_distance2": 8.0},
            }
        )
        assert dd.distance_restraint_type == "flat-bottomed2"
        assert dd.target_distance2 == pytest.approx(8.0)

    def test_flat_bottomed_raises_when_d1_gt_d2(self):
        with pytest.raises(ValueError, match="target_distance1 must be smaller"):
            _flat_bottomed(10.0, 3.0)

    def test_harmonic_missing_target_raises_with_clear_message(self):
        # shared parse_geom_type message (matches rmsd/angle/dihedral wording)
        with pytest.raises(ValueError, match="harmonic needs target_distance"):
            _make_dd(
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "harmonic": {},
                }
            )

    def test_no_type_key_raises_not_run(self):
        # no harmonic/flat-bottomed* block at all -> falls through to run_restr=False
        with pytest.raises(ValueError, match="distance restraints not run"):
            _make_dd({"atom_selection1": "chain A", "atom_selection2": "chain B"})

    def test_invalid_calc_method_raises(self):
        with pytest.raises(ValueError, match="calc_method"):
            _make_dd(
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "calc_method": "invalid",
                    "harmonic": {"target_distance": 5.0},
                }
            )

    def test_is_valid_after_set_config(self):
        dd = _harmonic(5.0)
        assert dd.is_valid() is True


_OMITTED = object()


def _move(mv) -> DistanceData:
    cfg = {
        "atom_selection1": "chain A",
        "atom_selection2": "chain B",
        "harmonic": {"target_distance": 5.0},
    }
    if mv is not _OMITTED:
        cfg["move"] = mv
    return _make_dd(cfg)


class TestDistanceDataMove:
    """The `move` key shares its vocabulary (int / list / comma-string / all/both) with
    the angle/dihedral `move` via parse_move_indices; a 2-group distance maps the index
    set onto the 0/1/2 move_mode enum. move_mode: 0=both / 1=grp1 / 2=grp2."""

    def test_default_is_both(self):
        assert _move(_OMITTED).move_mode == 0

    @pytest.mark.parametrize(
        "mv,expected",
        [
            ("both", 0),
            ("all", 0),
            (1, 1),
            (2, 2),
            ("1", 1),
            ("2", 2),
            ([1], 1),
            ([2], 2),
            ([1, 2], 0),
            ([2, 1], 0),  # order-independent (set)
            ("1,2", 0),
            ("1 2", 0),
            ([1, 1], 1),  # duplicates collapse (set)
        ],
    )
    def test_accepted_values(self, mv, expected):
        assert _move(mv).move_mode == expected

    @pytest.mark.parametrize("mv", [3, 0, [1, 3], [3], "1,3", "3", []])
    def test_out_of_range_or_empty_raises(self, mv):
        with pytest.raises(ValueError, match="move"):
            _move(mv)

    def test_non_integer_raises(self):
        with pytest.raises(ValueError, match="move"):
            _move("x")


class TestDistanceDataResolveSites:
    def test_resolve_sites_maps_correct_atoms(self):
        dd = _make_dd(
            {
                "atom_selection1": "chain A and resid 1",
                "atom_selection2": "chain B",
                "harmonic": {"target_distance": 5.0},
            }
        )
        atoms = [
            AtomRecord(chain="A", resid=1, index=0),
            AtomRecord(chain="A", resid=2, index=1),
            AtomRecord(chain="B", resid=1, index=2),
        ]
        dd.resolve_sites(MockAdapter(atoms))
        assert dd.target_sites1 == [0]
        assert dd.target_sites2 == [2]

    def test_resolve_sites_empty_selection_raises(self):
        dd = _make_dd(
            {
                "atom_selection1": "chain Z",
                "atom_selection2": "chain B",
                "harmonic": {"target_distance": 5.0},
            }
        )
        atoms = [AtomRecord(chain="A", resid=1, index=0)]
        with pytest.raises(ValueError, match="matched no atoms"):
            dd.resolve_sites(MockAdapter(atoms))
