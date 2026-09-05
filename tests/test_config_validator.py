"""Regression coverage for config parsing and the bundled standalone validator."""

import ast
import runpy
from pathlib import Path

import pytest

from rgi_utils.config import RestraintsConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def validator():
    return runpy.run_path(
        str(ROOT / ".claude/skills/generate-rgi-config/scripts/validate_config.py")
    )


def _distance(**overrides):
    return {
        "atom_selection1": "index 0",
        "atom_selection2": "index 1",
        "harmonic": {"target_distance": 2.0},
        **overrides,
    }


@pytest.mark.parametrize(
    "entry",
    [
        _distance(**{"flat-bottomed2": {"target_distance2": 3.0}}),
        _distance(harmonic=None),
        _distance(harmonic=3),
        _distance(harmonic={"target_distance": float("inf")}),
        _distance(harmonic={"target_distance": float("nan")}),
        _distance(weight=float("nan")),
        _distance(weight=float("inf")),
        _distance(move=1.9),
        _distance(move=True),
        _distance(move=[False, 2]),
        _distance(move=float("inf")),
        _distance(start_sigma=float("nan")),
        _distance(stop_step=float("nan")),
        _distance(start_sigma=1, stop_sigma=2),
        _distance(start_step=2, stop_step=1),
    ],
)
def test_invalid_distance_values_fail_at_parse(entry):
    with pytest.raises(ValueError):
        RestraintsConfig.from_dict({"distance_restraints_config": [entry]})


@pytest.mark.parametrize(
    "config",
    [
        [],
        False,
        {"distance_restraints_config": {}},
        {"distance_restraints_config": [None]},
        {"conformer_restraints_config": False},
        {"conformer_restraints_config": {"bond": 2}},
        {"conformer_restraints_config": {"plane": {"slack": float("nan")}}},
        {"conformer_restraints_config": {"angle": {"slack": -1}}},
        {"conformer_restraints_config": {"chiral": {"weight": float("inf")}}},
        {"max_iter": -1},
        {"max_iter": 1.5},
        {"max_iter": True},
        {"max_iter": "10"},
    ],
)
def test_invalid_config_shapes_and_numbers_fail_at_parse(config):
    with pytest.raises(ValueError):
        RestraintsConfig.from_dict(config)


def test_explicit_infinite_windows_and_integral_move_remain_valid():
    cfg = RestraintsConfig.from_dict(
        {
            "max_iter": 0,
            "distance_restraints_config": [
                _distance(move=1.0, start_step=float("-inf"), stop_step=float("inf"))
            ],
        }
    )
    assert cfg.max_iter == 0 and cfg.distance_data[0].move_mode == 1


@pytest.mark.parametrize("flag", ["hbonds", "coplanar"])
def test_base_pair_string_false(flag):
    cfg = RestraintsConfig.from_dict(
        {
            "base_pair_restraints_config": [
                {
                    "residue1": "resid 1",
                    "residue2": "resid 2",
                    flag: "false",
                }
            ]
        }
    )
    assert getattr(cfg.base_pair_data[0], flag) is False


@pytest.mark.parametrize("field", ["target", "coplanar_slack", "weight"])
def test_nonfinite_base_pair_values_fail_at_parse(field):
    with pytest.raises(ValueError, match="finite"):
        RestraintsConfig.from_dict(
            {
                "base_pair_restraints_config": [
                    {
                        "residue1": "resid 1",
                        "residue2": "resid 2",
                        field: float("nan"),
                    }
                ]
            }
        )


@pytest.mark.parametrize("selection,expected", [("index 3", 0), ("index", 1)])
def test_validator_discovers_and_checks_improper(validator, selection, expected):
    cfg = {
        "improper_restraints_config": [
            {
                "atom_selection1": "index 0",
                "atom_selection2": "index 1",
                "atom_selection3": "index 2",
                "atom_selection4": selection,
                "harmonic": {"target_improper": 0},
            }
        ]
    }
    found = list(validator["_find_configs"](cfg))
    assert len(found) == 1
    assert validator["_validate_one"](*found[0]) == expected


def test_validator_checks_base_triple_and_empty_windows(validator):
    for cfg in [
        {
            "base_pair_restraints_config": [
                {
                    "residue1": "resid 1",
                    "residue2": "resid 2",
                    "residue3": "resid",
                }
            ]
        },
        {"distance_restraints_config": [_distance(start_step=2, stop_step=1)]},
    ]:
        assert validator["_validate_one"]("test", cfg, cfg) == 1
    assert not validator["_has_conformer_optin"](
        {}, {"conformer_restraints": {"A": "false"}}
    )


# Bounded fixture discovery excludes generated prediction directories.
EXAMPLES = sorted(
    path
    for pattern in ("*/*/*", "custom/*/*/*")
    for path in (ROOT / "example").glob(pattern)
    if path.suffix in {".json", ".yaml", ".yml", ".py"} and path.is_file()
)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: str(p.relative_to(ROOT)))
def test_example_configs_pass_validator(validator, path):
    if path.suffix == ".py":
        tree = ast.parse(path.read_text())
        config = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "RESTRAINTS_CONFIG"
                for t in node.targets
            )
        )
        data = config
    else:
        data = validator["_load"](path)
    found = list(validator["_find_configs"](data))
    assert found, path
    assert all(validator["_validate_one"](*entry) == 0 for entry in found)
