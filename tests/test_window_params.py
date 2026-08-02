"""Characterization tests for the shared weight + gate-window parse.

``weight`` and the two mutually-exclusive gate windows (``start_sigma``/``stop_sigma``
vs ``start_step``/``stop_step``) are parsed identically by the distance / rmsd / angle /
dihedral / improper entry classes. These tests PIN the exact omitted-key defaults and
the null/zero normalisation of every type, so the consolidation of that parse into one
shared helper (``_config_util.apply_window_params``) cannot silently drift a default —
notably rmsd, whose ``weight`` field default is ``None`` (normalised to 1.0) while the
other entries pre-init 1.0, and whose ``stop_sigma`` re-asserts -1.0.
"""

import pytest

from rgi_utils.custom.data import CustomData
from rgi_utils.distance_restr_data import DistanceData
from rgi_utils.group_geom_restr_data import (
    AngleRestraintData,
    DihedralRestraintData,
    ImproperRestraintData,
)
from rgi_utils.rmsd_restr_data import RmsdData

# minimal valid configs per type, parameterised so the shared keys (weight + windows)
# can be merged in. Each carries exactly the type-specific selection/target keys needed
# for set_config to run without raising.
_BASE = {
    "distance": (
        DistanceData,
        {
            "atom_selection1": "chain A",
            "atom_selection2": "chain B",
            "harmonic": {"target_distance": 10.0},
        },
    ),
    "rmsd": (
        RmsdData,
        {"ref_pdb": "ref.pdb", "harmonic": {"target_rmsd": 1.0}},
    ),
    "angle": (
        AngleRestraintData,
        {
            "atom_selection1": "chain A",
            "atom_selection2": "chain B",
            "atom_selection3": "chain C",
            "harmonic": {"target_angle": 90.0},
        },
    ),
    "dihedral": (
        DihedralRestraintData,
        {
            "atom_selection1": "chain A",
            "atom_selection2": "chain B",
            "atom_selection3": "chain C",
            "atom_selection4": "chain D",
            "harmonic": {"target_dihedral": 180.0},
        },
    ),
    "improper": (
        ImproperRestraintData,
        {
            "atom_selection1": "chain A",
            "atom_selection2": "chain B",
            "atom_selection3": "chain C",
            "atom_selection4": "chain D",
            "harmonic": {"target_improper": 180.0},
        },
    ),
    "custom": (
        CustomData,
        {"energy": "rg(A)", "selections": {"A": "resid 1"}},
    ),
}

_TYPES = list(_BASE)


def _build(kind: str, extra: dict | None = None):
    cls, base = _BASE[kind]
    obj = cls()
    obj.set_config({**base, **(extra or {})})
    return obj


@pytest.mark.parametrize("kind", _TYPES)
def test_window_defaults_when_omitted(kind):
    """Every restraint type lands the SAME gate-window defaults when the keys are absent:
    weight 1.0, start_sigma None (config.from_dict fills +inf), stop_sigma -1, step window
    -inf/+inf. This is the contract the shared parse must preserve."""
    obj = _build(kind)
    assert obj.weight == pytest.approx(1.0)
    assert obj.start_sigma is None
    assert obj.stop_sigma == pytest.approx(-1.0)
    assert obj.start_step == float("-inf")
    assert obj.stop_step == float("inf")


@pytest.mark.parametrize("kind", _TYPES)
def test_window_sigma_values_parsed(kind):
    obj = _build(kind, {"start_sigma": 2.5, "stop_sigma": 0.5})
    assert obj.start_sigma == pytest.approx(2.5)
    assert obj.stop_sigma == pytest.approx(0.5)


@pytest.mark.parametrize("kind", _TYPES)
def test_window_step_values_parsed(kind):
    obj = _build(kind, {"start_step": 3, "stop_step": 8})
    assert obj.start_step == pytest.approx(3.0)
    assert obj.stop_step == pytest.approx(8.0)


@pytest.mark.parametrize("kind", _TYPES)
def test_weight_explicit_zero_stays_zero(kind):
    """An explicit weight 0 is a zero-weight no-op restraint — it must NOT be coerced to
    the 1.0 default (truthiness-based `or 1.0` would wrongly do so)."""
    obj = _build(kind, {"weight": 0})
    assert obj.weight == pytest.approx(0.0)


@pytest.mark.parametrize("kind", _TYPES)
def test_weight_value_parsed(kind):
    obj = _build(kind, {"weight": 0.25})
    assert obj.weight == pytest.approx(0.25)


@pytest.mark.parametrize("kind", _TYPES)
def test_explicit_null_treated_as_omitted(kind):
    """A literal ``null`` (YAML) / ``None`` (JSON) on a window key is treated as omitted
    (-> the default), not parsed as a value and not a crash."""
    obj = _build(
        kind,
        {"start_sigma": None, "stop_sigma": None, "weight": None},
    )
    assert obj.weight == pytest.approx(1.0)
    assert obj.start_sigma is None
    assert obj.stop_sigma == pytest.approx(-1.0)


@pytest.mark.parametrize("kind", _TYPES)
def test_sigma_step_windows_mutually_exclusive(kind):
    """Mixing the sigma window and the step window raises (per-type label preserved)."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build(kind, {"start_sigma": 1.0, "start_step": 2})
