"""Cross-path checks for the shared built-in/custom geometry implementation."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from rgi_utils import _geometry as G
from rgi_utils._array_ops import get_ops
from rgi_utils.custom import vocabulary as V
from rgi_utils.energy import numpy_energy
from rgi_utils.energy._terms import TERM_DEFS


def _scalar(value) -> float:
    return float(np.asarray(value))


def test_registered_terms_have_one_schema_and_dispatch_definition():
    keys = [term.key for term in TERM_DEFS]
    assert len(keys) == len(set(keys))
    for term in TERM_DEFS:
        fields = dict(term.fields)
        assert fields["mask"] == "f"
        assert term.args
        assert set(term.args) <= set(fields)
        if term.gate != "conf":
            assert {
                "start_sigma",
                "stop_sigma",
                "start_step",
                "stop_step",
            } <= set(fields)


def test_backend_adapter_preserves_leaf_call_signature():
    signature = inspect.signature(numpy_energy.rmsd_energy)
    assert "ops" not in signature.parameters
    assert list(signature.parameters)[:3] == ["positions", "fit_idx", "fit_mask"]


def test_builtin_and_custom_geometry_paths_match_numpy():
    ops = get_ops("numpy")
    positions = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.7],
        ],
        dtype=np.float64,
    )
    one = np.ones(1)

    angle_target = 1.1
    angle_builtin = numpy_energy.angle_energy(
        positions,
        np.array([[0, 1, 2]]),
        np.array([angle_target]),
        np.zeros(1),
        one,
        one,
    )
    angle_custom = V.harmonic(
        ops,
        V.angle(ops, positions[[0]], positions[[1]], positions[[2]]),
        angle_target,
    )
    assert _scalar(angle_builtin) == pytest.approx(_scalar(angle_custom), abs=1e-12)

    torsion_target = 0.4
    torsion_builtin = numpy_energy.cistrans_energy(
        positions,
        np.array([[0, 1, 2, 3]]),
        np.array([torsion_target]),
        np.zeros(1),
        one,
        one,
    )
    torsion_custom = (
        V.wrap(
            ops,
            V.dihedral(
                ops,
                positions[[0]],
                positions[[1]],
                positions[[2]],
                positions[[3]],
            )
            - torsion_target,
        )
        ** 2
    )
    assert _scalar(torsion_builtin) == pytest.approx(_scalar(torsion_custom), abs=1e-12)

    group1 = np.array([[0, 1]])
    group2 = np.array([[2, 3]])
    group_mask = np.ones((1, 2))
    distance_target = 1.3
    distance_builtin = numpy_energy.distance_energy(
        positions,
        group1,
        group2,
        group_mask,
        group_mask,
        np.array([distance_target]),
        np.zeros(1),
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        one,
        one,
    )
    distance_custom = V.harmonic(
        ops,
        V.distance(ops, positions[group1[0]], positions[group2[0]]),
        distance_target,
    )
    assert _scalar(distance_builtin) == pytest.approx(
        _scalar(distance_custom), abs=1e-12
    )


def test_builtin_and_custom_plane_and_rmsd_paths_match_numpy():
    ops = get_ops("numpy")
    moving = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
            [0.0, 1.0, -0.1],
            [1.0, 1.0, 0.3],
        ],
        dtype=np.float64,
    )
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.0, 0.9, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    one = np.ones(1)
    idx = np.arange(4).reshape(1, 4)
    atom_mask = np.ones((1, 4))

    plane_builtin = numpy_energy.plane_energy(
        moving, idx, atom_mask, np.zeros(1), one, one
    )
    plane_custom = V.plane(ops, moving) ** 2
    assert _scalar(plane_builtin) == pytest.approx(_scalar(plane_custom), abs=1e-12)

    rmsd_builtin = numpy_energy.rmsd_energy(
        moving,
        idx,
        atom_mask,
        reference[None, ...],
        idx,
        atom_mask,
        reference[None, ...],
        np.zeros(1),
        np.zeros(1),
        np.zeros(1, dtype=np.int64),
        one,
        one,
    )
    rmsd_custom = V.rmsd(ops, moving, reference) ** 2
    assert _scalar(rmsd_builtin) == pytest.approx(_scalar(rmsd_custom), abs=1e-12)


@pytest.mark.parametrize("type_code", range(4))
def test_custom_penalties_use_shared_type_code_semantics(type_code):
    ops = get_ops("numpy")
    value = np.array([2.5])
    target1 = np.array([1.0])
    target2 = np.array([2.0])
    common = G.restraint_delta(ops, value, target1, target2, type_code) ** 2
    custom = (
        V.harmonic(ops, value, target1)
        if type_code == 0
        else V.flat_bottomed(ops, value, target1, target2)
        if type_code == 1
        else V.flat_bottomed1(ops, value, target1)
        if type_code == 2
        else V.flat_bottomed2(ops, value, target2)
    )
    np.testing.assert_allclose(common, custom, atol=0.0, rtol=0.0)
