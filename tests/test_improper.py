"""Improper restraint coverage across config, backends, optimizers, and custom APIs."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from rgi_utils.config import RestraintsConfig
from rgi_utils.custom.closure import build_terms
from rgi_utils.energy import numpy_energy
from rgi_utils.featurizer import build_spec
from rgi_utils.group_geom_restr_data import (
    DihedralRestraintData,
    ImproperRestraintData,
)

_SELECTIONS = {
    "atom_selection1": "chain A",
    "atom_selection2": "chain B",
    "atom_selection3": "chain C",
    "atom_selection4": "chain D",
}
_CUSTOM_SELECTIONS = {name: f"chain {name}" for name in "ABCD"}


class _Adapter:
    def iter_atoms(self):
        for index, chain in enumerate("ABCD"):
            yield SimpleNamespace(
                chain=chain,
                resid=1,
                index=index,
                name="C",
                mol_type="ligand",
                resname="LIG",
            )


def _coords() -> np.ndarray:
    coords = np.zeros((4, 3), dtype=np.float64)
    coords[0] = [0.0, 1.0, 0.0]
    coords[1] = [0.0, 0.0, 0.0]
    coords[2] = [1.0, 0.0, 0.0]
    coords[3] = [1.0, 1.0, 0.0]
    return coords


def _resolved_data(cls, target_key: str, target_deg: float = 90.0):
    data = cls()
    data.set_config(
        {
            **_SELECTIONS,
            "harmonic": {target_key: target_deg},
            "start_sigma": 2.0,
        }
    )
    data.target_sites1 = [0]
    data.target_sites2 = [1]
    data.target_sites3 = [2]
    data.target_sites4 = [3]
    return data


def _improper_spec(target_deg: float = 90.0):
    data = _resolved_data(ImproperRestraintData, "target_improper", target_deg)
    return build_spec(improper_restraints=[data])


@pytest.mark.parametrize(
    ("block", "expected_type", "target1", "target2"),
    [
        ({"harmonic": {"target_improper": 90.0}}, "harmonic", 90.0, 0.0),
        (
            {
                "flat-bottomed": {
                    "target_improper1": -30.0,
                    "target_improper2": 30.0,
                }
            },
            "flat-bottomed",
            -30.0,
            30.0,
        ),
        (
            {"flat-bottomed1": {"target_improper1": -20.0}},
            "flat-bottomed1",
            -20.0,
            0.0,
        ),
        (
            {"flat-bottomed2": {"target_improper2": 20.0}},
            "flat-bottomed2",
            0.0,
            20.0,
        ),
    ],
)
def test_improper_config_types_and_units(block, expected_type, target1, target2):
    data = ImproperRestraintData()
    data.set_config({**_SELECTIONS, **block})
    assert data.geom_type == expected_type
    assert data.target1 == pytest.approx(math.radians(target1))
    assert data.target2 == pytest.approx(math.radians(target2))
    assert data.move_free == (True, False, False, True)


def test_improper_top_level_and_reference_routing():
    config = RestraintsConfig.from_dict(
        {
            "improper_restraints_config": [
                {**_SELECTIONS, "harmonic": {"target_improper": 45.0}},
                {
                    **_SELECTIONS,
                    "atom_selection4": "ref1 and chain D",
                    "refs": {"ref1": {"ref_pdb": "ref.pdb", "pairing": "identity"}},
                    "harmonic": {"target_improper": 0.0},
                },
            ]
        }
    )
    assert len(config.improper_data) == 1
    assert len(config.custom_data) == 1
    assert config.custom_data[0].geom == "improper"
    assert config.custom_data[0].move_free == (True, True, True, False)


def test_improper_reference_group_resolves_and_evaluates(tmp_path):
    ref_path = tmp_path / "ref.pdb"
    ref_path.write_text(
        "HETATM    1  C   LIG D   1       1.000   1.000   1.000  1.00  0.00           C\n"
        "END\n",
        encoding="utf-8",
    )
    config = RestraintsConfig.from_dict(
        {
            "improper_restraints_config": [
                {
                    **_SELECTIONS,
                    "atom_selection4": "ref1 and chain D",
                    "refs": {
                        "ref1": {
                            "ref_pdb": str(ref_path),
                            "pairing": "identity",
                        }
                    },
                    "harmonic": {"target_improper": 30.0},
                }
            ]
        }
    )
    data = config.custom_data[0]
    data.resolve_sites(_Adapter())
    spec = build_spec(custom_restraints=[data])
    closure = build_terms(spec.custom, "numpy")[0][-1]
    value = float(closure(_coords()[spec.active_sites]))
    assert spec.custom[0].geom == "improper"
    assert np.isfinite(value)


def test_improper_spec_and_energy_are_distinct_but_match_dihedral():
    improper_spec = _improper_spec()
    dihedral = _resolved_data(DihedralRestraintData, "target_dihedral")
    dihedral_spec = build_spec(dihedral_restraints=[dihedral])

    assert improper_spec.has_group_improper()
    assert improper_spec.is_active()
    assert improper_spec.max_start_sigma() == pytest.approx(2.0)
    assert list(improper_spec.active_sites) == [0, 1, 2, 3]

    pos = _coords()
    improper_bd = numpy_energy.energy_breakdown(
        pos, numpy_energy.prepare_spec(improper_spec), sigma=0.0
    )
    dihedral_bd = numpy_energy.energy_breakdown(
        pos, numpy_energy.prepare_spec(dihedral_spec), sigma=0.0
    )
    assert improper_bd["group_improper"] == pytest.approx(dihedral_bd["group_dihedral"])
    assert improper_bd["group_improper"] > 0.0
    assert improper_bd["group_dihedral"] == 0.0
    assert (
        numpy_energy.total_energy(
            pos, numpy_energy.prepare_spec(improper_spec), sigma=3.0
        )
        == 0.0
    )


def test_improper_backend_energy_and_gradient_parity():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy

    spec = _improper_spec(75.0)
    pos = _coords()
    numpy_value = numpy_energy.total_energy(
        pos, numpy_energy.prepare_spec(spec), sigma=0.0
    )

    torch_pos = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    torch_value = torch_energy.total_energy(
        torch_pos,
        torch_energy.prepare_spec(spec, dtype=torch.float64),
        sigma=0.0,
    )
    torch_value.backward()

    jax_prepared = jax_energy.prepare_spec(spec)
    jax_value, jax_grad = jax.value_and_grad(
        lambda p: jax_energy.total_energy(p, jax_prepared, sigma=0.0)
    )(jnp.asarray(pos))

    assert float(torch_value.detach()) == pytest.approx(float(numpy_value), rel=1e-10)
    assert float(jax_value) == pytest.approx(float(numpy_value), rel=1e-10)
    np.testing.assert_allclose(
        torch_pos.grad.detach().numpy(), np.asarray(jax_grad), rtol=1e-8, atol=1e-8
    )


def _custom_spec(entry: dict):
    config = RestraintsConfig.from_dict({"custom_restraints_config": [entry]})
    data = config.custom_data[0]
    data.resolve_sites(_Adapter())
    return build_spec(custom_restraints=[data])


def test_custom_improper_formula_and_code_paths_match():
    target = 0.7
    formula_spec = _custom_spec(
        {
            "energy": f"wrap(improper(A,B,C,D) - {target})**2",
            "selections": _CUSTOM_SELECTIONS,
        }
    )

    def energy(ctx):
        return ctx.wrap(ctx.improper("A", "B", "C", "D") - target) ** 2

    code_spec = _custom_spec(
        {
            "fn": energy,
            "selections": _CUSTOM_SELECTIONS,
        }
    )
    pos = _coords()
    formula_fn = build_terms(formula_spec.custom, "numpy")[0][-1]
    code_fn = build_terms(code_spec.custom, "numpy")[0][-1]
    assert float(formula_fn(pos)) == pytest.approx(float(code_fn(pos)))
    assert float(formula_fn(pos)) > 0.0


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_improper_optimizer_reduces_energy(backend):
    spec = _improper_spec(90.0)
    coords = _coords()[None, ...]
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

        value = torch.tensor(coords, dtype=torch.float64)
        optimizer = TorchRestraintOptimizer(spec, max_iter=500)
        before = optimizer.energy(value)
        optimizer.minimize(value, sigma=0.0)
        after = optimizer.energy(value)
    else:
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        from rgi_utils.optim.jax_optim import energy_of, make_minimizer

        value = jnp.asarray(coords)
        before = energy_of(spec, value)
        value = make_minimizer(spec, max_iter=500)(value, 0.0)
        after = energy_of(spec, value)

    assert float(after) < 1e-4 * float(before)
