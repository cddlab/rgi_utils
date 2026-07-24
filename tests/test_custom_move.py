from types import SimpleNamespace

import numpy as np
import pytest

from rgi_utils.config import RestraintsConfig
from rgi_utils.custom.closure import build_terms
from rgi_utils.featurizer import build_spec


class _FakeAdapter:
    def __init__(self, n: int) -> None:
        self.n = n

    def iter_atoms(self):
        for i in range(self.n):
            yield SimpleNamespace(
                chain="A",
                resid=i + 1,
                index=i,
                name="CA",
                mol_type="protein",
                resname="ALA",
            )


def _write_ca_pdb(path, coords):
    lines = []
    for i, (x, y, z) in enumerate(coords):
        lines.append(
            "ATOM  "
            f"{i + 1:>5} {'CA':<4} {'ALA':>3} A{i + 1:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {'C':>2}\n"
        )
    path.write_text("".join(lines) + "END\n")


def _resolve_custom(config, n):
    parsed = RestraintsConfig.from_dict(config)
    for data in parsed.custom_data:
        data.resolve_sites(_FakeAdapter(n))
    return build_spec(custom_restraints=parsed.custom_data)


def _ref_def(path):
    return {"ref_pdb": str(path), "pairing": "identity"}


def _angle_entry(path, move):
    return {
        "atom_selection1": "chain A and resid 1",
        "atom_selection2": "ref1 and chain A and resid 2",
        "atom_selection3": "chain A and resid 3",
        "refs": {"ref1": _ref_def(path)},
        "move": move,
        "harmonic": {"target_angle": 60.0},
    }


def _assert_only_row_has_grad(spec, positions, moving_global):
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    active = positions[spec.active_sites]
    moving_row = spec.active_sites.tolist().index(moving_global)

    torch_positions = torch.tensor(active, dtype=torch.float64, requires_grad=True)
    build_terms(spec.custom, "torch")[0][-1](torch_positions).backward()
    torch_grad = torch_positions.grad.detach().numpy()

    jax_closure = build_terms(spec.custom, "jax")[0][-1]
    jax_grad = np.asarray(jax.grad(jax_closure)(jnp.asarray(active)))

    for grad in (torch_grad, jax_grad):
        assert np.linalg.norm(grad[moving_row]) > 1e-6
        pinned = np.delete(grad, moving_row, axis=0)
        assert np.allclose(pinned, 0.0, atol=1e-10)


def test_ref_angle_move_selects_prediction_groups(tmp_path):
    path = tmp_path / "ref.pdb"
    _write_ca_pdb(path, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]))

    one = RestraintsConfig.from_dict(
        {"angle_restraints_config": [_angle_entry(path, 1)]}
    )
    assert one.custom_data[0].move_free == (True, False, False)

    all_prediction = RestraintsConfig.from_dict(
        {"angle_restraints_config": [_angle_entry(path, "all")]}
    )
    assert all_prediction.custom_data[0].move_free == (True, False, True)

    with pytest.raises(ValueError, match="move selects reference group"):
        RestraintsConfig.from_dict({"angle_restraints_config": [_angle_entry(path, 2)]})


def test_ref_angle_move_pins_unselected_prediction_group(tmp_path):
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    path = tmp_path / "ref.pdb"
    _write_ca_pdb(path, positions)
    spec = _resolve_custom(
        {"angle_restraints_config": [_angle_entry(path, 1)]},
        n=3,
    )
    _assert_only_row_has_grad(spec, positions, moving_global=0)


def test_custom_move_uses_selection_names_with_reference(tmp_path):
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    path = tmp_path / "ref.pdb"
    _write_ca_pdb(path, positions)
    entry = {
        "energy": "(angle(A, R, B) - 1.0)**2",
        "selections": {
            "A": "chain A and resid 1",
            "R": "ref1 and chain A and resid 2",
            "B": "chain A and resid 3",
        },
        "refs": {"ref1": _ref_def(path)},
        "move": "B",
    }
    spec = _resolve_custom({"custom_restraints_config": [entry]}, n=3)
    assert spec.custom[0].move_free == {"A": False, "B": True}
    _assert_only_row_has_grad(spec, positions, moving_global=2)

    entry["move"] = "R"
    with pytest.raises(ValueError, match="reference-backed selection"):
        _resolve_custom({"custom_restraints_config": [entry]}, n=3)


def test_add_custom_forwards_move():
    from rgi_utils import CombinedRestraints

    restraints = CombinedRestraints()
    restraints.add_custom(
        fn=lambda ctx: ctx.distance("resid 1", "resid 2") ** 2,
        move="resid 1",
    )
    restraints.setup(_FakeAdapter(2), config={"gpu": False})
    assert restraints.spec.custom[0].move_free == {
        "resid 1": True,
        "resid 2": False,
    }
