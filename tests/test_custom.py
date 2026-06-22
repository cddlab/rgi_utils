"""Verification harness for custom restraints (expression DSL + ctx function).

A custom restraint is a backend-agnostic energy ``energy(ctx) -> scalar`` written either as
a config ``energy`` formula string or a Python ctx function. Both compile to a closure that
the optimizers add to the CG objective. This harness drives BOTH authoring paths through:

  (a) 3-backend energy + gradient parity. Custom energies use PLAIN centroids (no
      rigid-translation stop-gradient trick), so unlike the built-in group restraints the
      autodiff gradient matches the numpy finite-difference ground truth — a strict check.
  (b) the torch eager CG: a custom distance restraint converges onto its target.
  (c) the jax minimizer (the AF3 lax.scan closure): converges, NaN-free.

Plus the DSL safety whitelist, the config top-level whitelist, and the ``ctx`` vocabulary.
Custom energies are closures (not array terms), so they are exercised here by building the
per-backend terms directly — the built-in energy layer is untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rgi_utils.config import RestraintsConfig
from rgi_utils.custom.closure import build_terms
from rgi_utils.featurizer import build_spec


class _FakeAdapter:
    def __init__(self, n: int = 12) -> None:
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


def _spec_from_entries(entries, n: int = 12):
    cfg = RestraintsConfig.from_dict({"custom_restraints_config": entries})
    for cd in cfg.custom_data:
        cd.resolve_sites(_FakeAdapter(n))
    return build_spec(custom_restraints=cfg.custom_data)


def _sym_spec():
    """A non-trivial custom restraint: a symmetric distance difference, all-positive energy."""
    return _spec_from_entries(
        [
            {
                "name": "sym",
                "energy": "(distance(A,B) - distance(C,D))**2 + 0.5*harmonic(rg(A), 1.5)",
                "selections": {
                    "A": "resid 1 to 3",
                    "B": "resid 4 to 6",
                    "C": "resid 7 to 8",
                    "D": "resid 9 to 10",
                },
                "weight": 1.0,
            }
        ]
    )


def _positions(spec, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((spec.n_active, 3)) * 2.0


def _fd_grad(f, x, eps: float = 1e-6):
    x = np.asarray(x, dtype=np.float64).copy()
    g = np.zeros_like(x)
    for i in range(x.size):
        o = x.flat[i]
        x.flat[i] = o + eps
        fp = f(x)
        x.flat[i] = o - eps
        fm = f(x)
        x.flat[i] = o
        g.flat[i] = (fp - fm) / (2.0 * eps)
    return g


# --------------------------------------------------------------------------------------
# (a) 3-backend energy + gradient parity
# --------------------------------------------------------------------------------------
def test_custom_energy_parity_3backend():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    spec = _sym_spec()
    assert spec.has_custom() and len(spec.custom) == 1
    pos = _positions(spec)

    e_np = float(build_terms(spec.custom, "numpy")[0][-1](pos))
    e_t = float(
        build_terms(spec.custom, "torch")[0][-1](torch.tensor(pos, dtype=torch.float64))
    )
    e_j = float(build_terms(spec.custom, "jax")[0][-1](jnp.asarray(pos)))
    assert e_np > 0.0
    assert abs(e_np - e_t) < 1e-6, f"numpy={e_np} torch={e_t}"
    assert abs(e_np - e_j) < 1e-6, f"numpy={e_np} jax={e_j}"


def test_custom_grad_matches_fd():
    """Custom uses plain centroids, so autodiff == numpy finite-difference (strict)."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    spec = _sym_spec()
    pos = _positions(spec, seed=2)
    np_clo = build_terms(spec.custom, "numpy")[0][-1]
    g_fd = _fd_grad(lambda x: float(np_clo(x.reshape(spec.n_active, 3))), pos.flatten())
    g_fd = g_fd.reshape(spec.n_active, 3)

    pt = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    build_terms(spec.custom, "torch")[0][-1](pt).backward()
    assert np.allclose(pt.grad.numpy(), g_fd, atol=1e-4), (
        f"torch vs FD {np.abs(pt.grad.numpy() - g_fd).max()}"
    )
    jax_clo = build_terms(spec.custom, "jax")[0][-1]
    g_j = np.asarray(jax.grad(lambda x: jax_clo(x))(jnp.asarray(pos)))
    assert np.allclose(g_j, g_fd, atol=1e-4), f"jax vs FD {np.abs(g_j - g_fd).max()}"


def test_ctx_vocabulary_parity():
    """A formula touching geometry + math + penalty agrees across backends."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    spec = _spec_from_entries(
        [
            {
                "energy": "exp(-distance(A,B)) + flat_bottomed(angle(A,B,C), 1.0, 2.0) "
                "+ where(rg(A) > 1.0, harmonic(rg(A), 1.0), 0.0)",
                "selections": {
                    "A": "resid 1 to 3",
                    "B": "resid 5 to 6",
                    "C": "resid 8 to 9",
                },
            }
        ]
    )
    pos = _positions(spec, seed=4)
    e_np = float(build_terms(spec.custom, "numpy")[0][-1](pos))
    e_t = float(
        build_terms(spec.custom, "torch")[0][-1](torch.tensor(pos, dtype=torch.float64))
    )
    e_j = float(build_terms(spec.custom, "jax")[0][-1](jnp.asarray(pos)))
    assert abs(e_np - e_t) < 1e-6 and abs(e_np - e_j) < 1e-6


# --------------------------------------------------------------------------------------
# (b) torch eager CG + (c) jax minimizer: a custom distance restraint converges to target
# --------------------------------------------------------------------------------------
def _dist_spec():
    return _spec_from_entries(
        [
            {
                "energy": "(distance(A,B) - 5.0)**2",
                "selections": {"A": "resid 1", "B": "resid 2"},
            }
        ],
        n=2,
    )


def test_custom_torch_minimize_converges():
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec = _dist_spec()
    coords = torch.zeros((1, 2, 3), dtype=torch.float64)
    coords[0, 1, 0] = 1.0  # start 1 A apart; target 5 A
    TorchRestraintOptimizer(spec, max_iter=200).minimize(coords)
    d = float(torch.linalg.norm(coords[0, 0] - coords[0, 1]))
    assert abs(d - 5.0) < 0.1, f"distance {d} did not reach 5.0"


def test_custom_jax_minimize_nan_free():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer

    spec = _dist_spec()
    coords = np.zeros((1, 2, 3))
    coords[0, 1, 0] = 1.0
    out = np.asarray(make_minimizer(spec, max_iter=200)(jnp.asarray(coords), 0.0))
    assert np.all(np.isfinite(out)), "custom restraint produced NaN/Inf under jax"
    d = float(np.linalg.norm(out[0, 0] - out[0, 1]))
    assert abs(d - 5.0) < 0.1, f"jax distance {d} did not reach 5.0"


# --------------------------------------------------------------------------------------
# code path: a registered ctx function + the direct add_custom quick path
# --------------------------------------------------------------------------------------
def test_code_ctx_restraint_parity_and_minimize():
    torch = pytest.importorskip("torch")
    from rgi_utils import custom_restraint
    from rgi_utils.custom.registry import clear_custom_fns
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    @custom_restraint("pull5")
    def energy(ctx):
        return (ctx.distance("resid 1", "resid 2") - 5.0) ** 2

    try:
        spec = _spec_from_entries([{"use": "pull5"}], n=2)
        assert spec.has_custom()
        # parity vs the equivalent formula
        fspec = _dist_spec()
        pos = np.random.default_rng(7).standard_normal((2, 3))
        a = float(build_terms(spec.custom, "numpy")[0][-1](pos))
        b = float(build_terms(fspec.custom, "numpy")[0][-1](pos))
        assert abs(a - b) < 1e-9, (a, b)
        # and it converges
        coords = torch.zeros((1, 2, 3), dtype=torch.float64)
        coords[0, 1, 0] = 1.0
        TorchRestraintOptimizer(spec, max_iter=200).minimize(coords)
        assert abs(float(torch.linalg.norm(coords[0, 0] - coords[0, 1])) - 5.0) < 0.1
    finally:
        clear_custom_fns()


def test_add_custom_direct_callable():
    torch = pytest.importorskip("torch")
    from rgi_utils import CombinedRestraints

    def energy(ctx):
        return (ctx.distance("resid 1", "resid 2") - 5.0) ** 2

    restr = CombinedRestraints()
    restr.add_custom(fn=energy)
    restr.setup(_FakeAdapter(2), config={"gpu": False, "max_iter": 200})
    assert restr.spec.has_custom()
    coords = torch.zeros((1, 2, 3), dtype=torch.float64)
    coords[0, 1, 0] = 1.0
    restr.minimize(coords, 0, 0.0)
    assert abs(float(torch.linalg.norm(coords[0, 0] - coords[0, 1])) - 5.0) < 0.1


# --------------------------------------------------------------------------------------
# DSL safety + config whitelist
# --------------------------------------------------------------------------------------
def test_dsl_rejects_unsafe():
    from rgi_utils.custom.dsl import parse_formula

    parse_formula("(distance(A,B) - 2.0)**2")  # ok
    for bad in [
        "__import__('os').system('x')",
        "open('f')",
        "a.b",
        "x[0]",
        "[i for i in y]",
        "lambda: 1",
        "distance(A,B) and 1",  # BoolOp not allowed (use & for elementwise)
    ]:
        with pytest.raises(ValueError):
            parse_formula(bad)


def test_config_whitelist():
    RestraintsConfig.from_dict(
        {
            "custom_restraints_config": [
                {"energy": "rg(A)", "selections": {"A": "resid 1 to 3"}}
            ]
        }
    )
    with pytest.raises(ValueError, match="unknown top-level"):
        RestraintsConfig.from_dict({"bogus_restraints_config": []})


def test_custom_entry_requires_one_source():
    cfg = RestraintsConfig.from_dict
    with pytest.raises(ValueError, match="exactly one"):
        cfg(
            {"custom_restraints_config": [{"selections": {"A": "resid 1"}}]}
        )  # no energy/use/fn


def test_penalty_vocabulary_is_flat_bottomed():
    """The penalty vocabulary mirrors the distance config keys: flat_bottomed /
    flat_bottomed1 (lower) / flat_bottomed2 (upper) / harmonic. The old lower / upper /
    flat_bottom names are GONE (hard rename, not aliased) -- a formula using the new
    names builds a closure; one using an old name raises at parse (not in the DSL
    whitelist)."""
    spec = _spec_from_entries(
        [
            {
                "energy": "flat_bottomed(distance(A,B), 1.0, 5.0) "
                "+ flat_bottomed1(rg(A), 1.0) + flat_bottomed2(rg(B), 5.0)",
                "selections": {"A": "resid 1 to 3", "B": "resid 5 to 6"},
            }
        ]
    )
    assert spec.custom  # new names build a closure
    for old in (
        "lower(rg(A), 1.0)",
        "upper(rg(A), 5.0)",
        "flat_bottom(rg(A), 1.0, 5.0)",
    ):
        with pytest.raises(ValueError, match="only calls|disallowed"):
            _spec_from_entries([{"energy": old, "selections": {"A": "resid 1 to 3"}}])
