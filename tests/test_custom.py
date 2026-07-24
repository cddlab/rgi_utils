"""Verification harness for custom restraints (expression DSL + ctx function).

A custom restraint is a backend-agnostic energy ``energy(ctx) -> scalar`` written either as
a config ``energy`` formula string or a Python ctx function. Both compile to a closure that
the optimizers add to the CG objective. This harness drives BOTH authoring paths through:

  (a) 3-backend energy + gradient parity. MOST custom energies use PLAIN centroids (no
      rigid-translation stop-gradient trick), so unlike the built-in group restraints the
      autodiff gradient matches the numpy finite-difference ground truth — a strict check
      (test_custom_grad_matches_fd). EXCEPTION: kabsch / rmsd freeze the Kabsch rotation
      with stop-gradient (the SVD is not differentiated), so — like the built-in rmsd term
      — an FD gradient is INAPPLICABLE; those are checked torch-vs-jax instead
      (test_custom_kabsch_grad_torch_vs_jax). Do NOT add a kabsch/rmsd case under the FD test.
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


def test_custom_dihedral_wrap_periodicity():
    """``wrap`` folds a dihedral deviation into [-pi, pi], so a penalty is periodicity-safe
    like the built-in ``group_dihedral`` / ``cistrans``. Contract: with the deviation set to
    straddle +-180 deg (358 deg), ``wrap(dihedral - t)**2`` reads it as ~2 deg while the naive
    ``harmonic(dihedral, t)`` reads it as ~358 deg (huge) — this is exactly what the fix buys.
    """
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    sels = {"A": "resid 1", "B": "resid 2", "C": "resid 3", "D": "resid 4"}

    # phi of the four single-atom centroids, computed on the SAME active-site ordering the
    # closures use (energy == dihedral(...) -> ops.sum(scalar) == scalar), so the target can
    # be placed exactly 358 deg away without tracking the A/B/C/D -> local-index mapping.
    # All specs here reference the same four atoms, so they share n_active == 4 and one
    # ``pos`` maps consistently across them.
    phi_spec = _spec_from_entries([{"energy": "dihedral(A,B,C,D)", "selections": sels}])
    pos = _positions(phi_spec, seed=7)
    phi = float(build_terms(phi_spec.custom, "numpy")[0][-1](pos))
    d = np.radians(358.0)
    t = phi - d  # so dihedral(...) - t == 358 deg, which wraps to -2 deg

    wrapped = _spec_from_entries(
        [{"energy": f"wrap(dihedral(A,B,C,D) - ({t}))**2", "selections": sels}]
    )
    naive = _spec_from_entries(
        [{"energy": f"harmonic(dihedral(A,B,C,D), ({t}))", "selections": sels}]
    )
    e_wrapped = float(build_terms(wrapped.custom, "numpy")[0][-1](pos))
    e_naive = float(build_terms(naive.custom, "numpy")[0][-1](pos))
    assert abs(e_wrapped - np.radians(2.0) ** 2) < 1e-6, e_wrapped  # ~ (2 deg)^2
    assert abs(e_naive - d**2) < 1e-6, e_naive  # ~ (358 deg)^2, the periodicity bug
    assert e_naive / e_wrapped > 1000.0  # the fix collapses a huge false penalty

    # the wrapped form agrees across all three backends (structural parity of ``wrap``)
    pt = torch.tensor(pos, dtype=torch.float64)
    e_t = float(build_terms(wrapped.custom, "torch")[0][-1](pt))
    e_j = float(build_terms(wrapped.custom, "jax")[0][-1](jnp.asarray(pos)))
    assert abs(e_wrapped - e_t) < 1e-6 and abs(e_wrapped - e_j) < 1e-6


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


def test_add_custom_re_setup_without_config_no_duplication():
    """Regression: setup() must not mutate cfg.custom_data in place. A reused instance whose
    SECOND setup() omits ``config`` reuses the same config object, so an in-place
    ``extend(self._pending_custom)`` would re-append the pending custom restraint every call
    and silently duplicate it. The local-merge fix keeps the spec at exactly one custom term
    and leaves config.custom_data untouched."""
    from rgi_utils import CombinedRestraints

    def energy(ctx):
        return (ctx.distance("resid 1", "resid 2") - 5.0) ** 2

    restr = CombinedRestraints()
    restr.add_custom(fn=energy)
    restr.setup(_FakeAdapter(2), config={"gpu": False, "max_iter": 200})
    assert len(restr.spec.custom) == 1
    # re-setup WITHOUT a fresh config reuses self.config -- the double-merge trigger.
    restr.setup(_FakeAdapter(2))
    assert len(restr.spec.custom) == 1  # still one, not duplicated
    assert restr.config.custom_data == []  # setup never mutated the config in place


# --------------------------------------------------------------------------------------
# gate regression: the sigma/step window actually gates the custom energy in the optimizer
# (the gate lives independently in torch_optim._custom_energy and jax_optim._descend, so a
# single-backend test would miss a divergence). A windowed `(distance-5)**2`: INSIDE the
# window it converges to 5 A, OUTSIDE the gate zeroes the term and the coords stay put.
# --------------------------------------------------------------------------------------
def _dist_spec_win(extra):
    return _spec_from_entries(
        [
            {
                "energy": "(distance(A,B) - 5.0)**2",
                "selections": {"A": "resid 1", "B": "resid 2"},
                **extra,
            }
        ],
        n=2,
    )


def _two_atoms_torch(torch):
    c = torch.zeros((1, 2, 3), dtype=torch.float64)
    c[0, 1, 0] = 1.0  # 1 A apart; target 5 A
    return c


def test_custom_gate_sigma_window():
    """A custom restraint with start_sigma=2 is active for sigma<=2 (converges) and OFF
    for sigma>2 (coords unchanged) — on BOTH torch and jax."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec = _dist_spec_win({"start_sigma": 2.0})

    # torch: inside the window -> reaches 5 A; outside -> stays at 1 A
    c_in = _two_atoms_torch(torch)
    TorchRestraintOptimizer(spec, max_iter=200).minimize(c_in, sigma=1.0, step=0)
    assert abs(float(torch.linalg.norm(c_in[0, 0] - c_in[0, 1])) - 5.0) < 0.1
    c_out = _two_atoms_torch(torch)
    TorchRestraintOptimizer(spec, max_iter=200).minimize(c_out, sigma=3.0, step=0)
    assert abs(float(torch.linalg.norm(c_out[0, 0] - c_out[0, 1])) - 1.0) < 0.1

    # jax: same gate, independent implementation
    base = np.zeros((1, 2, 3))
    base[0, 1, 0] = 1.0
    mz = make_minimizer(spec, max_iter=200)
    o_in = np.asarray(mz(jnp.asarray(base), 1.0))
    assert abs(np.linalg.norm(o_in[0, 0] - o_in[0, 1]) - 5.0) < 0.1
    o_out = np.asarray(mz(jnp.asarray(base), 3.0))
    assert abs(np.linalg.norm(o_out[0, 0] - o_out[0, 1]) - 1.0) < 0.1


def test_custom_gate_step_window():
    """A custom restraint with start_step=5/stop_step=10 (no sigma window) is active for
    step in [5,10] (converges) and OFF outside (coords unchanged) — on torch and jax."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec = _dist_spec_win({"start_step": 5, "stop_step": 10})

    # torch: step inside the window -> reaches 5 A; step outside -> stays at 1 A
    c_in = _two_atoms_torch(torch)
    TorchRestraintOptimizer(spec, max_iter=200).minimize(c_in, sigma=1.0, step=7)
    assert abs(float(torch.linalg.norm(c_in[0, 0] - c_in[0, 1])) - 5.0) < 0.1
    c_out = _two_atoms_torch(torch)
    TorchRestraintOptimizer(spec, max_iter=200).minimize(c_out, sigma=1.0, step=0)
    assert abs(float(torch.linalg.norm(c_out[0, 0] - c_out[0, 1])) - 1.0) < 0.1

    # jax: same gate, independent implementation (the minimizer threads `step`)
    base = np.zeros((1, 2, 3))
    base[0, 1, 0] = 1.0
    mz = make_minimizer(spec, max_iter=200)
    o_in = np.asarray(mz(jnp.asarray(base), 1.0, 7))
    assert abs(np.linalg.norm(o_in[0, 0] - o_in[0, 1]) - 5.0) < 0.1
    o_out = np.asarray(mz(jnp.asarray(base), 1.0, 0))
    assert abs(np.linalg.norm(o_out[0, 0] - o_out[0, 1]) - 1.0) < 0.1


def test_custom_weight_scaling():
    """``weight`` is folded into the closure (closure.py): halving the weight halves the
    energy at the same coords (an explicit 0 would zero it — a no-op restraint)."""
    pos = np.random.default_rng(3).standard_normal((2, 3))
    e_full = float(
        build_terms(_dist_spec_win({"weight": 1.0}).custom, "numpy")[0][-1](pos)
    )
    e_half = float(
        build_terms(_dist_spec_win({"weight": 0.5}).custom, "numpy")[0][-1](pos)
    )
    assert e_full > 0.0
    assert abs(e_half - 0.5 * e_full) < 1e-9


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


# --------------------------------------------------------------------------------------
# kabsch(A, B): Kabsch superposition returning coordinates (a (k, 3) block that composes)
# --------------------------------------------------------------------------------------
def _kabsch_spec():
    """kabsch(A, B) superposes A onto B; ``norm(... - coords(B))`` then sums the per-atom
    post-superposition deviations. A/B are equal-size (positional correspondence)."""
    return _spec_from_entries(
        [
            {
                "name": "sym",
                "energy": "norm(kabsch(A, B) - coords(B))",
                "selections": {"A": "resid 1 to 4", "B": "resid 5 to 8"},
            }
        ]
    )


def test_custom_kabsch_energy_parity_3backend():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    spec = _kabsch_spec()
    assert spec.has_custom() and len(spec.custom) == 1
    pos = _positions(spec, seed=4)
    e_np = float(build_terms(spec.custom, "numpy")[0][-1](pos))
    e_t = float(
        build_terms(spec.custom, "torch")[0][-1](torch.tensor(pos, dtype=torch.float64))
    )
    e_j = float(build_terms(spec.custom, "jax")[0][-1](jnp.asarray(pos)))
    assert e_np > 0.0
    assert abs(e_np - e_t) < 1e-6, f"numpy={e_np} torch={e_t}"
    assert abs(e_np - e_j) < 1e-6, f"numpy={e_np} jax={e_j}"


def test_custom_kabsch_grad_torch_vs_jax():
    """The kabsch rotation is stop-gradient'd (the SVD is never differentiated), so — like
    the built-in ``rmsd`` term — a numpy finite-difference gradient is INAPPLICABLE and the
    contract is torch-vs-jax gradient parity (both freeze the rotation identically). This is
    why kabsch/rmsd formulas are exercised HERE, not under ``test_custom_grad_matches_fd``."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    spec = _kabsch_spec()
    pos = _positions(spec, seed=6)
    pt = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    build_terms(spec.custom, "torch")[0][-1](pt).backward()
    g_t = pt.grad.numpy()
    jax_clo = build_terms(spec.custom, "jax")[0][-1]
    g_j = np.asarray(jax.grad(lambda x: jax_clo(x))(jnp.asarray(pos)))
    assert np.allclose(g_t, g_j, atol=1e-6), f"torch vs jax {np.abs(g_t - g_j).max()}"


def test_custom_kabsch_rigid_invariance():
    """kabsch(A, B) energy (post-superposition deviation of A vs B) is invariant to a rigid
    motion of A — the Kabsch property. Confirms the superposition removes rotation/translation."""
    spec = _kabsch_spec()
    clo = build_terms(spec.custom, "numpy")[0][-1]
    pos = _positions(spec, seed=8)
    e0 = float(clo(pos))
    th = 0.5
    rot = np.array(
        [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]
    )
    a_local = spec.custom[0].selections["A"]  # move only A's atoms rigidly
    moved = pos.copy()
    moved[a_local] = pos[a_local] @ rot.T + np.array([5.0, -3.0, 2.0])
    assert abs(e0 - float(clo(moved))) < 1e-6


def test_custom_kabsch_requires_equal_counts():
    """kabsch(A, B) needs |A| == |B| (positional correspondence) — a mismatch raises loudly
    at resolve (not a cryptic matmul shape error at runtime)."""
    with pytest.raises(ValueError, match="equal atom counts"):
        _spec_from_entries(
            [
                {
                    "energy": "norm(kabsch(A, B) - coords(B))",
                    "selections": {"A": "resid 1 to 4", "B": "resid 5 to 7"},
                }
            ]
        )


# --------------------------------------------------------------------------------------
# rmsd(A, ref): Kabsch-superposed RMSD against a per-call external reference structure
# --------------------------------------------------------------------------------------
def _write_ca_pdb(path, coords, chain="A"):
    """One CA atom per coord (its own ALA residue, chain ``chain``); the per-chain resid
    ordinal follows file order (1..n) — read_pdb_atoms' convention."""
    lines = []
    for i, (x, y, z) in enumerate(coords):
        lines.append(
            "ATOM  "
            f"{i + 1:>5} {'CA':<4} {'ALA':>3} {chain}{i + 1:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {'C':>2}\n"
        )
    path.write_text("".join(lines) + "END\n")


def _superposed_rmsd(P, Q):
    """Kabsch RMSD of moving P onto reference Q (matches vocabulary.rmsd)."""
    P0 = P - P.mean(0)
    Q0 = Q - Q.mean(0)
    U, _S, Vt = np.linalg.svd(P0.T @ Q0)
    V = Vt.T
    d = np.sign(np.linalg.det(V @ U.T)) or 1.0
    Vd = V.copy()
    Vd[:, 2] *= d
    R = Vd @ U.T  # R P0 ~ Q0
    return float(np.sqrt(((P0 @ R.T - Q0) ** 2).sum() / len(P)))


def _rmsd_spec(pdb, energy, *, pairing="identity", best_effort=True, n=12):
    return _spec_from_entries(
        [
            {
                "name": "rmsd",
                "energy": energy,
                "selections": {"g": "chain A"},
                "refs": {
                    "r1": {
                        "ref_pdb": str(pdb),
                        "pairing": pairing,
                        "best_effort": best_effort,
                    }
                },
            }
        ],
        n=n,
    )


def test_custom_rmsd_alignment_matched_subset(tmp_path):
    """THE alignment test: the reference covers only resid 1..8 (target has 12), so a
    best-effort pairing must (a) drop resid 9..12 and (b) keep ref_coords row-aligned to the
    MATCHED target subset that the closure gathers. The ref is a rigid transform of the
    target's first 8 atoms, so a correct pairing gives superposed RMSD ~ 0; any row
    misalignment or wrong subset would blow it up."""
    rng = np.random.default_rng(11)
    n = 12
    pos = rng.standard_normal((n, 3)) * 3.0  # target coords (the active coords)
    th = 0.7
    rz = np.array(
        [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    )
    ref_full = pos[:8] @ rz.T + np.array(
        [4.0, -2.0, 1.0]
    )  # ref = rigid transform of 1..8
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, ref_full)

    spec = _rmsd_spec(pdb, "rmsd(g, r1)", n=n)
    tgt_idx, ref_coords = spec.custom[0].refs[("g", "r1")]
    assert len(tgt_idx) == 8, f"expected the 8 matched atoms, got {len(tgt_idx)}"
    # ref rows are aligned to the matched target subset (resid 1..8 -> local 0..7)
    assert np.array_equal(tgt_idx, np.arange(8))
    assert np.allclose(ref_coords, ref_full, atol=2e-3)  # PDB 3-decimal rounding
    # the closure gathers exactly that subset and compares to the aligned ref (the ref the
    # closure actually uses, read back from the PDB) -> superposed RMSD ~ 0 (rigid transform)
    e = float(build_terms(spec.custom, "numpy")[0][-1](pos))
    assert e == pytest.approx(_superposed_rmsd(pos[:8], ref_coords), abs=1e-6)
    assert e < 5e-3, f"aligned superposed RMSD should be ~0, got {e}"


def test_custom_rmsd_energy_parity_3backend(tmp_path):
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    rng = np.random.default_rng(13)
    n = 12
    pos = rng.standard_normal((n, 3)) * 3.0
    ref = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, ref)
    spec = _rmsd_spec(pdb, "rmsd(g, r1)", n=n)

    e_np = float(build_terms(spec.custom, "numpy")[0][-1](pos))
    e_t = float(
        build_terms(spec.custom, "torch")[0][-1](torch.tensor(pos, dtype=torch.float64))
    )
    e_j = float(build_terms(spec.custom, "jax")[0][-1](jnp.asarray(pos)))
    assert e_np > 0.0
    assert abs(e_np - e_t) < 1e-6 and abs(e_np - e_j) < 1e-6
    # compare to the ref the closure actually uses (read back from the PDB, 3-decimal)
    _, ref_coords = spec.custom[0].refs[("g", "r1")]
    assert e_np == pytest.approx(_superposed_rmsd(pos, ref_coords), abs=1e-5)


def test_custom_rmsd_minimize_converges(tmp_path):
    """rmsd(g, r1)**2 (target 0) drives the moving group's superposed RMSD onto the ref down
    under the torch CG (the same convergence contract as the built-in RMSD restraint)."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    rng = np.random.default_rng(17)
    n = 8
    ref = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, ref)
    spec = _rmsd_spec(pdb, "rmsd(g, r1)**2", n=n)

    tgt = ref @ np.eye(3) + rng.standard_normal((n, 3)) * 1.5  # ref + noise
    coords = torch.tensor(tgt.reshape(1, n, 3), dtype=torch.float64)
    before = _superposed_rmsd(tgt, ref)
    TorchRestraintOptimizer(spec, max_iter=200).minimize(coords)
    after = _superposed_rmsd(coords[0].numpy(), ref)
    assert after < before, f"rmsd did not decrease: {before:.3f} -> {after:.3f}"


def test_custom_rmsd_jax_minimize_nan_free(tmp_path):
    """rmsd(g, r1)**2 under the pure-jax CG minimizer (the AF3 lax.scan closure path): the
    ``jnp.linalg.svd`` inside the stop-gradient Kabsch stays finite + converges. This covers
    the SVD-under-jax risk on CPU, so the GPU AF3 path is de-risked before it runs."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer

    rng = np.random.default_rng(19)
    n = 8
    ref = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, ref)
    spec = _rmsd_spec(pdb, "rmsd(g, r1)**2", n=n)

    tgt = ref + rng.standard_normal((n, 3)) * 1.5
    coords = tgt.reshape(1, n, 3)
    out = np.asarray(make_minimizer(spec, max_iter=200)(jnp.asarray(coords), 0.0))
    assert np.all(np.isfinite(out)), "custom rmsd produced NaN/Inf under jax"
    assert _superposed_rmsd(out[0], ref) < _superposed_rmsd(tgt, ref)


def test_custom_rmsd_add_custom_fn_with_refs(tmp_path):
    """The programmatic add_custom(fn=..., refs=...) path forwards refs, so a Python ctx
    function calling ctx.rmsd(sel, ref) resolves its reference (regression: refs must flow
    through add_custom like selections, not only via a config entry)."""
    torch = pytest.importorskip("torch")
    from rgi_utils import CombinedRestraints

    rng = np.random.default_rng(23)
    n = 6
    ref = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, ref)

    def energy(ctx):
        return ctx.rmsd("chain A", "r1") ** 2

    restr = CombinedRestraints()
    restr.add_custom(fn=energy, refs={"r1": {"ref_pdb": str(pdb)}})
    restr.setup(_FakeAdapter(n), config={"gpu": False, "max_iter": 200})
    assert restr.spec.has_custom()
    tgt = ref + rng.standard_normal((n, 3)) * 1.2
    coords = torch.tensor(tgt.reshape(1, n, 3), dtype=torch.float64)
    before = _superposed_rmsd(tgt, ref)
    restr.minimize(coords, 0, 0.0)
    assert _superposed_rmsd(coords[0].numpy(), ref) < before


def test_custom_rmsd_undefined_ref_raises(tmp_path):
    ref = np.zeros((4, 3))
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, ref)
    with pytest.raises(ValueError, match="not defined in the entry's 'refs'"):
        _spec_from_entries(
            [
                {
                    "energy": "rmsd(g, missing)",
                    "selections": {"g": "chain A"},
                    "refs": {"r1": {"ref_pdb": str(pdb)}},
                }
            ],
            n=4,
        )


def test_custom_rmsd_first_arg_must_be_selection(tmp_path):
    """rmsd's first arg must be a bare selection (the ref is aligned to it at setup), not a
    coord expression — the resolve pass rejects rmsd(kabsch(...), ref)."""
    ref = np.zeros((4, 3))
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, ref)
    with pytest.raises(TypeError, match="bare selection identifier"):
        _spec_from_entries(
            [
                {
                    "energy": "rmsd(kabsch(A, B), r1)",
                    "selections": {"A": "resid 1", "B": "resid 2"},
                    "refs": {"r1": {"ref_pdb": str(pdb)}},
                }
            ],
            n=4,
        )
