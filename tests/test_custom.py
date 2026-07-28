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
      (test_custom_kabsch_grad_torch_vs_jax). `move` pinning is likewise stop-gradient and lives
      in test_custom_move.py with torch-vs-jax checks. Do NOT add either case under the FD test.
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
# rmsd(A, B): Kabsch-superposed RMSD against a per-call external reference structure
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
                "selections": {"g": "chain A", "r": "ref1 and chain A"},
                "refs": {
                    "ref1": {
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

    spec = _rmsd_spec(pdb, "rmsd(g, r)", n=n)
    tgt_idx, ref_coords = spec.custom[0].refs[("g", "r")]
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
    spec = _rmsd_spec(pdb, "rmsd(g, r)", n=n)

    e_np = float(build_terms(spec.custom, "numpy")[0][-1](pos))
    e_t = float(
        build_terms(spec.custom, "torch")[0][-1](torch.tensor(pos, dtype=torch.float64))
    )
    e_j = float(build_terms(spec.custom, "jax")[0][-1](jnp.asarray(pos)))
    assert e_np > 0.0
    assert abs(e_np - e_t) < 1e-6 and abs(e_np - e_j) < 1e-6
    # compare to the ref the closure actually uses (read back from the PDB, 3-decimal)
    _, ref_coords = spec.custom[0].refs[("g", "r")]
    assert e_np == pytest.approx(_superposed_rmsd(pos, ref_coords), abs=1e-5)


def test_custom_rmsd_minimize_converges(tmp_path):
    """rmsd(g, r)**2 (target 0) drives the moving group's superposed RMSD onto the ref down
    under the torch CG (the same convergence contract as the built-in RMSD restraint)."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    rng = np.random.default_rng(17)
    n = 8
    ref = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, ref)
    spec = _rmsd_spec(pdb, "rmsd(g, r)**2", n=n)

    tgt = ref @ np.eye(3) + rng.standard_normal((n, 3)) * 1.5  # ref + noise
    coords = torch.tensor(tgt.reshape(1, n, 3), dtype=torch.float64)
    before = _superposed_rmsd(tgt, ref)
    TorchRestraintOptimizer(spec, max_iter=200).minimize(coords)
    after = _superposed_rmsd(coords[0].numpy(), ref)
    assert after < before, f"rmsd did not decrease: {before:.3f} -> {after:.3f}"


def test_custom_rmsd_jax_minimize_nan_free(tmp_path):
    """rmsd(g, r)**2 under the pure-jax CG minimizer (the AF3 lax.scan closure path): the
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
    spec = _rmsd_spec(pdb, "rmsd(g, r)**2", n=n)

    tgt = ref + rng.standard_normal((n, 3)) * 1.5
    coords = tgt.reshape(1, n, 3)
    out = np.asarray(make_minimizer(spec, max_iter=200)(jnp.asarray(coords), 0.0))
    assert np.all(np.isfinite(out)), "custom rmsd produced NaN/Inf under jax"
    assert _superposed_rmsd(out[0], ref) < _superposed_rmsd(tgt, ref)


def test_custom_rmsd_add_custom_fn_with_refs(tmp_path):
    """The programmatic add_custom(fn=..., refs=...) path forwards refs, so a Python ctx
    function calling ctx.rmsd(prediction, reference_selection) resolves its reference (regression: refs must flow
    through add_custom like selections, not only via a config entry)."""
    torch = pytest.importorskip("torch")
    from rgi_utils import CombinedRestraints

    rng = np.random.default_rng(23)
    n = 6
    ref = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, ref)

    def energy(ctx):
        return ctx.rmsd("chain A", "ref1 and chain A") ** 2

    restr = CombinedRestraints()
    restr.add_custom(fn=energy, refs={"ref1": {"ref_pdb": str(pdb)}})
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
    with pytest.raises(ValueError, match="undefined reference name"):
        _spec_from_entries(
            [
                {
                    "energy": "rmsd(g, missing)",
                    "selections": {
                        "g": "chain A",
                        "missing": "ref2 and chain A",
                    },
                    "refs": {"ref1": {"ref_pdb": str(pdb)}},
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
                    "energy": "rmsd(kabsch(A, B), R)",
                    "selections": {
                        "A": "resid 1",
                        "B": "resid 2",
                        "R": "ref1 and resid 1",
                    },
                    "refs": {"ref1": {"ref_pdb": str(pdb)}},
                }
            ],
            n=4,
        )


# --------------------------------------------------------------------------------------
# reference-backed selections: fitted-reference coordinate blocks — reference atoms placed in the
# prediction frame by a Kabsch fit, usable in distance/angle/dihedral alongside prediction
# groups. Like kabsch/rmsd the whole transform is stop-gradient'd (the fit anchor gets NO
# gradient), so these are checked torch-vs-jax, NOT under the numpy-FD test.
# --------------------------------------------------------------------------------------
def _ref_spec(pdb, energy, *, selections, fit=True, pairing="identity", n=12):
    """A custom entry using a reference-backed selection; the fit anchor is prediction/ref resid 1..6 (identity)."""
    rdef = {"ref_pdb": str(pdb), "pairing": pairing}
    if fit:
        rdef["atom_selection_target_fit"] = "chain A and resid 1 to 6"
        rdef["atom_selection_ref_fit"] = "chain A and resid 1 to 6"
    return _spec_from_entries(
        [
            {
                "name": "refgeom",
                "energy": energy,
                "selections": {**selections, "B": "ref1 and chain A and resid 8"},
                "refs": {"ref1": rdef},
            }
        ],
        n=n,
    )


def _rigid_ref(pred, seed=31):
    """A reference that is a rigid transform of the WHOLE prediction (rotation + translation),
    so fitting its anchor recovers the inverse exactly -> each fitted ref atom == the matching
    prediction atom (rigid invariance, the analytic ground truth used below)."""
    th = 0.7
    rz = np.array(
        [[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]]
    )
    return pred @ rz.T + np.array([4.0, -2.0, 1.0])


def test_custom_ref_rigid_invariance_and_parity(tmp_path):
    """WITH a fit, B lands on the prediction's resid-8 atom (the reference is
    a rigid transform of the prediction, so the fit recovers the inverse), hence
    distance(A, B) == ||pred_A - pred_B||. Also checks 3-backend energy parity."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    rng = np.random.default_rng(31)
    n = 12
    pred = rng.standard_normal((n, 3)) * 3.0
    ref = _rigid_ref(pred)
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, ref)

    spec = _ref_spec(
        pdb,
        "distance(A, B)",
        selections={"A": "chain A and resid 10"},
        n=n,
    )
    # the fit anchor atoms (resid 1..6 -> index 0..5) + the measured group (resid 10 -> 9) are
    # all in active_sites (the closure gathers the anchor to compute the per-eval fit).
    assert set(spec.active_sites.tolist()) == {0, 1, 2, 3, 4, 5, 9}
    active = pred[spec.active_sites]
    expect = float(np.linalg.norm(pred[9] - pred[7]))  # resid 10 vs (fitted) resid 8

    e_np = float(build_terms(spec.custom, "numpy")[0][-1](active))
    e_t = float(
        build_terms(spec.custom, "torch")[0][-1](
            torch.tensor(active, dtype=torch.float64)
        )
    )
    e_j = float(build_terms(spec.custom, "jax")[0][-1](jnp.asarray(active)))
    assert e_np == pytest.approx(expect, abs=3e-3), (
        f"{e_np} vs {expect}"
    )  # PDB rounding
    assert abs(e_np - e_t) < 1e-6 and abs(e_np - e_j) < 1e-6


def test_custom_ref_grad_torch_vs_jax_and_anchor_frozen(tmp_path):
    """The whole fit transform is stop-gradient'd, so (a) torch and jax grads agree while a
    numpy-FD is inapplicable (same carve-out as kabsch/rmsd) and (b) the fit anchor atoms get
    ZERO gradient — the restraint pulls only the measured prediction group, never the anchor."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    rng = np.random.default_rng(37)
    n = 12
    pred = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, _rigid_ref(pred))
    spec = _ref_spec(
        pdb,
        "harmonic(distance(A, B), 2.0)",
        selections={"A": "chain A and resid 10"},
        n=n,
    )
    active = pred[spec.active_sites]  # rows 0..5 = anchor, row 6 = measured group A

    pt = torch.tensor(active, dtype=torch.float64, requires_grad=True)
    build_terms(spec.custom, "torch")[0][-1](pt).backward()
    g_t = pt.grad.numpy()
    jax_clo = build_terms(spec.custom, "jax")[0][-1]
    g_j = np.asarray(jax.grad(lambda x: jax_clo(x))(jnp.asarray(active)))
    assert np.allclose(g_t, g_j, atol=1e-6), f"torch vs jax {np.abs(g_t - g_j).max()}"
    assert np.allclose(g_t[:6], 0.0, atol=1e-9), "fit anchor must receive no gradient"
    assert np.linalg.norm(g_t[6]) > 1e-6, "the measured group must receive a gradient"


def test_custom_ref_no_fit_own_frame(tmp_path):
    """Without fit selections the reference block is used in its OWN frame (still fixed), so
    distance(A, B) == ||pred_A - ref_B||."""
    rng = np.random.default_rng(41)
    n = 12
    pred = rng.standard_normal((n, 3)) * 3.0
    ref = _rigid_ref(pred)
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, ref)
    spec = _ref_spec(
        pdb,
        "distance(A, B)",
        selections={"A": "chain A and resid 10"},
        fit=False,
        n=n,
    )
    assert not spec.custom[0].ref_fits  # no fit resolved
    active = pred[spec.active_sites]
    e = float(build_terms(spec.custom, "numpy")[0][-1](active))
    assert e == pytest.approx(float(np.linalg.norm(pred[9] - ref[7])), abs=3e-3)


def test_custom_ref_minimize_converges(tmp_path):
    """harmonic(distance(A, B), 5.0) under the torch CG drives the measured prediction
    group to 5 A from the fitted reference landmark."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    rng = np.random.default_rng(43)
    n = 12
    pred = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, _rigid_ref(pred))
    energy = "harmonic(distance(A, B), 5.0)"
    spec = _ref_spec(pdb, energy, selections={"A": "chain A and resid 10"}, n=n)

    # the optimizer takes the FULL coordinate tensor and slices active_sites out of it.
    coords = torch.tensor(pred.reshape(1, n, 3), dtype=torch.float64)
    TorchRestraintOptimizer(spec, max_iter=300).minimize(coords)
    # measure the achieved distance with a distance-only closure on the minimized active coords
    dspec = _ref_spec(
        pdb,
        "distance(A, B)",
        selections={"A": "chain A and resid 10"},
        n=n,
    )
    final = coords[0].numpy()[dspec.active_sites]
    d = float(build_terms(dspec.custom, "numpy")[0][-1](final))
    assert d == pytest.approx(5.0, abs=0.1), f"distance did not reach 5.0: {d}"


def test_custom_ref_jax_minimize_nan_free(tmp_path):
    """The reference-selection fit (jnp.linalg.svd inside the stop-gradient Kabsch) stays finite + converges
    under the pure-jax CG (the AF3 lax.scan closure path) — de-risks the GPU AF3 run."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer

    rng = np.random.default_rng(47)
    n = 12
    pred = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, _rigid_ref(pred))
    energy = "harmonic(distance(A, B), 5.0)"
    spec = _ref_spec(pdb, energy, selections={"A": "chain A and resid 10"}, n=n)

    coords = pred.reshape(
        1, n, 3
    )  # FULL coordinate tensor (the minimizer slices active_sites)
    out = np.asarray(make_minimizer(spec, max_iter=300)(jnp.asarray(coords), 0.0))
    assert np.all(np.isfinite(out)), "reference selection produced NaN/Inf under jax"
    dspec = _ref_spec(
        pdb,
        "distance(A, B)",
        selections={"A": "chain A and resid 10"},
        n=n,
    )
    d = float(build_terms(dspec.custom, "numpy")[0][-1](out[0][dspec.active_sites]))
    assert d == pytest.approx(5.0, abs=0.2), f"jax distance did not reach 5.0: {d}"


def test_custom_ref_all_reference_raises(tmp_path):
    """A restraint whose every group is reference-backed is a constant (the fitted ref is
    stop-gradient'd) -> zero gradient; it must raise rather than silently do nothing."""
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, np.zeros((6, 3)))
    with pytest.raises(ValueError, match="only reference atoms"):
        _spec_from_entries(
            [
                {
                    "energy": "distance(A, B)",
                    "selections": {
                        "A": "ref1 and resid 1",
                        "B": "ref1 and resid 2",
                    },
                    "refs": {"ref1": {"ref_pdb": str(pdb), "pairing": "identity"}},
                }
            ],
            n=6,
        )


def test_custom_ref_fit_needs_three_atoms(tmp_path):
    """A Kabsch fit needs >= 3 anchor atoms; fewer raises loudly (an under-determined
    superposition would otherwise give a garbage rotation)."""
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, np.arange(36, dtype=float).reshape(12, 3))
    with pytest.raises(ValueError, match="at least 3"):
        _spec_from_entries(
            [
                {
                    "energy": "distance(A, B)",
                    "selections": {
                        "A": "chain A and resid 10",
                        "B": "ref1 and resid 8",
                    },
                    "refs": {
                        "ref1": {
                            "ref_pdb": str(pdb),
                            "pairing": "identity",
                            "atom_selection_target_fit": "chain A and resid 1 to 2",
                            "atom_selection_ref_fit": "chain A and resid 1 to 2",
                        }
                    },
                }
            ],
            n=12,
        )


def test_custom_ref_undefined_raises(tmp_path):
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, np.zeros((6, 3)))
    with pytest.raises(ValueError, match="undefined reference name"):
        _spec_from_entries(
            [
                {
                    "energy": "distance(A, B)",
                    "selections": {
                        "A": "chain A and resid 4",
                        "B": "ref2 and resid 2",
                    },
                    "refs": {"ref1": {"ref_pdb": str(pdb), "pairing": "identity"}},
                }
            ],
            n=6,
        )


# --------------------------------------------------------------------------------------
# Built-in reference-group distance/angle/dihedral (ref_geom): a distance/angle/dihedral
# config entry carrying reference keys is routed to a kind="ref_geom" CustomSpec closure.
# Prediction groups use a RIGID centroid (weight=1 moves a large group), reference groups are
# fitted-and-fixed. Grad is torch-vs-jax (the rigid-centroid + stop-gradient carve-out).
# --------------------------------------------------------------------------------------
def _refgeom_spec(cfgdict, n=12):
    cfg = RestraintsConfig.from_dict(cfgdict)
    for cd in cfg.custom_data:  # reference-group entries are routed into custom_data
        cd.resolve_sites(_FakeAdapter(n))
    return build_spec(custom_restraints=cfg.custom_data)


def _refgeom_dist_entry(pdb, sel1, sel2_ref, block, *, fit=True):
    ref_def = {"ref_pdb": str(pdb), "pairing": "identity"}
    if fit:
        ref_def["atom_selection_target_fit"] = "chain A and resid 1 to 6"
        ref_def["atom_selection_ref_fit"] = "chain A and resid 1 to 6"
    entry = {
        "atom_selection1": sel1,
        "atom_selection2": f"ref1 and {sel2_ref}",
        "refs": {"ref1": ref_def},
        **block,
    }
    return {"distance_restraints_config": [entry]}


def test_refgeom_distance_rigid_and_parity(tmp_path):
    """reference-group distance: group1 prediction, group2 fitted reference. The reference is a
    rigid transform of the prediction, so the fitted ref landmark lands on the matching
    prediction atom -> the measured distance == ||pred_A - pred_B||. Also 3-backend parity."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    rng = np.random.default_rng(51)
    n = 12
    pred = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, _rigid_ref(pred))
    spec = _refgeom_spec(
        _refgeom_dist_entry(
            pdb,
            "chain A and resid 10",
            "chain A and resid 8",
            {"harmonic": {"target_distance": 4.0}},
        ),
        n=n,
    )
    assert spec.custom[0].kind == "ref_geom" and spec.custom[0].geom == "distance"
    active = pred[spec.active_sites]
    d = float(np.linalg.norm(pred[9] - pred[7]))  # resid 10 vs (fitted) resid 8

    e_np = float(build_terms(spec.custom, "numpy")[0][-1](active))
    e_t = float(
        build_terms(spec.custom, "torch")[0][-1](
            torch.tensor(active, dtype=torch.float64)
        )
    )
    e_j = float(build_terms(spec.custom, "jax")[0][-1](jnp.asarray(active)))
    # measured distance = 4 + sqrt(energy) (harmonic (dist-4)^2, dist>4); comparing the distance
    # itself is robust to the ref PDB's 3-decimal rounding (squaring would amplify it).
    assert 4.0 + np.sqrt(e_np) == pytest.approx(d, abs=5e-3)
    assert abs(e_np - e_t) < 1e-6 and abs(e_np - e_j) < 1e-6


def test_refgeom_rigid_centroid_Ntimes_fd_and_anchor_frozen(tmp_path):
    """Decisive rigid-centroid + stop-gradient test. numpy has no autodiff, so a numpy
    finite-difference is the TRUE gradient (plain centroid, f'(c)/N per atom, and it DOES see
    the fit's dependence on the anchor). torch autodiff instead gives (a) N× that on the
    prediction group (the rigid-centroid distortion that moves a whole group at weight=1) and
    (b) ZERO on the fit anchor (the whole fit transform is stop-gradient'd)."""
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(53)
    n = 12
    pred = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, _rigid_ref(pred))
    spec = _refgeom_spec(
        _refgeom_dist_entry(
            pdb,
            "chain A and resid 9 to 12",  # a 4-atom prediction group
            "chain A and resid 8",
            {"harmonic": {"target_distance": 3.0}},
        ),
        n=n,
    )
    active = pred[spec.active_sites]
    globals_ = spec.active_sites.tolist()
    pred_rows = [
        i for i, g in enumerate(globals_) if g in (8, 9, 10, 11)
    ]  # resid 9..12
    anchor_rows = [i for i, g in enumerate(globals_) if g in (0, 1, 2, 3, 4, 5)]
    n_group = len(pred_rows)
    assert n_group == 4

    np_clo = build_terms(spec.custom, "numpy")[0][-1]
    g_fd = _fd_grad(
        lambda x: float(np_clo(x.reshape(active.shape))), active.flatten()
    ).reshape(active.shape)

    pt = torch.tensor(active, dtype=torch.float64, requires_grad=True)
    build_terms(spec.custom, "torch")[0][-1](pt).backward()
    g_t = pt.grad.numpy()

    for r in pred_rows:  # rigid centroid: autodiff == N × finite-difference
        assert np.allclose(g_t[r], n_group * g_fd[r], atol=1e-4), (
            f"row {r}: torch {g_t[r]} vs N*FD {n_group * g_fd[r]}"
        )
    for r in (
        anchor_rows
    ):  # fit anchor: autodiff 0, but FD sees the fit dependence (nonzero)
        assert np.allclose(g_t[r], 0.0, atol=1e-9), (
            f"anchor row {r} not frozen: {g_t[r]}"
        )
    assert np.abs(g_fd[anchor_rows]).max() > 1e-4, (
        "the finite-difference should see the fit's dependence on the anchor"
    )


def test_refgeom_large_group_converges(tmp_path):
    """A reference-group distance with a large (30-atom) prediction group converges to its target
    at weight=1 under the torch CG — the rigid centroid gives the whole group a full-step pull."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    rng = np.random.default_rng(55)
    n = 40
    pred = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, _rigid_ref(pred))
    spec = _refgeom_spec(
        _refgeom_dist_entry(
            pdb,
            "chain A and resid 10 to 39",  # 30-atom prediction group
            "chain A and resid 8",
            {"harmonic": {"target_distance": 12.0}},
        ),
        n=n,
    )
    clo = build_terms(spec.custom, "numpy")[0][-1]
    before = float(clo(pred[spec.active_sites]))
    coords = torch.tensor(pred.reshape(1, n, 3), dtype=torch.float64)  # FULL tensor
    TorchRestraintOptimizer(spec, max_iter=400).minimize(coords)
    after = float(clo(coords[0].numpy()[spec.active_sites]))
    assert after < 0.05 and after < 0.02 * before, (
        f"did not converge: {before} -> {after}"
    )


def test_refgeom_angle_and_dihedral_energy(tmp_path):
    """reference-group angle (3 groups) and dihedral (4 groups) compute the group-centroid angle /
    dihedral with a fitted reference group and apply the degrees->radians harmonic penalty."""
    rng = np.random.default_rng(57)
    n = 12
    pred = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    pdb2 = tmp_path / "ref2.pdb"
    pdb3 = tmp_path / "ref3.pdb"
    _write_ca_pdb(pdb, _rigid_ref(pred))
    _write_ca_pdb(pdb2, _rigid_ref(pred) + np.array([2.0, -1.0, 3.0]))
    _write_ca_pdb(pdb3, _rigid_ref(pred) + np.array([-3.0, 2.0, 1.0]))

    aspec = _refgeom_spec(
        {
            "angle_restraints_config": [
                {
                    "atom_selection1": "chain A and resid 10",
                    "atom_selection2": "ref1 and chain A and resid 11",
                    "atom_selection3": "ref2 and chain A and resid 8",
                    "refs": {
                        "ref1": {
                            "ref_pdb": str(pdb),
                            "pairing": "identity",
                            "atom_selection_target_fit": "chain A and resid 1 to 6",
                            "atom_selection_ref_fit": "chain A and resid 1 to 6",
                        },
                        "ref2": {
                            "ref_pdb": str(pdb2),
                            "pairing": "identity",
                            "atom_selection_target_fit": "chain A and resid 1 to 6",
                            "atom_selection_ref_fit": "chain A and resid 1 to 6",
                        },
                    },
                    "harmonic": {"target_angle": 90.0},
                }
            ]
        },
        n=n,
    )
    ea = float(build_terms(aspec.custom, "numpy")[0][-1](pred[aspec.active_sites]))
    v1, v2 = pred[9] - pred[10], pred[7] - pred[10]  # fitted resid 8 -> pred[7]
    ang = np.arccos(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    assert ea == pytest.approx((ang - np.radians(90.0)) ** 2, abs=5e-3)

    dspec = _refgeom_spec(
        {
            "dihedral_restraints_config": [
                {
                    "atom_selection1": "chain A and resid 10",
                    "atom_selection2": "ref1 and chain A and resid 11",
                    "atom_selection3": "ref2 and chain A and resid 12",
                    "atom_selection4": "ref3 and chain A and resid 8",
                    "refs": {
                        "ref1": {
                            "ref_pdb": str(pdb),
                            "pairing": "identity",
                            "atom_selection_target_fit": "chain A and resid 1 to 6",
                            "atom_selection_ref_fit": "chain A and resid 1 to 6",
                        },
                        "ref2": {
                            "ref_pdb": str(pdb2),
                            "pairing": "identity",
                            "atom_selection_target_fit": "chain A and resid 1 to 6",
                            "atom_selection_ref_fit": "chain A and resid 1 to 6",
                        },
                        "ref3": {
                            "ref_pdb": str(pdb3),
                            "pairing": "identity",
                            "atom_selection_target_fit": "chain A and resid 1 to 6",
                            "atom_selection_ref_fit": "chain A and resid 1 to 6",
                        },
                    },
                    "harmonic": {"target_dihedral": 45.0},
                }
            ]
        },
        n=n,
    )
    assert dspec.custom[0].geom == "dihedral"
    ed = float(build_terms(dspec.custom, "numpy")[0][-1](pred[dspec.active_sites]))
    assert ed >= 0.0 and np.isfinite(ed)


def test_refgeom_jax_minimize_nan_free(tmp_path):
    """ref_geom under the pure-jax CG (AF3 lax.scan path): the Kabsch SVD stays finite and the
    prediction group converges to the target distance from the fitted reference."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer

    rng = np.random.default_rng(59)
    n = 12
    pred = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, _rigid_ref(pred))
    spec = _refgeom_spec(
        _refgeom_dist_entry(
            pdb,
            "chain A and resid 9 to 12",
            "chain A and resid 8",
            {"harmonic": {"target_distance": 6.0}},
        ),
        n=n,
    )
    coords = pred.reshape(1, n, 3)  # FULL tensor (the minimizer slices active_sites)
    out = np.asarray(make_minimizer(spec, max_iter=300)(jnp.asarray(coords), 0.0))
    assert np.all(np.isfinite(out)), "ref_geom produced NaN/Inf under jax"
    clo = build_terms(spec.custom, "numpy")[0][-1]
    assert float(clo(out[0][spec.active_sites])) < 0.05, (
        "jax did not converge the ref_geom distance"
    )


def test_refgeom_config_validation(tmp_path):
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, np.zeros((6, 3)))

    ref_def = {"ref_pdb": str(pdb), "pairing": "identity"}

    with pytest.raises(ValueError, match="at least one group must select prediction"):
        _refgeom_spec(
            {
                "distance_restraints_config": [
                    {
                        "atom_selection1": "ref1 and resid 1",
                        "atom_selection2": "ref1 and resid 2",
                        "refs": {"ref1": ref_def},
                    }
                ]
            },
            n=6,
        )

    with pytest.raises(ValueError, match="move selects reference group"):
        _refgeom_spec(
            {
                "distance_restraints_config": [
                    {
                        "atom_selection1": "resid 5",
                        "atom_selection2": "ref1 and resid 2",
                        "refs": {"ref1": ref_def},
                        "move": 2,
                        "harmonic": {"target_distance": 3.0},
                    }
                ]
            },
            n=6,
        )

    with pytest.raises(ValueError, match="at most 1 distinct reference"):
        _refgeom_spec(
            {
                "distance_restraints_config": [
                    {
                        "atom_selection1": "ref1 and resid 1",
                        "atom_selection2": "ref2 and resid 2",
                        "refs": {
                            "ref1": ref_def,
                            "ref2": ref_def,
                        },
                    }
                ]
            },
            n=6,
        )


def test_refgeom_combined_lifecycle(tmp_path, capsys):
    """A ref_geom entry through the FULL CombinedRestraints path (setup -> minimize -> finalize)
    that boltz/AF3 actually use — not just the closure directly. Confirms the config routes to
    a ref_geom CustomSpec, the verbose setup log counts it (ref_distance=1), and minimize on
    gpu:false (torch on CPU) converges the prediction group to the target from the fitted ref."""
    torch = pytest.importorskip("torch")
    from rgi_utils import CombinedRestraints

    rng = np.random.default_rng(61)
    n = 12
    pred = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_ca_pdb(pdb, _rigid_ref(pred))
    config = {
        "gpu": False,
        "verbose": True,
        "max_iter": 300,
        "distance_restraints_config": [
            {
                "refs": {
                    "ref1": {
                        "ref_pdb": str(pdb),
                        "pairing": "identity",
                        "atom_selection_target_fit": "chain A and resid 1 to 6",
                        "atom_selection_ref_fit": "chain A and resid 1 to 6",
                    }
                },
                "atom_selection1": "chain A and resid 10",
                "atom_selection2": "ref1 and chain A and resid 8",
                "harmonic": {"target_distance": 5.0},
            }
        ],
    }
    restr = CombinedRestraints()
    restr.setup(_FakeAdapter(n), config=config)
    assert any(getattr(c, "kind", "") == "ref_geom" for c in restr.spec.custom)
    setup_log = capsys.readouterr().out
    assert "ref_distance=1" in setup_log, setup_log  # the setup-log breakdown counts it

    coords = torch.tensor(pred.reshape(1, n, 3), dtype=torch.float64)
    restr.minimize(coords, 0, 0.0)
    restr.finalize(
        coords, 0
    )  # the numpy _custom_breakdown finalize path must run clean

    dspec = _ref_spec(
        pdb,
        "distance(A, B)",
        selections={"A": "chain A and resid 10"},
        n=n,
    )
    d = float(
        build_terms(dspec.custom, "numpy")[0][-1](coords[0].numpy()[dspec.active_sites])
    )
    assert d == pytest.approx(5.0, abs=0.1), f"lifecycle did not converge: {d}"


# --------------------------------------------------------------------------------------
# (e) plane(): the best-fit-plane primitive in the DSL + the ref_geom plane closure
# --------------------------------------------------------------------------------------
def _puckered_ring(lift: float = 0.4, n: int = 6) -> np.ndarray:
    """A hexagon in z=0 with every other atom lifted -> out-of-plane RMS = lift/2."""
    ring = np.array(
        [
            [np.cos(t), np.sin(t), 0.0]
            for t in np.linspace(0, 2 * np.pi, n, endpoint=False)
        ]
    )
    ring[::2, 2] += lift
    return ring


def test_custom_plane_energy_parity_3backend():
    """plane(A) = A's out-of-plane RMS. Its normal is stop-gradient'd (like kabsch/rmsd),
    so the VALUE agrees on all three backends but the gradient is compared torch-vs-jax
    (test_custom_plane_grad_parity_torch_jax) rather than against a numpy FD."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    spec = _spec_from_entries(
        [
            {
                "name": "flat",
                "energy": "plane(A)**2",
                "selections": {"A": "resid 1 to 6"},
            }
        ]
    )
    pos = np.zeros((spec.n_active, 3))
    pos[: len(spec.active_sites)] = _puckered_ring()

    e_np = float(build_terms(spec.custom, "numpy")[0][-1](pos))
    e_t = float(
        build_terms(spec.custom, "torch")[0][-1](torch.tensor(pos, dtype=torch.float64))
    )
    e_j = float(build_terms(spec.custom, "jax")[0][-1](jnp.asarray(pos)))
    assert e_np == pytest.approx(0.2**2, abs=1e-6)  # (lift/2)^2
    assert abs(e_np - e_t) < 1e-6 and abs(e_np - e_j) < 1e-6


def test_custom_plane_grad_parity_torch_jax():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    spec = _spec_from_entries(
        [
            {
                "name": "flat",
                "energy": "plane(A) + plane(A, B)",
                "selections": {"A": "resid 1 to 6", "B": "resid 7 to 12"},
            }
        ]
    )
    rng = np.random.default_rng(5)
    pos = np.vstack([_puckered_ring(), _puckered_ring(0.2) + [4.0, 0.0, 0.3]])
    pos += rng.standard_normal(pos.shape) * 0.05

    pt = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    build_terms(spec.custom, "torch")[0][-1](pt).backward()
    jax_clo = build_terms(spec.custom, "jax")[0][-1]
    g_j = np.asarray(jax.grad(lambda x: jax_clo(x))(jnp.asarray(pos)))
    assert np.allclose(pt.grad.numpy(), g_j, atol=1e-6), (
        f"max|d|={np.abs(pt.grad.numpy() - g_j).max()}"
    )


def test_custom_plane_move_pins_the_other_selection():
    """`move` pins an unlisted selection, so a two-argument plane(A, B) with move: A
    leaves B's atoms without gradient — the plane B defines is then genuinely fixed."""
    torch = pytest.importorskip("torch")

    spec = _spec_from_entries(
        [
            {
                "name": "onto_b",
                "energy": "plane(A, B)**2",
                "selections": {"A": "resid 1 to 6", "B": "resid 7 to 12"},
                "move": "A",
            }
        ]
    )
    pos = np.vstack([_puckered_ring() + [0.0, 0.0, 0.8], _puckered_ring(0.0)])
    pt = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    build_terms(spec.custom, "torch")[0][-1](pt).backward()
    grad = pt.grad.numpy()
    assert not np.allclose(grad[:6], 0.0)
    assert np.allclose(grad[6:], 0.0)  # B pinned by `move`


def test_custom_plane_torch_minimize_flattens():
    """The closure actually flattens the selection under the torch CG."""
    torch = pytest.importorskip("torch")
    from rgi_utils.combined import CombinedRestraints

    pos = np.zeros((12, 3))
    pos[:6] = _puckered_ring(0.6)
    restr = CombinedRestraints()
    restr.setup(
        _FakeAdapter(12),
        1,
        {
            "gpu": False,
            "max_iter": 400,
            "custom_restraints_config": [
                {
                    "name": "flat",
                    "energy": "plane(A)**2",
                    "selections": {"A": "resid 1 to 6"},
                }
            ],
        },
    )
    out = restr.minimize(torch.tensor(pos, dtype=torch.float64), 0, 0.0)
    x = out.detach().numpy()[:6]
    x0 = x - x.mean(0)
    _w, vecs = np.linalg.eigh(x0.T @ x0)
    assert float(np.sqrt(((x0 @ vecs[:, 0]) ** 2).mean())) < 1e-3


def test_custom_plane_resolve_records_selections():
    """The setup-time resolve pass must see BOTH arguments, else the second selection is
    never resolved and the closure raises at evaluation time."""
    from rgi_utils.custom.context import ResolveContext
    from rgi_utils.custom.dsl import eval_formula, parse_formula

    ctx = ResolveContext()
    eval_formula(parse_formula("plane(A) + plane(B, C)"), ctx)
    assert ctx.selections == ["A", "B", "C"]


def test_refgeom_plane_pulls_prediction_onto_the_reference_plane(tmp_path, capsys):
    """A ref-anchored plane entry: the plane comes from the REFERENCE atoms alone (fixed),
    and the prediction group is pulled onto it. Distinct from the array-path plane term,
    which fits the plane to the prediction's own atoms."""
    torch = pytest.importorskip("torch")
    from rgi_utils.combined import CombinedRestraints

    n = 12
    ring = _puckered_ring(0.0)  # flat hexagon in z=0, resid 1..6 of the ref
    _write_ca_pdb(tmp_path / "ref.pdb", ring)

    pred = np.zeros((n, 3))
    pred[:6] = _puckered_ring(0.4) + [0.0, 0.0, 0.9]  # lifted + puckered

    restr = CombinedRestraints()
    restr.setup(
        _FakeAdapter(n),
        1,
        {
            "verbose": True,
            "gpu": False,
            "max_iter": 400,
            "plane_restraints_config": [
                {
                    "atom_selection1": "resid 1 to 6",
                    "atom_selection2": "ref1 and resid 1 to 6",
                    "refs": {
                        "ref1": {
                            "ref_pdb": str(tmp_path / "ref.pdb"),
                            "pairing": "identity",
                        }
                    },
                }
            ],
        },
    )
    assert any(getattr(c, "geom", "") == "plane" for c in restr.spec.custom)
    assert restr.spec.group_plane is None  # ref entries do NOT take the array path
    assert "ref_plane=1" in capsys.readouterr().out

    out = restr.minimize(torch.tensor(pred, dtype=torch.float64), 0, 0.0)
    z = out.detach().numpy()[:6, 2]
    assert np.abs(z).max() < 1e-2, f"not pulled onto the reference plane: {z}"


def test_refgeom_plane_needs_three_reference_atoms(tmp_path):
    """A 1- or 2-atom reference selection leaves the plane normal undefined -> raise
    rather than silently optimising a meaningless quantity."""
    _write_ca_pdb(tmp_path / "ref.pdb", np.zeros((6, 3)))
    cfg = RestraintsConfig.from_dict(
        {
            "plane_restraints_config": [
                {
                    "atom_selection1": "resid 1 to 6",
                    "atom_selection2": "ref1 and resid 1 to 2",
                    "refs": {
                        "ref1": {
                            "ref_pdb": str(tmp_path / "ref.pdb"),
                            "pairing": "identity",
                        }
                    },
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="best-fit plane needs at least 3"):
        cfg.custom_data[0].resolve_sites(_FakeAdapter(12))


def test_refgeom_plane_pools_multiple_groups(tmp_path):
    """A ref-anchored plane with SEVERAL prediction groups and SEVERAL reference groups:
    each side is pooled along the atom axis (``ops.concat_atoms``), which the single-group
    entries above short-circuit past. Also the only >2-group ``RefGeomData`` path."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    n = 12
    # reference: 12 atoms in the z=0 plane, so ANY subset of >= 3 defines the same plane
    ref = np.zeros((n, 3))
    ref[:, 0] = np.linspace(0.0, 6.0, n)
    ref[:, 1] = np.tile([0.0, 1.0, 2.0, 1.0], 3)
    _write_ca_pdb(tmp_path / "ref.pdb", ref)

    cfg = RestraintsConfig.from_dict(
        {
            "plane_restraints_config": [
                {
                    "atom_selection1": "resid 1 to 3",  # prediction group 1
                    "atom_selection2": "resid 4 to 6",  # prediction group 2 -> pooled
                    "atom_selection3": "ref1 and resid 7 to 9",  # reference group 1
                    "atom_selection4": "ref1 and resid 10 to 12",  # reference 2 -> pooled
                    "refs": {
                        "ref1": {
                            "ref_pdb": str(tmp_path / "ref.pdb"),
                            "pairing": "identity",
                        }
                    },
                }
            ]
        }
    )
    (rg,) = cfg.custom_data
    assert rg.n_groups == 4
    rg.resolve_sites(_FakeAdapter(n))
    spec = build_spec(custom_restraints=cfg.custom_data)

    rng = np.random.default_rng(7)
    pos = ref[spec.active_sites] + rng.standard_normal((spec.n_active, 3)) * 0.3
    pos[:, 2] += 0.8  # lift the prediction atoms off the reference plane

    e_t = float(
        build_terms(spec.custom, "torch")[0][-1](torch.tensor(pos, dtype=torch.float64))
    )
    e_j = float(build_terms(spec.custom, "jax")[0][-1](jnp.asarray(pos)))
    e_np = float(build_terms(spec.custom, "numpy")[0][-1](pos))
    assert e_np > 0.0
    assert abs(e_np - e_t) < 1e-6 and abs(e_np - e_j) < 1e-6

    pt = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    build_terms(spec.custom, "torch")[0][-1](pt).backward()
    jax_clo = build_terms(spec.custom, "jax")[0][-1]
    g_j = np.asarray(jax.grad(lambda x: jax_clo(x))(jnp.asarray(pos)))
    assert np.allclose(pt.grad.numpy(), g_j, atol=1e-6)
