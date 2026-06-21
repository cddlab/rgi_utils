"""Verification harness for the custom-restraint registry (the extension mechanism).

This is a first-class deliverable, not an afterthought: a registered restraint can pass
the numpy reference yet still break on the two hardest backends (the torch.compile GPU
path and jax inside ``lax.scan``) in ways CPU-only checks miss. So every registered
restraint — the built-in ``custom`` (pattern B) AND a synthetic external restraint
(pattern A) — is driven through:

  (a) 3-backend energy + gradient parity (numpy reference vs torch vs jax);
  (b) the torch optimizer / compiled GPU path (a ``@pytest.mark.gpu`` minimize on CUDA,
      plus the CPU pre-gate test that proves a custom per-entry term is folded — the
      "ungated on the compiled GPU path" footgun);
  (c) the jax minimizer (the AF3 ``lax.scan`` closure), NaN-free.

Plus the config-layer contract (a registered section extends the whitelist; an unknown
section still RAISES).
"""

from __future__ import annotations

import numpy as np
import pytest

from rgi_utils import registry
from rgi_utils.config import RestraintsConfig
from rgi_utils.energy import numpy_energy
from rgi_utils.featurizer import build_spec


class _FakeAdapter:
    """Minimal adapter: ``n`` protein atoms, chain A, one atom per residue (resid 1..n,
    global index 0..n-1), named CA — enough for the selection DSL used below."""

    def __init__(self, n: int = 12) -> None:
        self.n = n

    def iter_atoms(self):
        from types import SimpleNamespace

        for i in range(self.n):
            yield SimpleNamespace(
                chain="A",
                resid=i + 1,
                index=i,
                name="CA",
                mol_type="protein",
                resname="ALA",
            )


def _build(config: dict, n: int = 12):
    """config dict -> resolved spec via the REAL config + featurizer path."""
    cfg = RestraintsConfig.from_dict(config)
    ad = _FakeAdapter(n)
    for items in cfg.registered_data.values():
        for it in items:
            it.resolve_sites(ad)
    return build_spec(registered_restraints=cfg.registered_data)


def _custom_spec():
    """One custom restraint per measure (distance / angle / dihedral / radius_of_gyration),
    all groups free so the energy is non-trivial and torch-vs-jax grad parity holds."""
    return _build(
        {
            "custom_restraints_config": [
                {
                    "measure": "distance",
                    "atom_selection1": "resid 1 to 2",
                    "atom_selection2": "resid 3 to 4",
                    "form": "harmonic",
                    "target": 3.0,
                    "weight": 1.0,
                    "move": "all",
                },
                {
                    "measure": "angle",
                    "atom_selection1": "resid 5",
                    "atom_selection2": "resid 6",
                    "atom_selection3": "resid 7",
                    "form": "harmonic",
                    "target": 70.0,
                    "weight": 0.5,
                    "move": "all",
                },
                {
                    "measure": "dihedral",
                    "atom_selection1": "resid 8",
                    "atom_selection2": "resid 9",
                    "atom_selection3": "resid 10",
                    "atom_selection4": "resid 11",
                    "form": "harmonic",
                    "target": 40.0,
                    "weight": 0.8,
                    "move": "all",
                },
                {
                    "measure": "radius_of_gyration",
                    "atom_selection": "resid 1 to 5",
                    "form": "harmonic",
                    "target": 2.0,
                    "weight": 0.3,
                },
            ]
        }
    )


def _positions(spec, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((spec.n_active, 3)) * 2.0


def _fd_grad(f, x, eps: float = 1e-6):
    x = np.asarray(x, dtype=np.float64).copy()
    g = np.zeros_like(x)
    for i in range(x.size):
        orig = x.flat[i]
        x.flat[i] = orig + eps
        fp = f(x)
        x.flat[i] = orig - eps
        fm = f(x)
        x.flat[i] = orig
        g.flat[i] = (fp - fm) / (2.0 * eps)
    return g


# --------------------------------------------------------------------------------------
# (a) 3-backend energy + gradient parity for the built-in custom restraint
# --------------------------------------------------------------------------------------
def test_custom_config_builds_spec():
    """The config + featurizer path produces a spec.registered['custom'] with one row per
    measure, and the whole structure is its active_sites union."""
    spec = _custom_spec()
    assert "custom" in spec.registered
    ca = spec.registered["custom"]
    assert ca.measure_type.tolist() == [0, 1, 2, 3]  # distance/angle/dihedral/rg
    assert spec.has_registered()
    assert spec.is_active()
    # atoms referenced: resid 1..11 -> indices 0..10
    assert spec.n_active == 11


def test_custom_energy_parity_3backend():
    """numpy reference == torch == jax for the custom restraint energy (all measures)."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy

    spec = _custom_spec()
    pos = _positions(spec)
    e_np = float(numpy_energy.total_energy(pos, numpy_energy.prepare_spec(spec)))
    e_t = float(
        torch_energy.total_energy(
            torch.tensor(pos, dtype=torch.float64),
            torch_energy.prepare_spec(spec, dtype=torch.float64),
        )
    )
    e_j = float(
        jax_energy.total_energy(jnp.asarray(pos), jax_energy.prepare_spec(spec))
    )
    assert e_np > 0.0
    assert abs(e_np - e_t) < 1e-6, f"numpy={e_np} torch={e_t}"
    assert abs(e_np - e_j) < 1e-6, f"numpy={e_np} jax={e_j}"


def test_custom_grad_parity_torch_jax():
    """torch autograd == jax grad for the custom restraint. (distance/angle/dihedral use
    the rigid-translation _move_centroid, whose N x-rescaled gradient does NOT match a
    numpy finite-difference — the same carve-out as the built-in group restraints — so the
    parity here is torch-vs-jax; the rg measure's FD gradient is checked separately.)"""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy

    spec = _custom_spec()
    pos = _positions(spec)
    pt = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    torch_energy.total_energy(
        pt, torch_energy.prepare_spec(spec, dtype=torch.float64)
    ).backward()
    g_t = pt.grad.numpy()
    prep_j = jax_energy.prepare_spec(spec)
    g_j = np.asarray(
        jax.grad(lambda x: jax_energy.total_energy(x, prep_j))(jnp.asarray(pos))
    )
    assert np.all(np.isfinite(g_t)) and np.all(np.isfinite(g_j))
    assert np.allclose(g_t, g_j, atol=1e-6), f"max diff {np.abs(g_t - g_j).max()}"


def test_custom_rg_grad_matches_fd():
    """The radius_of_gyration measure uses the PLAIN centroid (no rigid-translation
    rescale), so its autodiff gradient must match the numpy finite-difference ground
    truth — this proves the gradient is CORRECT, not merely torch==jax."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy

    spec = _build(
        {
            "custom_restraints_config": [
                {
                    "measure": "radius_of_gyration",
                    "atom_selection": "resid 1 to 6",
                    "form": "harmonic",
                    "target": 2.0,
                    "weight": 1.0,
                }
            ]
        }
    )
    pos = _positions(spec, seed=3)
    prep_np = numpy_energy.prepare_spec(spec)

    def f(x):
        return float(numpy_energy.total_energy(x.reshape(spec.n_active, 3), prep_np))

    g_fd = _fd_grad(f, pos.flatten()).reshape(spec.n_active, 3)
    pt = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    torch_energy.total_energy(
        pt, torch_energy.prepare_spec(spec, dtype=torch.float64)
    ).backward()
    g_t = pt.grad.numpy()
    prep_j = jax_energy.prepare_spec(spec)
    g_j = np.asarray(
        jax.grad(lambda x: jax_energy.total_energy(x, prep_j))(jnp.asarray(pos))
    )
    assert np.allclose(g_t, g_fd, atol=1e-4), f"torch vs FD {np.abs(g_t - g_fd).max()}"
    assert np.allclose(g_j, g_fd, atol=1e-4), f"jax vs FD {np.abs(g_j - g_fd).max()}"


# --------------------------------------------------------------------------------------
# (a') a SYNTHETIC external restraint (pattern A): leaf fns supplied by the caller as
# plain callables, registered + driven through the same dispatch. Proves the registry is
# a real extension point, not just the home of the one built-in custom restraint.
# --------------------------------------------------------------------------------------
def _synth_centroid(np_, positions, grp1_idx, grp1_mask):
    pos = positions[..., grp1_idx, :]
    m = grp1_mask[..., None]
    return np_.sum(pos * m, axis=-2) / (np_.sum(grp1_mask, axis=-1)[..., None] + 1e-12)


def _synth_numpy(positions, grp1_idx, grp1_mask, ref, weight, mask):
    c = _synth_centroid(np, positions, grp1_idx, grp1_mask)
    return np.sum(weight * np.sum((c - ref) ** 2, axis=-1) * mask)


def _synth_torch(positions, grp1_idx, grp1_mask, ref, weight, mask):
    import torch

    c = _synth_centroid(torch, positions, grp1_idx, grp1_mask)
    return torch.sum(weight * torch.sum((c - ref) ** 2, dim=-1) * mask)


def _synth_jax(positions, grp1_idx, grp1_mask, ref, weight, mask):
    import jax.numpy as jnp

    c = _synth_centroid(jnp, positions, grp1_idx, grp1_mask)
    return jnp.sum(weight * jnp.sum((c - ref) ** 2, axis=-1) * mask)


class _SynthArrays:
    """A pattern-A restraint: pull group-1's centroid toward a fixed reference point."""

    def __init__(self):
        self.grp1_idx = np.array([[0, 1], [2, 3]], dtype=np.int64)
        self.grp1_mask = np.array([[1.0, 1.0], [1.0, 0.0]])  # row 2: group of 1
        self.ref = np.array([[5.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
        self.weight = np.array([1.0, 0.5])
        self.mask = np.array([1.0, 1.0])
        self.start_sigma = np.array([np.inf, np.inf])
        self.stop_sigma = np.array([-1.0, -1.0])


_SYNTH_TYPE = registry.RestraintType(
    name="synthpull",
    config_section="synthpull_restraints_config",
    data_class=object,
    data_builder=lambda items, g2l: None,
    spec_schema=(
        ("grp1_idx", "i"),
        ("grp1_mask", "f"),
        ("ref", "f"),
        ("weight", "f"),
        ("mask", "f"),
        ("start_sigma", "f"),
        ("stop_sigma", "f"),
    ),
    term_args=("grp1_idx", "grp1_mask", "ref", "weight"),
    leaf_fns={"numpy": _synth_numpy, "torch": _synth_torch, "jax": _synth_jax},
)


@pytest.fixture
def synth_registered():
    registry.register_restraint(_SYNTH_TYPE)
    try:
        yield
    finally:
        registry.unregister_restraint("synthpull")


def _synth_spec():
    from rgi_utils.spec import RestraintSpec

    return RestraintSpec(
        n_active=4,
        active_sites=np.arange(4),
        registered={"synthpull": _SynthArrays()},
    )


def test_synth_restraint_energy_and_grad_parity(synth_registered):
    """A caller-registered restraint flows through the dispatch on all three backends, with
    matching energy AND (plain-centroid) gradient vs the numpy finite-difference truth."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy

    spec = _synth_spec()
    assert spec.has_registered()
    pos = _positions(spec, seed=1)
    prep_np = numpy_energy.prepare_spec(spec)
    e_np = float(numpy_energy.total_energy(pos, prep_np))
    e_t = float(
        torch_energy.total_energy(
            torch.tensor(pos, dtype=torch.float64),
            torch_energy.prepare_spec(spec, dtype=torch.float64),
        )
    )
    e_j = float(
        jax_energy.total_energy(jnp.asarray(pos), jax_energy.prepare_spec(spec))
    )
    assert e_np > 0.0
    assert abs(e_np - e_t) < 1e-6 and abs(e_np - e_j) < 1e-6

    g_fd = _fd_grad(
        lambda x: float(numpy_energy.total_energy(x.reshape(4, 3), prep_np)),
        pos.flatten(),
    ).reshape(4, 3)
    pt = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    torch_energy.total_energy(
        pt, torch_energy.prepare_spec(spec, dtype=torch.float64)
    ).backward()
    assert np.allclose(pt.grad.numpy(), g_fd, atol=1e-4)
    prep_j = jax_energy.prepare_spec(spec)
    g_j = np.asarray(
        jax.grad(lambda x: jax_energy.total_energy(x, prep_j))(jnp.asarray(pos))
    )
    assert np.allclose(g_j, g_fd, atol=1e-4)


# --------------------------------------------------------------------------------------
# (b) torch optimizer + the GPU pre-gate footgun guard
# --------------------------------------------------------------------------------------
def test_custom_torch_minimize_converges():
    """The torch CG drives a custom centroid-distance restraint onto its target (eager
    CPU path; the same algorithm runs fused on CUDA)."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec = _build(
        {
            "custom_restraints_config": [
                {
                    "measure": "distance",
                    "atom_selection1": "resid 1",
                    "atom_selection2": "resid 2",
                    "form": "harmonic",
                    "target": 5.0,
                    "move": "all",
                }
            ]
        },
        n=2,
    )
    coords = torch.tensor(np.zeros((1, 2, 3)), dtype=torch.float64)
    coords[0, 1, 0] = 1.0  # start 1 A apart on x; target is 5 A
    TorchRestraintOptimizer(spec, max_iter=200).minimize(coords)
    d = float(torch.linalg.norm(coords[0, 0] - coords[0, 1]))
    assert abs(d - 5.0) < 0.1, f"distance {d} did not reach target 5.0"


def test_gated_prepared_folds_custom_gate():
    """The torch GPU pre-gate must fold a custom (per-entry) restraint's sigma gate into
    its mask — otherwise the restraint runs UNGATED on the compiled CUDA path (a bug CPU
    CI can't otherwise catch). 'custom' must appear in per_entry_keys()."""
    pytest.importorskip("torch")
    from rgi_utils.energy._terms import per_entry_keys
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    assert "custom" in per_entry_keys()
    spec = _build(
        {
            "custom_restraints_config": [
                {
                    "measure": "distance",
                    "atom_selection1": "resid 1",
                    "atom_selection2": "resid 2",
                    "form": "harmonic",
                    "target": 5.0,
                    "start_sigma": 1.0,  # active only when sigma <= 1
                }
            ]
        },
        n=2,
    )
    import torch

    opt = TorchRestraintOptimizer(spec, max_iter=10)
    opt._ensure(torch.device("cpu"), torch.float64)
    base = opt._prepared["custom"]["mask"]
    assert float(base.sum()) > 0
    # above the gate (sigma=5 > start_sigma 1): mask folded to all-zero (inactive)
    assert float(opt._gated_prepared(5.0)["custom"]["mask"].sum()) == 0.0
    # below the gate (sigma=0.5 <= 1): mask preserved (active)
    assert float(opt._gated_prepared(0.5)["custom"]["mask"].sum()) > 0.0


@pytest.mark.gpu
def test_custom_gpu_minimize_nan_free():
    """On CUDA the torch optimizer runs the inductor-fused compiled CG. A custom restraint
    must converge there AND stay NaN-free (the compile path is where a new term most
    easily breaks). Mirrors test_custom_torch_minimize_converges on the GPU."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec = _build(
        {
            "gpu": True,
            "custom_restraints_config": [
                {
                    "measure": "distance",
                    "atom_selection1": "resid 1",
                    "atom_selection2": "resid 2",
                    "form": "harmonic",
                    "target": 5.0,
                    "move": "all",
                }
            ],
        },
        n=2,
    )
    coords = torch.zeros((1, 2, 3), dtype=torch.float32, device="cuda")
    coords[0, 1, 0] = 1.0
    TorchRestraintOptimizer(spec, max_iter=200).minimize(coords)
    assert torch.isfinite(coords).all(), "custom restraint produced NaN/Inf on CUDA"
    d = float(torch.linalg.norm(coords[0, 0] - coords[0, 1]))
    assert abs(d - 5.0) < 0.2, f"distance {d} did not reach target 5.0 on CUDA"


# --------------------------------------------------------------------------------------
# (c) jax minimizer — the AF3 lax.scan closure — NaN-free
# --------------------------------------------------------------------------------------
def test_custom_jax_minimize_converges_nan_free():
    """make_minimizer returns the pure closure AF3 calls inside lax.scan. A custom
    restraint must converge there and stay NaN-free (jax static-shape / JIT path)."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer

    spec = _build(
        {
            "backend": "jax",
            "custom_restraints_config": [
                {
                    "measure": "distance",
                    "atom_selection1": "resid 1",
                    "atom_selection2": "resid 2",
                    "form": "harmonic",
                    "target": 5.0,
                    "move": "all",
                }
            ],
        },
        n=2,
    )
    coords = np.zeros((1, 2, 3))
    coords[0, 1, 0] = 1.0
    out = np.asarray(make_minimizer(spec, max_iter=200)(jnp.asarray(coords), 0.0))
    assert np.all(np.isfinite(out)), "custom restraint produced NaN/Inf under jax"
    d = float(np.linalg.norm(out[0, 0] - out[0, 1]))
    assert abs(d - 5.0) < 0.1, f"jax: distance {d} did not reach target 5.0"


# --------------------------------------------------------------------------------------
# config-layer contract: a registered section extends the whitelist; unknown still RAISES
# --------------------------------------------------------------------------------------
def test_registered_section_extends_whitelist():
    # the built-in custom section is accepted
    cfg = RestraintsConfig.from_dict(
        {
            "custom_restraints_config": [
                {
                    "measure": "distance",
                    "atom_selection1": "resid 1",
                    "atom_selection2": "resid 2",
                    "form": "harmonic",
                    "target": 5.0,
                }
            ]
        }
    )
    assert "custom" in cfg.registered_data
    # an unknown top-level section still raises (the silent-drop footgun guard)
    with pytest.raises(ValueError, match="unknown top-level"):
        RestraintsConfig.from_dict({"bogus_restraints_config": []})


def test_register_validation():
    """Registration rejects reserved names / sections / gates and missing backends."""

    def mk(**kw):
        base = dict(
            name="okname",
            config_section="okname_restraints_config",
            data_class=object,
            data_builder=lambda i, g: None,
            spec_schema=(("mask", "f"),),
            term_args=(),
            leaf_fns={"numpy": "m:f", "torch": "m:f", "jax": "m:f"},
        )
        base.update(kw)
        return registry.RestraintType(**base)

    with pytest.raises(ValueError, match="built-in term"):
        registry.register_restraint(mk(name="distance"))
    with pytest.raises(ValueError, match="reserved"):
        registry.register_restraint(mk(gate="dist"))
    with pytest.raises(ValueError, match="built-in top-level"):
        registry.register_restraint(mk(config_section="distance_restraints_config"))
    with pytest.raises(ValueError, match="all backends"):
        registry.register_restraint(mk(leaf_fns={"numpy": "m:f"}))
