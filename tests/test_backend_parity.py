"""Cross-backend parity: numpy / torch / jax must agree on energy and gradient.

This is the contract that lets us trust autodiff (torch/jax) against the
hand-written numpy reference. Energies must match to ~1e-6; autodiff gradients
must match the numpy finite-difference gradient to ~1e-4.
"""

from __future__ import annotations

import numpy as np
import pytest

from rgi_utils.energy import numpy_energy
from rgi_utils.spec import (
    AngleArrays,
    BondArrays,
    ChiralArrays,
    DihedralArrays,
    DistanceArrays,
    RestraintSpec,
    VdwArrays,
)

N_ACTIVE = 12


def _make_spec() -> RestraintSpec:
    """A small spec exercising every restraint type with non-zero energy."""
    bond = BondArrays(
        idx=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64),
        r0=np.array([1.0, 1.5, 1.2]),
        slack=np.array([0.0, 0.0, 0.0]),
        weight=np.array([0.05, 0.05, 0.1]),
        half=np.array([0.0, 1.0, 0.0]),  # second bond is half (stretch-only)
        mask=np.array([1.0, 1.0, 1.0]),
    )
    angle = AngleArrays(
        idx=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        th0=np.array([1.9, 2.0]),
        slack=np.array([0.0, 0.0]),
        weight=np.array([0.05, 0.05]),
        mask=np.array([1.0, 1.0]),
    )
    chiral = ChiralArrays(
        idx=np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64),
        vol0=np.array([0.5, -0.5]),
        slack=np.array([0.0, 0.0]),
        weight=np.array([0.1, 0.1]),
        mask=np.array([1.0, 1.0]),
    )
    # non-trivial phi0 so the random positions violate the torsion (energy > 0);
    # second entry is masked-out padding (mask=0) to exercise the padding path.
    dihedral = DihedralArrays(
        idx=np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64),
        phi0=np.array([0.7, -2.5]),
        slack=np.array([0.0, 0.2]),
        weight=np.array([0.15, 0.15]),
        mask=np.array([1.0, 0.0]),
    )
    distance = DistanceArrays(
        grp1_idx=np.array([[0, 1], [8, 9]], dtype=np.int64),
        grp2_idx=np.array([[5, 6], [10, 11]], dtype=np.int64),
        grp1_mask=np.array([[1.0, 1.0], [1.0, 0.0]]),  # second restr: group of 1
        grp2_mask=np.array([[1.0, 1.0], [1.0, 1.0]]),
        target1=np.array([3.0, 2.0]),
        target2=np.array([6.0, 5.0]),
        dist_type=np.array([0, 1], dtype=np.int64),  # harmonic + flat-bottomed
        mask=np.array([1.0, 1.0]),
        start_sigma=np.array([100.0, 5.0]),  # different per-distance start_sigma
    )
    # large r_min so these pairs are "clashing" (non-zero VdW energy) for the
    # random positions; the third pair is masked out (padding).
    vdw = VdwArrays(
        idx=np.array([[0, 4], [1, 7], [2, 9]], dtype=np.int64),
        r_min=np.array([5.0, 4.5, 5.5]),
        weight=np.array([0.2, 0.2, 0.2]),
        mask=np.array([1.0, 1.0, 0.0]),
    )
    return RestraintSpec(
        n_active=N_ACTIVE,
        active_sites=np.arange(N_ACTIVE),
        bond=bond,
        angle=angle,
        chiral=chiral,
        dihedral=dihedral,
        vdw=vdw,
        distance=distance,
        conf_start_sigma=10.0,  # conformer terms active when sigma <= 10
    )


def _positions(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_ACTIVE, 3)) * 2.0


def test_energy_parity():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy

    spec = _make_spec()
    pos = _positions()

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

    assert e_np > 0.0  # sanity: restraints are violated
    assert abs(e_np - e_t) < 1e-6, f"numpy={e_np} torch={e_t}"
    assert abs(e_np - e_j) < 1e-6, f"numpy={e_np} jax={e_j}"


def test_grad_parity():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from scipy.optimize import approx_fprime

    from rgi_utils.energy import jax_energy, torch_energy

    spec = _make_spec()
    pos = _positions()

    # numpy finite-difference gradient (ground truth for autodiff)
    prep_np = numpy_energy.prepare_spec(spec)

    def f(x):
        return float(numpy_energy.total_energy(x.reshape(N_ACTIVE, 3), prep_np))

    g_fd = approx_fprime(pos.flatten(), f, 1e-6).reshape(N_ACTIVE, 3)

    # torch autograd
    pt = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    e_t = torch_energy.total_energy(
        pt, torch_energy.prepare_spec(spec, dtype=torch.float64)
    )
    e_t.backward()
    g_t = pt.grad.numpy()

    # jax grad
    prep_j = jax_energy.prepare_spec(spec)
    g_j = np.asarray(
        jax.grad(lambda x: jax_energy.total_energy(x, prep_j))(jnp.asarray(pos))
    )

    assert np.allclose(g_t, g_fd, atol=1e-4), f"max diff {np.abs(g_t - g_fd).max()}"
    assert np.allclose(g_j, g_fd, atol=1e-4), f"max diff {np.abs(g_j - g_fd).max()}"
    assert np.allclose(g_t, g_j, atol=1e-6), f"max diff {np.abs(g_t - g_j).max()}"


def test_sigma_gating_parity():
    """Per-restraint start_sigma gating agrees across backends and actually gates."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy

    spec = _make_spec()  # conf_start_sigma=10; distance start_sigma=[100, 5]
    pos = _positions()
    prep_np = numpy_energy.prepare_spec(spec)
    prep_t = torch_energy.prepare_spec(spec, dtype=torch.float64)
    prep_j = jax_energy.prepare_spec(spec)
    pt = torch.tensor(pos, dtype=torch.float64)
    pj = jnp.asarray(pos)

    def e_all(sigma):
        return (
            float(numpy_energy.total_energy(pos, prep_np, sigma)),
            float(torch_energy.total_energy(pt, prep_t, sigma)),
            float(jax_energy.total_energy(pj, prep_j, sigma)),
        )

    # cross-backend agreement at several noise levels
    for sigma in (200.0, 50.0, 8.0, 3.0):
        en, et, ej = e_all(sigma)
        assert abs(en - et) < 1e-6 and abs(en - ej) < 1e-6, f"sigma={sigma}"

    e_ungated = float(numpy_energy.total_energy(pos, prep_np))  # sigma=None
    # above every start_sigma -> nothing active -> zero
    assert e_all(200.0)[0] == 0.0
    # sigma=3 is <= every start_sigma -> all active -> equals the ungated energy
    assert abs(e_all(3.0)[0] - e_ungated) < 1e-9
    # sigma=50 (conf off, dist start_sigma=5 off, only dist start_sigma=100 on)
    # has strictly fewer active terms than sigma=3
    assert e_all(50.0)[0] <= e_all(3.0)[0]


def test_dihedral_degenerate_gradient_parity():
    """Degenerate dihedral geometry must give FINITE, mutually-equal gradients in
    all three backends. Regression for the atan2(0,0) divergence (jax=NaN vs
    torch=0) at exact collinearity / coincident atoms in dihedral_energy."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy

    # one dihedral, nonzero target so a degenerate phi still yields a nonzero delta
    spec = RestraintSpec(
        n_active=4,
        active_sites=np.arange(4),
        dihedral=DihedralArrays(
            idx=np.array([[0, 1, 2, 3]], dtype=np.int64),
            phi0=np.array([1.0]),
            slack=np.array([0.0]),
            weight=np.array([1.0]),
            mask=np.array([1.0]),
        ),
        conf_start_sigma=10.0,
    )
    cases = {
        "collinear_ijk": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0]],
        "collinear_jkl": [[0, 1, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]],
        "coincident_jk": [[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]],
        "near_collinear": [[0, 0, 0], [1, 0, 0], [2, 1e-7, 0], [2, 1, 0]],
    }
    prep_np = numpy_energy.prepare_spec(spec)
    prep_t = torch_energy.prepare_spec(spec, dtype=torch.float64)
    prep_j = jax_energy.prepare_spec(spec)
    for name, p in cases.items():
        pos = np.array(p, dtype=np.float64)
        # energy: finite and equal across backends
        e_np = float(numpy_energy.total_energy(pos, prep_np))
        e_t = float(
            torch_energy.total_energy(torch.tensor(pos, dtype=torch.float64), prep_t)
        )
        e_j = float(jax_energy.total_energy(jnp.asarray(pos), prep_j))
        assert np.isfinite([e_np, e_t, e_j]).all(), f"{name}: non-finite energy"
        assert abs(e_np - e_t) < 1e-6 and abs(e_np - e_j) < 1e-6, (
            f"{name}: energy mismatch"
        )
        # gradient: finite in both autodiff backends and equal to each other (the
        # cross-backend parity invariant; FD is unreliable exactly at the singularity)
        pt = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
        torch_energy.total_energy(pt, prep_t).backward()
        g_t = pt.grad.numpy()
        g_j = np.asarray(
            jax.grad(lambda x: jax_energy.total_energy(x, prep_j))(jnp.asarray(pos))
        )
        assert np.isfinite(g_t).all(), f"{name}: torch grad non-finite"
        assert np.isfinite(g_j).all(), f"{name}: jax grad non-finite"
        assert np.allclose(g_t, g_j, rtol=1e-5, atol=1e-6), (
            f"{name}: torch/jax grad mismatch, max|d|={np.abs(g_t - g_j).max()}"
        )


def test_distance_closed_form_backend_parity():
    """The closed-form COM-distance shift agrees across numpy/torch/jax and lands the
    COM separation exactly on target in one step (this replaced the per-atom CG)."""
    torch = pytest.importorskip("torch")
    jnp = pytest.importorskip("jax.numpy")
    from rgi_utils.optim import distance_shift as ds

    d_np = dict(
        grp1_idx=np.array([[0, 1]]),
        grp2_idx=np.array([[2, 3]]),
        grp1_mask=np.array([[1.0, 1.0]]),
        grp2_mask=np.array([[1.0, 1.0]]),
        target1=np.array([5.0]),
        target2=np.array([0.0]),
        dist_type=np.array([0]),
        mask=np.array([1.0]),
        start_sigma=np.array([1e30]),
    )
    active = np.zeros((4, 3))
    active[2:, 0] = 20.0  # COM1 at x=0, COM2 at x=20 -> dist 20

    a_np = ds.apply_distance_shift_numpy(active, d_np, 0.0)
    a_t = ds.apply_distance_shift_torch(
        torch.as_tensor(active), {k: torch.as_tensor(v) for k, v in d_np.items()}, 0.0
    ).numpy()
    a_j = np.asarray(
        ds.apply_distance_shift_jax(
            jnp.asarray(active), {k: jnp.asarray(v) for k, v in d_np.items()}, 0.0
        )
    )

    def com_dist(a):
        return np.linalg.norm(a[2:].mean(0) - a[:2].mean(0))

    assert abs(com_dist(a_np) - 5.0) < 1e-6
    assert np.allclose(a_np, a_t, atol=1e-6)
    assert np.allclose(a_np, a_j, atol=1e-5)


def test_distance_closed_form_coupled_restraints():
    """Two distance restraints that SHARE an atom converge via the Jacobi iteration
    (a single pass would satisfy neither); numpy/torch/jax agree."""
    torch = pytest.importorskip("torch")
    jnp = pytest.importorskip("jax.numpy")
    from rgi_utils.optim import distance_shift as ds

    d = dict(
        grp1_idx=np.array([[0], [1]]),
        grp2_idx=np.array([[1], [2]]),  # restraints A-B and B-C share atom 1 (B)
        grp1_mask=np.array([[1.0], [1.0]]),
        grp2_mask=np.array([[1.0], [1.0]]),
        target1=np.array([5.0, 5.0]),
        target2=np.array([0.0, 0.0]),
        dist_type=np.array([0, 0]),
        mask=np.array([1.0, 1.0]),
        start_sigma=np.array([1e30, 1e30]),
    )
    a = np.zeros((3, 3))
    a[1, 0] = 10.0
    a[2, 0] = 20.0  # A=0, B=10, C=20 -> both gaps 10, target 5

    a_np = ds.apply_distance_shift_numpy(a, d, 0.0)
    assert abs(np.linalg.norm(a_np[1] - a_np[0]) - 5.0) < 1e-3
    assert abs(np.linalg.norm(a_np[2] - a_np[1]) - 5.0) < 1e-3
    a_t = ds.apply_distance_shift_torch(
        torch.as_tensor(a), {k: torch.as_tensor(v) for k, v in d.items()}, 0.0
    ).numpy()
    a_j = np.asarray(
        ds.apply_distance_shift_jax(
            jnp.asarray(a), {k: jnp.asarray(v) for k, v in d.items()}, 0.0
        )
    )
    assert np.allclose(a_np, a_t, atol=1e-6)
    assert np.allclose(a_np, a_j, atol=1e-5)
