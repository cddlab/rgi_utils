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
    CisTransArrays,
    DistanceArrays,
    GroupAngleArrays,
    GroupDihedralArrays,
    PlanarityArrays,
    RestraintSpec,
    VdwArrays,
)

N_ACTIVE = 12


def _make_spec(
    include_groups: bool = True,
    include_planarity: bool = True,
    include_distance: bool = True,
) -> RestraintSpec:
    """A small spec exercising every restraint type with non-zero energy.

    ``include_groups`` adds the group-centroid angle/dihedral terms (default). The fuzzy
    torch-vs-jax CG-convergence test opts out (``include_groups=False``) so its
    calibrated tolerance keeps its original group-free landscape — the periodic group
    dihedral makes the two backends' CG minima diverge a touch more than that tolerance
    (group CG convergence is covered directly in test_optim.py). ``include_planarity``
    (default on) adds the planarity term; the same fuzzy CG-convergence test
    opts out of it too (its stiff near-zero-volume target shifts the fixed-iteration
    minimum a touch — planarity energy/grad parity is covered by the other tests)."""
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
    # planarity: same signed-volume maths as chiral but target vol0 ~ 0 (a
    # planar sp2 centre). Random positions give a non-zero volume so energy > 0; the
    # second entry is masked-out padding. Reuses chiral_energy via the _terms dispatch,
    # so this row is what proves planarity is wired + parity-correct across backends.
    planarity = (
        None
        if not include_planarity
        else PlanarityArrays(
            idx=np.array([[8, 9, 10, 11], [0, 2, 4, 6]], dtype=np.int64),
            vol0=np.array([0.0, 0.0]),
            slack=np.array([0.0, 0.05]),
            weight=np.array([0.1, 0.1]),
            mask=np.array([1.0, 0.0]),
        )
    )
    # non-trivial phi0 so the random positions violate the torsion (energy > 0);
    # second entry is masked-out padding (mask=0) to exercise the padding path.
    cistrans = CisTransArrays(
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
        move_mode=np.array([0, 0], dtype=np.int64),  # both groups move (default)
        weight=np.array([1.0, 0.5]),  # exercise the weighted energy multiply in parity
        mask=np.array([1.0, 1.0]),
        start_sigma=np.array([100.0, 5.0]),  # different per-distance start_sigma
        stop_sigma=np.array([-1.0, -1.0]),  # -1 = never released (off)
        start_step=np.full(2, float("-inf")),  # step window unused -> always-on
        stop_step=np.full(2, float("inf")),
    )
    # large r_min so these pairs are "clashing" (non-zero VdW energy) for the
    # random positions; the third pair is masked out (padding).
    vdw = VdwArrays(
        idx=np.array([[0, 4], [1, 7], [2, 9]], dtype=np.int64),
        r_min=np.array([5.0, 4.5, 5.5]),
        weight=np.array([0.2, 0.2, 0.2]),
        mask=np.array([1.0, 1.0, 0.0]),
    )
    # group-centroid angle: 2 restraints (vertex = group 2), exercising the harmonic + flat-
    # bottomed types (geom_type 0/1). Row 1 uses groups of 1 (intra-group padding).
    # move_free all 1 (every group free) so the numpy-FD gradient parity below holds
    # (a pinned group diverges from FD; tested torch-vs-jax in test_optim). start_sigma
    # [100, 5] keeps the gating-test invariants.
    group_angle = (
        None
        if not include_groups
        else GroupAngleArrays(
            grp1_idx=np.array([[0, 1], [6, 0]], dtype=np.int64),
            grp2_idx=np.array([[2, 3], [7, 0]], dtype=np.int64),
            grp3_idx=np.array([[4, 5], [8, 0]], dtype=np.int64),
            grp1_mask=np.array([[1.0, 1.0], [1.0, 0.0]]),  # row1: group of 1
            grp2_mask=np.array([[1.0, 1.0], [1.0, 0.0]]),
            grp3_mask=np.array([[1.0, 1.0], [1.0, 0.0]]),
            target1=np.array([1.2, 2.0]),  # harmonic target / flat-bottomed lower
            target2=np.array([0.0, 2.4]),  # flat-bottomed upper (unused for harmonic)
            geom_type=np.array([0, 1], dtype=np.int64),  # harmonic + flat-bottomed
            move_free=np.ones((2, 3)),  # all groups free (FD-grad parity needs it)
            weight=np.array([1.0, 0.5]),
            mask=np.array([1.0, 1.0]),
            start_sigma=np.array([100.0, 5.0]),  # different per-restraint start_sigma
            stop_sigma=np.array([-1.0, -1.0]),
            start_step=np.full(2, float("-inf")),  # step window unused -> always-on
            stop_step=np.full(2, float("inf")),
        )
    )
    # group-centroid dihedral: 1 harmonic restraint + 1 masked padding row (per-restraint
    # mask path). axis = group2-group3; harmonic is periodicity-safe.
    group_dihedral = (
        None
        if not include_groups
        else GroupDihedralArrays(
            grp1_idx=np.array([[0, 1], [4, 5]], dtype=np.int64),
            grp2_idx=np.array([[2, 3], [6, 7]], dtype=np.int64),
            grp3_idx=np.array([[4, 5], [8, 9]], dtype=np.int64),
            grp4_idx=np.array([[6, 7], [10, 11]], dtype=np.int64),
            grp1_mask=np.array([[1.0, 1.0], [1.0, 1.0]]),
            grp2_mask=np.array([[1.0, 1.0], [1.0, 1.0]]),
            grp3_mask=np.array([[1.0, 1.0], [1.0, 1.0]]),
            grp4_mask=np.array([[1.0, 1.0], [1.0, 1.0]]),
            target1=np.array([0.7, -1.0]),
            target2=np.array([0.0, 1.0]),
            geom_type=np.array(
                [0, 1], dtype=np.int64
            ),  # harmonic (active) + flat (masked)
            move_free=np.ones((2, 4)),
            weight=np.array([0.8, 0.8]),
            mask=np.array([1.0, 0.0]),  # second restraint is masked padding
            start_sigma=np.array([100.0, 100.0]),
            stop_sigma=np.array([-1.0, -1.0]),
            start_step=np.full(2, float("-inf")),  # step window unused -> always-on
            stop_step=np.full(2, float("inf")),
        )
    )
    return RestraintSpec(
        n_active=N_ACTIVE,
        active_sites=np.arange(N_ACTIVE),
        bond=bond,
        angle=angle,
        chiral=chiral,
        planarity=planarity,
        cistrans=cistrans,
        vdw=vdw,
        distance=distance if include_distance else None,
        group_angle=group_angle,
        group_dihedral=group_dihedral,
        conf_start_sigma=10.0,  # conformer terms active when sigma <= 10
    )


def _positions(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_ACTIVE, 3)) * 2.0


def _fd_grad(f, x, eps: float = 1e-6):
    """Central finite-difference gradient of scalar ``f`` at ``x`` (numpy-only,
    replaces scipy.optimize.approx_fprime so the suite needs no scipy)."""
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

    from rgi_utils.energy import jax_energy, torch_energy

    # group AND distance terms excluded: both have an intentionally N x-rescaled centroid
    # gradient (centroid_eff, so the group/centroid moves rigidly at weight=1) — distance now
    # uses the reduced-mass scale mu=N1*N2/(N1+N2) — so neither matches a numpy finite-
    # difference of the true energy. Group grad parity is checked torch-vs-jax in test_optim;
    # distance grad parity torch-vs-jax is test_distance_grad_parity_torch_jax below.
    spec = _make_spec(include_groups=False, include_distance=False)
    pos = _positions()

    # numpy finite-difference gradient (ground truth for autodiff)
    prep_np = numpy_energy.prepare_spec(spec)

    def f(x):
        return float(numpy_energy.total_energy(x.reshape(N_ACTIVE, 3), prep_np))

    g_fd = _fd_grad(f, pos.flatten()).reshape(N_ACTIVE, 3)

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


def test_interligand_vdw_energy_parity():
    """Inter-ligand VdW pairs built by build_spec for two overlapping ligands score
    identically across numpy/torch/jax. They are ordinary VdwArrays rows (so the vdw rows
    in _make_spec already prove the energy-layer parity), but this guards the
    build_spec -> prepare_spec assembly of the cross-ligand pairs specifically."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from rgi_utils.atom_context import LigandConf
    from rgi_utils.energy import jax_energy, torch_energy
    from rgi_utils.featurizer import build_spec

    m = Chem.MolFromSmiles("CC")
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=1)
    m = Chem.RemoveHs(m)  # heavy-only ethane: 0 intramolecular pairs
    c = np.asarray(m.GetConformer().GetPositions())
    n = m.GetNumAtoms()
    lcA = LigandConf(
        mol=m, conf_coords=c, global_indices=np.arange(n), conformer_restraints=True
    )
    lcB = LigandConf(
        mol=m, conf_coords=c, global_indices=np.arange(n) + n, conformer_restraints=True
    )
    spec = build_spec(
        [lcA, lcB], [], {"vdw": {"weight": 1.0, "scale": 0.9}}, conf_start_sigma=1e30
    )
    assert spec.vdw is not None and spec.vdw.idx.shape[0] == n * n

    pos = np.zeros((spec.n_active, 3))
    pos[:n] = c
    pos[n:] = c + np.array([0.3, 0.0, 0.0])  # overlap -> non-zero inter VdW energy

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
    assert e_np > 0.0  # the overlap is penalised
    assert abs(e_np - e_t) < 1e-6, f"numpy={e_np} torch={e_t}"
    assert abs(e_np - e_j) < 1e-6, f"numpy={e_np} jax={e_j}"


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


def test_cistrans_degenerate_gradient_parity():
    """Degenerate cis/trans geometry must give FINITE, mutually-equal gradients in
    all three backends. Regression for the atan2(0,0) divergence (jax=NaN vs
    torch=0) at exact collinearity / coincident atoms in cistrans_energy."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy

    # one cistrans, nonzero target so a degenerate phi still yields a nonzero delta
    spec = RestraintSpec(
        n_active=4,
        active_sites=np.arange(4),
        cistrans=CisTransArrays(
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


def test_distance_grad_parity_torch_jax():
    """Distance is now CG-minimised with the reduced-mass ``centroid_eff`` rescale
    (``scale = N1*N2/(N1+N2)``), so its autodiff gradient is intentionally N×-rescaled and
    does NOT match a numpy finite-difference of the true energy — the same carve-out as
    rmsd/group. The contract that survives: the energy VALUE agrees across numpy/torch/jax
    (centroid_eff leaves the value unchanged), and the autodiff GRADIENT agrees
    torch-vs-jax (what the CG optimizers rely on). N1=3 != N2=1 so the rescale is genuinely
    active (mu = 3/4 != 1, unlike the N1=N2=2 case where mu collapses to 1)."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy

    # group1 = 3 atoms, group2 = 1 atom (mu = 3*1/(3+1) = 0.75), harmonic, off target so
    # energy + gradient are non-zero.
    spec = RestraintSpec(
        n_active=8,
        active_sites=np.arange(8),
        distance=DistanceArrays(
            grp1_idx=np.array([[0, 1, 2]], dtype=np.int64),
            grp2_idx=np.array([[7, 0, 0]], dtype=np.int64),
            grp1_mask=np.array([[1.0, 1.0, 1.0]]),
            grp2_mask=np.array([[1.0, 0.0, 0.0]]),  # group2 is a single atom
            target1=np.array([3.0]),
            target2=np.array([0.0]),
            dist_type=np.array([0], dtype=np.int64),
            move_mode=np.array([0], dtype=np.int64),  # both groups free
            weight=np.array([1.0]),
            mask=np.array([1.0]),
            start_sigma=np.array([float("inf")]),
            stop_sigma=np.array([-1.0]),
            start_step=np.full(1, float("-inf")),
            stop_step=np.full(1, float("inf")),
        ),
        conf_start_sigma=-1.0,
    )
    pos = _positions(2)[:8]
    prep_np = numpy_energy.prepare_spec(spec)
    prep_t = torch_energy.prepare_spec(spec, dtype=torch.float64)
    prep_j = jax_energy.prepare_spec(spec)

    # energy value parity across the three backends (centroid_eff is value-preserving)
    e_np = float(numpy_energy.total_energy(pos, prep_np))
    e_t = float(
        torch_energy.total_energy(torch.tensor(pos, dtype=torch.float64), prep_t)
    )
    e_j = float(jax_energy.total_energy(jnp.asarray(pos), prep_j))
    assert e_np > 0.0
    assert abs(e_np - e_t) < 1e-6 and abs(e_np - e_j) < 1e-6

    # gradient parity: torch vs jax (both apply the reduced-mass rescale; a numpy-FD of the
    # true energy would NOT match because the gradient is deliberately N×-rescaled).
    pt = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    torch_energy.total_energy(pt, prep_t).backward()
    g_t = pt.grad.numpy()
    g_j = np.asarray(
        jax.grad(lambda x: jax_energy.total_energy(x, prep_j))(jnp.asarray(pos))
    )
    assert np.allclose(g_t, g_j, atol=1e-6), f"max|d|={np.abs(g_t - g_j).max()}"


def _rmsd_case(seed=3, n=6):
    """One RMSD restraint over n atoms: reference + a rotated/translated/noised target."""
    rng = np.random.default_rng(seed)
    ref = rng.standard_normal((n, 3)) * 3.0
    th = 0.7
    rz = np.array(
        [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    )
    pos = ref @ rz.T + np.array([5.0, 2.0, -3.0]) + rng.standard_normal((n, 3)) * 0.4
    idx = np.arange(n).reshape(1, n)
    m = np.ones((1, n))
    refc = ref.reshape(1, n, 3)
    args = dict(  # fit == calc (plain superposed RMSD)
        fit_idx=idx,
        fit_mask=m,
        fit_ref=refc,
        calc_idx=idx,
        calc_mask=m,
        calc_ref=refc,
        target1=np.array([0.0]),
        target2=np.array([0.0]),
        geom_type=np.array([0]),
        weight=np.array([1.0]),
        mask=np.array([1.0]),
    )
    return ref, pos, args


def test_rmsd_kabsch_backend_parity():
    """The Kabsch-superposed RMSD energy agrees across numpy/torch/jax, its
    autodiff gradient agrees torch-vs-jax (the detached rotation makes a numpy
    finite-difference gradient inapplicable, like the cistrans degenerate case),
    and the energy is invariant to a rigid motion of the target (the Kabsch
    property)."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy

    ref, pos, args = _rmsd_case()

    e_np = float(numpy_energy.rmsd_energy(pos, **args))
    e_t = float(
        torch_energy.rmsd_energy(
            torch.tensor(pos), **{k: torch.tensor(v) for k, v in args.items()}
        )
    )
    e_j = float(
        jax_energy.rmsd_energy(
            jnp.asarray(pos), **{k: jnp.asarray(v) for k, v in args.items()}
        )
    )
    assert e_np > 0.0
    assert abs(e_np - e_t) < 1e-6 and abs(e_np - e_j) < 1e-6

    # gradient parity: torch vs jax (both stop-gradient the rotation)
    pt = torch.tensor(pos, requires_grad=True)
    torch_energy.rmsd_energy(
        pt, **{k: torch.tensor(v) for k, v in args.items()}
    ).backward()
    g_t = pt.grad.numpy()
    g_j = np.asarray(
        jax.grad(
            lambda x: jax_energy.rmsd_energy(
                x, **{k: jnp.asarray(v) for k, v in args.items()}
            )
        )(jnp.asarray(pos))
    )
    assert np.allclose(g_t, g_j, atol=1e-6), f"max|d|={np.abs(g_t - g_j).max()}"

    # Kabsch property: energy is invariant under a rigid motion of the target
    th2 = 1.3
    ry = np.array(
        [[np.cos(th2), 0, np.sin(th2)], [0, 1, 0], [-np.sin(th2), 0, np.cos(th2)]]
    )
    pos2 = pos @ ry.T + np.array([-7.0, 11.0, 4.0])
    e_np2 = float(numpy_energy.rmsd_energy(pos2, **args))
    assert abs(e_np - e_np2) < 1e-6


def test_rmsd_flat_bottom_backend_parity():
    """The flat-bottomed RMSD types (flat-bottomed / flat-bottomed1 lower / flat-bottomed2
    upper) agree across numpy/torch/jax on energy AND torch-vs-jax on the autodiff
    gradient. The detached Kabsch rotation rules out a numpy finite-difference check (as
    in test_rmsd_kabsch_backend_parity), so this EXTENDS that case's harness rather than
    routing through the FD harness. Bounds are placed around the case's actual RMSD so
    each one-sided type is genuinely active (E>0) and a wide window sits in the dead
    zone (E~0) -- guarding against a silently no-op flat-bottom branch."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy

    ref, pos, base = _rmsd_case()  # base is harmonic (geom_type 0, target1 0)
    rmsd0 = float(np.sqrt(numpy_energy.rmsd_energy(pos, **base)))  # weight 1 -> sqrt(E)

    # (label, geom_type, target1, target2, expect_active)
    cases = [
        ("flat-bottomed dead zone", 1, 0.0, rmsd0 + 1.0, False),
        ("flat-bottomed below window", 1, rmsd0 + 0.5, rmsd0 + 1.5, True),
        ("flat-bottomed1 lower bound", 2, rmsd0 + 1.0, 0.0, True),
        ("flat-bottomed2 upper bound", 3, 0.0, rmsd0 * 0.5, True),
    ]
    for label, gt, t1, t2, active in cases:
        args = dict(base)
        args["target1"] = np.array([float(t1)])
        args["target2"] = np.array([float(t2)])
        args["geom_type"] = np.array([gt])
        e_np = float(numpy_energy.rmsd_energy(pos, **args))
        e_t = float(
            torch_energy.rmsd_energy(
                torch.tensor(pos), **{k: torch.tensor(v) for k, v in args.items()}
            )
        )
        e_j = float(
            jax_energy.rmsd_energy(
                jnp.asarray(pos), **{k: jnp.asarray(v) for k, v in args.items()}
            )
        )
        assert abs(e_np - e_t) < 1e-6 and abs(e_np - e_j) < 1e-6, (
            label,
            e_np,
            e_t,
            e_j,
        )
        if active:
            assert e_np > 1e-6, (label, e_np)
        else:
            assert e_np < 1e-9, (label, e_np)
        # gradient parity: torch vs jax (both stop-gradient the rotation)
        pt = torch.tensor(pos, requires_grad=True)
        torch_energy.rmsd_energy(
            pt, **{k: torch.tensor(v) for k, v in args.items()}
        ).backward()
        g_t = pt.grad.numpy()
        g_j = np.asarray(
            jax.grad(
                lambda x, a=args: jax_energy.rmsd_energy(
                    x, **{k: jnp.asarray(v) for k, v in a.items()}
                )
            )(jnp.asarray(pos))
        )
        assert np.allclose(g_t, g_j, atol=1e-6), (label, np.abs(g_t - g_j).max())


def test_energy_breakdown_sums_to_total():
    """energy_breakdown (now schema-driven via _terms) must sum to total_energy, expose
    exactly the BREAKDOWN_KEYS, and agree across backends — it powers the finalize
    per-term log (CLAUDE.md's 'decisive signals'), which nothing else tests."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import _terms, jax_energy, torch_energy

    spec = _make_spec()
    pos = _positions()
    per_key = {}
    for name, mod, prep, p in (
        ("numpy", numpy_energy, numpy_energy.prepare_spec(spec), pos),
        (
            "torch",
            torch_energy,
            torch_energy.prepare_spec(spec, dtype=torch.float64),
            torch.tensor(pos, dtype=torch.float64),
        ),
        ("jax", jax_energy, jax_energy.prepare_spec(spec), jnp.asarray(pos)),
    ):
        bd = mod.energy_breakdown(p, prep)
        assert set(bd) == set(_terms.BREAKDOWN_KEYS), name
        tot = float(mod.total_energy(p, prep))
        assert abs(sum(bd.values()) - tot) < 1e-6, (name, sum(bd.values()), tot)
        per_key[name] = bd
    for k in _terms.BREAKDOWN_KEYS:  # per-term cross-backend agreement
        assert abs(per_key["numpy"][k] - per_key["torch"][k]) < 1e-6, k
        assert abs(per_key["numpy"][k] - per_key["jax"][k]) < 1e-6, k


def test_jax_torch_cg_same_minimum_at_default_iters():
    """The jax CG (_cg_minimize) and the torch CG (_cg_minimize_torch, == the CUDA
    algorithm) reach the same minimum on _make_spec at the default max_iter. (XLA float
    reordering can diverge at very high max_iter; this pins the default-setting parity —
    the cross-tool invariant that one restraints_config converges the same everywhere.)"""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy
    from rgi_utils.optim._torch_cg_gpu import _cg_minimize_torch
    from rgi_utils.optim.jax_optim import _cg_minimize

    # conformer + vdw only (no rmsd, group, planarity, OR distance); the CG handles conf
    # here. This test originally excluded distance via include_distance=False (distance was
    # closed-form); with that flag gone, distance is dropped from the spec instead — its
    # reduced-mass centroid_eff rescale shifts the fixed-iteration CG minimum past this fuzzy
    # tolerance (same reason groups/planarity are excluded). Distance CG-convergence parity is
    # covered directly in test_optim; group convergence parity likewise.
    spec = _make_spec(
        include_groups=False, include_planarity=False, include_distance=False
    )
    pos = _positions(0)
    prep_t = torch_energy.prepare_spec(spec, dtype=torch.float64)
    prep_j = jax_energy.prepare_spec(spec)

    def et(a):
        return torch_energy.total_energy(a, prep_t, sigma=None)

    def ej(a):
        return jax_energy.total_energy(a, prep_j, sigma=None)

    xt = _cg_minimize_torch(
        torch.func.grad_and_value(et), torch.tensor(pos, dtype=torch.float64), 100
    )
    Et = float(et(xt))
    Ej = float(ej(_cg_minimize(ej, jnp.asarray(pos), 100)))
    assert abs(Et - Ej) < 1e-3 + 0.05 * abs(Et), (Et, Ej)


def test_rmsd_known_value_and_target():
    """ref == target -> superposed RMSD 0 -> E = weight * target_rmsd**2."""
    rng = np.random.default_rng(0)
    n = 5
    ref = rng.standard_normal((n, 3)) * 2.0
    idx = np.arange(n).reshape(1, n)
    m = np.ones((1, n))
    refc = ref.reshape(1, n, 3)
    e = float(
        numpy_energy.rmsd_energy(
            ref,
            idx,
            m,
            refc,
            idx,
            m,
            refc,
            target1=np.array([2.0]),
            target2=np.array([0.0]),
            geom_type=np.array([0]),
            weight=np.array([1.5]),
            mask=np.array([1.0]),
        )
    )
    assert abs(e - 1.5 * 2.0**2) < 1e-5


def test_rmsd_stop_sigma_release_window():
    """stop_sigma RELEASES the rmsd restraint below it: the term is gated to 0 for
    sigma < stop_sigma (the model's final low-sigma steps run restraint-free so they can
    re-idealise the boundary geometry), is active in [stop_sigma, start_sigma], and is 0
    above start_sigma. The same window holds across numpy/torch/jax (the gate lives in
    the shared sigma_gate). Regression for the RMSD dangling-terminus bond fix."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy
    from rgi_utils.spec import RestraintSpec, RmsdArrays

    n = 4
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((n, 3)) * 2.0
    pos = ref + rng.standard_normal((n, 3)) * 0.5  # nonzero rmsd -> nonzero energy
    idx = np.arange(n).reshape(1, n)
    spec = RestraintSpec(
        n_active=n,
        active_sites=np.arange(n),
        rmsd=RmsdArrays(
            fit_idx=idx,
            fit_mask=np.ones((1, n)),
            fit_ref=ref.reshape(1, n, 3),
            calc_idx=idx,
            calc_mask=np.ones((1, n)),
            calc_ref=ref.reshape(1, n, 3),
            target1=np.array([0.0]),
            target2=np.array([0.0]),
            geom_type=np.array([0]),
            weight=np.array([1.0]),
            start_sigma=np.array([10.0]),
            stop_sigma=np.array([2.0]),
            start_step=np.full(1, float("-inf")),
            stop_step=np.full(1, float("inf")),
            mask=np.array([1.0]),
        ),
        conf_start_sigma=-1.0,
    )
    for name, mod, p, prep in (
        ("numpy", numpy_energy, pos, numpy_energy.prepare_spec(spec)),
        (
            "torch",
            torch_energy,
            torch.tensor(pos),
            torch_energy.prepare_spec(spec, dtype=torch.float64),
        ),
        ("jax", jax_energy, jnp.asarray(pos), jax_energy.prepare_spec(spec)),
    ):
        e_below = float(mod.total_energy(p, prep, sigma=1.0))  # < stop -> released
        e_in = float(mod.total_energy(p, prep, sigma=5.0))  # in window -> active
        e_above = float(mod.total_energy(p, prep, sigma=20.0))  # > start -> not on yet
        assert e_below == pytest.approx(0.0, abs=1e-9), (name, e_below)
        assert e_in > 1e-3, (name, e_in)
        assert e_above == pytest.approx(0.0, abs=1e-9), (name, e_above)


def test_conformer_distance_stop_sigma_window():
    """stop_sigma releases the CONFORMER terms (shared conf_stop_sigma) and the DISTANCE
    terms (per-restraint stop_sigma) below it, exactly like rmsd: the energy is 0 for
    sigma < stop, active in [stop, start], 0 above start — identical across the backends
    (the conformer cg + distance sigma_gate share one window). -1 = never released."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy
    from rgi_utils.spec import BondArrays, DistanceArrays, RestraintSpec

    # bond 0-1 stretched (3 vs r0 1.5) and dist groups off target (centroid gap 8 vs 2)
    pos = np.array([[0.0, 0, 0], [3.0, 0, 0], [0, 5.0, 0], [0, 8.0, 0]])
    spec = RestraintSpec(
        n_active=4,
        active_sites=np.arange(4),
        bond=BondArrays(
            idx=np.array([[0, 1]], dtype=np.int64),
            r0=np.array([1.5]),
            slack=np.array([0.0]),
            weight=np.array([1.0]),
            half=np.array([0.0]),
            mask=np.array([1.0]),
        ),
        distance=DistanceArrays(
            grp1_idx=np.array([[0, 1]], dtype=np.int64),
            grp2_idx=np.array([[2, 3]], dtype=np.int64),
            grp1_mask=np.array([[1.0, 1.0]]),
            grp2_mask=np.array([[1.0, 1.0]]),
            target1=np.array([2.0]),
            target2=np.array([0.0]),
            dist_type=np.array([0], dtype=np.int64),
            move_mode=np.array([0], dtype=np.int64),
            weight=np.array([1.0]),
            mask=np.array([1.0]),
            start_sigma=np.array([10.0]),
            stop_sigma=np.array([2.0]),
            start_step=np.full(1, float("-inf")),
            stop_step=np.full(1, float("inf")),
        ),
        conf_start_sigma=10.0,
        conf_stop_sigma=2.0,
    )
    for name, mod, p, prep in (
        ("numpy", numpy_energy, pos, numpy_energy.prepare_spec(spec)),
        (
            "torch",
            torch_energy,
            torch.tensor(pos),
            torch_energy.prepare_spec(spec, dtype=torch.float64),
        ),
        ("jax", jax_energy, jnp.asarray(pos), jax_energy.prepare_spec(spec)),
    ):
        e_below = float(mod.total_energy(p, prep, sigma=1.0))  # < stop -> released
        e_in = float(mod.total_energy(p, prep, sigma=5.0))  # in window -> active
        e_above = float(mod.total_energy(p, prep, sigma=20.0))  # > start -> off
        assert e_below == pytest.approx(0.0, abs=1e-9), (name, e_below)
        assert e_in > 1e-3, (name, e_in)
        assert e_above == pytest.approx(0.0, abs=1e-9), (name, e_above)


def test_step_window_gating_parity():
    """The STEP window (start_step/stop_step) gates the CONFORMER terms (shared
    conf_start_step/conf_stop_step) and the DISTANCE terms (per-restraint) on the
    diffusion STEP index, exactly as the sigma window gates on noise: energy is 0 for
    step < start_step, active in [start_step, stop_step], 0 above — identical across
    backends. ``sigma=None`` so only the step axis gates (the two axes are mutually
    exclusive per restraint anyway). Mirrors test_conformer_distance_stop_sigma_window."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy
    from rgi_utils.spec import BondArrays, DistanceArrays, RestraintSpec

    # bond 0-1 stretched (3 vs r0 1.5) and dist groups off target (centroid gap ~6.7 vs 2)
    pos = np.array([[0.0, 0, 0], [3.0, 0, 0], [0, 5.0, 0], [0, 8.0, 0]])
    spec = RestraintSpec(
        n_active=4,
        active_sites=np.arange(4),
        bond=BondArrays(
            idx=np.array([[0, 1]], dtype=np.int64),
            r0=np.array([1.5]),
            slack=np.array([0.0]),
            weight=np.array([1.0]),
            half=np.array([0.0]),
            mask=np.array([1.0]),
        ),
        distance=DistanceArrays(
            grp1_idx=np.array([[0, 1]], dtype=np.int64),
            grp2_idx=np.array([[2, 3]], dtype=np.int64),
            grp1_mask=np.array([[1.0, 1.0]]),
            grp2_mask=np.array([[1.0, 1.0]]),
            target1=np.array([2.0]),
            target2=np.array([0.0]),
            dist_type=np.array([0], dtype=np.int64),
            move_mode=np.array([0], dtype=np.int64),
            weight=np.array([1.0]),
            mask=np.array([1.0]),
            # sigma window always-on (the unused axis); the step window [5, 10] gates.
            start_sigma=np.array([float("inf")]),
            stop_sigma=np.array([-1.0]),
            start_step=np.array([5.0]),
            stop_step=np.array([10.0]),
        ),
        conf_start_sigma=float("inf"),
        conf_stop_sigma=-1.0,
        conf_start_step=5.0,
        conf_stop_step=10.0,
    )
    for name, mod, p, prep in (
        ("numpy", numpy_energy, pos, numpy_energy.prepare_spec(spec)),
        (
            "torch",
            torch_energy,
            torch.tensor(pos),
            torch_energy.prepare_spec(spec, dtype=torch.float64),
        ),
        ("jax", jax_energy, jnp.asarray(pos), jax_energy.prepare_spec(spec)),
    ):
        e_before = float(
            mod.total_energy(p, prep, sigma=None, step=3)
        )  # < start -> off
        e_in = float(
            mod.total_energy(p, prep, sigma=None, step=7)
        )  # in window -> active
        e_after = float(mod.total_energy(p, prep, sigma=None, step=12))  # > stop -> off
        assert e_before == pytest.approx(0.0, abs=1e-9), (name, e_before)
        assert e_in > 1e-3, (name, e_in)
        assert e_after == pytest.approx(0.0, abs=1e-9), (name, e_after)


def test_chiral_flat_bottom_zero_at_reference():
    """chiral_energy is flat-bottomed around vol0: ZERO within ±slack (so the
    reference geometry has zero energy), quadratic outside, equal across backends.
    Regression for the old vol0∓slack shifted harmonic (nonzero floor at the
    reference + a minimum biased toward chiral inversion)."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy

    # atom 0 is the chiral center; vol0 = the reference scalar triple product
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    v1, v2, v3 = pos[1] - pos[0], pos[2] - pos[0], pos[3] - pos[0]
    vol0 = float(np.dot(v1, np.cross(v2, v3)))  # = 1.0
    spec = RestraintSpec(
        n_active=4,
        active_sites=np.arange(4),
        chiral=ChiralArrays(
            idx=np.array([[0, 1, 2, 3]], dtype=np.int64),
            vol0=np.array([vol0]),
            slack=np.array([0.05]),
            weight=np.array([0.1]),
            mask=np.array([1.0]),
        ),
        conf_start_sigma=10.0,
    )
    prep_np = numpy_energy.prepare_spec(spec)
    # at the reference geometry vol == vol0 -> inside the band -> ZERO (old code
    # returned weight*slack**2 = 2.5e-4 here)
    assert float(numpy_energy.total_energy(pos, prep_np)) == pytest.approx(
        0.0, abs=1e-12
    )

    # outside the band the penalty is non-zero and identical across backends
    pos2 = pos.copy()
    pos2[1, 0] = 3.0  # stretches the volume well past vol0 + slack
    prep_t = torch_energy.prepare_spec(spec, dtype=torch.float64)
    prep_j = jax_energy.prepare_spec(spec)
    e_np = float(numpy_energy.total_energy(pos2, prep_np))
    e_t = float(
        torch_energy.total_energy(torch.tensor(pos2, dtype=torch.float64), prep_t)
    )
    e_j = float(jax_energy.total_energy(jnp.asarray(pos2), prep_j))
    assert e_np > 0.0
    assert abs(e_np - e_t) < 1e-6 and abs(e_np - e_j) < 1e-6


def test_planarity_flat_bottom_zero_at_reference():
    """planarity reuses chiral_energy: a planar sp2 centre (vol0 ~ 0) has
    ZERO energy at the reference geometry, quadratic once it pyramidalises out of plane,
    equal across backends. Proves planarity is wired through _terms to chiral_energy."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy

    # atom 0 is the sp2 centre; 1/2/3 its three neighbours, all coplanar -> vol0 = 0
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    v1, v2, v3 = pos[1] - pos[0], pos[2] - pos[0], pos[3] - pos[0]
    vol0 = float(np.dot(v1, np.cross(v2, v3)))  # = 0.0 (planar)
    assert vol0 == pytest.approx(0.0, abs=1e-12)
    spec = RestraintSpec(
        n_active=4,
        active_sites=np.arange(4),
        planarity=PlanarityArrays(
            idx=np.array([[0, 1, 2, 3]], dtype=np.int64),
            vol0=np.array([vol0]),
            slack=np.array([0.05]),
            weight=np.array([0.1]),
            mask=np.array([1.0]),
        ),
        conf_start_sigma=10.0,
    )
    prep_np = numpy_energy.prepare_spec(spec)
    # at the planar reference the volume is 0 == vol0 -> inside the band -> ZERO
    assert float(numpy_energy.total_energy(pos, prep_np)) == pytest.approx(
        0.0, abs=1e-12
    )
    # the energy_breakdown reports the term under the dedicated "planarity" key
    bd = numpy_energy.energy_breakdown(pos, prep_np)
    assert "planarity" in bd

    # pyramidalise the centre out of its neighbours' plane -> non-zero, parity-equal
    pos2 = pos.copy()
    pos2[3, 2] = 1.0  # lift atom 3 out of the z=0 plane
    prep_t = torch_energy.prepare_spec(spec, dtype=torch.float64)
    prep_j = jax_energy.prepare_spec(spec)
    e_np = float(numpy_energy.total_energy(pos2, prep_np))
    e_t = float(
        torch_energy.total_energy(torch.tensor(pos2, dtype=torch.float64), prep_t)
    )
    e_j = float(jax_energy.total_energy(jnp.asarray(pos2), prep_j))
    assert e_np > 0.0
    assert abs(e_np - e_t) < 1e-6 and abs(e_np - e_j) < 1e-6


def test_rmsd_fit_calc_separation():
    """Superpose on the FIT atoms, measure RMSD on the CALC atoms. For a target that
    is a RIGID transform of the reference, fitting on the fit subset aligns the calc
    atoms perfectly too -> calc RMSD ~ 0 (energy -> 0)."""
    rng = np.random.default_rng(7)
    nf, nc = 5, 4
    ref = rng.standard_normal((nf + nc, 3)) * 3.0  # 0..nf-1 = fit, nf.. = calc
    th = 0.5
    rz = np.array(
        [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    )
    pos = ref @ rz.T + np.array([2.0, -1.0, 4.0])  # rigid (no noise)
    e = float(
        numpy_energy.rmsd_energy(
            pos,
            np.arange(nf).reshape(1, nf),
            np.ones((1, nf)),
            ref[:nf].reshape(1, nf, 3),
            np.arange(nf, nf + nc).reshape(1, nc),
            np.ones((1, nc)),
            ref[nf:].reshape(1, nc, 3),
            target1=np.array([0.0]),
            target2=np.array([0.0]),
            geom_type=np.array([0]),
            weight=np.array([1.0]),
            mask=np.array([1.0]),
        )
    )
    assert e < 1e-6, e


def test_vdw_fixed_background_torch_jax_parity():
    """The dynamic fixed-background VdW lives in the OPTIMIZERS (not the energy layer that
    the numpy reference covers), so the jnp port is otherwise unguarded. Here torch and
    jax must agree on value AND gradient w.r.t. the moving active coords."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim._torch_cg_gpu import _vdw_pair_energy as t_vdw
    from rgi_utils.optim.jax_optim import _vdw_pair_energy as j_vdw

    rng = np.random.default_rng(0)
    n_active, n_bg = 6, 5
    active = rng.standard_normal((n_active, 3))
    bg = rng.standard_normal((n_bg, 3))  # same cluster -> guaranteed contacts
    lig_local = np.array([0, 2, 4], dtype=np.int64)  # 3 of 6 active atoms are ligand
    lig_r = np.array([1.7, 1.5, 1.6])
    bg_r = np.array([1.7, 1.5, 1.6, 1.55, 1.8])
    scale, weight = 0.9, 2.0

    at = torch.tensor(active, requires_grad=True)
    e_t = t_vdw(
        at,
        torch.tensor(bg),
        torch.tensor(lig_local),
        torch.tensor(lig_r),
        torch.tensor(bg_r),
        scale,
        weight,
    )
    e_t.backward()
    g_t = at.grad.numpy()

    def jf(a):
        return j_vdw(
            a,
            jnp.asarray(bg),
            jnp.asarray(lig_local),
            jnp.asarray(lig_r),
            jnp.asarray(bg_r),
            scale,
            weight,
        )

    e_j, g_j = jax.value_and_grad(jf)(jnp.asarray(active))
    assert float(e_t.detach()) > 0.0  # the case actually exercises the repulsion
    assert abs(float(e_t.detach()) - float(e_j)) < 1e-8
    assert np.allclose(g_t, np.asarray(g_j), atol=1e-8), (
        f"max|d|={np.abs(g_t - np.asarray(g_j)).max()}"
    )
