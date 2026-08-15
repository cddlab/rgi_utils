"""Optimizer sanity: each backend reduces restraint energy on a distorted ligand."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rgi_utils.atom_context import LigandConf
from rgi_utils.featurizer import build_spec
from rgi_utils.group_geom_restr_data import AngleRestraintData, DihedralRestraintData
from rgi_utils.plane_restr_data import PlaneRestraintData


def _distorted_ethane():
    m = Chem.MolFromSmiles("CC")
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=1)
    c = np.asarray(m.GetConformer().GetPositions())
    n = m.GetNumAtoms()
    lc = LigandConf(
        mol=m, conf_coords=c, global_indices=np.arange(n), conformer_restraints=True
    )
    # non-empty conformer_config so the opt-in gate builds conformer terms;
    # conf_start_sigma large so they are active at any sigma
    spec = build_spec(
        [lc],
        [],
        {"bond": {"weight": 1.0}, "angle": {"weight": 1.0}},
        conf_start_sigma=1e30,
    )

    n_atom = n + 5  # padding atoms beyond the ligand
    coords = np.zeros((1, n_atom, 3))
    coords[0, :n, :] = c
    rng = np.random.default_rng(0)
    coords[0, spec.active_sites, :] += rng.standard_normal((spec.n_active, 3)) * 0.3
    return spec, coords


def test_torch_minimize_reduces_energy():
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec, coords_np = _distorted_ethane()
    coords = torch.tensor(coords_np, dtype=torch.float64)
    opt = TorchRestraintOptimizer(spec, max_iter=100)
    e0 = opt.energy(coords)
    opt.minimize(coords)
    e1 = opt.energy(coords)
    assert e1 < 0.5 * e0, f"{e0} -> {e1}"


def test_jax_minimize_reduces_energy():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import energy_of, make_minimizer

    spec, coords_np = _distorted_ethane()
    coords = jnp.asarray(coords_np)
    e0 = energy_of(spec, coords)
    # gating uses spec.max_start_sigma() (here conf_start_sigma=1e30), so sigma=0 runs
    minimize = make_minimizer(spec, max_iter=2000)
    coords = minimize(coords, 0.0)
    e1 = energy_of(spec, coords)
    assert e1 < 0.5 * e0, f"{e0} -> {e1}"


def test_jax_minimizer_step_window_traced_under_jit():
    """The jax minimizer (the exact path AF3 drives in its hk.scan) gates a distance
    restraint on its STEP window with ``step`` as a TRACED value under ``jax.jit`` — not a
    concrete Python int. This covers the distance term's per-entry step gate + ``_descend``
    /``minimize``'s step threading, which the concrete-``step`` energy parity test does NOT
    exercise (different module). OUTSIDE [start_step, stop_step] the distance term is gated
    off (centroid separation untouched); INSIDE it the CG lands the separation on target."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer
    from rgi_utils.spec import DistanceArrays, RestraintSpec

    # distance restraint with a STEP window [5, 10]; sigma window stays always-on so only
    # the step axis gates (start_sigma=+inf -> max_start_sigma=+inf -> lax.cond never skips).
    spec = RestraintSpec(
        n_active=4,
        active_sites=np.arange(4),
        distance=DistanceArrays(
            grp1_idx=np.array([[0, 1]], dtype=np.int64),
            grp2_idx=np.array([[2, 3]], dtype=np.int64),
            grp1_mask=np.array([[1.0, 1.0]]),
            grp2_mask=np.array([[1.0, 1.0]]),
            target1=np.array([7.0]),
            target2=np.array([0.0]),
            dist_type=np.array([0], dtype=np.int64),
            move_mode=np.array([0], dtype=np.int64),
            weight=np.array([1.0]),
            mask=np.array([1.0]),
            start_sigma=np.array([float("inf")]),
            stop_sigma=np.array([-1.0]),
            start_step=np.array([5.0]),
            stop_step=np.array([10.0]),
        ),
    )
    # jit so all of (coords, sigma, step) become tracers — the AF3-in-scan condition.
    mini = jax.jit(make_minimizer(spec, max_iter=50))
    base = np.zeros((4, 3))
    base[2:, 0] = 20.0  # centroid separation 20 along x (off the 7.0 target)

    def cdist(c):
        c = np.asarray(c)
        return float(np.linalg.norm(c[2:].mean(0) - c[:2].mean(0)))

    out_before = mini(jnp.asarray(base), 5.0, 3)  # step 3 < start_step -> gated off
    out_in = mini(jnp.asarray(base), 5.0, 7)  # step 7 in window -> active
    out_after = mini(jnp.asarray(base), 5.0, 12)  # step 12 > stop_step -> gated off
    assert cdist(out_before) == pytest.approx(20.0, abs=1e-5), cdist(out_before)
    assert abs(cdist(out_in) - 7.0) < 1e-4, cdist(out_in)
    assert cdist(out_after) == pytest.approx(20.0, abs=1e-5), cdist(out_after)


def test_jax_minimizer_move_mode_end_to_end():
    """jax E2E (build_spec -> jax_energy.prepare_spec -> pack_spec -> make_minimizer -> the
    CG): move_mode=2 pins group1 and moves ONLY group2 to meet the centroid-distance target.
    Confirms move_mode flows via the featurizer and the SHARED pack_spec into the jax prepared
    dict, then through the jax minimizer (the AF3 closure path). Distance is now CG-minimised
    (the old apply_distance_shift_jax closed-form is gone); the move_mode pin is the
    centroid_eff free=0 stop_gradient, so group1 stays EXACTLY fixed. NOTE: unlike the
    closed-form (which shifted along the axis to the NEAR side), the CG can land the moving
    group on EITHER side of the pin (both satisfy |c2-c1|=target), so the side is not
    asserted -- the contract is pin + target reached."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.distance_restr_data import DistanceData
    from rgi_utils.optim.jax_optim import make_minimizer

    dd = DistanceData()
    dd.target_sites1 = [0, 1]
    dd.target_sites2 = [2, 3]
    dd.distance_restraint_type = "harmonic"
    dd.target_distance = 7.0
    dd.move_mode = 2  # only group2 (atom_selection2) moves
    dd.run_restr = True
    dd.start_sigma = 1e30
    spec = build_spec(
        [], [dd], {}, elements=np.zeros(4, dtype=np.int64), conf_start_sigma=1e30
    )
    assert list(spec.distance.move_mode) == [2]  # flowed into the spec via featurizer

    coords_np = np.zeros((1, 4, 3))
    coords_np[0, 2:, 0] = (
        20.0  # centroid1 (group1) x=0, centroid2 (group2) x=20 -> dist 20
    )
    coords = jnp.asarray(coords_np)
    g1_before = np.asarray(coords[0, :2])
    minimize = make_minimizer(spec, max_iter=100)
    coords = minimize(coords, 0.0)
    a = np.asarray(coords)
    gap = np.linalg.norm(a[0, 2:].mean(0) - a[0, :2].mean(0))
    assert abs(gap - 7.0) < 1e-5  # centroid gap on target
    assert np.allclose(
        a[0, :2], g1_before, atol=1e-9
    )  # group1 EXACTLY pinned (move_mode=2)
    # group2 alone moved to meet the restraint; group1 is fixed at x=0, so |centroid2_x| = 7.
    # The CG may land it on either side (-7 or +7) -- both give gap 7 -- so check |x|, not sign.
    assert abs(abs(a[0, 2:].mean(0)[0]) - 7.0) < 1e-5  # only group2 moved, to |x| = 7


def test_distance_minimal_displacement_split_torch_jax():
    """move_mode=0 (both groups move) reproduces the old closed-form's minimal-displacement
    split via the reduced-mass centroid_eff scale (``mu = N1*N2/(N1+N2)``): the per-group
    centroid shifts satisfy ``|s1|:|s2| = N2:N1`` (the SMALLER group moves more), the gap
    reaches target, and torch and jax agree. N1=3 != N2=1 so the split is non-trivial (1:3);
    the gap 12 -> 4 never crosses 0, so there is no near/far-side ambiguity here."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer
    from rgi_utils.spec import DistanceArrays, RestraintSpec

    spec = RestraintSpec(
        n_active=4,
        active_sites=np.arange(4),
        distance=DistanceArrays(
            grp1_idx=np.array([[0, 1, 2]], dtype=np.int64),  # N1 = 3
            grp2_idx=np.array([[3, 0, 0]], dtype=np.int64),  # N2 = 1
            grp1_mask=np.array([[1.0, 1.0, 1.0]]),
            grp2_mask=np.array([[1.0, 0.0, 0.0]]),  # group2 is a single atom
            target1=np.array([4.0]),
            target2=np.array([0.0]),
            dist_type=np.array([0], dtype=np.int64),
            move_mode=np.array([0], dtype=np.int64),  # both move (minimal-displacement)
            weight=np.array([1.0]),
            mask=np.array([1.0]),
            start_sigma=np.array([1e30]),
            stop_sigma=np.array([-1.0]),
            start_step=np.full(1, float("-inf")),
            stop_step=np.full(1, float("inf")),
        ),
    )
    # group1 centroid at x=0 (3 atoms), group2 (single atom) at x=12 -> gap 12, target 4
    base = np.array(
        [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [12.0, 0.0, 0.0]]
    )
    c1_0, c2_0 = base[:3].mean(0), base[3]

    def split(a):
        c1, c2 = a[:3].mean(0), a[3]
        return (
            float(np.linalg.norm(c2 - c1)),  # gap
            float(np.linalg.norm(c1 - c1_0)),  # |s1|
            float(np.linalg.norm(c2 - c2_0)),  # |s2|
        )

    at = torch.tensor(base, dtype=torch.float64)
    TorchRestraintOptimizer(spec, max_iter=200).minimize(at)
    aj = np.asarray(make_minimizer(spec, max_iter=200)(jnp.asarray(base), 0.0))

    for gap, s1, s2 in (split(at.numpy()), split(aj)):
        assert abs(gap - 4.0) < 1e-3, gap  # target reached
        assert abs(s1 / s2 - 1.0 / 3.0) < 1e-2, (s1, s2)  # |s1|:|s2| = N2:N1 = 1:3
    assert np.allclose(at.numpy(), aj, atol=1e-4)  # torch == jax


def test_distance_move_mode1_pins_group2_torch_jax():
    """move_mode=1 moves ONLY group1 (atom_selection1) and pins group2 -- the 1/3 distance
    move path that no optimizer test exercised (existing tests cover mode 0 and 2). Built
    through the featurizer (asserts move_mode flowed into the spec) and run on BOTH torch and
    jax: group2 stays EXACTLY fixed (the centroid_eff free=0 stop_gradient) and the centroid
    gap reaches target."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.distance_restr_data import DistanceData
    from rgi_utils.optim.jax_optim import make_minimizer
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    dd = DistanceData()
    dd.target_sites1, dd.target_sites2 = [0, 1], [2, 3]
    dd.distance_restraint_type = "harmonic"
    dd.target_distance = 7.0
    dd.move_mode = 1  # only group1 (atom_selection1) moves
    dd.run_restr = True
    dd.start_sigma = 1e30
    spec = build_spec(
        [], [dd], {}, elements=np.zeros(4, dtype=np.int64), conf_start_sigma=1e30
    )
    assert list(spec.distance.move_mode) == [1]  # flowed into the spec via featurizer

    base = np.zeros((4, 3))
    base[2:, 0] = 20.0  # group1 centroid x=0, group2 centroid x=20 -> gap 20, target 7
    g2_before = base[2:].copy()

    at = torch.tensor(base, dtype=torch.float64)
    TorchRestraintOptimizer(spec, max_iter=100).minimize(at)
    aj = np.asarray(make_minimizer(spec, max_iter=100)(jnp.asarray(base), 0.0))

    for a in (at.numpy(), aj):
        gap = np.linalg.norm(a[2:].mean(0) - a[:2].mean(0))
        assert abs(gap - 7.0) < 1e-4, gap  # centroid gap on target
        # group2 fixed at x=20 stays EXACTLY pinned; centroid1 lands at 20 +/- 7.
        assert np.allclose(a[2:], g2_before, atol=1e-9)
        assert abs(abs(a[:2].mean(0)[0] - 20.0) - 7.0) < 1e-4  # only group1 moved
    assert np.allclose(at.numpy(), aj, atol=1e-3)  # torch == jax


def test_distance_coupled_weight_balance_torch_jax():
    """An atom shared by two OVER-CONSTRAINED distance restraints settles at the least-squares
    weighted balance ``B = (t1*w1 + t2*w2)/(w1+w2)`` -- the ONLY case where the per-entry
    ``weight`` is not a no-op (a single / disjoint restraint reaches its target regardless of
    weight, since the CG fully converges). Both restraints pin the shared anchor (move_mode=2)
    at the origin and pull the shared atom B to targets t1/t2 along x, so B converges to their
    weighted average; torch and jax agree."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer
    from rgi_utils.spec import DistanceArrays, RestraintSpec

    t1, t2, w1, w2 = 4.0, 10.0, 1.0, 3.0
    expected = (t1 * w1 + t2 * w2) / (w1 + w2)  # 8.5 (weighted toward t2)
    spec = RestraintSpec(
        n_active=2,
        active_sites=np.arange(2),
        distance=DistanceArrays(
            grp1_idx=np.array([[0], [0]], dtype=np.int64),  # anchor (shared, pinned)
            grp2_idx=np.array([[1], [1]], dtype=np.int64),  # atom B (shared, free)
            grp1_mask=np.array([[1.0], [1.0]]),
            grp2_mask=np.array([[1.0], [1.0]]),
            target1=np.array([t1, t2]),
            target2=np.array([0.0, 0.0]),
            dist_type=np.array([0, 0], dtype=np.int64),  # harmonic
            move_mode=np.array([2, 2], dtype=np.int64),  # pin group1 (the anchor)
            weight=np.array([w1, w2]),
            mask=np.array([1.0, 1.0]),
            start_sigma=np.array([1e30, 1e30]),
            stop_sigma=np.array([-1.0, -1.0]),
            start_step=np.full(2, float("-inf")),
            stop_step=np.full(2, float("inf")),
        ),
    )
    base = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])  # anchor at origin, B at x=6

    at = torch.tensor(base, dtype=torch.float64)
    TorchRestraintOptimizer(spec, max_iter=300).minimize(at)
    aj = np.asarray(make_minimizer(spec, max_iter=300)(jnp.asarray(base), 0.0))

    for a in (at.numpy(), aj):
        assert np.allclose(a[0], [0.0, 0.0, 0.0], atol=1e-9)  # anchor pinned (mode 2)
        assert abs(a[1, 0] - expected) < 1e-3, a[1, 0]  # B at weighted balance (8.5)
    assert np.allclose(at.numpy(), aj, atol=1e-4)  # torch == jax


# --- group-centroid angle / dihedral restraints ---------------------------------------


def _angle_deg(p1, p2, p3):
    """Angle (degrees) at vertex p2 of the points p1-p2-p3."""
    v1, v3 = p1 - p2, p3 - p2
    cos = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _dihedral_deg(p0, p1, p2, p3):
    """Dihedral (degrees) about the p1-p2 axis."""
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    b2n = b2 / np.linalg.norm(b2)
    m = np.cross(n1, b2n)
    return float(np.degrees(np.arctan2(np.dot(m, n2), np.dot(n1, n2))))


def _wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def _group_angle_spec(target_deg, start_sigma=1e30, move_free=(True, True, True)):
    """Spec with one harmonic group-centroid angle restraint: groups {0,1}/{2,3}/{4,5}
    (vertex=2). ``move_free`` is the per-group free mask (default all free)."""
    ad = AngleRestraintData()
    ad.target_sites1, ad.target_sites2, ad.target_sites3 = [0, 1], [2, 3], [4, 5]
    ad.geom_type, ad.target1, ad.target2 = "harmonic", math.radians(target_deg), 0.0
    ad.move_free, ad.weight, ad.run_restr = move_free, 1.0, True
    ad.start_sigma, ad.stop_sigma = start_sigma, -1.0
    return build_spec(angle_restraints=[ad], conf_start_sigma=1e30)


def _group_plane_spec(start_sigma=1e30, move_free=(True,), groups=None):
    """Spec with one standalone best-fit-plane restraint (``plane_restraints_config``).
    ``groups`` defaults to one 6-atom group (atoms 0..5) pooled into a single plane."""
    pr = PlaneRestraintData()
    pr.target_sites = groups or [[0, 1, 2, 3, 4, 5]]
    pr.atom_selections = [f"group{i + 1}" for i in range(len(pr.target_sites))]
    pr.geom_type, pr.target1, pr.target2 = "harmonic", 0.0, 0.0
    pr.move_free, pr.weight, pr.run_restr = move_free, 1.0, True
    pr.start_sigma, pr.stop_sigma = start_sigma, -1.0
    return build_spec(plane_restraints=[pr], conf_start_sigma=1e30)


def _group_angle_coords():
    """group1 centroid (8,0,0), vertex centroid (0,0,0), group3 centroid (0,8,0) -> initial 90 deg."""
    coords = np.zeros((1, 6, 3))
    coords[0, 0], coords[0, 1] = [8.0, 0.5, 0.0], [8.0, -0.5, 0.0]
    coords[0, 2], coords[0, 3] = [0.0, 0.5, 0.0], [0.0, -0.5, 0.0]
    coords[0, 4], coords[0, 5] = [0.5, 8.0, 0.0], [-0.5, 8.0, 0.0]
    return coords


def _coms(a):
    return a[[0, 1]].mean(0), a[[2, 3]].mean(0), a[[4, 5]].mean(0)


def _group_dihedral_spec(target_deg, move_free=(True, True, True, True)):
    """Spec with one group-centroid dihedral restraint over single-atom groups 0-1-2-3.
    ``move_free`` is the per-group free mask (default all free)."""
    dd = DihedralRestraintData()
    dd.target_sites1, dd.target_sites2 = [0], [1]
    dd.target_sites3, dd.target_sites4 = [2], [3]
    dd.geom_type, dd.target1, dd.target2 = "harmonic", math.radians(target_deg), 0.0
    dd.move_free, dd.weight, dd.run_restr = move_free, 1.0, True
    dd.start_sigma, dd.stop_sigma = 1e30, -1.0
    return build_spec(dihedral_restraints=[dd], conf_start_sigma=1e30)


def _group_dihedral_coords():
    """p0(0,1,0)-p1(0,0,0)-p2(1,0,0)-p3(1,1,0): planar, initial dihedral 0 deg."""
    coords = np.zeros((1, 4, 3))
    coords[0, 0] = [0.0, 1.0, 0.0]
    coords[0, 1] = [0.0, 0.0, 0.0]
    coords[0, 2] = [1.0, 0.0, 0.0]
    coords[0, 3] = [1.0, 1.0, 0.0]
    return coords


def test_torch_group_angle_converges():
    """The torch CG bends the centroid1-centroid2-centroid3 angle from 90 deg onto a 120 deg target,
    moving each group rigidly (the centroid-only energy gives a group's atoms equal grad)."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec = _group_angle_spec(120.0)
    coords = torch.tensor(_group_angle_coords(), dtype=torch.float64)
    c1, c2, c3 = _coms(coords.numpy()[0])
    assert abs(_angle_deg(c1, c2, c3) - 90.0) < 1e-6  # initial
    TorchRestraintOptimizer(spec, max_iter=500).minimize(coords)
    c1, c2, c3 = _coms(coords.numpy()[0])
    assert abs(_angle_deg(c1, c2, c3) - 120.0) < 1.0


def test_jax_group_angle_converges():
    """Same group-centroid angle convergence on the jax minimizer (the AF3 path)."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer

    spec = _group_angle_spec(120.0)
    coords = make_minimizer(spec, max_iter=500)(jnp.asarray(_group_angle_coords()), 0.0)
    c1, c2, c3 = _coms(np.asarray(coords)[0])
    assert abs(_angle_deg(c1, c2, c3) - 120.0) < 1.0


def test_torch_group_dihedral_converges():
    """The torch CG drives the centroid dihedral from 0 deg onto a 90 deg target (90 deg is
    mid-range, away from the +-180 wrap boundary, so convergence is unambiguous)."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec = _group_dihedral_spec(90.0)
    coords = torch.tensor(_group_dihedral_coords(), dtype=torch.float64)
    a = coords.numpy()[0]
    assert abs(_dihedral_deg(a[0], a[1], a[2], a[3])) < 1e-6  # initial 0 deg
    TorchRestraintOptimizer(spec, max_iter=500).minimize(coords)
    a = coords.numpy()[0]
    assert abs(_wrap180(_dihedral_deg(a[0], a[1], a[2], a[3]) - 90.0)) < 1.0


def test_torch_group_angle_gating_noop_above_start_sigma():
    """A step at sigma above the restraint's start_sigma leaves coords unchanged
    (the per-restraint gate); below it the restraint activates and converges."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec = _group_angle_spec(120.0, start_sigma=1.0)
    coords = torch.tensor(_group_angle_coords(), dtype=torch.float64)
    before = coords.clone()
    TorchRestraintOptimizer(spec, max_iter=500).minimize(coords, sigma=5.0)  # 5 > 1
    assert torch.allclose(coords, before, atol=1e-12)  # gated off -> no change
    TorchRestraintOptimizer(spec, max_iter=500).minimize(coords, sigma=0.5)  # 0.5 <= 1
    c1, c2, c3 = _coms(coords.numpy()[0])
    assert abs(_angle_deg(c1, c2, c3) - 120.0) < 1.0  # now active -> converged


def test_torch_group_angle_move_pins_other_groups():
    """move:1 moves ONLY group 1; groups 2 and 3 stay EXACTLY fixed (atol=1e-9) while
    the angle still reaches target — the group analogue of the distance move E2E."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec = _group_angle_spec(120.0, move_free=(True, False, False))  # only group 1 free
    coords = torch.tensor(_group_angle_coords(), dtype=torch.float64)  # initial 90 deg
    pinned_before = coords[0, [2, 3, 4, 5], :].clone()  # groups 2 + 3 atoms
    TorchRestraintOptimizer(spec, max_iter=500).minimize(coords)
    c1, c2, c3 = _coms(coords.numpy()[0])
    assert abs(_angle_deg(c1, c2, c3) - 120.0) < 1.0  # target reached
    assert torch.allclose(coords[0, [2, 3, 4, 5], :], pinned_before, atol=1e-9)


def test_torch_group_angle_default_move_converges():
    """The DEFAULT angle move (groups 1+3 free, vertex group 2 pinned — what a bare
    angle_restraints_config entry gets) reaches target while the vertex stays put."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec = _group_angle_spec(120.0, move_free=(True, False, True))  # the default
    coords = torch.tensor(_group_angle_coords(), dtype=torch.float64)  # initial 90 deg
    pinned_before = coords[0, [2, 3], :].clone()  # vertex group (atoms 2,3)
    TorchRestraintOptimizer(spec, max_iter=500).minimize(coords)
    c1, c2, c3 = _coms(coords.numpy()[0])
    assert abs(_angle_deg(c1, c2, c3) - 120.0) < 1.0
    assert torch.allclose(coords[0, [2, 3], :], pinned_before, atol=1e-9)


def test_group_angle_step_gated_off_is_noop_torch_jax():
    """A group-angle SOLVER term with a STEP window no-ops OUTSIDE its window and converges
    INSIDE it — exercised end-to-end through the CG on both backends (distance has its own
    step-window e2e test, but it is closed-form; this is the gradient-solver path). Unlike
    the custom closure path (a gated-off custom term is DROPPED -> a constant objective ->
    the value_grad guard), an array term's gate is a multiplicative 0 inside total_energy,
    so the graph stays connected (grad 0) and the CG no-ops without needing the guard. Both
    backends must agree the gated-off step leaves the coords untouched."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    ad = AngleRestraintData()
    ad.target_sites1, ad.target_sites2, ad.target_sites3 = [0, 1], [2, 3], [4, 5]
    ad.geom_type, ad.target1, ad.target2 = "harmonic", math.radians(120.0), 0.0
    ad.move_free, ad.weight, ad.run_restr = (True, True, True), 1.0, True
    ad.start_sigma, ad.stop_sigma = (
        1e30,
        -1.0,
    )  # sigma always-on (gate is the step window)
    ad.start_step, ad.stop_step = 5.0, 10.0  # active only for step in [5, 10]
    spec = build_spec(angle_restraints=[ad], conf_start_sigma=1e30)

    init = _group_angle_coords()  # 90 deg

    # torch: step OUTSIDE the window -> gated off -> coords UNCHANGED (and no crash)
    c_off = torch.tensor(init, dtype=torch.float64)
    TorchRestraintOptimizer(spec, max_iter=200).minimize(c_off, sigma=5.0, step=0)
    assert torch.allclose(c_off, torch.tensor(init), atol=1e-9)

    # torch: step INSIDE the window -> the term IS wired, just gated -> reaches 120 deg
    c_on = torch.tensor(init, dtype=torch.float64)
    TorchRestraintOptimizer(spec, max_iter=500).minimize(c_on, sigma=5.0, step=7)
    a1, a2, a3 = _coms(c_on.numpy()[0])
    assert abs(_angle_deg(a1, a2, a3) - 120.0) < 1.0

    # jax: the same gate-off is already a no-op (jnp.where -> 0 grad); parity check
    out = np.asarray(make_minimizer(spec, max_iter=200)(jnp.asarray(init), 5.0, 0))
    assert np.allclose(out, init, atol=1e-9)


def test_group_move_grad_parity_torch_jax():
    """For move!=both the pinned groups are stop-gradient'd, so the gradient diverges
    from a numpy finite-difference (which moves every atom) — but torch and jax must
    AGREE (both detach identically). The rmsd-style parity carve-out, plus a check that
    the pinned groups really get zero gradient."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy, torch_energy

    spec = _group_angle_spec(120.0, move_free=(False, True, False))  # only group 2 free
    pos = np.random.default_rng(0).standard_normal((spec.n_active, 3)) * 3.0
    pt = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    torch_energy.total_energy(
        pt, torch_energy.prepare_spec(spec, dtype=torch.float64)
    ).backward()
    g_t = pt.grad.numpy()
    g_j = np.asarray(
        jax.grad(lambda x: jax_energy.total_energy(x, jax_energy.prepare_spec(spec)))(
            jnp.asarray(pos)
        )
    )
    assert np.allclose(g_t, g_j, atol=1e-6), np.abs(g_t - g_j).max()
    # groups 1 (atoms 0,1) and 3 (atoms 4,5) are pinned -> exactly zero gradient
    assert np.linalg.norm(g_t[[0, 1, 4, 5]]) < 1e-12
    assert np.linalg.norm(g_t[[2, 3]]) > 1e-6  # group 2 (free) moves


def test_torch_group_dihedral_multi_move():
    """`move` can free SEVERAL groups at once (the user's `move: 1,4`): a dihedral with
    groups 1+4 free and 2+3 pinned. The two pinned axis atoms stay EXACTLY put
    (atol=1e-9) while the dihedral still reaches target."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    # single-atom groups 0,1,2,3; free groups 1 and 4 -> atoms 0 and 3 move
    spec = _group_dihedral_spec(90.0, move_free=(True, False, False, True))
    coords = torch.tensor(_group_dihedral_coords(), dtype=torch.float64)  # init 0 deg
    pinned_before = coords[0, [1, 2], :].clone()  # groups 2,3 (the axis atoms)
    TorchRestraintOptimizer(spec, max_iter=500).minimize(coords)
    a = coords.numpy()[0]
    assert abs(_wrap180(_dihedral_deg(a[0], a[1], a[2], a[3]) - 90.0)) < 1.0
    assert torch.allclose(coords[0, [1, 2], :], pinned_before, atol=1e-9)


def test_gated_prepared_folds_group_gate_cpu():
    """The GPU pre-gate (_gated_prepared) must fold each group restraint's per-entry
    gate into its mask. CPU runs this directly (only torch.compile EXECUTION needs CUDA;
    the mask folding is pure torch). Regression for a per-entry term (group_angle/
    dihedral) silently going UNGATED on the compiled path (the `else: pg[k]=v` branch),
    which the eager CPU CG cannot reveal because it gates live via sigma."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec = _group_angle_spec(120.0, start_sigma=1.0)  # stop_sigma defaults -1 (never)
    opt = TorchRestraintOptimizer(spec, max_iter=10)
    opt._ensure(torch.device("cpu"), torch.float64)
    base = opt._prepared["group_angle"]["mask"].clone()
    assert float(base.sum()) > 0
    # sigma ABOVE start_sigma -> the gate must zero the mask (term off)
    off = opt._gated_prepared(5.0)["group_angle"]["mask"]
    assert float(off.sum()) == 0.0, "group gate not folded: active above start_sigma"
    # sigma INSIDE the window -> mask unchanged (term active)
    on = opt._gated_prepared(0.5)["group_angle"]["mask"]
    assert torch.allclose(on, base), "group term wrongly gated inside its active window"
    # distinct gate states must NOT collide in the compile cache
    assert opt._gated_prepared(5.0) is not opt._gated_prepared(0.5)


def test_gated_prepared_folds_group_plane_gate_cpu():
    """Same regression for the standalone plane term: it is a per-entry (PER_ENTRY_KEYS)
    term, so the GPU pre-gate must fold its window into the mask. Registering it in
    ``TERM_DEFS`` with an entry gate is what makes this work — a ``"conf"`` gate would
    silently tie it to the conformer window instead."""
    torch = pytest.importorskip("torch")
    from rgi_utils.energy._terms import CONF_KEYS, PER_ENTRY_KEYS
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    assert "group_plane" in PER_ENTRY_KEYS and "group_plane" not in CONF_KEYS

    spec = _group_plane_spec(start_sigma=1.0)
    opt = TorchRestraintOptimizer(spec, max_iter=10)
    opt._ensure(torch.device("cpu"), torch.float64)
    base = opt._prepared["group_plane"]["mask"].clone()
    assert float(base.sum()) > 0
    assert float(opt._gated_prepared(5.0)["group_plane"]["mask"].sum()) == 0.0
    assert torch.allclose(opt._gated_prepared(0.5)["group_plane"]["mask"], base)
    assert opt._gated_prepared(5.0) is not opt._gated_prepared(0.5)


def test_torch_vdw_pushes_ligand_off_fixed_protein():
    """Dynamic fixed-background VdW: a ligand atom clashing with a fixed protein
    atom is pushed away, while the protein atom (not in active_sites) stays put."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    m = Chem.MolFromSmiles("CC")
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=1)
    c = np.asarray(m.GetConformer().GetPositions())
    n = m.GetNumAtoms()
    lc = LigandConf(
        mol=m, conf_coords=c, global_indices=np.arange(n), conformer_restraints=True
    )

    n_atom = n + 1  # one extra "protein" atom right after the ligand
    elements = np.zeros(n_atom, dtype=np.int64)
    for i, atom in enumerate(m.GetAtoms()):
        elements[i] = atom.GetAtomicNum()
    elements[n] = 6  # a carbon protein atom (heavy -> VdW background)

    spec = build_spec(
        [lc], [], {"vdw": {"weight": 1.0, "scale": 0.9}}, elements=elements
    )
    assert spec.vdw_config is not None
    assert n in set(int(x) for x in spec.vdw_config.background_global)

    coords_np = np.zeros((1, n_atom, 3))
    coords_np[0, :n, :] = c
    # drop the protein atom 0.5 A from ligand atom 0: a severe clash
    coords_np[0, n, :] = c[0] + np.array([0.5, 0.0, 0.0])
    coords = torch.tensor(coords_np, dtype=torch.float64)
    bg_before = coords[0, n].clone()

    opt = TorchRestraintOptimizer(spec, max_iter=200)
    e0 = opt.energy(coords)
    d0 = float(torch.linalg.norm(coords[0, 0] - coords[0, n]))
    opt.minimize(coords)
    e1 = opt.energy(coords)
    d1 = float(torch.linalg.norm(coords[0, 0] - coords[0, n]))

    assert e1 < 0.5 * e0, f"{e0} -> {e1}"
    assert d1 > d0, f"ligand not pushed away: {d0} -> {d1}"
    # the fixed protein atom must not move (it is not in active_sites)
    assert torch.allclose(coords[0, n], bg_before, atol=1e-9)


def test_jax_vdw_pushes_ligand_off_fixed_protein():
    """The JAX port of the dynamic fixed-background VdW: the jax minimizer pushes a
    clashing ligand atom off a FIXED protein atom (not in active_sites), same as the
    torch optimizer. This is what makes `intermolecular` / `both` work on AF3."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer

    m = Chem.MolFromSmiles("CC")
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=1)
    c = np.asarray(m.GetConformer().GetPositions())
    n = m.GetNumAtoms()
    lc = LigandConf(
        mol=m, conf_coords=c, global_indices=np.arange(n), conformer_restraints=True
    )
    n_atom = n + 1  # one fixed "protein" atom after the ligand
    elements = np.zeros(n_atom, dtype=np.int64)
    for i, atom in enumerate(m.GetAtoms()):
        elements[i] = atom.GetAtomicNum()
    elements[n] = 6  # heavy -> VdW background

    # conf_start_sigma high so sigma=0 passes the gate (the config path defaults it to
    # +inf; a direct build_spec call defaults it to -1.0, which gates everything off)
    spec = build_spec(
        [lc],
        [],
        {"vdw": {"weight": 1.0, "scale": 0.9}},
        elements=elements,
        conf_start_sigma=1e30,
    )
    assert spec.vdw_config is not None

    coords_np = np.zeros((1, n_atom, 3))
    coords_np[0, :n, :] = c
    coords_np[0, n, :] = c[0] + np.array([0.5, 0.0, 0.0])  # severe clash
    coords = jnp.asarray(coords_np)
    bg_before = np.asarray(coords[0, n])
    d0 = float(np.linalg.norm(np.asarray(coords[0, 0]) - np.asarray(coords[0, n])))

    minimize = make_minimizer(spec, max_iter=200)
    coords = minimize(coords, 0.0)
    d1 = float(np.linalg.norm(np.asarray(coords[0, 0]) - np.asarray(coords[0, n])))

    assert d1 > d0, f"ligand not pushed away: {d0} -> {d1}"
    # the fixed protein atom must not move (it is not in active_sites)
    assert np.allclose(np.asarray(coords[0, n]), bg_before, atol=1e-9)


def _heavy_ethane():
    """Heavy-only ethane (2 carbons, no intramolecular VdW pair) for isolating the
    inter-ligand term."""
    m = Chem.MolFromSmiles("CC")
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=1)
    m = Chem.RemoveHs(m)
    return m, np.asarray(m.GetConformer().GetPositions())


def test_torch_interligand_vdw_separates_two_ligands():
    """Inter-ligand VdW: two restrained ligands overlapping (BOTH in active_sites) are
    pushed apart, and — unlike the fixed-background term where the protein is fixed — BOTH
    ligands move (neither is a fixed background)."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    m, c = _heavy_ethane()
    n = m.GetNumAtoms()  # 2 heavy atoms
    lcA = LigandConf(
        mol=m, conf_coords=c, global_indices=np.arange(n), conformer_restraints=True
    )
    lcB = LigandConf(
        mol=m, conf_coords=c, global_indices=np.arange(n) + n, conformer_restraints=True
    )
    spec = build_spec([lcA, lcB], [], {"vdw": {"weight": 1.0, "scale": 0.9}})
    assert spec.vdw is not None and spec.vdw.idx.shape[0] == n * n
    assert spec.vdw_config is None  # no elements -> inter-ligand only

    coords_np = np.zeros((1, 2 * n, 3))
    coords_np[0, :n, :] = c
    coords_np[0, n:, :] = c + np.array([0.3, 0.0, 0.0])  # severe overlap
    coords = torch.tensor(coords_np, dtype=torch.float64)
    a_before = coords[0, :n].clone()
    b_before = coords[0, n:].clone()
    cen = lambda t: t.mean(dim=0)  # noqa: E731
    d0 = float(torch.linalg.norm(cen(coords[0, :n]) - cen(coords[0, n:])))

    opt = TorchRestraintOptimizer(spec, max_iter=300)
    e0 = opt.energy(coords)
    opt.minimize(coords)
    e1 = opt.energy(coords)
    d1 = float(torch.linalg.norm(cen(coords[0, :n]) - cen(coords[0, n:])))

    assert e1 < 0.5 * e0, f"{e0} -> {e1}"
    assert d1 > d0, f"ligands not separated: {d0} -> {d1}"
    # BOTH ligands moved (inter-ligand has no fixed background)
    assert not torch.allclose(coords[0, :n], a_before, atol=1e-4)
    assert not torch.allclose(coords[0, n:], b_before, atol=1e-4)


def test_jax_interligand_vdw_separates_two_ligands():
    """The JAX port: two overlapping restrained ligands separate under the jax minimizer
    (the inter-ligand VdW pairs are ordinary spec.vdw rows, so they ride lax.scan/AF3)."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer

    m, c = _heavy_ethane()
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
    assert spec.vdw is not None and spec.vdw_config is None

    coords_np = np.zeros((1, 2 * n, 3))
    coords_np[0, :n, :] = c
    coords_np[0, n:, :] = c + np.array([0.3, 0.0, 0.0])
    coords = jnp.asarray(coords_np)
    a_before = np.asarray(coords[0, :n])
    b_before = np.asarray(coords[0, n:])
    cen = lambda arr: np.asarray(arr).mean(axis=0)  # noqa: E731
    d0 = float(np.linalg.norm(cen(coords[0, :n]) - cen(coords[0, n:])))

    minimize = make_minimizer(spec, max_iter=300)
    coords = minimize(coords, 0.0)
    d1 = float(np.linalg.norm(cen(coords[0, :n]) - cen(coords[0, n:])))

    assert d1 > d0, f"ligands not separated: {d0} -> {d1}"
    # BOTH ligands moved (no fixed background)
    assert not np.allclose(np.asarray(coords[0, :n]), a_before, atol=1e-4)
    assert not np.allclose(np.asarray(coords[0, n:]), b_before, atol=1e-4)


# --- sync-free GPU CG (optim/_torch_cg_gpu.py) -----------------------------------


def _rmsd_spec(n=6, seed=3):
    """A RestraintSpec with one RMSD restraint (ref) + a rotated/translated/noised
    distorted starting pose, for the sync-free CG tests."""
    from rgi_utils.spec import RestraintSpec, RmsdArrays

    rng = np.random.default_rng(seed)
    ref = rng.standard_normal((n, 3)) * 3.0
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
            start_sigma=np.array([1e30]),
            stop_sigma=np.array([0.0]),
            start_step=np.full(1, float("-inf")),
            stop_step=np.full(1, float("inf")),
            mask=np.array([1.0]),
        ),
        conf_start_sigma=1e30,
    )
    th = 0.7
    rz = np.array(
        [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    )
    pos = ref @ rz.T + np.array([2.0, 1.0, -1.0]) + rng.standard_normal((n, 3)) * 0.3
    return spec, pos.reshape(1, n, 3)


def test_func_grad_matches_backward_conformer():
    """torch.func.grad_and_value (the GPU CG's gradient source) must equal the
    .backward() gradient for the conformer energy — guards the grad-source switch."""
    torch = pytest.importorskip("torch")
    from rgi_utils.energy import torch_energy

    spec, coords_np = _distorted_ethane()
    prepared = torch_energy.prepare_spec(spec, dtype=torch.float64)
    a0 = torch.tensor(coords_np[0, spec.active_sites, :], dtype=torch.float64)

    def e_of(a):
        return torch_energy.total_energy(a, prepared, sigma=None)

    ab = a0.clone().requires_grad_(True)
    e_of(ab).backward()
    g_func, _v = torch.func.grad_and_value(e_of)(a0)
    assert torch.allclose(g_func, ab.grad, atol=1e-8), (g_func - ab.grad).abs().max()


def test_func_grad_matches_backward_rmsd():
    """Same, for rmsd_energy: confirms the detached Kabsch rotation (_kabsch_R) stops
    the gradient identically under torch.func (the no_grad context is a functorch no-op;
    the .detach() is what must hold)."""
    torch = pytest.importorskip("torch")
    from rgi_utils.energy import torch_energy

    _spec, pos = _rmsd_spec()
    n = pos.shape[1]
    ref = _spec.rmsd.fit_ref
    idx = torch.arange(n).reshape(1, n)
    m = torch.ones((1, n), dtype=torch.float64)
    refc = torch.tensor(ref, dtype=torch.float64)
    kw = dict(
        fit_idx=idx,
        fit_mask=m,
        fit_ref=refc,
        calc_idx=idx,
        calc_mask=m,
        calc_ref=refc,
        target1=torch.zeros(1, dtype=torch.float64),
        target2=torch.zeros(1, dtype=torch.float64),
        geom_type=torch.zeros(1, dtype=torch.int64),
        weight=torch.ones(1, dtype=torch.float64),
        mask=torch.ones(1, dtype=torch.float64),
    )
    p = torch.tensor(pos[0], dtype=torch.float64)

    def e_of(a):
        return torch_energy.rmsd_energy(a, **kw)

    ab = p.clone().requires_grad_(True)
    e_of(ab).backward()
    g_func, _v = torch.func.grad_and_value(e_of)(p)
    assert torch.allclose(g_func, ab.grad, atol=1e-8), (g_func - ab.grad).abs().max()


def test_sync_free_cg_reduces_conformer_energy():
    """The sync-free CG (gpu_cg) is device-agnostic; verify the algorithm reduces the
    conformer energy on CPU (the GPU path runs the identical code, checked E2E via
    sbatch)."""
    torch = pytest.importorskip("torch")
    from rgi_utils.energy import torch_energy
    from rgi_utils.optim._torch_cg_gpu import gpu_cg

    spec, coords_np = _distorted_ethane()
    prepared = torch_energy.prepare_spec(spec, dtype=torch.float64)
    a0 = torch.tensor(coords_np[0, spec.active_sites, :], dtype=torch.float64)

    def e_of(a):
        return torch_energy.total_energy(a, prepared, sigma=None)

    e0 = float(e_of(a0))
    a1 = gpu_cg(prepared, a0, 200)
    assert float(e_of(a1)) < 0.5 * e0


def test_sync_free_cg_reduces_rmsd_energy():
    """The sync-free CG drives the Kabsch RMSD energy down (the detached-rotation
    gradient that stalls jaxopt NonlinearCG; this CG converges it like the jax port)."""
    torch = pytest.importorskip("torch")
    from rgi_utils.energy import torch_energy
    from rgi_utils.optim._torch_cg_gpu import gpu_cg

    spec, pos = _rmsd_spec()
    prepared = torch_energy.prepare_spec(spec, dtype=torch.float64)
    a0 = torch.tensor(pos[0], dtype=torch.float64)

    def e_of(a):
        return torch_energy.total_energy(a, prepared, sigma=None)

    e0 = float(e_of(a0))
    a1 = gpu_cg(prepared, a0, 500)
    assert float(e_of(a1)) < 0.1 * e0, f"{e0} -> {float(e_of(a1))}"


def test_gpu_cg_converges_stiff_chiral():
    """Regression: the GPU CG must drive a STIFF chiral term to ~0. A fixed-width vmap
    line search (an earlier attempt) could not reach the fine backtracking steps a stiff
    chiral needs and silently let it diverge; the sequential early-exit line search does.
    Runs on CPU (eager functional CG, same algorithm as the CUDA path)."""
    torch = pytest.importorskip("torch")
    from rgi_utils.energy import torch_energy
    from rgi_utils.optim._torch_cg_gpu import gpu_cg

    m = Chem.MolFromSmiles("C[C@H](N)O")  # one tetrahedral stereocentre
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=1)
    Chem.AssignStereochemistryFrom3D(m)
    c = np.asarray(m.GetConformer().GetPositions())
    n = m.GetNumAtoms()
    lc = LigandConf(
        mol=m, conf_coords=c, global_indices=np.arange(n), conformer_restraints=True
    )
    spec = build_spec(
        [lc],
        [],
        {"bond": {"weight": 1.0}, "angle": {"weight": 1.0}, "chiral": {"weight": 1.0}},
        conf_start_sigma=1e30,
    )
    assert spec.chiral is not None and spec.chiral.mask.sum() > 0, "no chiral restraint"

    prepared = torch_energy.prepare_spec(spec, dtype=torch.float64)
    rng = np.random.default_rng(0)
    a0 = torch.tensor(
        c[spec.active_sites] + rng.standard_normal((spec.n_active, 3)) * 0.3,
        dtype=torch.float64,
    )

    def e_of(a):
        return torch_energy.total_energy(a, prepared, sigma=None)

    def chiral_e(a):
        ch = prepared["chiral"]
        return float(
            torch_energy.chiral_energy(
                a, ch["idx"], ch["vol0"], ch["slack"], ch["weight"], ch["mask"]
            )
        )

    ch0 = chiral_e(a0)
    ch1 = chiral_e(gpu_cg(prepared, a0, 300))
    assert ch0 > 0.0 and ch1 < 0.1 * ch0, f"chiral not converged: {ch0} -> {ch1}"


def _benzene_plane_spec():
    """A benzene ligand + opt-in plane restraint (its single aromatic ring, 6 atoms)."""
    m = Chem.MolFromSmiles("c1ccccc1")
    mh = Chem.AddHs(m)
    AllChem.EmbedMolecule(mh, randomSeed=1)
    AllChem.UFFOptimizeMolecule(mh)
    m = Chem.RemoveHs(mh)
    c = np.asarray(m.GetConformer(0).GetPositions())
    n = m.GetNumAtoms()
    lc = LigandConf(
        mol=m, conf_coords=c, global_indices=np.arange(n), conformer_restraints=True
    )
    spec = build_spec([lc], [], {"plane": {"weight": 1.0}}, conf_start_sigma=1e30)
    assert spec.plane is not None and spec.plane.mask.sum() > 0, "no plane restraint"
    return spec, c


def _plane_rms_dev(p):
    """RMS out-of-plane deviation of points ``p`` (n,3) from their best-fit plane —
    the same quantity the plane energy penalises (sqrt(lambda_min / N))."""
    x0 = p - p.mean(axis=0)
    _w, v = np.linalg.eigh(x0.T @ x0)
    return float(np.sqrt(np.mean((x0 @ v[:, 0]) ** 2)))


def _pucker_ring(c, active_sites, out=0.6):
    """Push atom 0 of the active (ring) set ~``out`` A along the ring normal -> a strongly
    non-planar start, where lambda_min is least separated from lambda_mid (the plane normal
    is worst-conditioned — the hardest case for the stop-gradient CG to converge from)."""
    ring = c[active_sites].copy()
    x0 = ring - ring.mean(axis=0)
    _w, v = np.linalg.eigh(x0.T @ x0)
    ring[0] = ring[0] + out * v[:, 0]
    return ring


def test_gpu_cg_flattens_plane_group():
    """Convergence: the torch CG must FLATTEN a plane group despite the stop-gradient plane
    normal (the same detached-decomposition + CG interaction that stalls jaxopt on RMSD; the
    hand-rolled CG converges it). Pucker a benzene ring ~0.6 A out of plane, run the CG,
    assert its out-of-plane RMS deviation collapses. CPU (same algorithm as the CUDA path)."""
    torch = pytest.importorskip("torch")
    from rgi_utils.energy import torch_energy
    from rgi_utils.optim._torch_cg_gpu import gpu_cg

    spec, c = _benzene_plane_spec()
    prepared = torch_energy.prepare_spec(spec, dtype=torch.float64)
    pos = _pucker_ring(c, spec.active_sites)
    dev0 = _plane_rms_dev(pos)
    assert dev0 > 0.1, f"start not puckered: {dev0}"
    a1 = gpu_cg(prepared, torch.tensor(pos, dtype=torch.float64), 300)
    dev1 = _plane_rms_dev(a1.numpy())
    assert dev1 < 0.2 * dev0, f"plane not flattened: {dev0} -> {dev1}"


def test_jax_cg_flattens_plane_group():
    """Same plane-group flattening on the jax hand-rolled CG (the AF3 / lax.scan path),
    confirming the stop-gradient plane normal does not stall it there either."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.energy import jax_energy
    from rgi_utils.optim.jax_optim import _cg_minimize

    spec, c = _benzene_plane_spec()
    prep_j = jax_energy.prepare_spec(spec)
    pos = _pucker_ring(c, spec.active_sites)
    dev0 = _plane_rms_dev(pos)
    assert dev0 > 0.1, f"start not puckered: {dev0}"

    def ej(a):
        return jax_energy.total_energy(a, prep_j, sigma=None)

    a1 = _cg_minimize(ej, jnp.asarray(pos), 300)
    dev1 = _plane_rms_dev(np.asarray(a1))
    assert dev1 < 0.2 * dev0, f"plane not flattened: {dev0} -> {dev1}"


def _standalone_plane_coords(out=0.6, n=6):
    """A hexagon with atom 0 pushed ``out`` A along the ring normal — the same
    worst-conditioned start as ``_pucker_ring``, but for the selection-driven term (which
    needs no ligand / RDKit mol)."""
    ring = np.array(
        [
            [np.cos(t), np.sin(t), 0.0]
            for t in np.linspace(0, 2 * np.pi, n, endpoint=False)
        ]
    )
    ring[0, 2] += out
    return ring


def test_torch_cg_flattens_standalone_plane():
    """The standalone plane term (``plane_restraints_config``) converges under the torch CG
    exactly like the conformer one — its gate is per-entry, so this also confirms the
    restraint is actually reached by the solver-run condition (``has_group_plane``)."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec = _group_plane_spec()
    pos = _standalone_plane_coords()
    dev0 = _plane_rms_dev(pos)
    coords = torch.tensor(pos.reshape(1, 6, 3), dtype=torch.float64)
    TorchRestraintOptimizer(spec, max_iter=300).minimize(coords, sigma=1.0)
    dev1 = _plane_rms_dev(coords.numpy()[0])
    assert dev1 < 0.05 * dev0, f"plane not flattened: {dev0} -> {dev1}"


def test_jax_cg_flattens_standalone_plane():
    """Same on the jax minimizer (the AF3 / lax.scan path)."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer

    spec = _group_plane_spec()
    pos = _standalone_plane_coords()
    dev0 = _plane_rms_dev(pos)
    mini = make_minimizer(spec, max_iter=300)
    out = jax.jit(lambda x: mini(x, 0, jnp.float64(1.0)))(
        jnp.asarray(pos.reshape(1, 6, 3))
    )
    dev1 = _plane_rms_dev(np.asarray(out)[0])
    assert dev1 < 0.05 * dev0, f"plane not flattened: {dev0} -> {dev1}"


def test_standalone_plane_gated_off_above_start_sigma():
    """Above ``start_sigma`` the whole step is a no-op — the coords come back untouched."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec = _group_plane_spec(start_sigma=1.0)
    pos = _standalone_plane_coords()
    coords = torch.tensor(pos.reshape(1, 6, 3), dtype=torch.float64)
    TorchRestraintOptimizer(spec, max_iter=100).minimize(coords, sigma=5.0)  # 5 > 1
    assert np.allclose(coords.numpy()[0], pos)
    TorchRestraintOptimizer(spec, max_iter=300).minimize(
        coords, sigma=0.5
    )  # now active
    assert _plane_rms_dev(coords.numpy()[0]) < 0.05 * _plane_rms_dev(pos)


def test_standalone_plane_move_pins_the_other_group():
    """``move`` frees one pooled group and pins the other: the pinned group's atoms must not
    move, while the plane still flattens by moving the free group onto it."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    # group 1 = atoms 0..2 (lifted out of plane), group 2 = atoms 3..5 (in z=0)
    pos = np.array(
        [
            [0.0, 0.0, 0.7],
            [1.0, 0.0, 0.7],
            [0.0, 1.0, 0.7],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )
    spec = _group_plane_spec(move_free=(True, False), groups=[[0, 1, 2], [3, 4, 5]])
    coords = torch.tensor(pos.reshape(1, 6, 3), dtype=torch.float64)
    TorchRestraintOptimizer(spec, max_iter=300).minimize(coords, sigma=1.0)
    x = coords.numpy()[0]
    assert np.allclose(x[3:], pos[3:], atol=1e-8), "pinned group moved"
    # the pinned group holds the plane, so the free group has to travel onto it; 0.2 is the
    # same convergence bar the conformer-plane CG tests use
    assert _plane_rms_dev(x) < 0.2 * _plane_rms_dev(pos)


def test_gated_prepared_matches_energy_gate():
    """The GPU pre-gated masks (_gated_prepared + total_energy(sigma=None)) reproduce the
    energy layer's own sigma gating for conf-on/off and rmsd-on/off, so the compiled GPU
    minimum can't silently diverge from the CPU/jax one."""
    torch = pytest.importorskip("torch")
    from rgi_utils.energy import torch_energy
    from rgi_utils.optim._torch_cg_gpu import _energy
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer
    from rgi_utils.spec import BondArrays, RestraintSpec, RmsdArrays

    rng = np.random.default_rng(2)
    n = 6
    ref = rng.standard_normal((n, 3)) * 3.0
    idx = np.arange(n).reshape(1, n)
    spec = RestraintSpec(
        n_active=n,
        active_sites=np.arange(n),
        bond=BondArrays(
            idx=np.array([[0, 1], [2, 3]], dtype=np.int64),
            r0=np.array([1.0, 1.5]),
            slack=np.zeros(2),
            weight=np.ones(2),
            half=np.zeros(2),
            mask=np.ones(2),
        ),
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
            start_sigma=np.array([5.0]),
            stop_sigma=np.array([0.0]),
            start_step=np.full(1, float("-inf")),
            stop_step=np.full(1, float("inf")),
            mask=np.array([1.0]),
        ),
        conf_start_sigma=10.0,
    )
    opt = TorchRestraintOptimizer(spec)
    opt._ensure(torch.device("cpu"), torch.float64)
    pos = torch.tensor(rng.standard_normal((n, 3)), dtype=torch.float64)
    for sigma in (
        50.0,
        8.0,
        3.0,
        None,
    ):  # conf+rmsd off / conf-on-rmsd-off / both / both
        pg = opt._gated_prepared(sigma)
        e_gpu = float(_energy(pos, pg))
        e_ref = float(torch_energy.total_energy(pos, opt._prepared, sigma))
        assert abs(e_gpu - e_ref) < 1e-9, (sigma, e_gpu, e_ref)
        # the dropped scalar leaf must not be in the compiled pytree (per-value recompile)
        assert "conf_start_sigma" not in pg
    # conf gated off (sigma 50 > conf_start_sigma 10) zeros the conformer masks
    assert float(opt._gated_prepared(50.0)["bond"]["mask"].sum()) == 0.0


def test_compiled_energy_matches_eager():
    """torch.compile of the GPU energy+grad must equal eager grad_and_value (compiling
    fuses kernels; it must NOT change the maths), incl. the detached Kabsch SVD in the
    RMSD term AND the group-centroid angle/dihedral terms (gather + masked centroid + cross +
    atan2 wrap). Runs on CPU (inductor, no CUDA graph) so CI guards the invariant;
    otherwise the compiled group energy is only ever exercised by an sbatch GPU run."""
    torch = pytest.importorskip("torch")
    from rgi_utils.energy import torch_energy
    from rgi_utils.optim import _torch_cg_gpu as g
    from rgi_utils.spec import GroupAngleArrays, GroupDihedralArrays

    spec, pos_np = _rmsd_spec()  # active_sites = arange(6); add group terms over those
    # a pinned group on the angle (move_free col 0) exercises the detach-select
    # (torch.where + .detach) UNDER torch.compile — the one inductor path the rmsd
    # precedent doesn't cover.
    spec.group_angle = GroupAngleArrays(
        grp1_idx=np.array([[0, 1]]),
        grp2_idx=np.array([[2, 3]]),
        grp3_idx=np.array([[4, 5]]),
        grp1_mask=np.array([[1.0, 1.0]]),
        grp2_mask=np.array([[1.0, 1.0]]),
        grp3_mask=np.array([[1.0, 1.0]]),
        target1=np.array([1.3]),
        target2=np.array([0.0]),
        geom_type=np.array([0]),
        move_free=np.array([[0.0, 1.0, 0.0]]),
        weight=np.array([1.0]),
        mask=np.array([1.0]),
        start_sigma=np.array([1e30]),
        stop_sigma=np.array([-1.0]),
        start_step=np.full(1, float("-inf")),
        stop_step=np.full(1, float("inf")),
    )
    spec.group_dihedral = GroupDihedralArrays(
        grp1_idx=np.array([[0]]),
        grp2_idx=np.array([[1]]),
        grp3_idx=np.array([[2]]),
        grp4_idx=np.array([[3]]),
        grp1_mask=np.array([[1.0]]),
        grp2_mask=np.array([[1.0]]),
        grp3_mask=np.array([[1.0]]),
        grp4_mask=np.array([[1.0]]),
        target1=np.array([0.5]),
        target2=np.array([0.0]),
        geom_type=np.array([0]),
        move_free=np.array([[1.0, 1.0, 1.0, 1.0]]),
        weight=np.array([1.0]),
        mask=np.array([1.0]),
        start_sigma=np.array([1e30]),
        stop_sigma=np.array([-1.0]),
        start_step=np.full(1, float("-inf")),
        stop_step=np.full(1, float("inf")),
    )
    prepared = torch_energy.prepare_spec(spec, dtype=torch.float64)
    pos = torch.tensor(pos_np[0], dtype=torch.float64)
    eager_vg = torch.func.grad_and_value(g._energy, argnums=0)
    comp_vg = torch.compile(torch.func.grad_and_value(g._energy, argnums=0))
    ge, ve = eager_vg(pos, prepared)
    gc, vc = comp_vg(pos, prepared)
    assert torch.allclose(ge, gc, atol=1e-8), (ge - gc).abs().max()
    assert abs(float(ve) - float(vc)) < 1e-8


def test_dynamic_vdw_pair_energy_matches_optimizer():
    """The pure _vdw_pair_energy (folded into the compiled GPU energy) must equal the
    optimizer's _vdw_energy method (used on the CPU/eager path) — so compiling the
    dynamic fixed-background VdW conformer path doesn't change the energy."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim._torch_cg_gpu import _vdw_pair_energy
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    m = Chem.MolFromSmiles("CC")
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=1)
    c = np.asarray(m.GetConformer().GetPositions())
    n = m.GetNumAtoms()
    lc = LigandConf(
        mol=m, conf_coords=c, global_indices=np.arange(n), conformer_restraints=True
    )
    n_atom = n + 1
    elements = np.zeros(n_atom, dtype=np.int64)
    for i, atom in enumerate(m.GetAtoms()):
        elements[i] = atom.GetAtomicNum()
    elements[n] = 6  # a heavy "protein" background atom
    spec = build_spec(
        [lc], [], {"vdw": {"weight": 1.0, "scale": 0.9}}, elements=elements
    )
    assert spec.vdw_config is not None  # dynamic fixed-background VdW

    coords = torch.zeros((1, n_atom, 3), dtype=torch.float64)
    coords[0, :n, :] = torch.tensor(c)
    coords[0, n, :] = torch.tensor(c[0] + np.array([0.5, 0.0, 0.0]))  # a clash
    opt = TorchRestraintOptimizer(spec, max_iter=10)
    opt._ensure(coords.device, coords.dtype)  # builds opt._vdw
    active = coords[0, opt._active_idx, :]
    bg_pos = coords[0, opt._vdw["bg_global"], :]
    v = opt._vdw
    fixed_pairs = opt._fixed_vdw_pairs(active, bg_pos)

    e_method = float(opt._vdw_energy(active, bg_pos, fixed_pairs))
    e_pure = float(
        _vdw_pair_energy(
            active,
            bg_pos,
            v["lig_local"],
            fixed_pairs[0],
            fixed_pairs[1],
            v["lig_r"],
            v["bg_r"],
            v["scale"],
            v["weight"],
        )
    )
    assert e_method > 0.0 and abs(e_method - e_pure) < 1e-10, (e_method, e_pure)


def _vdw_and_custom_spec(n_bg: int = 1):
    """A spec carrying, at once: conformer terms, the dynamic fixed-background VdW
    (``vdw_config``), the active-active polymer VdW (``active_vdw_config``) and a custom
    restraint — i.e. every ingredient of the compiled custom-inclusive energy."""
    from rgi_utils.config import RestraintsConfig
    from rgi_utils.spec import ActiveVdwConfig

    m = Chem.MolFromSmiles("CC")
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=1)
    c = np.asarray(m.GetConformer().GetPositions())
    n = m.GetNumAtoms()
    lc = LigandConf(
        mol=m, conf_coords=c, global_indices=np.arange(n), conformer_restraints=True
    )
    n_atom = n + n_bg
    elements = np.zeros(n_atom, dtype=np.int64)
    for i, atom in enumerate(m.GetAtoms()):
        elements[i] = atom.GetAtomicNum()
    elements[n:] = 6  # heavy "protein" background atoms

    cfg = RestraintsConfig.from_dict(
        {
            "custom_restraints_config": [
                {
                    "name": "c",
                    "energy": "(distance(A, B) - 3.0)**2",
                    "selections": {"A": "index 0", "B": "index 1"},
                }
            ]
        }
    )
    records = [
        SimpleNamespace(
            chain="A",
            resid=i + 1,
            index=i,
            name="C",
            mol_type="ligand",
            resname="LIG",
        )
        for i in range(n_atom)
    ]
    adapter = SimpleNamespace(iter_atoms=lambda: iter(records))
    for cd in cfg.custom_data:
        cd.resolve_sites(adapter)

    spec = build_spec(
        [lc],
        [],
        {"bond": {"weight": 1.0}, "vdw": {"weight": 1.0, "scale": 0.9}},
        elements=elements,
        custom_restraints=cfg.custom_data,
    )
    assert spec.vdw_config is not None and spec.has_custom()
    # active-active polymer VdW is only built from a polymer geometry, which this synthetic
    # ligand-only spec has none of -- attach it directly so mode 2/3 are exercised.
    n_active = spec.n_active
    spec.active_vdw_config = ActiveVdwConfig(
        weight=1.0,
        radii=np.full(n_active, 1.7),
        polymer_mask=np.ones(n_active, dtype=bool),
        excluded_codes=np.zeros(0, dtype=np.int64),
        scale=0.9,
        dmax=5.0,
        max_neighbors=4,
    )

    coords = np.zeros((1, n_atom, 3))
    coords[0, :n, :] = c
    for j in range(n_bg):
        coords[0, n + j, :] = c[j] + np.array([0.5, 0.0, 0.0])  # a clash
    return spec, coords


def _mode_args(opt, coords):
    """``{mode: extra-args tuple}`` for ``_ENERGY_BY_MODE``, built from an ``_ensure``d
    optimizer the same way ``minimize`` builds them."""
    from rgi_utils.optim._torch_cg_gpu import build_active_vdw_pairs

    active = coords[0, opt._active_idx, :]
    bg_pos = coords[0, opt._vdw["bg_global"], :]
    v, av = opt._vdw, opt._active_vdw
    fixed_neighbours, fixed_pair_mask = opt._fixed_vdw_pairs(active, bg_pos)
    vdw = (
        bg_pos,
        v["lig_local"],
        fixed_neighbours,
        fixed_pair_mask,
        v["lig_r"],
        v["bg_r"],
        v["scale"],
        v["weight"],
    )
    neighbours, pair_factor = build_active_vdw_pairs(
        active,
        av["radii"],
        av["polymer_mask"],
        av["excluded_codes"],
        av["dmax"],
        av["max_neighbors"],
    )
    active_vdw = (neighbours, pair_factor, av["radii"], av["scale"], av["weight"])
    return (
        active,
        bg_pos,
        {
            0: (),
            1: vdw,
            2: active_vdw,
            3: vdw + active_vdw,
        },
    )


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_compiled_vdw_energy_matches_eager(mode):
    """Every ``_ENERGY_BY_MODE`` variant must compile to the same maths as eager — not
    just mode 0 (``test_compiled_energy_matches_eager``). Modes 1/2/3 fold in the DYNAMIC
    VdW terms (fixed background / active-active neighbour list), which the CPU suite
    otherwise never runs through inductor at all. Skips where no compile toolchain."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim import _torch_cg_gpu as g
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec, coords_np = _vdw_and_custom_spec()
    coords = torch.tensor(coords_np, dtype=torch.float64)
    opt = TorchRestraintOptimizer(spec, max_iter=10)
    opt._ensure(coords.device, coords.dtype)
    active, _bg, extras = _mode_args(opt, coords)
    prepared = opt._gated_prepared(None, None)

    base = g._ENERGY_BY_MODE[mode]
    extra = extras[mode]
    ge, ve = torch.func.grad_and_value(base, argnums=0)(active, prepared, *extra)
    try:
        comp = torch.compile(torch.func.grad_and_value(base, argnums=0))
        gc, vc = comp(active, prepared, *extra)
    except Exception as exc:  # no C++ toolchain / unsupported inductor env
        pytest.skip(f"torch.compile unavailable: {exc}")
    assert float(ve) > 0.0, "degenerate fixture: the VdW term contributes nothing"
    assert torch.allclose(ge, gc, atol=1e-8), (ge - gc).abs().max()
    assert abs(float(ve) - float(vc)) < 1e-8


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_custom_compiled_energy_includes_vdw(mode):
    """The per-optimizer custom-inclusive compiled energy must equal the eager CG's own
    objective (``total_energy`` + ``_vdw_energy`` + ``active_vdw_pair_energy`` +
    ``_custom_energy``) for EVERY VdW mode. Before this, custom + dynamic VdW had no
    compiled artifact at all and the whole CG dropped to eager on CUDA; the risk of the
    fix is a mis-assembled argument tuple, which this cross-check catches."""
    torch = pytest.importorskip("torch")
    from rgi_utils.energy import torch_energy
    from rgi_utils.optim._torch_cg_gpu import active_vdw_pair_energy
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec, coords_np = _vdw_and_custom_spec()
    coords = torch.tensor(coords_np, dtype=torch.float64)
    opt = TorchRestraintOptimizer(spec, max_iter=10)
    opt._ensure(coords.device, coords.dtype)
    active, bg_pos, extras = _mode_args(opt, coords)
    prepared = opt._gated_prepared(None, None)
    gates = torch.ones(len(opt._custom_terms), dtype=coords.dtype)

    cvg = opt._get_custom_cvg(mode)
    if cvg is None:
        pytest.skip("torch.compile unavailable")
    try:
        _grad, value = cvg(active, prepared, gates, *extras[mode])
    except Exception as exc:
        pytest.skip(f"torch.compile unavailable: {exc}")

    # independent reference: exactly what the eager `energy_fn` in minimize() sums
    ref = torch_energy.total_energy(active, prepared, None, None)
    ref = ref + opt._custom_energy(active, None, None)
    if mode & 1:
        ref = ref + opt._vdw_energy(active, bg_pos, extras[1][2:4])
    if mode & 2:
        neighbours, pair_factor, radii, scale, weight = extras[2]
        ref = ref + active_vdw_pair_energy(
            active, neighbours, pair_factor, radii, scale, weight
        )
    assert abs(float(value) - float(ref)) < 1e-8, (float(value), float(ref))


def test_sync_free_cg_nonfinite_guard():
    """A non-finite gradient/energy returns the input coords unchanged (on-device
    guard, mirroring the jax backend) — no NaN written into the structure."""
    torch = pytest.importorskip("torch")
    from rgi_utils.optim._torch_cg_gpu import _cg_minimize_torch

    x0 = torch.zeros((4, 3), dtype=torch.float64)

    def vg_nan(x):
        return torch.full_like(x, float("nan")), torch.tensor(float("nan"))

    out = _cg_minimize_torch(vg_nan, x0, max_iter=5)
    assert torch.isfinite(out).all() and torch.allclose(out, x0)


@pytest.mark.gpu
def test_gpu_cg_matches_cpu_minimum():
    """The GPU (sync-free) CG and the CPU (early-exit) CG reach the same minimum."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no cuda device")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer

    spec, coords_np = _distorted_ethane()
    cc = torch.tensor(coords_np, dtype=torch.float64)
    oc = TorchRestraintOptimizer(spec, max_iter=200)
    e0c = oc.energy(cc)
    oc.minimize(cc)
    e1c = oc.energy(cc)

    cg = torch.tensor(coords_np, dtype=torch.float64, device="cuda")
    og = TorchRestraintOptimizer(spec, max_iter=200)
    e0g = og.energy(cg)
    og.minimize(cg)
    e1g = og.energy(cg)

    assert e1c < 0.5 * e0c and e1g < 0.5 * e0g
    assert abs(e1c - e1g) < 1e-3 + 0.1 * abs(e1c), f"cpu {e1c} vs gpu {e1g}"


@pytest.mark.parametrize("weight", [1.0, 32.0])
def test_torch_vdw_cg_step_cap_prevents_overshoot(weight):
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer
    from rgi_utils.spec import RestraintSpec, VdwArrays

    spec = RestraintSpec(
        n_active=2,
        active_sites=np.arange(2),
        vdw=VdwArrays(
            idx=np.array([[0, 1]], dtype=np.int64),
            r_min=np.array([2.55]),
            weight=np.array([weight]),
            mask=np.ones(1),
        ),
        conf_start_sigma=float("inf"),
        vdw_max_atom_step=0.1,
    )
    coords = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64)

    TorchRestraintOptimizer(spec, max_iter=100, method="CG").minimize(coords)
    distance = float(torch.linalg.norm(coords[0] - coords[1]))

    assert 2.5 <= distance <= 2.75


@pytest.mark.parametrize("weight", [1.0, 32.0])
def test_jax_vdw_cg_step_cap_prevents_overshoot(weight):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer
    from rgi_utils.spec import RestraintSpec, VdwArrays

    spec = RestraintSpec(
        n_active=2,
        active_sites=np.arange(2),
        vdw=VdwArrays(
            idx=np.array([[0, 1]], dtype=np.int64),
            r_min=np.array([2.55]),
            weight=np.array([weight]),
            mask=np.ones(1),
        ),
        conf_start_sigma=float("inf"),
        vdw_max_atom_step=0.1,
    )
    coords = jnp.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

    coords = make_minimizer(spec, max_iter=100, method="CG")(coords, 0.0)
    distance = float(jnp.linalg.norm(coords[0] - coords[1]))

    assert 2.5 <= distance <= 2.75


def test_torch_dynamic_vdw_rebuilds_before_new_contact():
    torch = pytest.importorskip("torch")
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer
    from rgi_utils.spec import DistanceArrays, RestraintSpec, VdwConfig

    spec = RestraintSpec(
        n_active=2,
        active_sites=np.array([0, 1]),
        distance=DistanceArrays(
            grp1_idx=np.array([[0]], dtype=np.int64),
            grp2_idx=np.array([[1]], dtype=np.int64),
            grp1_mask=np.ones((1, 1)),
            grp2_mask=np.ones((1, 1)),
            target1=np.zeros(1),
            target2=np.zeros(1),
            dist_type=np.zeros(1, dtype=np.int64),
            move_mode=np.ones(1, dtype=np.int64),
            weight=np.ones(1),
            mask=np.ones(1),
            start_sigma=np.array([float("inf")]),
            stop_sigma=np.array([-1.0]),
            start_step=np.array([float("-inf")]),
            stop_step=np.array([float("inf")]),
        ),
        vdw_config=VdwConfig(
            weight=1.0,
            ligand_local=np.array([0]),
            ligand_radii=np.array([1.7]),
            background_global=np.array([2]),
            background_radii=np.array([1.7]),
            scale=0.75,
            dmax=5.0,
            max_neighbors=4,
        ),
        conf_start_sigma=float("inf"),
        vdw_max_atom_step=0.1,
        vdw_neighbor_rebuild_interval=4,
        # Pinned, not inherited: the fixture depends on the pair at 5.3 A being OUTSIDE the
        # built list (dmax 5.0, r_min 2.55, movement 0.4 -> cutoff max(5.0, 4.95) = 5.0). A
        # larger default skin would list it from the first build and the test would pass
        # vacuously, verifying nothing about the rebuild.
        vdw_neighbor_skin=2.0,
    )
    coords = torch.tensor(
        [[5.3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    TorchRestraintOptimizer(spec, max_iter=100, method="CG").minimize(coords)
    fresh_distance = float(torch.linalg.norm(coords[0] - coords[2]))
    assert 0.8 < fresh_distance < 2.55


def test_jax_dynamic_vdw_rebuilds_before_new_contact():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer
    from rgi_utils.spec import DistanceArrays, RestraintSpec, VdwConfig

    spec = RestraintSpec(
        n_active=2,
        active_sites=np.array([0, 1]),
        distance=DistanceArrays(
            grp1_idx=np.array([[0]], dtype=np.int64),
            grp2_idx=np.array([[1]], dtype=np.int64),
            grp1_mask=np.ones((1, 1)),
            grp2_mask=np.ones((1, 1)),
            target1=np.zeros(1),
            target2=np.zeros(1),
            dist_type=np.zeros(1, dtype=np.int64),
            move_mode=np.ones(1, dtype=np.int64),
            weight=np.ones(1),
            mask=np.ones(1),
            start_sigma=np.array([float("inf")]),
            stop_sigma=np.array([-1.0]),
            start_step=np.array([float("-inf")]),
            stop_step=np.array([float("inf")]),
        ),
        vdw_config=VdwConfig(
            weight=1.0,
            ligand_local=np.array([0]),
            ligand_radii=np.array([1.7]),
            background_global=np.array([2]),
            background_radii=np.array([1.7]),
            scale=0.75,
            dmax=5.0,
            max_neighbors=4,
        ),
        conf_start_sigma=float("inf"),
        vdw_max_atom_step=0.1,
        vdw_neighbor_rebuild_interval=4,
        # Pinned, not inherited: the fixture depends on the pair at 5.3 A being OUTSIDE the
        # built list (dmax 5.0, r_min 2.55, movement 0.4 -> cutoff max(5.0, 4.95) = 5.0). A
        # larger default skin would list it from the first build and the test would pass
        # vacuously, verifying nothing about the rebuild.
        vdw_neighbor_skin=2.0,
    )
    coords = jnp.asarray([[5.3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    coords = make_minimizer(spec, max_iter=100, method="CG")(coords, 0.0)
    fresh_distance = float(jnp.linalg.norm(coords[0] - coords[2]))

    assert 0.8 < fresh_distance < 2.55


def test_torch_dynamic_vdw_stops_rebuilding_after_convergence(monkeypatch):
    torch = pytest.importorskip("torch")
    from rgi_utils.optim import _torch_cg_gpu
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer
    from rgi_utils.spec import RestraintSpec, VdwConfig

    calls = 0
    original = _torch_cg_gpu.build_fixed_vdw_pairs

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_torch_cg_gpu, "build_fixed_vdw_pairs", counted)
    spec = RestraintSpec(
        n_active=1,
        active_sites=np.array([0]),
        vdw_config=VdwConfig(
            weight=1.0,
            ligand_local=np.array([0]),
            ligand_radii=np.array([1.7]),
            background_global=np.array([1]),
            background_radii=np.array([1.7]),
            scale=0.75,
            dmax=5.0,
            max_neighbors=4,
        ),
        conf_start_sigma=float("inf"),
        vdw_max_atom_step=0.1,
        vdw_neighbor_rebuild_interval=1,
    )
    coords = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)

    TorchRestraintOptimizer(spec, max_iter=20, method="CG").minimize(coords)

    assert calls == 1


def test_cg_warm_start_cuts_line_search_evals():
    """Regression pin for the line-search warm start (see optim/_cg_config.py).

    ``e(x) = 0.5 * sum(k_i * x_i**2)`` with a 16:1 curvature ratio: a unit step along
    ``-grad`` overshoots the stiff axis, so Armijo accepts only around 2**-4 and the FIRST
    iteration pays 5 evaluations getting there. Every later iteration needs the same scale.
    A cold start re-derives it every time (~5 x 20 = ~100 evaluations); carrying the step
    costs 5 + ~2 per iteration (measured 42). The bound below separates the two.

    The curvature ratio is deliberately moderate. At 64:1 this solver does not converge at
    all — Armijo only asks for *sufficient* decrease, so a step that reflects the stiff
    coordinate from +1 to -1 is accepted on the strength of the soft axes alone. That is
    ordinary steepest-descent behaviour, unrelated to the warm start, and it would make the
    energy assertion below untestable.
    """
    torch = pytest.importorskip("torch")
    from rgi_utils.optim._torch_cg_gpu import _cg_minimize_torch

    k = torch.tensor([16.0, 1.0, 1.0], dtype=torch.float64)

    def energy(x):
        return 0.5 * torch.sum(k * x * x)

    calls = 0

    def vg(x):
        nonlocal calls
        calls += 1
        return torch.func.grad_and_value(energy)(x)

    x = _cg_minimize_torch(vg, torch.ones(3, dtype=torch.float64), 20)
    assert calls <= 60, f"line search cost {calls} evaluations (cold start costs ~100)"
    # keeps a solver that "saves" evaluations by stopping early from passing
    assert float(energy(x)) < 0.01, "warm start must still reach the minimum"


def test_cg_warm_start_recovers_after_shrinking():
    """The carried step must be able to grow back, not ratchet monotonically down.

    Optimising the stiff quadratic drives the accepted step to ~2**-6; the returned point
    is then used as the start of a run on an ISOTROPIC quadratic, where step 1.0 is
    optimal. With growth capped at one backtrack per iteration the solver climbs back
    within a few iterations, so a handful of iterations suffices to converge. Without any
    growth the step would stay at 2**-6 forever and this would not converge.
    """
    torch = pytest.importorskip("torch")
    from rgi_utils.optim._torch_cg_gpu import _cg_minimize_torch

    def easy(x):
        return 0.5 * torch.sum(x * x)

    def vg(x):
        return torch.func.grad_and_value(easy)(x)

    x0 = torch.full((3,), 0.5, dtype=torch.float64)
    x = _cg_minimize_torch(vg, x0, 8)
    assert float(easy(x)) < 1e-12, (
        "isotropic quadratic must converge in a few iterations"
    )


@pytest.mark.parametrize(
    "skin,expect_rebuilds",
    [(2.0, 1), (0.5, 3)],  # travel is 1.5 A: under a 2.0 skin, over a 0.5 one
)
def test_torch_dynamic_vdw_rebuild_follows_measured_displacement(
    monkeypatch, skin, expect_rebuilds
):
    """The neighbour list is rebuilt on MEASURED displacement, not every N iterations.

    A distance restraint drags the ligand atom 1.5 A while ``max_atom_step=0.1`` forces at
    least 15 iterations, i.e. at least 4 staleness checks at ``interval=4``. With a 2.0 A
    skin the atom never travels far enough and the initial list stands; with a 0.5 A skin it
    crosses the budget repeatedly. The background atom is parked 20 A away so VdW never
    interferes with the motion, and the movement assertion stops a solver that simply
    stalls from satisfying the ``calls == 1`` case for the wrong reason.
    """
    torch = pytest.importorskip("torch")
    from rgi_utils.optim import _torch_cg_gpu
    from rgi_utils.optim.torch_optim import TorchRestraintOptimizer
    from rgi_utils.spec import DistanceArrays, RestraintSpec, VdwConfig

    calls = 0
    original = _torch_cg_gpu.build_fixed_vdw_pairs

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_torch_cg_gpu, "build_fixed_vdw_pairs", counted)
    spec = RestraintSpec(
        n_active=2,
        active_sites=np.array([0, 1]),
        distance=DistanceArrays(
            grp1_idx=np.array([[0]], dtype=np.int64),
            grp2_idx=np.array([[1]], dtype=np.int64),
            grp1_mask=np.ones((1, 1)),
            grp2_mask=np.ones((1, 1)),
            target1=np.zeros(1),
            target2=np.zeros(1),
            dist_type=np.zeros(1, dtype=np.int64),
            move_mode=np.ones(1, dtype=np.int64),  # only group 1 moves
            weight=np.ones(1),
            mask=np.ones(1),
            start_sigma=np.array([float("inf")]),
            stop_sigma=np.array([-1.0]),
            start_step=np.array([float("-inf")]),
            stop_step=np.array([float("inf")]),
        ),
        vdw_config=VdwConfig(
            weight=1.0,
            ligand_local=np.array([0]),
            ligand_radii=np.array([1.7]),
            background_global=np.array([2]),
            background_radii=np.array([1.7]),
            scale=0.75,
            dmax=5.0,
            max_neighbors=4,
        ),
        conf_start_sigma=float("inf"),
        vdw_max_atom_step=0.1,
        vdw_neighbor_rebuild_interval=4,
        vdw_neighbor_skin=skin,
    )
    coords = torch.tensor(
        [[1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    TorchRestraintOptimizer(spec, max_iter=20, method="CG").minimize(coords)

    moved = 1.5 - float(torch.linalg.norm(coords[0] - coords[1]))
    assert moved > 1.4, f"the ligand barely moved ({moved:.3f} A); test is vacuous"
    if expect_rebuilds == 1:
        assert calls == 1, f"list rebuilt {calls}x despite staying inside the skin"
    else:
        assert calls >= expect_rebuilds, f"only {calls} rebuild(s) for 1.5 A of travel"


def test_vdw_skin_does_not_change_the_listed_energy():
    """Widening the search radius by the skin must not change what the VdW term scores.

    The K-nearest cap (`max_neighbors`) is applied AFTER ranking by VdW clearance, so a
    larger radius can only add candidates that rank WORSE than everything already kept —
    and those sit beyond contact, where ``clamp(d - r_min, max=0)**2`` is exactly zero.
    This is the cheap guard on the truncation hazard the skin introduces: if a wider list
    ever displaced a contacting pair, the two energies would differ.
    """
    torch = pytest.importorskip("torch")
    from rgi_utils.optim._torch_cg_gpu import _vdw_pair_energy, build_fixed_vdw_pairs

    rng = np.random.default_rng(0)
    lig = torch.tensor(rng.uniform(-4, 4, (12, 3)), dtype=torch.float64)
    bg = torch.tensor(rng.uniform(-9, 9, (200, 3)), dtype=torch.float64)
    lig_local = torch.arange(12)
    lig_r = torch.full((12,), 1.7, dtype=torch.float64)
    bg_r = torch.full((200,), 1.7, dtype=torch.float64)
    scale = torch.tensor(0.75, dtype=torch.float64)
    weight = torch.tensor(1.0, dtype=torch.float64)

    def energy_at(cutoff):
        pairs = build_fixed_vdw_pairs(
            lig,
            bg,
            lig_local,
            torch.tensor(cutoff, dtype=torch.float64),
            32,
            lig_r,
            bg_r,
            scale,
        )
        return float(
            _vdw_pair_energy(
                lig, bg, lig_local, pairs[0], pairs[1], lig_r, bg_r, scale, weight
            )
        )

    bare, skinned = energy_at(5.0), energy_at(7.0)
    assert bare > 0.0, "fixture has no contacts; the comparison would be vacuous"
    assert abs(bare - skinned) < 1e-12, f"skin changed the energy: {bare} vs {skinned}"
