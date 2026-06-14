"""Optimizer sanity: each backend reduces restraint energy on a distorted ligand."""

from __future__ import annotations

import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rgi_utils.atom_context import LigandConf
from rgi_utils.featurizer import build_spec
from rgi_utils.group_geom_restr_data import AngleRestraintData, DihedralRestraintData


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
        [lc], [], {"bond": {"weight": 1.0}, "angle": {"weight": 1.0}}, conf_start_sigma=1e30
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


def test_jax_minimizer_move_mode_end_to_end():
    """jax E2E (build_spec -> jax_energy.prepare_spec -> pack_spec -> make_minimizer ->
    apply_distance_shift_jax): move_mode=2 moves ONLY group2; group1 stays fixed while
    group2 lands the COM gap on target. Confirms move_mode flows via the featurizer and
    the SHARED pack_spec into the jax prepared dict, then through the jax minimizer (the
    AF3 closure path) -- not just the directly-tested apply_distance_shift_jax."""
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
    coords_np[0, 2:, 0] = 20.0  # COM1 (group1) x=0, COM2 (group2) x=20 -> dist 20
    coords = jnp.asarray(coords_np)
    g1_before = np.asarray(coords[0, :2])
    minimize = make_minimizer(spec, max_iter=100)
    coords = minimize(coords, 0.0)
    a = np.asarray(coords)
    gap = np.linalg.norm(a[0, 2:].mean(0) - a[0, :2].mean(0))
    assert abs(gap - 7.0) < 1e-5  # COM gap on target
    assert np.allclose(a[0, :2], g1_before, atol=1e-9)  # group1 EXACTLY fixed
    assert abs(a[0, 2:].mean(0)[0] - 7.0) < 1e-5  # only group2 moved (to x=7)


# --- group-COM angle / dihedral restraints ---------------------------------------


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
    """Spec with one harmonic group-COM angle restraint: groups {0,1}/{2,3}/{4,5}
    (vertex=2). ``move_free`` is the per-group free mask (default all free)."""
    ad = AngleRestraintData()
    ad.target_sites1, ad.target_sites2, ad.target_sites3 = [0, 1], [2, 3], [4, 5]
    ad.geom_type, ad.target1, ad.target2 = "harmonic", math.radians(target_deg), 0.0
    ad.move_free, ad.weight, ad.run_restr = move_free, 1.0, True
    ad.start_sigma, ad.stop_sigma = start_sigma, -1.0
    return build_spec(angle_restraints=[ad], conf_start_sigma=1e30)


def _group_angle_coords():
    """group1 COM (8,0,0), vertex COM (0,0,0), group3 COM (0,8,0) -> initial 90 deg."""
    coords = np.zeros((1, 6, 3))
    coords[0, 0], coords[0, 1] = [8.0, 0.5, 0.0], [8.0, -0.5, 0.0]
    coords[0, 2], coords[0, 3] = [0.0, 0.5, 0.0], [0.0, -0.5, 0.0]
    coords[0, 4], coords[0, 5] = [0.5, 8.0, 0.0], [-0.5, 8.0, 0.0]
    return coords


def _coms(a):
    return a[[0, 1]].mean(0), a[[2, 3]].mean(0), a[[4, 5]].mean(0)


def _group_dihedral_spec(target_deg, move_free=(True, True, True, True)):
    """Spec with one group-COM dihedral restraint over single-atom groups 0-1-2-3.
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
    """The torch CG bends the COM1-COM2-COM3 angle from 90 deg onto a 120 deg target,
    moving each group rigidly (the COM-only energy gives a group's atoms equal grad)."""
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
    """Same group-COM angle convergence on the jax minimizer (the AF3 path)."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rgi_utils.optim.jax_optim import make_minimizer

    spec = _group_angle_spec(120.0)
    coords = make_minimizer(spec, max_iter=500)(jnp.asarray(_group_angle_coords()), 0.0)
    c1, c2, c3 = _coms(np.asarray(coords)[0])
    assert abs(_angle_deg(c1, c2, c3) - 120.0) < 1.0


def test_torch_group_dihedral_converges():
    """The torch CG drives the COM dihedral from 0 deg onto a 90 deg target (90 deg is
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


def test_torch_vdw_pushes_ligand_off_fixed_protein():
    """Dynamic ligand-protein VdW: a ligand atom clashing with a fixed protein
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
    assert n in set(int(x) for x in spec.vdw_config.protein_global)

    coords_np = np.zeros((1, n_atom, 3))
    coords_np[0, :n, :] = c
    # drop the protein atom 0.5 A from ligand atom 0: a severe clash
    coords_np[0, n, :] = c[0] + np.array([0.5, 0.0, 0.0])
    coords = torch.tensor(coords_np, dtype=torch.float64)
    prot_before = coords[0, n].clone()

    opt = TorchRestraintOptimizer(spec, max_iter=200)
    e0 = opt.energy(coords)
    d0 = float(torch.linalg.norm(coords[0, 0] - coords[0, n]))
    opt.minimize(coords)
    e1 = opt.energy(coords)
    d1 = float(torch.linalg.norm(coords[0, 0] - coords[0, n]))

    assert e1 < 0.5 * e0, f"{e0} -> {e1}"
    assert d1 > d0, f"ligand not pushed away: {d0} -> {d1}"
    # the fixed protein atom must not move (it is not in active_sites)
    assert torch.allclose(coords[0, n], prot_before, atol=1e-9)


def test_jax_vdw_pushes_ligand_off_fixed_protein():
    """The JAX port of the dynamic ligand-protein VdW: the jax minimizer pushes a
    clashing ligand atom off a FIXED protein atom (not in active_sites), same as the
    torch optimizer. This is what makes `ligand_protein` / `both` work on AF3."""
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
    prot_before = np.asarray(coords[0, n])
    d0 = float(np.linalg.norm(np.asarray(coords[0, 0]) - np.asarray(coords[0, n])))

    minimize = make_minimizer(spec, max_iter=200)
    coords = minimize(coords, 0.0)
    d1 = float(np.linalg.norm(np.asarray(coords[0, 0]) - np.asarray(coords[0, n])))

    assert d1 > d0, f"ligand not pushed away: {d0} -> {d1}"
    # the fixed protein atom must not move (it is not in active_sites)
    assert np.allclose(np.asarray(coords[0, n]), prot_before, atol=1e-9)


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
            fit_idx=idx, fit_mask=np.ones((1, n)), fit_ref=ref.reshape(1, n, 3),
            calc_idx=idx, calc_mask=np.ones((1, n)), calc_ref=ref.reshape(1, n, 3),
            target_rmsd=np.array([0.0]), weight=np.array([1.0]),
            start_sigma=np.array([1e30]), stop_sigma=np.array([0.0]),
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
        return torch_energy.total_energy(a, prepared, sigma=None, include_distance=False)

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
        fit_idx=idx, fit_mask=m, fit_ref=refc, calc_idx=idx, calc_mask=m, calc_ref=refc,
        target_rmsd=torch.zeros(1, dtype=torch.float64),
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
        return torch_energy.total_energy(a, prepared, sigma=None, include_distance=False)

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
        return torch_energy.total_energy(a, prepared, sigma=None, include_distance=False)

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
        [lc], [],
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
        return torch_energy.total_energy(a, prepared, sigma=None, include_distance=False)

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
        n_active=n, active_sites=np.arange(n),
        bond=BondArrays(
            idx=np.array([[0, 1], [2, 3]], dtype=np.int64), r0=np.array([1.0, 1.5]),
            slack=np.zeros(2), weight=np.ones(2), half=np.zeros(2), mask=np.ones(2),
        ),
        rmsd=RmsdArrays(
            fit_idx=idx, fit_mask=np.ones((1, n)), fit_ref=ref.reshape(1, n, 3),
            calc_idx=idx, calc_mask=np.ones((1, n)), calc_ref=ref.reshape(1, n, 3),
            target_rmsd=np.array([0.0]), weight=np.array([1.0]),
            start_sigma=np.array([5.0]), stop_sigma=np.array([0.0]),
            mask=np.array([1.0]),
        ),
        conf_start_sigma=10.0,
    )
    opt = TorchRestraintOptimizer(spec)
    opt._ensure(torch.device("cpu"), torch.float64)
    pos = torch.tensor(rng.standard_normal((n, 3)), dtype=torch.float64)
    for sigma in (50.0, 8.0, 3.0, None):  # conf+rmsd off / conf-on-rmsd-off / both / both
        pg = opt._gated_prepared(sigma)
        e_gpu = float(_energy(pos, pg))
        e_ref = float(
            torch_energy.total_energy(pos, opt._prepared, sigma, include_distance=False)
        )
        assert abs(e_gpu - e_ref) < 1e-9, (sigma, e_gpu, e_ref)
        # the dropped scalar leaf must not be in the compiled pytree (per-value recompile)
        assert "conf_start_sigma" not in pg
    # conf gated off (sigma 50 > conf_start_sigma 10) zeros the conformer masks
    assert float(opt._gated_prepared(50.0)["bond"]["mask"].sum()) == 0.0


def test_compiled_energy_matches_eager():
    """torch.compile of the GPU energy+grad must equal eager grad_and_value (compiling
    fuses kernels; it must NOT change the maths), incl. the detached Kabsch SVD in the
    RMSD term AND the group-COM angle/dihedral terms (gather + masked centroid + cross +
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
        grp1_idx=np.array([[0, 1]]), grp2_idx=np.array([[2, 3]]),
        grp3_idx=np.array([[4, 5]]), grp1_mask=np.array([[1.0, 1.0]]),
        grp2_mask=np.array([[1.0, 1.0]]), grp3_mask=np.array([[1.0, 1.0]]),
        target1=np.array([1.3]), target2=np.array([0.0]),
        geom_type=np.array([0]), move_free=np.array([[0.0, 1.0, 0.0]]),
        weight=np.array([1.0]),
        mask=np.array([1.0]), start_sigma=np.array([1e30]), stop_sigma=np.array([-1.0]),
    )
    spec.group_dihedral = GroupDihedralArrays(
        grp1_idx=np.array([[0]]), grp2_idx=np.array([[1]]), grp3_idx=np.array([[2]]),
        grp4_idx=np.array([[3]]), grp1_mask=np.array([[1.0]]),
        grp2_mask=np.array([[1.0]]), grp3_mask=np.array([[1.0]]),
        grp4_mask=np.array([[1.0]]), target1=np.array([0.5]), target2=np.array([0.0]),
        geom_type=np.array([0]), move_free=np.array([[1.0, 1.0, 1.0, 1.0]]),
        weight=np.array([1.0]),
        mask=np.array([1.0]), start_sigma=np.array([1e30]), stop_sigma=np.array([-1.0]),
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
    dynamic ligand-protein VdW conformer path doesn't change the energy."""
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
    spec = build_spec([lc], [], {"vdw": {"weight": 1.0, "scale": 0.9}}, elements=elements)
    assert spec.vdw_config is not None  # dynamic ligand-protein VdW

    coords = torch.zeros((1, n_atom, 3), dtype=torch.float64)
    coords[0, :n, :] = torch.tensor(c)
    coords[0, n, :] = torch.tensor(c[0] + np.array([0.5, 0.0, 0.0]))  # a clash
    opt = TorchRestraintOptimizer(spec, max_iter=10)
    opt._ensure(coords.device, coords.dtype)  # builds opt._vdw
    active = coords[0, opt._active_idx, :]
    prot_pos = coords[0, opt._vdw["prot_global"], :]
    v = opt._vdw

    e_method = float(opt._vdw_energy(active, prot_pos))
    e_pure = float(
        _vdw_pair_energy(
            active, prot_pos, v["lig_local"], v["lig_r"], v["prot_r"],
            v["scale"], v["weight"],
        )
    )
    assert e_method > 0.0 and abs(e_method - e_pure) < 1e-10, (e_method, e_pure)


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
