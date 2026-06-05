"""Optimizer sanity: each backend reduces restraint energy on a distorted ligand."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rgi_utils.atom_context import LigandConf
from rgi_utils.energy import numpy_energy
from rgi_utils.featurizer import build_spec


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


def _energy(spec, coords_np):
    prep = numpy_energy.prepare_spec(spec)
    return float(numpy_energy.total_energy(coords_np[:, spec.active_sites, :], prep))


def test_numpy_minimize_reduces_energy():
    from rgi_utils.optim.numpy_optim import NumpyRestraintOptimizer

    spec, coords = _distorted_ethane()
    e0 = _energy(spec, coords)
    NumpyRestraintOptimizer(spec, max_iter=300, method="CG").minimize(coords)
    e1 = _energy(spec, coords)
    assert e1 < 0.5 * e0, f"{e0} -> {e1}"


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
    minimize = make_minimizer(spec, max_iter=2000, learning_rate=0.02)
    coords = minimize(coords, 0.0)
    e1 = energy_of(spec, coords)
    assert e1 < 0.5 * e0, f"{e0} -> {e1}"


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
