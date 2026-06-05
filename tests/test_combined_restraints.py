"""Integration tests for the new CombinedRestraints (set_config / setup / minimize).

Uses the numpy backend (CPU, no GPU) with a mock adapter, exercising the full
flow: config parse -> distance resolution -> conformer spec build -> minimize.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rgi_utils import CombinedRestraints
from rgi_utils.atom_context import AtomRecord, LigandConf


def _write_pdb(path, coords, chain="A"):
    """Write one CA atom per coord, each its own residue (chain ``chain``)."""
    lines = []
    for i, (x, y, z) in enumerate(coords):
        lines.append(
            "ATOM  "
            f"{i + 1:>5} "
            f"{'CA':<4}"
            " "
            f"{'ALA':>3} "
            f"{chain}"
            f"{i + 1:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
            "  1.00  0.00          "
            f"{'C':>2}\n"
        )
    path.write_text("".join(lines) + "END\n")


def _superposed_rmsd(P, Q):
    """Kabsch RMSD between target P and reference Q (matches rmsd_energy)."""
    P0 = P - P.mean(0)
    Q0 = Q - Q.mean(0)
    U, _S, Vt = np.linalg.svd(Q0.T @ P0)
    V = Vt.T
    d = np.sign(np.linalg.det(V @ U.T)) or 1.0
    Vd = V.copy()
    Vd[:, 2] *= d
    R = Vd @ U.T  # R Q0 ~ P0
    return float(np.sqrt(((P0 - Q0 @ R.T) ** 2).sum() / len(P)))


class MockAdapter:
    """Mock FrameworkAdapter + ConformerAdapter for testing without a framework."""

    def __init__(self, atoms, ligand_confs=None):
        self._atoms = atoms
        self._ligand_confs = ligand_confs or []

    def iter_atoms(self):
        yield from self._atoms

    def iter_ligand_confs(self):
        yield from self._ligand_confs

    def num_atoms(self):
        return max((a.index for a in self._atoms), default=-1) + 1


@pytest.fixture(autouse=True)
def _reset_singleton():
    CombinedRestraints.reset()
    yield
    CombinedRestraints.reset()


def test_singleton():
    assert CombinedRestraints.get_instance() is CombinedRestraints.get_instance()


def test_config_defaults():
    cr = CombinedRestraints.get_instance()
    cr.set_config({})
    assert cr.config.verbose is False
    assert cr.config.gpu is False
    assert cr.config.method == "CG"
    # gpu:false now defaults to torch (run on CPU), not the numpy/scipy fallback;
    # gpu:true also torch (on the accelerator); numpy is opt-in via backend:numpy.
    assert cr.config.resolve_backend() == "torch"
    cr.set_config({"gpu": True})
    assert cr.config.resolve_backend() == "torch"
    cr.set_config({"backend": "numpy"})
    assert cr.config.resolve_backend() == "numpy"


def test_start_sigma_validation():
    """Top-level start_sigma is rejected; omitting it per restraint defaults to +inf
    (active at every step); explicit per-restraint values are honored."""
    import math

    cr = CombinedRestraints.get_instance()
    dist = {
        "atom_selection1": "chain A",
        "atom_selection2": "chain B",
        "start_sigma": 1e30,
        "harmonic": {"target_distance": 5.0},
    }
    # (a) top-level start_sigma is rejected
    with pytest.raises(ValueError, match="top-level"):
        cr.set_config({"start_sigma": 1e30, "distance_restraints_config": [dist]})
    # (b) a distance entry without start_sigma defaults to +inf (every step)
    no_ss = {k: v for k, v in dist.items() if k != "start_sigma"}
    cr.set_config({"distance_restraints_config": [no_ss]})
    assert math.isinf(cr.config.distance_data[0].start_sigma)
    # (c) conformer terms without start_sigma default to +inf (every step)
    cr.set_config({"conformer_restraints_config": {"bond": {"weight": 1.0}}})
    assert math.isinf(cr.config.conf_start_sigma)
    # (d) explicit per-restraint values are honored
    cr.set_config(
        {
            "distance_restraints_config": [dist],
            "conformer_restraints_config": {"start_sigma": 1.0, "bond": {"weight": 1.0}},
        }
    )
    assert cr.config.conf_start_sigma == 1.0


def test_setup_empty_is_inactive():
    cr = CombinedRestraints.get_instance()
    cr.set_config({})
    cr.setup(MockAdapter([AtomRecord("A", 1, 0), AtomRecord("A", 2, 1)]))
    assert not cr.is_active()
    coords = np.zeros((1, 2, 3))
    # minimize is a no-op and must not raise
    cr.minimize(coords, 0, sigma=0.0)


def test_distance_resolve_and_minimize():
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "backend": "numpy",
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "start_sigma": 1e30,
                    "harmonic": {"target_distance": 5.0},
                }
            ],
        }
    )
    atoms = [
        AtomRecord("A", 1, 0),
        AtomRecord("A", 2, 1),
        AtomRecord("B", 1, 2),
        AtomRecord("B", 2, 3),
    ]
    cr.setup(MockAdapter(atoms))
    assert cr.is_active()
    dd = cr.config.distance_data[0]
    assert set(dd.target_sites1) == {0, 1}
    assert set(dd.target_sites2) == {2, 3}

    coords = np.zeros((1, 4, 3))
    coords[0, 2:, 0] = 20.0  # group B far away on x
    d0 = np.linalg.norm(coords[0, 2:].mean(0) - coords[0, :2].mean(0))
    cr.minimize(coords, 0, sigma=0.0)
    d1 = np.linalg.norm(coords[0, 2:].mean(0) - coords[0, :2].mean(0))
    assert abs(d1 - 5.0) < abs(d0 - 5.0)


def test_distance_closed_form_hits_target_exactly():
    """The COM-distance restraint is applied in closed form (rigid translation), so one
    minimize lands the group COM distance exactly on target -- no iterative residual,
    and no conformer solver is needed for a distance-only spec."""
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "backend": "numpy",
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "start_sigma": 1e30,
                    "harmonic": {"target_distance": 7.0},
                }
            ],
        }
    )
    atoms = [
        AtomRecord("A", 1, 0),
        AtomRecord("A", 2, 1),
        AtomRecord("B", 1, 2),
        AtomRecord("B", 2, 3),
    ]
    cr.setup(MockAdapter(atoms))
    coords = np.zeros((1, 4, 3))
    coords[0, 2:, 0] = 20.0  # COM1 at x=0, COM2 at x=20 -> dist 20
    cr.minimize(coords, 0, sigma=0.0)
    d = np.linalg.norm(coords[0, 2:].mean(0) - coords[0, :2].mean(0))
    assert abs(d - 7.0) < 1e-6
    # minimal-displacement split (equal group sizes) -> COMs meet symmetrically
    assert abs(coords[0, :2].mean(0)[0] - 6.5) < 1e-6
    assert abs(coords[0, 2:].mean(0)[0] - 13.5) < 1e-6


def test_minimize_skipped_above_start_sigma():
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "backend": "numpy",
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "start_sigma": 1.0,
                    "harmonic": {"target_distance": 5.0},
                }
            ],
        }
    )
    cr.setup(MockAdapter([AtomRecord("A", 1, 0), AtomRecord("B", 1, 1)]))
    coords = np.zeros((1, 2, 3))
    coords[0, 1, 0] = 20.0
    original = coords.copy()
    cr.minimize(coords, 0, sigma=2.0)  # sigma > start_sigma -> skip
    assert np.allclose(coords, original)


def test_conformer_opt_in():
    """Conformer restraints are opt-in: built only when conformer_restraints_config is
    present AND the ligand is flagged conformer_restraints=True."""
    m = Chem.AddHs(Chem.MolFromSmiles("CC"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    c = np.asarray(m.GetConformer().GetPositions())
    n = m.GetNumAtoms()
    atoms = [AtomRecord("A", i + 1, i) for i in range(n)]
    conf_cfg = {
        "backend": "numpy",
        "conformer_restraints_config": {"start_sigma": 1e30, "bond": {"weight": 0.1}},
    }
    cr = CombinedRestraints.get_instance()

    # (a) conformer_restraints_config present + ligand flagged -> conformer active
    cr.set_config(conf_cfg)
    cr.setup(MockAdapter(atoms, [LigandConf(m, c, np.arange(n), conformer_restraints=True)]))
    assert cr.spec.has_conformer()
    # (b) config present + ligand NOT flagged -> opt-in: no conformer
    cr.set_config(conf_cfg)
    cr.setup(MockAdapter(atoms, [LigandConf(m, c, np.arange(n))]))
    assert not cr.spec.has_conformer()
    # (c) ligand flagged but NO conformer_restraints_config -> config gate: no conformer
    cr.set_config({"backend": "numpy"})
    cr.setup(MockAdapter(atoms, [LigandConf(m, c, np.arange(n), conformer_restraints=True)]))
    assert not cr.spec.has_conformer()


def test_multiligand_conformer_setup():
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "backend": "numpy",
            "conformer_restraints_config": {"start_sigma": 1e30, "bond": {"weight": 0.1}},
        }
    )
    m = Chem.AddHs(Chem.MolFromSmiles("CC"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    c = np.asarray(m.GetConformer().GetPositions())
    n = m.GetNumAtoms()
    lcs = [
        LigandConf(m, c, np.arange(n), conformer_restraints=True),
        LigandConf(m, c, np.arange(n) + n, conformer_restraints=True),
    ]
    atoms = [AtomRecord("A", i + 1, i) for i in range(2 * n)]
    cr.setup(MockAdapter(atoms, ligand_confs=lcs))
    assert cr.is_active()
    assert cr.spec.n_active == 2 * n
    assert cr.spec.bond.idx.shape[0] == 2 * m.GetNumBonds()


def test_rmsd_resolve_and_minimize(tmp_path):
    """RMSD restraint (target_rmsd=0) drives the moving group's superposed RMSD to
    the reference down. Exercises pdb_ref + RmsdData.resolve_sites + the CG solver."""
    rng = np.random.default_rng(5)
    n = 6
    ref = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_pdb(pdb, ref)
    th = 0.6
    rz = np.array(
        [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    )
    tgt = ref @ rz.T + np.array([4.0, -2.0, 1.0]) + rng.standard_normal((n, 3)) * 0.8

    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "backend": "numpy",
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(pdb),
                    "target_rmsd": 0.0,
                    "atom_selection_ref": "chain A",
                    "atom_selection_target": "chain A",
                    "start_sigma": 1e30,
                }
            ],
        }
    )
    atoms = [AtomRecord("A", i + 1, i) for i in range(n)]
    cr.setup(MockAdapter(atoms))
    assert cr.is_active() and cr.spec.has_rmsd()
    assert cr.config.rmsd_data[0].ref_coords.shape == (n, 3)

    coords = tgt.reshape(1, n, 3).copy()
    before = _superposed_rmsd(coords[0], ref)
    cr.minimize(coords, 0, sigma=0.0)
    after = _superposed_rmsd(coords[0], ref)
    assert after < before, f"rmsd did not decrease: {before:.3f} -> {after:.3f}"


def test_rmsd_count_mismatch_raises(tmp_path):
    """User requirement: mismatched ref/target atom counts must error out."""
    ref = np.random.default_rng(1).standard_normal((6, 3))
    pdb = tmp_path / "ref.pdb"
    _write_pdb(pdb, ref)
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "backend": "numpy",
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(pdb),
                    "target_rmsd": 1.0,
                    "atom_selection_ref": "chain A",  # 6 atoms
                    "atom_selection_target": "chain A and resid 1 to 3",  # 3 atoms
                }
            ],
        }
    )
    atoms = [AtomRecord("A", i + 1, i) for i in range(6)]
    with pytest.raises(ValueError, match="count mismatch"):
        cr.setup(MockAdapter(atoms))


def test_rmsd_missing_pdb_raises(tmp_path):
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "backend": "numpy",
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(tmp_path / "does_not_exist.pdb"),
                    "target_rmsd": 1.0,
                    "atom_selection_ref": "chain A",
                    "atom_selection_target": "chain A",
                }
            ],
        }
    )
    atoms = [AtomRecord("A", i + 1, i) for i in range(3)]
    with pytest.raises(ValueError, match="could not be read"):
        cr.setup(MockAdapter(atoms))


def _dist_atoms():
    return [
        AtomRecord("A", 1, 0),
        AtomRecord("A", 2, 1),
        AtomRecord("B", 1, 2),
        AtomRecord("B", 2, 3),
    ]


def _dist_config(**extra):
    return {
        **extra,
        "distance_restraints_config": [
            {
                "atom_selection1": "chain A",
                "atom_selection2": "chain B",
                "start_sigma": 1e30,
                "harmonic": {"target_distance": 5.0},
            }
        ],
    }


def test_gpu_false_uses_torch_on_cpu():
    """gpu:false (no explicit backend) resolves to the torch backend and runs on a
    CPU tensor (replacing the old numpy/scipy fallback)."""
    torch = pytest.importorskip("torch")
    cr = CombinedRestraints.get_instance()
    cr.set_config(_dist_config())  # gpu omitted -> False; no explicit backend
    assert cr.config.resolve_backend() == "torch"
    cr.setup(MockAdapter(_dist_atoms()))
    assert cr._backend == "torch"
    coords = torch.zeros((1, 4, 3))  # CPU tensor
    coords[0, 2:, 0] = 20.0
    cr.minimize(coords, 0, sigma=0.0)
    assert coords.device.type == "cpu"
    d = float(torch.norm(coords[0, 2:].mean(0) - coords[0, :2].mean(0)))
    assert abs(d - 5.0) < 1e-4  # closed-form COM-distance shift hits target on CPU


@pytest.mark.gpu
def test_gpu_false_cuda_coords_compute_on_cpu():
    """gpu:false with CUDA coords: the restraint is computed on CPU but the result is
    written back to the original CUDA device (the model stays on GPU)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cr = CombinedRestraints.get_instance()
    cr.set_config(_dist_config(gpu=False))
    cr.setup(MockAdapter(_dist_atoms()))
    assert cr._backend == "torch"
    coords = torch.zeros((1, 4, 3), device="cuda")
    coords[0, 2:, 0] = 20.0
    cr.minimize(coords, 0, sigma=0.0)
    assert coords.device.type == "cuda"  # written back to the original device
    d = float(torch.norm(coords[0, 2:].mean(0) - coords[0, :2].mean(0)))
    assert abs(d - 5.0) < 1e-4
