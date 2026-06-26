"""Integration tests for the new CombinedRestraints (set_config / setup / minimize).

Uses the inferred torch backend (CPU, no GPU) with a mock adapter, exercising the full
flow: config parse -> distance resolution -> conformer spec build -> minimize.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rgi_utils import CombinedRestraints
from rgi_utils._align import THREE_TO_ONE
from rgi_utils.atom_context import AtomRecord, LigandConf

_ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}


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
    assert cr.config.gpu is True
    assert cr.config.method == "CG"
    # backend is no longer a config field — it is inferred at minimize/get_minimizer
    # time (numpy/torch coords -> torch; get_minimizer() -> jax).
    cr.set_config({"gpu": True})
    assert cr.config.gpu is True
    # a leftover `backend` key is rejected with a migration hint (it is now inferred)
    with pytest.raises(ValueError, match="backend"):
        cr.set_config({"backend": "torch"})


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
            "conformer_restraints_config": {
                "start_sigma": 1.0,
                "bond": {"weight": 1.0},
            },
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


class _ThrowingElementsAdapter(MockAdapter):
    """MockAdapter whose get_elements() raises -- exercises the B3 path in setup():
    elements feed ONLY the VdW term, so a failure is tolerated (warn) when VdW is off
    but re-raised when vdw.weight > 0."""

    def get_elements(self):
        raise RuntimeError("boom: simulated broken get_elements")


def test_get_elements_failure_loud_only_when_vdw_requested(caplog):
    """B3: a broken get_elements() must not silently drop a REQUESTED VdW term.
    weight 0 (default) -> tolerate (warn + continue); weight > 0 -> re-raise."""
    atoms = [AtomRecord("A", 1, 0), AtomRecord("A", 2, 1)]
    # (a) VdW OFF (default): tolerate the failure -> warn and continue, no raise.
    cr = CombinedRestraints.get_instance()
    cr.set_config({})
    with caplog.at_level(logging.WARNING):
        cr.setup(_ThrowingElementsAdapter(atoms))  # must NOT raise
    assert any("get_elements failed" in r.getMessage() for r in caplog.records)
    # (b) VdW requested (weight > 0): the failure is fatal -> re-raise the real error.
    CombinedRestraints.reset()
    cr = CombinedRestraints.get_instance()
    cr.set_config({"conformer_restraints_config": {"vdw": {"weight": 1.0}}})
    with pytest.raises(RuntimeError, match="boom"):
        cr.setup(_ThrowingElementsAdapter(atoms))


def test_distance_resolve_and_minimize():
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
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


def test_distance_hits_target():
    """The centroid-distance restraint is optimized by the autodiff CG (reduced-mass
    rescale -> rigid group translation), landing the group centroid distance on target
    within CG tolerance. Equal groups + move_mode=0 -> minimal-displacement symmetric split."""
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
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
    coords[0, 2:, 0] = 20.0  # centroid1 at x=0, centroid2 at x=20 -> dist 20
    cr.minimize(coords, 0, sigma=0.0)
    d = np.linalg.norm(coords[0, 2:].mean(0) - coords[0, :2].mean(0))
    assert abs(d - 7.0) < 1e-3  # centroid gap lands on target (the physical invariant)
    # NOTE: the CG minimizes the sign-agnostic (d - target)^2, so for a LARGE one-shot
    # move it may settle on the reflected (groups-crossed) solution rather than the
    # minimal-displacement split the old closed-form guaranteed. We assert only the gap;
    # the per-step diffusion regime (small moves) stays in the minimal-displacement basin,
    # and the exact N2:N1 split is covered by the parity test.


def test_distance_move_mode_end_to_end():
    """End-to-end (config -> DistanceData.move_mode -> featurizer -> DistanceArrays ->
    pack_spec -> CG with pinned group1): `move: 2` moves ONLY atom_selection2's group, so
    chain A (group1) stays put while chain B lands the centroid gap on target. Guards the
    middle wiring that the parity tests (move_mode=0) and the hand-built-dict tests both
    skip -- a dropped schema field / wrong featurizer attr would silently fall back to 0
    (both) and chain A would move."""
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "start_sigma": 1e30,
                    "move": 2,  # only chain B (atom_selection2) moves
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
    assert cr.config.distance_data[0].move_mode == 2  # parsed onto the DistanceData
    coords = np.zeros((1, 4, 3))
    coords[0, 2:, 0] = 20.0  # centroid1 (A) at x=0, centroid2 (B) at x=20 -> dist 20
    a_before = coords[0, :2].copy()
    cr.minimize(coords, 0, sigma=0.0)
    d = np.linalg.norm(coords[0, 2:].mean(0) - coords[0, :2].mean(0))
    assert abs(d - 7.0) < 1e-3  # centroid gap lands on target
    assert np.allclose(
        coords[0, :2], a_before
    )  # group1 (chain A) EXACTLY fixed (pinned, grad 0)
    # group2 carries the whole shift (group1 pinned). The CG may land it on the reflected
    # side for this large one-shot move, so we assert only that the pinned group did not move.


def test_distance_name_ca_selects_backbone_only():
    """`name CA` inside a distance selection filters the centroid group to CA atoms only
    (the user-facing 'name CA' for distance restraints). Each residue here carries a CA
    and a CB, so the group must hold only the CA indices -- a regression that dropped
    `name` from the distance candidate dict would admit every atom and break these
    sets."""
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A and name CA",
                    "atom_selection2": "chain B and name CA",
                    "start_sigma": 1e30,
                    "harmonic": {"target_distance": 5.0},
                }
            ],
        }
    )
    # two residues per chain, each a CA (selected) + a CB (excluded by `name CA`)
    atoms = [
        AtomRecord("A", 1, 0, name="CA"),
        AtomRecord("A", 1, 1, name="CB"),
        AtomRecord("A", 2, 2, name="CA"),
        AtomRecord("A", 2, 3, name="CB"),
        AtomRecord("B", 1, 4, name="CA"),
        AtomRecord("B", 1, 5, name="CB"),
        AtomRecord("B", 2, 6, name="CA"),
        AtomRecord("B", 2, 7, name="CB"),
    ]
    cr.setup(MockAdapter(atoms))
    dd = cr.config.distance_data[0]
    assert set(dd.target_sites1) == {0, 2}  # chain A CAs only (CB 1,3 excluded)
    assert set(dd.target_sites2) == {4, 6}  # chain B CAs only (CB 5,7 excluded)
    # end-to-end: the CG centroid shift lands the CA-group distance on target
    coords = np.zeros((1, 8, 3))
    coords[0, 4:, 0] = 20.0  # chain B far on x
    cr.minimize(coords, 0, sigma=0.0)
    centroid1 = coords[0, [0, 2]].mean(0)
    centroid2 = coords[0, [4, 6]].mean(0)
    assert abs(np.linalg.norm(centroid2 - centroid1) - 5.0) < 1e-3


def test_distance_moltype_selector_matches():
    """Parity with RMSD: protein/dna/rna selectors resolve in distance centroid groups too
    (the distance candidate dict now carries mol_type). Guards the silent-failure
    footgun where 'protein'/'dna' would match nothing and an OR with a chain term would
    quietly return a wrong, non-empty group."""
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "distance_restraints_config": [
                {
                    "atom_selection1": "protein",
                    "atom_selection2": "dna",
                    "start_sigma": 1e30,
                    "harmonic": {"target_distance": 5.0},
                }
            ],
        }
    )
    atoms = [
        AtomRecord("A", 1, 0, name="CA", mol_type="protein"),
        AtomRecord("A", 2, 1, name="CA", mol_type="protein"),
        AtomRecord("B", 1, 2, name="P", mol_type="dna"),
        AtomRecord("B", 2, 3, name="P", mol_type="dna"),
    ]
    cr.setup(MockAdapter(atoms))
    dd = cr.config.distance_data[0]
    assert set(dd.target_sites1) == {0, 1}  # protein atoms only
    assert set(dd.target_sites2) == {2, 3}  # dna atoms only


def test_distance_backbone_sidechain_resname_gated():
    """`backbone`/`sidechain` resolve in distance centroid groups via the resname-gated
    polymer path: each atom carries resname but NO mol_type (the chai/of3/protenix
    case), so the distance candidate dict's `resname` is what fires the polymer gate.
    CA -> group1 (backbone), CB -> group2 (sidechain)."""
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "distance_restraints_config": [
                {
                    "atom_selection1": "backbone",
                    "atom_selection2": "sidechain",
                    "start_sigma": 1e30,
                    "harmonic": {"target_distance": 5.0},
                }
            ],
        }
    )
    atoms = [
        AtomRecord("A", 1, 0, name="CA", resname="ALA"),
        AtomRecord("A", 1, 1, name="CB", resname="ALA"),
        AtomRecord("A", 2, 2, name="CA", resname="VAL"),
        AtomRecord("A", 2, 3, name="CB", resname="VAL"),
    ]
    cr.setup(MockAdapter(atoms))
    dd = cr.config.distance_data[0]
    assert set(dd.target_sites1) == {0, 2}  # backbone CAs
    assert set(dd.target_sites2) == {1, 3}  # sidechain CBs


def test_minimize_skipped_above_start_sigma():
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
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
        "conformer_restraints_config": {"start_sigma": 1e30, "bond": {"weight": 0.1}},
    }
    cr = CombinedRestraints.get_instance()

    # (a) conformer_restraints_config present + ligand flagged -> conformer active
    cr.set_config(conf_cfg)
    cr.setup(
        MockAdapter(atoms, [LigandConf(m, c, np.arange(n), conformer_restraints=True)])
    )
    assert cr.spec.has_conformer()
    # (b) config present + ligand NOT flagged -> opt-in: no conformer
    cr.set_config(conf_cfg)
    cr.setup(MockAdapter(atoms, [LigandConf(m, c, np.arange(n))]))
    assert not cr.spec.has_conformer()
    # (c) ligand flagged but NO conformer_restraints_config -> config gate: no conformer
    cr.set_config({})
    cr.setup(
        MockAdapter(atoms, [LigandConf(m, c, np.arange(n), conformer_restraints=True)])
    )
    assert not cr.spec.has_conformer()


def test_multiligand_conformer_setup():
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "conformer_restraints_config": {
                "start_sigma": 1e30,
                "bond": {"weight": 0.1},
            },
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


def test_multiligand_interligand_vdw_setup():
    """Two restrained ligands + vdw on (default mode='both') build inter-ligand VdW pairs
    in spec.vdw (the cross product). Heavy-only ethane has no intramolecular pair, so the
    count is purely inter (n*n)."""
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "conformer_restraints_config": {
                "start_sigma": 1e30,
                "vdw": {"weight": 1.0},
            },
        }
    )
    m = Chem.AddHs(Chem.MolFromSmiles("CC"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    m = Chem.RemoveHs(m)  # 2 heavy atoms, 0 intramolecular pairs
    c = np.asarray(m.GetConformer().GetPositions())
    n = m.GetNumAtoms()
    lcs = [
        LigandConf(m, c, np.arange(n), conformer_restraints=True),
        LigandConf(m, c, np.arange(n) + n, conformer_restraints=True),
    ]
    atoms = [AtomRecord("A", i + 1, i) for i in range(2 * n)]
    cr.setup(MockAdapter(atoms, ligand_confs=lcs))
    assert cr.spec.vdw is not None
    assert cr.spec.vdw.idx.shape[0] == n * n  # inter cross product (intra=0 for ethane)


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
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(pdb),
                    "harmonic": {"target_rmsd": 0.0},
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
    assert cr.config.rmsd_data[0].calc_ref_coords.shape == (n, 3)

    coords = tgt.reshape(1, n, 3).copy()
    before = _superposed_rmsd(coords[0], ref)
    cr.minimize(coords, 0, sigma=0.0)
    after = _superposed_rmsd(coords[0], ref)
    assert after < before, f"rmsd did not decrease: {before:.3f} -> {after:.3f}"


def test_rmsd_stop_sigma_releases_below(tmp_path):
    """stop_sigma RELEASES the restraint below it: minimize at sigma < stop_sigma leaves
    the coordinates untouched (the model's final low-sigma steps run restraint-free),
    while minimize in the active window still drives the superposed RMSD down. Exercises
    config -> RmsdData.stop_sigma -> optimizer gate wiring (dangling-terminus fix)."""
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
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(pdb),
                    "harmonic": {"target_rmsd": 0.0},
                    "atom_selection_ref": "chain A",
                    "atom_selection_target": "chain A",
                    "start_sigma": 1e30,
                    "stop_sigma": 2.0,  # released for sigma < 2.0
                }
            ],
        }
    )
    assert cr.config.rmsd_data[0].stop_sigma == 2.0
    atoms = [AtomRecord("A", i + 1, i) for i in range(n)]
    cr.setup(MockAdapter(atoms))

    # sigma below stop_sigma -> released -> coords unchanged
    coords = tgt.reshape(1, n, 3).copy()
    before = _superposed_rmsd(coords[0], ref)
    cr.minimize(coords, 0, sigma=1.0)
    assert _superposed_rmsd(coords[0], ref) == pytest.approx(before, abs=1e-6)

    # sigma inside [stop_sigma, start_sigma] -> active -> rmsd decreases
    coords = tgt.reshape(1, n, 3).copy()
    cr.minimize(coords, 0, sigma=5.0)
    assert _superposed_rmsd(coords[0], ref) < before


def test_rmsd_stop_sigma_above_start_raises(tmp_path):
    """stop_sigma > start_sigma inverts the active window to EMPTY -> the restraint can
    never act. setup() must RAISE (a warning would be muted by the package NullHandler,
    leaving a silent no-op that reads as satisfied)."""
    rng = np.random.default_rng(5)
    n = 4
    ref = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_pdb(pdb, ref)
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(pdb),
                    "harmonic": {"target_rmsd": 0.0},
                    "atom_selection_ref": "chain A",
                    "atom_selection_target": "chain A",
                    "start_sigma": 1.0,
                    "stop_sigma": 5.0,  # > start_sigma -> empty active window
                }
            ],
        }
    )
    atoms = [AtomRecord("A", i + 1, i) for i in range(n)]
    with pytest.raises(ValueError, match="stop_sigma > start_sigma"):
        cr.setup(MockAdapter(atoms))


def test_distance_move_mode_parsing():
    """The distance `move` key parses to move_mode 0/1/2 (both / 1 / 2), accepts int or
    string, defaults to 0 (both) when omitted, and raises on an unknown value."""
    from rgi_utils.distance_restr_data import DistanceData

    def mode(move):
        dd = DistanceData()
        cfg = {
            "atom_selection1": "chain A and resid 1",
            "atom_selection2": "chain A and resid 2",
            "harmonic": {"target_distance": 5.0},
        }
        if move is not None:
            cfg["move"] = move
        dd.set_config(cfg)
        return dd.move_mode

    assert mode(None) == 0  # omitted -> both
    assert mode("both") == 0
    assert mode(1) == 1 and mode("1") == 1  # int or string form
    assert mode(2) == 2 and mode("2") == 2
    with pytest.raises(ValueError, match="move"):
        mode("group1")  # unknown value raises (no silent fallback)
    with pytest.raises(ValueError, match="move"):
        mode(3)


def test_distance_weight_parsing():
    """The distance `weight` key parses to DistanceData.weight (float), defaulting to 1.0
    when omitted. It is accepted alongside move (no `warn_unknown_keys` warning)."""
    from rgi_utils.distance_restr_data import DistanceData

    def weight(w):
        dd = DistanceData()
        cfg = {
            "atom_selection1": "chain A and resid 1",
            "atom_selection2": "chain A and resid 2",
            "harmonic": {"target_distance": 5.0},
        }
        if w is not None:
            cfg["weight"] = w
        dd.set_config(cfg)
        return dd.weight

    assert weight(None) == 1.0  # omitted -> default 1.0
    assert weight(2.5) == 2.5
    assert weight(3) == 3.0  # int coerced to float


def test_distance_stop_sigma_releases_below():
    """A DISTANCE restraint with stop_sigma is RELEASED below it (the per-entry sigma gate
    turns it off): minimize at sigma < stop leaves the centroid separation untouched,
    while in the active window it lands on target. Exercises
    DistanceData.stop_sigma -> per-entry gate (stop_sigma on a distance term)."""
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "start_sigma": 1e30,
                    "stop_sigma": 2.0,  # released for sigma < 2.0
                    "harmonic": {"target_distance": 7.0},
                }
            ],
        }
    )
    assert cr.config.distance_data[0].stop_sigma == 2.0
    atoms = [
        AtomRecord("A", 1, 0),
        AtomRecord("A", 2, 1),
        AtomRecord("B", 1, 2),
        AtomRecord("B", 2, 3),
    ]
    cr.setup(MockAdapter(atoms))

    def centroid_dist(c):
        return np.linalg.norm(c[0, 2:].mean(0) - c[0, :2].mean(0))

    # sigma below stop_sigma -> released -> centroid separation unchanged (still 20)
    coords = np.zeros((1, 4, 3))
    coords[0, 2:, 0] = 20.0
    cr.minimize(coords, 0, sigma=1.0)
    assert centroid_dist(coords) == pytest.approx(20.0, abs=1e-6)

    # sigma in [stop, start] -> active -> CG shift lands on target
    coords = np.zeros((1, 4, 3))
    coords[0, 2:, 0] = 20.0
    cr.minimize(coords, 0, sigma=5.0)
    assert abs(centroid_dist(coords) - 7.0) < 1e-3


def test_distance_stop_sigma_above_start_raises():
    """Empty window (stop_sigma > start_sigma) on a DISTANCE restraint must RAISE,
    mirroring the rmsd / conformer checks in _warn_never_active."""
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "start_sigma": 1.0,
                    "stop_sigma": 5.0,  # > start_sigma -> empty window
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
    with pytest.raises(ValueError, match="stop_sigma > start_sigma"):
        cr.setup(MockAdapter(atoms))


def test_distance_step_window_gates_e2e():
    """A DISTANCE restraint with a STEP window (start_step/stop_step) is gated on the
    diffusion step index instead of sigma: minimize OUTSIDE [start_step, stop_step] leaves
    the centroid separation untouched, INSIDE it lands on target. End-to-end exercise of
    config -> DistanceData.start_step/stop_step -> featurizer -> the distance per-entry step gate.
    The 2nd positional minimize arg is the step index (istep)."""
    cr = CombinedRestraints()
    cr.set_config(
        {
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    # step window only; sigma window stays at its always-on default
                    "start_step": 5,
                    "stop_step": 10,
                    "harmonic": {"target_distance": 7.0},
                }
            ],
        }
    )
    assert cr.config.distance_data[0].start_step == 5.0
    assert cr.config.distance_data[0].stop_step == 10.0
    atoms = [
        AtomRecord("A", 1, 0),
        AtomRecord("A", 2, 1),
        AtomRecord("B", 1, 2),
        AtomRecord("B", 2, 3),
    ]
    cr.setup(MockAdapter(atoms))

    def centroid_dist(c):
        return np.linalg.norm(c[0, 2:].mean(0) - c[0, :2].mean(0))

    # step before the window -> gated off -> centroid separation unchanged (still 20)
    coords = np.zeros((1, 4, 3))
    coords[0, 2:, 0] = 20.0
    cr.minimize(coords, 3, sigma=5.0)
    assert centroid_dist(coords) == pytest.approx(20.0, abs=1e-6)

    # step inside [start_step, stop_step] -> active -> CG shift lands on target
    coords = np.zeros((1, 4, 3))
    coords[0, 2:, 0] = 20.0
    cr.minimize(coords, 7, sigma=5.0)
    assert abs(centroid_dist(coords) - 7.0) < 1e-3

    # step after the window -> gated off again -> unchanged
    coords = np.zeros((1, 4, 3))
    coords[0, 2:, 0] = 20.0
    cr.minimize(coords, 12, sigma=5.0)
    assert centroid_dist(coords) == pytest.approx(20.0, abs=1e-6)


def test_distance_empty_step_window_raises():
    """Empty STEP window (stop_step < start_step) on a DISTANCE restraint must RAISE,
    mirroring the empty sigma-window check in _warn_never_active."""
    cr = CombinedRestraints()
    cr.set_config(
        {
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "start_step": 50,
                    "stop_step": 10,  # < start_step -> empty step window
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
    with pytest.raises(ValueError, match="stop_step < start_step"):
        cr.setup(MockAdapter(atoms))


def test_conformer_stop_sigma_above_start_raises():
    """Empty window (conf_stop_sigma > conf_start_sigma) on the CONFORMER block must
    RAISE. Builds a real conformer (flagged ligand) so has_conformer() is True and the
    check is reached."""
    m = Chem.AddHs(Chem.MolFromSmiles("CC"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    c = np.asarray(m.GetConformer().GetPositions())
    n = m.GetNumAtoms()
    atoms = [AtomRecord("A", i + 1, i) for i in range(n)]
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "conformer_restraints_config": {
                "start_sigma": 1.0,
                "stop_sigma": 5.0,  # > conf_start_sigma -> empty window
                "bond": {"weight": 1.0},
            },
        }
    )
    lc = LigandConf(m, c, np.arange(n), conformer_restraints=True)
    with pytest.raises(ValueError, match="conf_stop_sigma > conf_start_sigma"):
        cr.setup(MockAdapter(atoms, [lc]))


def test_rmsd_count_mismatch_raises(tmp_path):
    """User requirement: mismatched ref/target atom counts must error out."""
    ref = np.random.default_rng(1).standard_normal((6, 3))
    pdb = tmp_path / "ref.pdb"
    _write_pdb(pdb, ref)
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(pdb),
                    "harmonic": {"target_rmsd": 1.0},
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
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(tmp_path / "does_not_exist.pdb"),
                    "harmonic": {"target_rmsd": 1.0},
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
    """gpu:false (backend inferred torch) runs on a CPU tensor (replacing the old
    numpy/scipy fallback). Backend is inferred at minimize time, so _backend is None
    until the first minimize."""
    torch = pytest.importorskip("torch")
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        _dist_config(gpu=False)
    )  # gpu defaults True now; set False for the CPU path
    cr.setup(MockAdapter(_dist_atoms()))
    assert cr._backend is None  # lazy: not resolved until first minimize/get_minimizer
    coords = torch.zeros((1, 4, 3))  # CPU tensor
    coords[0, 2:, 0] = 20.0
    cr.minimize(coords, 0, sigma=0.0)
    assert cr._backend == "torch"  # inferred from the torch tensor
    assert coords.device.type == "cpu"
    d = float(torch.norm(coords[0, 2:].mean(0) - coords[0, :2].mean(0)))
    assert abs(d - 5.0) < 1e-3  # CG centroid-distance shift hits target on CPU


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
    coords = torch.zeros((1, 4, 3), device="cuda")
    coords[0, 2:, 0] = 20.0
    cr.minimize(coords, 0, sigma=0.0)
    assert cr._backend == "torch"  # inferred from the torch tensor (lazy)
    assert coords.device.type == "cuda"  # written back to the original device
    d = float(torch.norm(coords[0, 2:].mean(0) - coords[0, :2].mean(0)))
    assert abs(d - 5.0) < 1e-4


def test_backend_inferred_torch_from_numpy():
    """backend is inferred at minimize time: a numpy coords array -> the torch path.
    setup leaves _backend None (lazy); the first minimize resolves it."""
    cr = CombinedRestraints.get_instance()
    cr.set_config(_dist_config())
    cr.setup(MockAdapter(_dist_atoms()))
    assert cr._backend is None  # not resolved until first minimize/get_minimizer
    coords = np.zeros((1, 4, 3))
    coords[0, 2:, 0] = 20.0
    cr.minimize(coords, 0, sigma=0.0)
    assert cr._backend == "torch"


def test_backend_inferred_jax_via_get_minimizer():
    """get_minimizer() selects the jax backend (the AF3 path) -- no config key needed."""
    pytest.importorskip("jax")
    cr = CombinedRestraints.get_instance()
    cr.set_config(_dist_config())
    cr.setup(MockAdapter(_dist_atoms()))
    m = cr.get_minimizer()
    assert cr._backend == "jax"
    assert m is not None


def _write_pdb_records(path, records, chain="A"):
    """records: list of (resid, name, x, y, z). Atoms are written in the given order
    (so callers can shuffle within a residue to exercise identity pairing)."""
    lines = []
    for i, (resid, name, x, y, z) in enumerate(records):
        lines.append(
            "ATOM  "
            f"{i + 1:>5} "
            f"{name:<4}"
            " "
            f"{'ALA':>3} "
            f"{chain}"
            f"{resid:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
            "  1.00  0.00          "
            f"{(name[0] if name else 'C'):>2}\n"
        )
    path.write_text("".join(lines) + "END\n")


def test_rmsd_identity_pairing_within_residue_order(tmp_path):
    """Atoms shuffled WITHIN each residue in the ref PDB still pair by (chain, resid,
    name) — the AF3 failure mode. The resolved ref coords come back in the adapter's
    atom order (order-pairing would have mis-paired them)."""
    rng = np.random.default_rng(8)
    res_atoms = ["N", "CA", "C"]
    nres = 4
    coords = {}  # (resid, name) -> xyz
    atoms = []  # adapter order: residue-major, atoms in N/CA/C order
    idx = 0
    for r in range(1, nres + 1):
        for nm in res_atoms:
            coords[(r, nm)] = rng.standard_normal(3) * 3.0
            atoms.append(AtomRecord("A", r, idx, name=nm))
            idx += 1
    # PDB: residues in order, but atoms WITHIN each residue in a different order
    records = []
    for r in range(1, nres + 1):
        for nm in ["CA", "C", "N"]:
            records.append((r, nm, *coords[(r, nm)]))
    pdb = tmp_path / "ref.pdb"
    _write_pdb_records(pdb, records)
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(pdb),
                    "harmonic": {"target_rmsd": 0.0},
                    "atom_selection_ref": "chain A",
                    "atom_selection_target": "chain A",
                    "start_sigma": 1e30,
                }
            ],
        }
    )
    cr.setup(MockAdapter(atoms))
    rr = cr.config.rmsd_data[0]
    expected = np.array([coords[(a.resid, a.name)] for a in atoms])
    assert np.allclose(rr.calc_ref_coords, expected, atol=1e-3)


def test_rmsd_fit_calc_resolves(tmp_path):
    """Separate fit/calc selections resolve to their own atom groups."""
    rng = np.random.default_rng(9)
    n = 6
    ref = rng.standard_normal((n, 3)) * 3.0
    pdb = tmp_path / "ref.pdb"
    _write_pdb_records(pdb, [(i + 1, "CA", *ref[i]) for i in range(n)])
    atoms = [AtomRecord("A", i + 1, i, name="CA") for i in range(n)]
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(pdb),
                    "harmonic": {"target_rmsd": 0.0},
                    "atom_selection_ref_fit": "chain A and (resid 1 to 3)",
                    "atom_selection_target_fit": "chain A and (resid 1 to 3)",
                    "atom_selection_ref_calc": "chain A and (resid 4 to 6)",
                    "atom_selection_target_calc": "chain A and (resid 4 to 6)",
                    "start_sigma": 1e30,
                }
            ],
        }
    )
    cr.setup(MockAdapter(atoms))
    rr = cr.config.rmsd_data[0]
    assert rr.fit_target_sites == [0, 1, 2]
    assert rr.calc_target_sites == [3, 4, 5]
    assert rr.fit_ref_coords.shape == (3, 3) and rr.calc_ref_coords.shape == (3, 3)


def _missing_atom_cfg(pdb, strict):
    cfg = {
        "ref_pdb": str(pdb),
        "harmonic": {"target_rmsd": 0.0},
        "atom_selection_ref": "chain A",
        "atom_selection_target": "chain A",
    }
    if strict:
        cfg["best_effort"] = False
    return {"rmsd_restraints_config": [cfg]}


def test_rmsd_strict_missing_atom_raises(tmp_path):
    """best_effort:false -> a target atom with no (chain, resid, name) match raises."""
    rng = np.random.default_rng(10)
    ref = rng.standard_normal((5, 3)) * 3.0  # ref has residues 1..5
    pdb = tmp_path / "ref.pdb"
    _write_pdb_records(pdb, [(i + 1, "CA", *ref[i]) for i in range(5)])
    atoms = [AtomRecord("A", i + 1, i, name="CA") for i in range(6)]  # 1..6
    cr = CombinedRestraints.get_instance()
    cr.set_config(_missing_atom_cfg(pdb, strict=True))
    with pytest.raises(ValueError, match="no matching"):
        cr.setup(MockAdapter(atoms))


def test_rmsd_default_best_effort_skips_missing(tmp_path):
    """Default (best_effort omitted) is now tolerant: the unmatched atom is skipped."""
    rng = np.random.default_rng(10)
    ref = rng.standard_normal((5, 3)) * 3.0  # ref has residues 1..5
    pdb = tmp_path / "ref.pdb"
    _write_pdb_records(pdb, [(i + 1, "CA", *ref[i]) for i in range(5)])
    atoms = [AtomRecord("A", i + 1, i, name="CA") for i in range(6)]  # 1..6 (6 absent)
    cr = CombinedRestraints.get_instance()
    cr.set_config(_missing_atom_cfg(pdb, strict=False))
    cr.setup(MockAdapter(atoms))  # must NOT raise
    assert cr.config.rmsd_data[0].fit_target_sites == [0, 1, 2, 3, 4]


def test_rmsd_best_effort_skips_missing(tmp_path):
    """best_effort: an EXPLICIT selection tolerates a target atom missing from the ref
    (PyMOL align/super-like) -- skip it instead of raising, fitting over the overlap."""
    rng = np.random.default_rng(10)
    ref = rng.standard_normal((5, 3)) * 3.0  # ref has residues 1..5
    pdb = tmp_path / "ref.pdb"
    _write_pdb_records(pdb, [(i + 1, "CA", *ref[i]) for i in range(5)])
    atoms = [AtomRecord("A", i + 1, i, name="CA") for i in range(6)]  # 6 not in ref
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(pdb),
                    "harmonic": {"target_rmsd": 0.0},
                    "atom_selection_ref": "chain A",
                    "atom_selection_target": "chain A",
                    "best_effort": True,
                    "start_sigma": 1e30,
                }
            ],
        }
    )
    cr.setup(MockAdapter(atoms))  # must NOT raise
    rr = cr.config.rmsd_data[0]
    # resid 6 has no ref match -> skipped; 1..5 matched (target indices 0..4)
    assert rr.fit_target_sites == [0, 1, 2, 3, 4]
    assert rr.calc_target_sites == [0, 1, 2, 3, 4]
    assert rr.fit_ref_coords.shape == (5, 3)


def test_rmsd_best_effort_no_overlap_still_raises(tmp_path):
    """best_effort still raises if NOTHING overlaps (e.g. a resid-numbering offset):
    tolerance skips unmatched atoms, it does not invent a pairing."""
    rng = np.random.default_rng(11)
    ref = rng.standard_normal((3, 3)) * 3.0  # ref residues 1..3
    pdb = tmp_path / "ref.pdb"
    _write_pdb_records(pdb, [(i + 1, "CA", *ref[i]) for i in range(3)])
    atoms = [AtomRecord("A", i + 11, i, name="CA") for i in range(3)]  # disjoint
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(pdb),
                    "harmonic": {"target_rmsd": 0.0},
                    "atom_selection_ref": "chain A",
                    "atom_selection_target": "chain A",
                    "best_effort": True,
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="no target atom matched"):
        cr.setup(MockAdapter(atoms))


def _write_seq_atoms_pdb(path, seq, names, coords, chain="A"):
    """`names` atoms per residue with real 3-letter resnames (for align + name CA
    tests). coords is (len(seq) * len(names), 3), residue-major then name order."""
    lines = []
    k = 0
    for i, c in enumerate(seq):
        for nm in names:
            x, y, z = coords[k]
            k += 1
            lines.append(
                "ATOM  "
                f"{k:>5} "
                f"{nm:<4}"
                " "
                f"{_ONE_TO_THREE[c]:>3} "
                f"{chain}"
                f"{i + 1:>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                "  1.00  0.00           C\n"
            )
    path.write_text("".join(lines) + "END\n")


def _write_seq_ca_pdb(path, seq, coords, chain="A"):
    """One CA atom per residue with real 3-letter resnames (for align pairing)."""
    _write_seq_atoms_pdb(path, seq, ["CA"], coords, chain)


_ALIGN_BASE = "MKLAVDEFGHIKLMNPQRST"  # 20 residues


def test_rmsd_align_homolog_indel(tmp_path):
    """pairing='align': a homolog reference missing an internal residue maps onto the
    prediction by sequence alignment, skipping the deleted position (align-like)."""
    rng = np.random.default_rng(3)
    # target: one CA per residue, resname carried, mol_type protein
    atoms = [
        AtomRecord(
            "A", i + 1, i, name="CA", mol_type="protein", resname=_ONE_TO_THREE[c]
        )
        for i, c in enumerate(_ALIGN_BASE)
    ]
    # ref homolog: residue 11 ("I") deleted -> 19 residues
    ref_seq = _ALIGN_BASE[:10] + _ALIGN_BASE[11:]
    _write_seq_ca_pdb(tmp_path / "ref.pdb", ref_seq, rng.standard_normal((19, 3)) * 3)
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(tmp_path / "ref.pdb"),
                    "harmonic": {"target_rmsd": 0.0},
                    "pairing": "align",
                    "start_sigma": 1e30,
                }
            ],
        }
    )
    cr.setup(MockAdapter(atoms))
    rr = cr.config.rmsd_data[0]
    # target 1..10 -> ref 1..10; target 11 (deleted) -> gap; target 12..20 -> ref 11..19
    assert rr.resid_map[("A", 10)] == 10
    assert ("A", 11) not in rr.resid_map
    assert rr.resid_map[("A", 12)] == 11 and rr.resid_map[("A", 20)] == 19
    # 19 atoms paired (resid 11 skipped); its index (10) is absent
    assert len(rr.fit_target_sites) == 19
    assert 10 not in rr.fit_target_sites
    assert rr.fit_ref_coords.shape == (19, 3)


def test_rmsd_pairing_defaults_to_align(tmp_path):
    """pairing DEFAULTS to 'align' for polymers: the same homolog-deletion case maps on
    by sequence even though the config OMITS `pairing` entirely."""
    rng = np.random.default_rng(3)
    atoms = [
        AtomRecord(
            "A", i + 1, i, name="CA", mol_type="protein", resname=_ONE_TO_THREE[c]
        )
        for i, c in enumerate(_ALIGN_BASE)
    ]
    ref_seq = _ALIGN_BASE[:10] + _ALIGN_BASE[11:]  # residue 11 deleted
    _write_seq_ca_pdb(tmp_path / "ref.pdb", ref_seq, rng.standard_normal((19, 3)) * 3)
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(tmp_path / "ref.pdb"),
                    "harmonic": {"target_rmsd": 0.0},
                    # NO `pairing` key -> defaults to align (polymer present)
                    "start_sigma": 1e30,
                }
            ],
        }
    )
    cr.setup(MockAdapter(atoms))
    rr = cr.config.rmsd_data[0]
    assert rr.pairing == "align"
    assert rr.resid_map[("A", 12)] == 11  # register recovered across the deletion
    assert ("A", 11) not in rr.resid_map  # deleted residue -> gap
    assert len(rr.fit_target_sites) == 19


def test_rmsd_default_no_polymer_uses_identity(tmp_path):
    """The align DEFAULT degrades to identity when there is nothing to align: a
    ligand-only structure (no polymer type) pairs by ordinal identity and does NOT
    raise, even though `pairing` is unset (-> align)."""
    from rgi_utils.rmsd_restr_data import RmsdData

    pdb = tmp_path / "lig.pdb"
    pdb.write_text(
        _pdb_atom_line("HETATM", "B", 900, "C1", 0.0)
        + _pdb_atom_line("HETATM", "B", 900, "C2", 1.0)
        + _pdb_atom_line("HETATM", "B", 900, "C3", 2.0)
        + "END\n"
    )
    atoms = [
        AtomRecord("B", 1, 10, name="C1"),
        AtomRecord("B", 2, 11, name="C2"),
        AtomRecord("B", 3, 12, name="C3"),
    ]
    rr = RmsdData()
    rr.set_config(
        {
            "ref_pdb": str(pdb),
            "harmonic": {"target_rmsd": 0.0},
            "atom_selection_ref": "chain B",
            "atom_selection_target": "chain B",
        }
    )  # no pairing -> align, but no polymer -> identity (no crash)
    assert rr.pairing == "align"
    rr.resolve_sites(MockAdapter(atoms))  # must NOT raise
    assert rr.calc_target_sites == [10, 11, 12]


def test_rmsd_align_strict_gap_raises(tmp_path):
    """best_effort:false is HONOURED under align (no longer a silent no-op): a target
    residue aligning to a gap in a homolog ref raises instead of being skipped. Needs an
    EXPLICIT selection -- whole-structure (no selection) RMSD is always best-effort."""
    rng = np.random.default_rng(3)
    atoms = [
        AtomRecord(
            "A", i + 1, i, name="CA", mol_type="protein", resname=_ONE_TO_THREE[c]
        )
        for i, c in enumerate(_ALIGN_BASE)
    ]
    ref_seq = _ALIGN_BASE[:10] + _ALIGN_BASE[11:]  # residue 11 deleted -> a gap
    _write_seq_ca_pdb(tmp_path / "ref.pdb", ref_seq, rng.standard_normal((19, 3)) * 3)
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(tmp_path / "ref.pdb"),
                    "harmonic": {"target_rmsd": 0.0},
                    "best_effort": False,  # strict: the deleted residue must raise
                    "atom_selection_target": "name CA",  # explicit -> strict applies
                    "atom_selection_ref": "name CA",
                    "start_sigma": 1e30,
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="aligned to a gap"):
        cr.setup(MockAdapter(atoms))


def test_rmsd_align_requires_target_resname(tmp_path):
    """align needs residue names on the target; an unplumbed adapter errors loudly."""
    rng = np.random.default_rng(4)
    atoms = [  # mol_type protein but NO resname -> align cannot build the sequence
        AtomRecord("A", i + 1, i, name="CA", mol_type="protein")
        for i in range(len(_ALIGN_BASE))
    ]
    _write_seq_ca_pdb(tmp_path / "ref.pdb", _ALIGN_BASE, rng.standard_normal((20, 3)))
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(tmp_path / "ref.pdb"),
                    "harmonic": {"target_rmsd": 0.0},
                    "pairing": "align",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="needs residue names on the target"):
        cr.setup(MockAdapter(atoms))


def test_rmsd_align_derives_polymer_from_resname(tmp_path):
    """Adapters that don't set mol_type (protenix/of3/chai) still align: the polymer
    type is derived from the residue name, so only resname needs plumbing."""
    rng = np.random.default_rng(7)
    # target like protenix: mol_type LEFT UNSET, but resname present
    atoms = [
        AtomRecord("A", i + 1, i, name="CA", resname=_ONE_TO_THREE[c])
        for i, c in enumerate(_ALIGN_BASE)
    ]
    assert all(a.mol_type is None for a in atoms)  # mimics an un-typed adapter
    ref_seq = _ALIGN_BASE[:10] + _ALIGN_BASE[11:]  # residue 11 deleted
    _write_seq_ca_pdb(tmp_path / "ref.pdb", ref_seq, rng.standard_normal((19, 3)))
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(tmp_path / "ref.pdb"),
                    "harmonic": {"target_rmsd": 0.0},
                    "pairing": "align",
                    "start_sigma": 1e30,
                }
            ],
        }
    )
    cr.setup(MockAdapter(atoms))
    rr = cr.config.rmsd_data[0]
    assert rr.resid_map[("A", 12)] == 11  # register recovered across the deletion
    assert len(rr.fit_target_sites) == 19


def test_rmsd_align_name_ca_only_excludes_side_chain(tmp_path):
    """pairing='align' + 'name CA': only CA atoms enter the superposition, so a
    substituted homolog residue's side chain is NOT pinned onto the reference -- the
    backbone-only escape from the side-chain-pinning limitation. The reference DOES
    contain CB, so a broken name filter would pair it (this catches a silent
    regression where 'name CA' stops filtering)."""
    rng = np.random.default_rng(12)
    # target: CA + CB per residue (side chain present), resname + mol_type carried
    atoms, ca_indices, idx = [], [], 0
    for i, c in enumerate(_ALIGN_BASE):
        atoms.append(
            AtomRecord(
                "A",
                i + 1,
                idx,
                name="CA",
                mol_type="protein",
                resname=_ONE_TO_THREE[c],
            )
        )
        ca_indices.append(idx)
        idx += 1
        atoms.append(
            AtomRecord(
                "A",
                i + 1,
                idx,
                name="CB",
                mol_type="protein",
                resname=_ONE_TO_THREE[c],
            )
        )
        idx += 1
    # ref homolog: residue 1 substituted (M->A, different side chain), CA+CB each
    ref_seq = "A" + _ALIGN_BASE[1:]
    _write_seq_atoms_pdb(
        tmp_path / "ref.pdb",
        ref_seq,
        ["CA", "CB"],
        rng.standard_normal((len(ref_seq) * 2, 3)) * 3,
    )
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(tmp_path / "ref.pdb"),
                    "harmonic": {"target_rmsd": 0.0},
                    "pairing": "align",
                    "atom_selection_target_fit": "name CA",
                    "atom_selection_ref_fit": "name CA",
                    "atom_selection_target_calc": "name CA",
                    "atom_selection_ref_calc": "name CA",
                    "start_sigma": 1e30,
                }
            ],
        }
    )
    cr.setup(MockAdapter(atoms))
    rr = cr.config.rmsd_data[0]
    # only CA atoms entered fit/calc -- CB excluded though it EXISTS in the ref
    assert rr.fit_target_sites == ca_indices
    assert rr.calc_target_sites == ca_indices
    assert rr.fit_ref_coords.shape == (len(_ALIGN_BASE), 3)


def test_rmsd_backbone_selection_resname_gated(tmp_path):
    """`backbone` resolves in an RMSD selection via the resname-gated polymer path:
    the target carries resname but NO mol_type (the chai/of3/protenix case), so the
    rmsd candidate dict's `resname` is what fires the polymer gate. Only CA enters
    fit/calc though target+ref both also carry CB. Pinned to pairing='identity' to
    test the selection independently of the (now default) align pairing."""
    seq = "ACDEFG"
    atoms, ca_indices, idx = [], [], 0
    for i, c in enumerate(seq):
        atoms.append(AtomRecord("A", i + 1, idx, name="CA", resname=_ONE_TO_THREE[c]))
        ca_indices.append(idx)
        idx += 1
        atoms.append(AtomRecord("A", i + 1, idx, name="CB", resname=_ONE_TO_THREE[c]))
        idx += 1
    _write_seq_atoms_pdb(
        tmp_path / "ref.pdb", seq, ["CA", "CB"], np.zeros((len(seq) * 2, 3))
    )
    cr = CombinedRestraints.get_instance()
    cr.set_config(
        {
            "rmsd_restraints_config": [
                {
                    "ref_pdb": str(tmp_path / "ref.pdb"),
                    "harmonic": {"target_rmsd": 0.0},
                    "pairing": "identity",
                    # the four independent fit/calc keys (no bare `atom_selection`)
                    "atom_selection_target_fit": "backbone",
                    "atom_selection_ref_fit": "backbone",
                    "atom_selection_target_calc": "backbone",
                    "atom_selection_ref_calc": "backbone",
                    "start_sigma": 1e30,
                }
            ],
        }
    )
    cr.setup(MockAdapter(atoms))
    rr = cr.config.rmsd_data[0]
    # backbone-only: CA in, CB out, gated purely by resname (target has no mol_type)
    assert rr.fit_target_sites == ca_indices
    assert rr.calc_target_sites == ca_indices


def _pdb_atom_line(rec, chain, resseq, name, x=0.0, y=0.0, z=0.0):
    """One ATOM/HETATM line in the fixed columns read_pdb_atoms parses."""
    return (
        f"{rec:<6}"
        f"{1:>5} "
        f"{name:<4}"
        " "
        f"{'LIG':>3} "
        f"{chain}"
        f"{resseq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        "  1.00  0.00          "
        f"{(name[0] if name else 'C'):>2}\n"
    )


def test_pdb_ref_hetatm_per_atom_ordinal(tmp_path):
    """HETATM atoms get a per-atom ordinal (matching the adapters' one-token-per-atom
    ligand convention); ATOM atoms in one residue still share a single ordinal."""
    from rgi_utils.pdb_ref import read_pdb_atoms

    pdb = tmp_path / "mix.pdb"
    pdb.write_text(
        _pdb_atom_line("ATOM  ", "A", 5, "N")
        + _pdb_atom_line("ATOM  ", "A", 5, "CA")  # same residue -> same ordinal
        + _pdb_atom_line("HETATM", "B", 900, "C1", 0.0)
        + _pdb_atom_line("HETATM", "B", 900, "C2", 1.0)  # same resSeq, own ordinal
        + _pdb_atom_line("HETATM", "B", 900, "C3", 2.0)
        + "END\n"
    )
    by = {(a.chain, a.name): a.resid for a in read_pdb_atoms(str(pdb))}
    assert by[("A", "N")] == 1 and by[("A", "CA")] == 1
    assert by[("B", "C1")] == 1 and by[("B", "C2")] == 2 and by[("B", "C3")] == 3


def test_rmsd_ligand_identity_pairing(tmp_path):
    """A ligand reference (HETATM, single resSeq) identity-pairs with an adapter that
    gives each ligand atom its own per-atom ordinal. Regression: pdb_ref used to give
    every ligand atom one ordinal -> the (chain, resid, name) key never matched."""
    from rgi_utils.rmsd_restr_data import RmsdData

    pdb = tmp_path / "lig.pdb"
    pdb.write_text(
        _pdb_atom_line("HETATM", "B", 900, "C1", 0.0)
        + _pdb_atom_line("HETATM", "B", 900, "C2", 1.0)
        + _pdb_atom_line("HETATM", "B", 900, "C3", 2.0)
        + "END\n"
    )
    atoms = [
        AtomRecord("B", 1, 10, name="C1"),
        AtomRecord("B", 2, 11, name="C2"),
        AtomRecord("B", 3, 12, name="C3"),
    ]
    rr = RmsdData()
    rr.set_config(
        {
            "ref_pdb": str(pdb),
            "harmonic": {"target_rmsd": 0.0},
            "atom_selection_ref": "chain B",
            "atom_selection_target": "chain B",
        }
    )
    rr.resolve_sites(MockAdapter(atoms))
    assert rr.calc_target_sites == [10, 11, 12]
    assert np.allclose(rr.calc_ref_coords, [[0, 0, 0], [1, 0, 0], [2, 0, 0]], atol=1e-3)


def test_rmsd_duplicate_ref_key_raises():
    """Duplicate (chain, resid, name) reference atoms must raise rather than silently
    collapse last-wins (e.g. an altloc collision within one polymer residue)."""
    from rgi_utils.pdb_ref import PdbAtom
    from rgi_utils.rmsd_restr_data import RmsdData

    rr = RmsdData()
    rr.ref_pdb = "dup.pdb"
    tgt = [AtomRecord("A", 1, 0, name="CA")]
    ref = [
        PdbAtom("A", 1, 0, "CA", "C", 0.0, 0.0, 0.0),
        PdbAtom("A", 1, 1, "CA", "C", 9.0, 9.0, 9.0),  # same (chain, resid, name)
    ]
    with pytest.raises(ValueError, match="duplicate reference atom"):
        rr._pair(tgt, ref, "chain A", "chain A", "calc")


def test_rmsd_weight_zero_preserved_and_default():
    """weight: 0 stays 0 (a no-op restraint); an omitted weight defaults to 1.0.
    The old `or 1.0` truthiness coerced an explicit 0 to full weight."""
    from rgi_utils.rmsd_restr_data import RmsdData

    base = {
        "ref_pdb": "x.pdb",
        "harmonic": {"target_rmsd": 1.0},
        "atom_selection_ref": "chain A",
        "atom_selection_target": "chain A",
    }
    rr0 = RmsdData()
    rr0.set_config({**base, "weight": 0})
    assert rr0.weight == 0.0
    rr1 = RmsdData()
    rr1.set_config(base)  # omitted -> default 1.0
    assert rr1.weight == 1.0


def test_boltz_adapter_decodes_atom_names_from_ref_chars():
    """boltz adapter must source atom names from ref_atom_name_chars (one-hot,
    ord(c)-32 codes), not the nonexistent record[0].atoms. Regression for the silent
    name=None that degraded RMSD identity pairing to selection-order pairing."""
    torch = pytest.importorskip("torch")
    from rgi_utils.boltz.adapter import BoltzFeatsAdapter

    names = ["N", "CA", "C", "O", "CB"]
    n_pad = 3
    arr = torch.zeros((1, len(names) + n_pad, 4, 64))  # (batch, n_atom, 4, vocab)
    for i, nm in enumerate(names):
        for c, ch in enumerate(nm):
            arr[0, i, c, ord(ch) - 32] = 1.0
    name_of = BoltzFeatsAdapter({"ref_atom_name_chars": arr})._atom_name_lookup()
    assert [name_of(i) for i in range(len(names))] == names
    assert name_of(len(names)) is None  # all-zero padding row -> None
    assert name_of(10_000) is None  # out of range -> None
    # missing field -> all None (graceful fall back to order pairing), no crash
    assert BoltzFeatsAdapter({})._atom_name_lookup()(0) is None


def test_rmsd_no_selection_whole_structure(tmp_path):
    """No atom_selection -> fit + RMSD over the WHOLE structure (every atom paired to the
    reference by identity); only ref_pdb + target_rmsd are required."""
    from rgi_utils.rmsd_restr_data import RmsdData

    rng = np.random.default_rng(11)
    n = 8  # 4 residues x (N, CA)
    spec = [(r, nm) for r in range(1, 5) for nm in ("N", "CA")]
    xyz = rng.standard_normal((n, 3)) * 3.0
    atoms = [AtomRecord("A", spec[i][0], 100 + i, name=spec[i][1]) for i in range(n)]
    pdb = tmp_path / "ref.pdb"
    _write_pdb_records(pdb, [(spec[i][0], spec[i][1], *xyz[i]) for i in range(n)])

    rr = RmsdData()
    rr.set_config(
        {"ref_pdb": str(pdb), "harmonic": {"target_rmsd": 0.0}}
    )  # NO selection
    assert rr.is_valid() and rr.sel_target_fit is None
    rr.resolve_sites(MockAdapter(atoms))
    assert rr.fit_target_sites == [a.index for a in atoms]  # the whole structure
    assert rr.calc_target_sites == [a.index for a in atoms]
    assert rr.fit_ref_coords.shape == (n, 3)


def test_rmsd_no_selection_best_effort_skips_unmatched(tmp_path):
    """No selection + a reference that lacks some structure atoms (e.g. no hydrogens):
    best-effort fits/measures over the matched subset instead of raising."""
    from rgi_utils.rmsd_restr_data import RmsdData

    rng = np.random.default_rng(12)
    atoms, recs = [], []
    idx = 0
    for r in range(1, 5):  # 4 residues x (N, CA, H); the ref omits every H
        for nm in ("N", "CA", "H"):
            v = rng.standard_normal(3) * 3.0
            atoms.append(AtomRecord("A", r, idx, name=nm))
            if nm != "H":
                recs.append((r, nm, *v))
            idx += 1
    pdb = tmp_path / "ref.pdb"
    _write_pdb_records(pdb, recs)

    rr = RmsdData()
    rr.set_config({"ref_pdb": str(pdb), "harmonic": {"target_rmsd": 0.0}})
    rr.resolve_sites(MockAdapter(atoms))  # must NOT raise (4 H's skipped)
    matched = {a.index for a in atoms if a.name != "H"}
    assert set(rr.fit_target_sites) == matched and len(rr.fit_target_sites) == 8
    assert set(rr.calc_target_sites) == matched


def test_rmsd_requires_ref_pdb_and_target_only():
    """Selections are optional now, but ref_pdb + target_rmsd are still required."""
    from rgi_utils.rmsd_restr_data import RmsdData

    with pytest.raises(ValueError, match="ref_pdb and a restraint-type"):
        RmsdData().set_config({"harmonic": {"target_rmsd": 0.0}})  # missing ref_pdb
    with pytest.raises(ValueError, match="ref_pdb and a restraint-type"):
        RmsdData().set_config({"ref_pdb": "x.pdb"})  # missing target_rmsd
    rr = RmsdData()
    rr.set_config(
        {"ref_pdb": "x.pdb", "harmonic": {"target_rmsd": 1.0}}
    )  # no selection -> valid
    assert rr.is_valid() and rr.sel_target_calc is None
