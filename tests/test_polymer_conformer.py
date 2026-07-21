from __future__ import annotations

import numpy as np
import pytest

from rgi_utils.atom_context import AtomRecord
from rgi_utils.combined import CombinedRestraints
from rgi_utils.config import RestraintsConfig


class _PolymerAdapter:
    def __init__(
        self,
        molecule_type: str,
        names: list[str],
        coords: np.ndarray,
        chain_flags: tuple[tuple[str, bool], ...] = (("A", True),),
    ):
        self._type = molecule_type
        self._names = names
        self._one = np.asarray(coords, dtype=np.float64)
        self._chain_flags = chain_flags
        self._positions = np.concatenate(
            [self._one, self._one] * len(chain_flags), axis=0
        )
        self._elements = np.array(
            [
                7
                if name == "N"
                else 8
                if name.startswith("O")
                else 15
                if name == "P"
                else 6
                for name in names
            ]
            * (2 * len(chain_flags)),
            dtype=np.int64,
        )
        self._uid = np.repeat(np.arange(2 * len(chain_flags)), len(names))

    def iter_atoms(self):
        for chain_index, (chain, enabled) in enumerate(self._chain_flags):
            for resid in (1, 2):
                offset = (2 * chain_index + resid - 1) * len(self._names)
                for local, name in enumerate(self._names):
                    yield AtomRecord(
                        chain=chain,
                        resid=resid,
                        index=offset + local,
                        name=name,
                        mol_type=self._type,
                        resname="ALA" if self._type == "protein" else "A",
                        conformer_restraints=enabled,
                    )

    def get_elements(self):
        return self._elements

    def get_reference_positions(self):
        return self._positions

    def get_reference_space_uid(self):
        return self._uid


class _AtomTokenizedPolymerAdapter(_PolymerAdapter):
    def iter_atoms(self):
        for residue_index in (0, 1):
            offset = residue_index * len(self._names)
            for local, name in enumerate(self._names):
                yield AtomRecord(
                    chain="A",
                    resid=offset + local + 1,
                    index=offset + local,
                    name=name,
                    mol_type=self._type,
                    resname="MSE",
                    conformer_restraints=True,
                )


_ALA_NAMES = ["N", "CA", "C", "O", "CB"]
_ALA_COORDS = np.array(
    [
        [-1.20, 0.60, 0.00],
        [0.00, 0.00, 0.00],
        [1.50, 0.10, 0.00],
        [2.10, 1.10, 0.00],
        [-0.10, -0.80, 1.20],
    ]
)


def _config():
    return {
        "gpu": False,
        "max_iter": 100,
        "conformer_restraints_config": {
            "bond": {},
            "angle": {},
            "chiral": {},
            "vdw": {"max_neighbors": 8},
        },
    }


def test_protein_builds_peptide_link_chiral_and_vdw_exclusions():
    restr = CombinedRestraints()
    restr.setup(_PolymerAdapter("protein", _ALA_NAMES, _ALA_COORDS), config=_config())
    spec = restr.spec

    assert spec.bond is not None
    peptide = np.where(np.all(spec.bond.idx == np.array([2, 5]), axis=1))[0]
    assert len(peptide) == 1
    assert spec.bond.r0[peptide[0]] == pytest.approx(1.329)
    assert spec.angle is not None
    assert any(np.array_equal(row, [1, 2, 5]) for row in spec.angle.idx)
    assert any(np.array_equal(row, [2, 5, 6]) for row in spec.angle.idx)
    assert spec.chiral is not None and int(spec.chiral.mask.sum()) >= 4
    peptide_planes = np.where(spec.chiral.vol0 == 0.0)[0]
    assert len(peptide_planes) >= 2

    av = spec.active_vdw_config
    assert av is not None
    assert av.polymer_mask.all()
    assert spec.conf_start_sigma == float("inf")
    # Peptide C-N and the CA-C-N 1-3 pair must never receive VdW repulsion.
    assert 2 * spec.n_active + 5 in set(av.excluded_codes.tolist())
    assert 1 * spec.n_active + 5 in set(av.excluded_codes.tolist())


def test_reference_uid_groups_atom_tokenized_modified_residues():
    restr = CombinedRestraints()
    restr.setup(
        _AtomTokenizedPolymerAdapter("protein", _ALA_NAMES, _ALA_COORDS),
        config=_config(),
    )
    assert any(np.array_equal(row, [2, 5]) for row in restr.spec.bond.idx)


def test_phosphodiester_link_targets_are_present():
    names = ["P", "O5'", "C5'", "C3'", "O3'"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
            [2.8, 0.5, 0.0],
            [5.0, 1.0, 0.0],
            [6.2, 1.4, 0.0],
        ]
    )
    restr = CombinedRestraints()
    restr.setup(_PolymerAdapter("rna", names, coords), config=_config())
    spec = restr.spec

    link = np.where(np.all(spec.bond.idx == np.array([4, 5]), axis=1))[0]
    assert len(link) == 1
    assert spec.bond.r0[link[0]] == pytest.approx(1.607)
    assert any(np.array_equal(row, [3, 4, 5]) for row in spec.angle.idx)
    assert any(np.array_equal(row, [4, 5, 6]) for row in spec.angle.idx)
    assert not any(np.array_equal(row, [0, 6, 7]) for row in spec.angle.idx)


def test_polymer_restraint_repairs_peptide_link_at_high_sigma():
    torch = pytest.importorskip("torch")
    restr = CombinedRestraints()
    restr.setup(_PolymerAdapter("protein", _ALA_NAMES, _ALA_COORDS), config=_config())
    coords = np.concatenate([_ALA_COORDS, _ALA_COORDS + np.array([5.0, 0.0, 0.0])])
    coords = torch.tensor(coords, dtype=torch.float64)
    before = abs(float(torch.linalg.vector_norm(coords[2] - coords[5])) - 1.329)
    restr.minimize(coords, istep=0, sigma=100.0)
    after = abs(float(torch.linalg.vector_norm(coords[2] - coords[5])) - 1.329)
    assert after < before
    assert after < 0.05


def test_active_vdw_exclusion_gradient_and_backend_parity():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from rgi_utils.optim._torch_cg_gpu import (
        active_vdw_pair_energy,
        build_active_vdw_pairs,
    )
    from rgi_utils.optim.jax_optim import (
        _active_vdw_pair_energy,
        _build_active_vdw_pairs,
    )

    coords_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [8.0, 0.0, 0.0]]
    )
    radii_np = np.full(4, 1.7)
    polymer_np = np.array([True, False, False, False])
    # Exclude covalent 0-1. The only remaining contact involving polymer is 0-2.
    excluded_np = np.array([1], dtype=np.int64)

    coords_t = torch.tensor(coords_np, dtype=torch.float64, requires_grad=True)
    radii_t = torch.tensor(radii_np, dtype=torch.float64)
    polymer_t = torch.tensor(polymer_np)
    excluded_t = torch.tensor(excluded_np)
    neighbours_t, factor_t = build_active_vdw_pairs(
        coords_t, radii_t, polymer_t, excluded_t, torch.tensor(5.0), 3
    )
    energy_t = active_vdw_pair_energy(
        coords_t, neighbours_t, factor_t, radii_t, torch.tensor(0.75), torch.tensor(1.0)
    )
    energy_t.backward()
    assert float(energy_t.detach()) == pytest.approx((0.5 - 0.75 * 3.4) ** 2)
    assert float(coords_t.grad[2, 0]) < 0.0

    coords_j = jnp.asarray(coords_np)
    neighbours_j, factor_j = _build_active_vdw_pairs(
        coords_j,
        jnp.asarray(radii_np),
        jnp.asarray(polymer_np),
        jnp.asarray(excluded_np, dtype=jnp.int32),
        jnp.asarray(5.0),
        3,
    )

    def energy_j(x):
        return _active_vdw_pair_energy(
            x,
            neighbours_j,
            factor_j,
            jnp.asarray(radii_np),
            jnp.asarray(0.75),
            jnp.asarray(1.0),
        )

    value_j, grad_j = jax.value_and_grad(energy_j)(coords_j)
    assert float(value_j) == pytest.approx(float(energy_t.detach()), rel=1e-6)
    np.testing.assert_allclose(np.asarray(grad_j), coords_t.grad.numpy(), atol=1e-5)


def test_polymer_restraint_runs_in_jitted_jax_minimizer():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    restr = CombinedRestraints()
    restr.setup(_PolymerAdapter("protein", _ALA_NAMES, _ALA_COORDS), config=_config())
    coords = np.concatenate([_ALA_COORDS, _ALA_COORDS + np.array([5.0, 0.0, 0.0])])
    before = abs(float(np.linalg.norm(coords[2] - coords[5])) - 1.329)
    minimize = jax.jit(restr.get_minimizer())
    out = minimize(jnp.asarray(coords, dtype=jnp.float32), 100.0, 0)
    after = abs(float(jnp.linalg.norm(out[2] - out[5])) - 1.329)
    assert bool(jnp.all(jnp.isfinite(out)))
    assert after < before


@pytest.mark.gpu
def test_polymer_active_vdw_runs_through_compiled_gpu_cg():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = _config()
    config["gpu"] = True
    restr = CombinedRestraints()
    restr.setup(_PolymerAdapter("protein", _ALA_NAMES, _ALA_COORDS), config=config)
    coords = np.concatenate([_ALA_COORDS, _ALA_COORDS + np.array([5.0, 0.0, 0.0])])
    coords = torch.tensor(coords, dtype=torch.float32, device="cuda")
    before = abs(float(torch.linalg.vector_norm(coords[2] - coords[5])) - 1.329)
    restr.minimize(coords, istep=0, sigma=100.0)
    after = abs(float(torch.linalg.vector_norm(coords[2] - coords[5])) - 1.329)
    assert bool(torch.isfinite(coords).all())
    assert after < before


def test_unknown_conformer_key_is_rejected_during_config_parse():
    with pytest.raises(ValueError, match="unknown key"):
        RestraintsConfig.from_dict({"conformer_restraints_config": {"not_a_term": {}}})


def test_active_vdw_int32_guard_rejects_oversized_polymer():
    # The JAX active-active VdW encodes pairs as min(i,j)*n_active+max(i,j) in int32.
    # 46340**2 still fits (~2.147e9 < 2**31-1); 46341**2 overflows -> loud failure so the
    # covalent-pair exclusion cannot be silently corrupted (torch uses int64, unaffected).
    from rgi_utils.spec import check_active_vdw_int32_safe

    check_active_vdw_int32_safe(1)
    check_active_vdw_int32_safe(46340)
    with pytest.raises(ValueError, match="int32"):
        check_active_vdw_int32_safe(46341)


def test_only_explicitly_enabled_polymer_chain_is_restrained():
    adapter = _PolymerAdapter(
        "protein",
        _ALA_NAMES,
        _ALA_COORDS,
        chain_flags=(("A", True), ("B", False)),
    )
    restr = CombinedRestraints()
    restr.setup(adapter, config=_config())

    assert restr.spec.active_sites.tolist() == list(range(2 * len(_ALA_NAMES)))
    assert any(np.array_equal(row, [2, 5]) for row in restr.spec.bond.idx)


def test_polymer_chain_defaults_to_unrestrained():
    adapter = _PolymerAdapter(
        "protein",
        _ALA_NAMES,
        _ALA_COORDS,
        chain_flags=(("A", False),),
    )
    restr = CombinedRestraints()
    restr.setup(adapter, config=_config())

    assert not restr.spec.is_active()
