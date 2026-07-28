"""Standalone best-fit-plane restraint (``plane_restraints_config``): config parse, site
resolution, spec wiring.

Mirrors ``test_group_geom_data.py`` (the four restraint types, ``move``, the gate windows)
with the plane-specific differences that are easy to regress:

  * targets are ANGSTROM (no ``unit`` key), unlike the angular group restraints;
  * the restraint-type block is OPTIONAL and defaults to ``harmonic`` toward 0, because a
    plane's target is essentially always 0 (this is also what lets the base-pair macro
    express ``coplanar_slack: 0``);
  * ``move`` defaults to EVERY group free (a plane has no anchor group to pin);
  * several groups in one entry are POOLED into a single plane, so the spec's ``free`` mask
    is per-ATOM, not per-group;
  * a ``refN and ...`` entry is a different energy and is routed to ``RefGeomData`` instead.
"""

import numpy as np
import pytest

from rgi_utils.atom_context import AtomRecord
from rgi_utils.config import RestraintsConfig
from rgi_utils.plane_restr_data import PlaneRestraintData


class MockAdapter:
    def __init__(self, atoms: list[AtomRecord]):
        self._atoms = atoms

    def iter_atoms(self):
        yield from self._atoms


_SEL1 = {"atom_selection1": "chain A"}
_SEL2 = {**_SEL1, "atom_selection2": "chain B"}


def _plane(extra: dict, base: dict | None = None) -> PlaneRestraintData:
    pr = PlaneRestraintData()
    pr.set_config({**(base or _SEL1), **extra})
    return pr


def _adapter(*chains) -> MockAdapter:
    """One atom per (chain, name) pair; index is the running position."""
    atoms, i = [], 0
    for chain, n_atoms in chains:
        for k in range(n_atoms):
            atoms.append(AtomRecord(chain, 1, i, f"C{k + 1}", "protein", "ALA"))
            i += 1
    return MockAdapter(atoms)


class TestTypes:
    def test_type_block_omitted_defaults_to_harmonic_zero(self):
        pr = _plane({})
        assert pr.geom_type == "harmonic"
        assert pr.target1 == 0.0 and pr.target2 == 0.0
        assert pr.run_restr is True

    def test_harmonic_angstrom_not_converted(self):
        # unlike angle/dihedral there is no degrees->radians conversion
        pr = _plane({"harmonic": {"target_plane": 0.25}})
        assert pr.target1 == pytest.approx(0.25)

    def test_flat_bottomed2_is_the_slack_form(self):
        pr = _plane({"flat-bottomed2": {"target_plane2": 0.1}})
        assert pr.geom_type == "flat-bottomed2"
        assert pr.target2 == pytest.approx(0.1)

    def test_flat_bottomed1_lower_bound_accepted(self):
        # "stay at least this far from planar" — odd but part of the shared four types
        pr = _plane({"flat-bottomed1": {"target_plane1": 0.5}})
        assert pr.geom_type == "flat-bottomed1"
        assert pr.target1 == pytest.approx(0.5)

    def test_flat_bottomed_window(self):
        pr = _plane({"flat-bottomed": {"target_plane1": 0.1, "target_plane2": 0.3}})
        assert pr.geom_type == "flat-bottomed"
        assert (pr.target1, pr.target2) == pytest.approx((0.1, 0.3))

    def test_flat_bottomed_requires_t1_lt_t2(self):
        with pytest.raises(ValueError, match="must be smaller"):
            _plane({"flat-bottomed": {"target_plane1": 0.3, "target_plane2": 0.1}})

    def test_harmonic_missing_target_raises(self):
        with pytest.raises(ValueError, match="harmonic needs target_plane"):
            _plane({"harmonic": {}})

    def test_unit_key_is_not_recognised(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            _plane({"unit": "radians"})
        assert any("unknown config key" in r.message for r in caplog.records)


class TestSelections:
    def test_missing_selection1_raises(self):
        pr = PlaneRestraintData()
        with pytest.raises(ValueError, match="atom_selection1 is required"):
            pr.set_config({"weight": 2.0})

    def test_non_contiguous_selections_raise(self):
        # a gap would silently drop a group and shift what `move` indices name
        pr = PlaneRestraintData()
        with pytest.raises(ValueError, match="contiguously"):
            pr.set_config({"atom_selection1": "chain A", "atom_selection3": "chain C"})

    def test_up_to_four_groups(self):
        pr = PlaneRestraintData()
        pr.set_config(
            {f"atom_selection{i}": f"chain {c}" for i, c in enumerate("ABCD", start=1)}
        )
        assert len(pr.atom_selections) == 4
        assert pr.move_free == (True, True, True, True)


class TestMove:
    def test_default_is_every_group_free(self):
        pr = _plane({}, base=_SEL2)
        assert pr.move_free == (True, True)

    def test_single_index_pins_the_rest(self):
        pr = _plane({"move": 1}, base=_SEL2)
        assert pr.move_free == (True, False)

    def test_list_and_all(self):
        assert _plane({"move": [2]}, base=_SEL2).move_free == (False, True)
        assert _plane({"move": "all"}, base=_SEL2).move_free == (True, True)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="within 1..2"):
            _plane({"move": 3}, base=_SEL2)


class TestWindows:
    def test_defaults(self):
        pr = _plane({})
        assert pr.start_sigma is None  # from_dict turns this into +inf
        assert pr.stop_sigma == -1.0
        assert pr.start_step == float("-inf") and pr.stop_step == float("inf")

    def test_sigma_and_step_windows_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _plane({"start_sigma": 2.0, "start_step": 5})

    def test_weight_zero_is_kept(self):
        assert _plane({"weight": 0}).weight == 0.0


class TestResolve:
    def test_pools_groups_and_logs_sizes(self):
        pr = _plane({}, base=_SEL2)
        pr.resolve_sites(_adapter(("A", 4), ("B", 3)))
        assert [len(g) for g in pr.target_sites] == [4, 3]

    def test_empty_group_raises(self):
        pr = _plane({}, base=_SEL2)
        with pytest.raises(ValueError, match="group 2 selection matched no atoms"):
            pr.resolve_sites(_adapter(("A", 4)))

    def test_fewer_than_three_atoms_raises(self):
        pr = _plane({})
        with pytest.raises(ValueError, match="best-fit plane needs at least 3"):
            pr.resolve_sites(_adapter(("A", 2)))


class TestConfigWiring:
    def test_from_dict_populates_plane_data_and_default_start_sigma(self):
        cfg = RestraintsConfig.from_dict(
            {"plane_restraints_config": [{**_SEL1, "weight": 2.0}]}
        )
        assert len(cfg.plane_data) == 1
        assert cfg.plane_data[0].start_sigma == float("inf")  # None -> +inf
        assert cfg.plane_data[0].weight == 2.0

    def test_ref_anchored_entry_goes_to_custom_data(self):
        cfg = RestraintsConfig.from_dict(
            {
                "plane_restraints_config": [
                    {
                        "atom_selection1": "chain A",
                        "atom_selection2": "ref1 and chain A",
                        "refs": {"ref1": {"ref_pdb": "x.pdb", "pairing": "identity"}},
                    }
                ]
            }
        )
        assert cfg.plane_data == []
        assert len(cfg.custom_data) == 1
        rg = cfg.custom_data[0]
        assert rg.geom == "plane" and rg.n_groups == 2
        # plane's type block is optional on the ref path too
        assert rg.geom_type == "harmonic" and rg.target1 == 0.0

    def test_ref_anchored_all_reference_groups_raise(self):
        # both groups reference the SAME ref, so the ref-count cap (n_groups - 1 = 1) is
        # satisfied and the "needs a prediction group" guard is what fires
        with pytest.raises(
            ValueError, match="at least one group must select prediction"
        ):
            RestraintsConfig.from_dict(
                {
                    "plane_restraints_config": [
                        {
                            "atom_selection1": "ref1 and chain A",
                            "atom_selection2": "ref1 and chain B",
                            "refs": {"ref1": {"ref_pdb": "x.pdb"}},
                        }
                    ]
                }
            )

    def test_ref_anchored_single_group_cannot_be_the_reference(self):
        # a 1-group entry has no room for a reference at all (max_refs = n_groups - 1 = 0)
        with pytest.raises(ValueError, match="at most 0 distinct reference"):
            RestraintsConfig.from_dict(
                {
                    "plane_restraints_config": [
                        {
                            "atom_selection1": "ref1 and chain A",
                            "refs": {"ref1": {"ref_pdb": "x.pdb"}},
                        }
                    ]
                }
            )

    def test_ref_anchored_move_cannot_select_a_reference(self):
        with pytest.raises(ValueError, match="move selects reference group"):
            RestraintsConfig.from_dict(
                {
                    "plane_restraints_config": [
                        {
                            "atom_selection1": "chain A",
                            "atom_selection2": "ref1 and chain A",
                            "move": 2,
                            "refs": {"ref1": {"ref_pdb": "x.pdb"}},
                        }
                    ]
                }
            )


class TestSpecArrays:
    def _spec(self, config):
        from rgi_utils.combined import CombinedRestraints

        cr = CombinedRestraints()
        cr.setup(_adapter(("A", 4), ("B", 3)), 1, {"gpu": False, **config})
        return cr.spec

    def test_pooled_padding_and_local_indices(self):
        spec = self._spec(
            {
                "plane_restraints_config": [
                    {**_SEL2},  # 7 atoms pooled
                    {"atom_selection1": "chain A"},  # 4 atoms -> padded to 7
                ]
            }
        )
        gp = spec.group_plane
        assert gp.idx.shape == (2, 7)
        assert gp.grp_mask[0].sum() == 7 and gp.grp_mask[1].sum() == 4
        # padding columns hold local index 0 (a valid atom), neutralised by grp_mask
        assert gp.idx[1, 4:].tolist() == [0, 0, 0]
        # local indices into active_sites, which is every referenced atom here
        assert spec.n_active == 7
        assert sorted(gp.idx[0].tolist()) == list(range(7))

    def test_move_expands_to_a_per_atom_free_mask(self):
        spec = self._spec({"plane_restraints_config": [{**_SEL2, "move": 1}]})
        gp = spec.group_plane
        # chain A (4 atoms) free, chain B (3) pinned — per ATOM, not per group
        assert gp.free[0].tolist() == [1.0] * 4 + [0.0] * 3

    def test_has_group_plane_drives_is_active(self):
        spec = self._spec({"plane_restraints_config": [{**_SEL1}]})
        assert spec.has_group_plane() and spec.is_active()
        # the conformer plane term stays empty: this is a separate term
        assert spec.plane is None
        assert not spec.has_conformer()

    def test_max_start_sigma_includes_plane(self):
        spec = self._spec({"plane_restraints_config": [{**_SEL1, "start_sigma": 3.5}]})
        assert spec.max_start_sigma() == pytest.approx(3.5)

    def test_empty_sigma_window_raises(self):
        with pytest.raises(ValueError, match="active window is EMPTY"):
            self._spec(
                {
                    "plane_restraints_config": [
                        {**_SEL1, "start_sigma": 1.0, "stop_sigma": 2.0}
                    ]
                }
            )

    def test_geom_type_codes_reach_the_spec(self):
        spec = self._spec(
            {
                "plane_restraints_config": [
                    {**_SEL1, "flat-bottomed2": {"target_plane2": 0.1}}
                ]
            }
        )
        from rgi_utils.spec import DIST_TYPE_CODES

        assert int(spec.group_plane.geom_type[0]) == DIST_TYPE_CODES["flat-bottomed2"]
        assert float(spec.group_plane.target2[0]) == pytest.approx(0.1)

    def test_top_level_key_is_whitelisted(self):
        # a misspelled section name must still raise (the whitelist is a hard reject)
        with pytest.raises(ValueError, match="unknown top-level key"):
            RestraintsConfig.from_dict({"plane_restraint_config": []})


class TestEnergyWiring:
    """The array term reaches all three backends with the same value, and `move` changes
    only the gradient (so the numpy value reference still matches)."""

    def _pucker(self):
        ring = np.array(
            [
                [np.cos(t), np.sin(t), 0.0]
                for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)
            ]
        )
        pos = ring.copy()
        pos[::2, 2] += 0.4  # out-of-plane RMS = 0.2 A
        return pos

    def _spec(self, **extra):
        from rgi_utils.combined import CombinedRestraints

        cr = CombinedRestraints()
        cr.setup(
            _adapter(("A", 6)),
            1,
            {"gpu": False, "plane_restraints_config": [{**_SEL1, **extra}]},
        )
        return cr.spec

    def test_three_backend_value_parity(self):
        torch = pytest.importorskip("torch")
        jnp = pytest.importorskip("jax.numpy")
        from rgi_utils.energy import jax_energy as je
        from rgi_utils.energy import numpy_energy as ne
        from rgi_utils.energy import torch_energy as te

        spec, pos = self._spec(), self._pucker()
        want = 0.2**2  # harmonic toward 0, weight 1
        assert float(
            ne.total_energy(pos, ne.prepare_spec(spec), sigma=1.0)
        ) == pytest.approx(want, abs=1e-6)
        assert float(
            te.total_energy(torch.tensor(pos), te.prepare_spec(spec), sigma=1.0)
        ) == pytest.approx(want, abs=1e-6)
        assert float(
            je.total_energy(jnp.asarray(pos), je.prepare_spec(spec), sigma=1.0)
        ) == pytest.approx(want, abs=1e-5)

    def test_breakdown_key_is_separate_from_the_conformer_plane(self):
        from rgi_utils.energy import numpy_energy as ne

        spec = self._spec()
        bd = ne.energy_breakdown(self._pucker(), ne.prepare_spec(spec), sigma=1.0)
        assert bd["group_plane"] == pytest.approx(0.04, abs=1e-6)
        assert bd["plane"] == 0.0

    def test_gate_above_start_sigma_is_a_noop(self):
        from rgi_utils.energy import numpy_energy as ne

        spec = self._spec(start_sigma=2.0)
        prepared = ne.prepare_spec(spec)
        assert float(ne.total_energy(self._pucker(), prepared, sigma=5.0)) == 0.0
        assert float(ne.total_energy(self._pucker(), prepared, sigma=1.0)) > 0.0

    def test_step_window_gate(self):
        from rgi_utils.energy import numpy_energy as ne

        spec = self._spec(start_step=3, stop_step=6)
        prepared = ne.prepare_spec(spec)
        assert (
            float(ne.total_energy(self._pucker(), prepared, sigma=1.0, step=1)) == 0.0
        )
        assert float(ne.total_energy(self._pucker(), prepared, sigma=1.0, step=4)) > 0.0

    def test_torch_cg_flattens_the_group(self):
        torch = pytest.importorskip("torch")
        from rgi_utils.combined import CombinedRestraints

        pos = self._pucker()
        cr = CombinedRestraints()
        cr.setup(
            _adapter(("A", 6)),
            1,
            {
                "gpu": False,
                "max_iter": 300,
                "plane_restraints_config": [{**_SEL1}],
            },
        )
        out = cr.minimize(torch.tensor(pos, dtype=torch.float64), 0, 1.0)
        x = out.detach().cpu().numpy()
        x0 = x - x.mean(0)
        _w, vecs = np.linalg.eigh(x0.T @ x0)
        assert float(np.sqrt(((x0 @ vecs[:, 0]) ** 2).mean())) < 1e-3

    def test_pinned_atoms_get_no_gradient(self):
        torch = pytest.importorskip("torch")
        from rgi_utils.energy import torch_energy as te

        spec = self._spec()
        spec.group_plane.free[0, 1::2] = 0.0  # pin every other atom
        t = torch.tensor(self._pucker(), requires_grad=True)
        te.total_energy(t, te.prepare_spec(spec), sigma=1.0).backward()
        grad = t.grad.numpy()
        assert np.allclose(grad[1::2], 0.0)
        assert not np.allclose(grad[0::2], 0.0)
