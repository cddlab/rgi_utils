"""Nucleic-acid base-pair restraints: config parse, WC expansion, fail-loud, convergence.

The base-pair restraint is a config-time MACRO: each entry expands into WC H-bond
distance restraints (+ an optional coplanarity plane restraint). These tests cover the
expansion (right atom pairs / counts), the fail-loud guards, and that the generated
restraints actually converge under the torch CG (H-bonds reach the target window, the
two bases flatten).

The coplanarity plane is a standalone ``PlaneRestraintData`` (the ``plane_restraints_config``
term), so it carries the entry's OWN gate window and ``move``. It used to be injected into
the conformer ``plane`` arrays, where it rode the shared conformer gate — hence the
``_plane_atoms`` / ``_plane_slack`` helpers below, which read the same two quantities the
old ``(atoms, slack, weight)`` tuple carried.
"""

from __future__ import annotations

import numpy as np
import pytest

from rgi_utils.atom_context import AtomRecord
from rgi_utils.base_pair_restr_data import BasePairData
from rgi_utils.combined import CombinedRestraints
from rgi_utils.config import RestraintsConfig

# base + a few backbone atoms (P/O5'/C1') so the coplanarity filter is exercised
_G_BASE = ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"]
_C_BASE = ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"]
_A_BASE = ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"]
_U_BASE = ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"]
_BACKBONE = ["P", "O5'", "C1'"]


class MockAdapter:
    def __init__(self, atoms):
        self._atoms = atoms

    def iter_atoms(self):
        yield from self._atoms


def _residue(chain, resid, resname, base_atoms, start, mol_type="rna"):
    """Build one residue's AtomRecords (base atoms + backbone). Returns (records, next)."""
    recs = []
    idx = start
    for nm in list(base_atoms) + _BACKBONE:
        recs.append(AtomRecord(chain, resid, idx, nm, mol_type, resname))
        idx += 1
    return recs, idx


def _adapter(*residues):
    """residues: (chain, resid, resname, base_atoms[, mol_type]) tuples."""
    recs, idx = [], 0
    for r in residues:
        mt = r[4] if len(r) > 4 else "rna"
        rr, idx = _residue(r[0], r[1], r[2], r[3], idx, mt)
        recs.extend(rr)
    return MockAdapter(recs)


def _entry(**kw):
    e = {"residue1": "chain A and resid 1", "residue2": "chain B and resid 1"}
    e.update(kw)
    return e


def _resolve(adapter, **kw):
    bp = BasePairData()
    bp.set_config(_entry(**kw))
    if bp.start_sigma is None:
        bp.start_sigma = float("inf")
    return bp.resolve_sites(adapter)


def _plane_atoms(plane):
    """Pooled atom list of a coplanarity restraint. The macro now emits a
    ``PlaneRestraintData`` with ONE group per residue (all pooled into a single best-fit
    plane), so flatten them — the plane's identity is the pooled set."""
    return [a for grp in plane.target_sites for a in grp]


def _plane_slack(plane):
    """The coplanarity tolerance, whichever restraint type encodes it: slack 0 becomes a
    pure ``harmonic`` toward 0, a positive slack the upper-bound-only ``flat-bottomed2``.
    ``target2`` holds the slack in both cases."""
    return plane.target2


# --------------------------------------------------------------------------- config
class TestConfig:
    def test_defaults(self):
        bp = BasePairData()
        bp.set_config(_entry())
        assert bp.coplanar is True
        assert bp.weight == pytest.approx(1.0)
        assert bp.move_mode == 0
        assert bp.target == (2.7, 3.1)  # default flat-bottomed window
        assert bp.run_restr is True

    def test_missing_residue_raises(self):
        with pytest.raises(ValueError, match="residue1 and residue2"):
            BasePairData().set_config({"residue1": "chain A and resid 1"})

    def test_pair_override_validation(self):
        with pytest.raises(ValueError, match="two base letters"):
            BasePairData().set_config(_entry(pair="XY"))
        bp = BasePairData()
        bp.set_config(_entry(pair="gc"))
        assert bp.pair == "GC"  # upper-cased

    def test_target_scalar_is_harmonic(self):
        bp = BasePairData()
        bp.set_config(_entry(target=2.9))
        assert bp.target == pytest.approx(2.9)  # scalar -> harmonic

    def test_target_bad_list_raises(self):
        with pytest.raises(ValueError, match="low, high"):
            BasePairData().set_config(_entry(target=[3.1, 2.7]))

    def test_move_parse(self):
        assert BasePairData.__init__ is not None
        for mv, expect in [("both", 0), (1, 1), (2, 2), ("1,2", 0)]:
            bp = BasePairData()
            bp.set_config(_entry(move=mv))
            assert bp.move_mode == expect

    def test_window_exclusive_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            BasePairData().set_config(_entry(start_sigma=1.0, start_step=2))

    def test_unknown_key_warns_not_raises(self, caplog):
        # a typo'd key is a soft warning (like every other restraint), not a raise
        import logging

        with caplog.at_level(logging.WARNING):
            BasePairData().set_config(_entry(typo_key=1))
        assert any("unknown config key" in r.message for r in caplog.records)


# ------------------------------------------------------------------------- resolve
class TestResolve:
    def test_gc_auto_three_hbonds(self):
        dists, plane = _resolve(
            _adapter(("A", 1, "G", _G_BASE), ("B", 1, "C", _C_BASE))
        )
        assert len(dists) == 3
        # all flat-bottomed toward the default window
        for d in dists:
            assert d.distance_restraint_type == "flat-bottomed"
            assert (d.target_distance1, d.target_distance2) == (2.7, 3.1)
            assert len(d.target_sites1) == 1 and len(d.target_sites2) == 1
        assert plane is not None

    def test_au_auto_two_hbonds(self):
        dists, plane = _resolve(
            _adapter(("A", 1, "A", _A_BASE), ("B", 1, "U", _U_BASE))
        )
        assert len(dists) == 2
        assert plane is not None

    def test_dna_auto(self):
        dists, _ = _resolve(
            _adapter(("A", 1, "DG", _G_BASE, "dna"), ("B", 1, "DC", _C_BASE, "dna"))
        )
        assert len(dists) == 3

    def test_reverse_order_cg(self):
        # residue1 = C, residue2 = G: same 3 h-bonds, atom pairs mirrored
        dists, _ = _resolve(_adapter(("A", 1, "C", _C_BASE), ("B", 1, "G", _G_BASE)))
        assert len(dists) == 3

    def test_coplanar_off(self):
        dists, plane = _resolve(
            _adapter(("A", 1, "G", _G_BASE), ("B", 1, "C", _C_BASE)), coplanar=False
        )
        assert len(dists) == 3
        assert plane is None

    def test_coplanar_group_excludes_backbone(self):
        adapter = _adapter(("A", 1, "G", _G_BASE), ("B", 1, "C", _C_BASE))
        _, plane = _resolve(adapter)
        atoms = _plane_atoms(plane)
        # only the base atoms (11 + 8), never P/O5'/C1'
        assert len(atoms) == len(_G_BASE) + len(_C_BASE)
        bb_indices = {a.index for a in adapter._atoms if a.name in _BACKBONE}
        assert not (set(atoms) & bb_indices)
        assert _plane_slack(plane) == 0.0 and plane.weight == pytest.approx(1.0)
        # slack 0 -> a pure harmonic toward 0 out-of-plane RMS
        assert plane.geom_type == "harmonic"
        # one group per residue, and the entry's gate window (NOT the conformer gate)
        assert len(plane.target_sites) == 2
        assert plane.start_sigma == float("inf") and plane.stop_sigma == -1.0
        assert plane.move_free == (True, True)  # move omitted -> both bases free

    def test_pair_override_wobble(self):
        # G-U wobble is opt-in via pair (never auto): 2 h-bonds
        dists, _ = _resolve(
            _adapter(("A", 1, "G", _G_BASE), ("B", 1, "U", _U_BASE)),
            pair="GU",
            coplanar=False,
        )
        assert len(dists) == 2

    def test_pair_override_resname_none(self):
        # resname unavailable -> explicit pair still works
        dists, _ = _resolve(
            _adapter(("A", 1, None, _G_BASE), ("B", 1, None, _C_BASE)),
            pair="GC",
            coplanar=False,
        )
        assert len(dists) == 3

    def test_weight_move_flow_into_hbonds(self):
        dists, _ = _resolve(
            _adapter(("A", 1, "G", _G_BASE), ("B", 1, "C", _C_BASE)),
            weight=3.0,
            move=1,
            coplanar=False,
        )
        assert all(d.weight == pytest.approx(3.0) for d in dists)
        assert all(d.move_mode == 1 for d in dists)


# ------------------------------------------------------------------------ fail-loud
class TestFailLoud:
    def test_more_than_one_residue(self):
        adapter = _adapter(
            ("A", 1, "G", _G_BASE), ("A", 2, "G", _G_BASE), ("B", 1, "C", _C_BASE)
        )
        with pytest.raises(ValueError, match="exactly one residue"):
            bp = BasePairData()
            bp.set_config({"residue1": "chain A", "residue2": "chain B and resid 1"})
            bp.resolve_sites(adapter)

    def test_non_wc_auto(self):
        with pytest.raises(ValueError, match="not a canonical"):
            _resolve(_adapter(("A", 1, "G", _G_BASE), ("B", 1, "G", _G_BASE)))

    def test_gu_auto_rejected(self):
        with pytest.raises(ValueError, match="not a canonical"):
            _resolve(_adapter(("A", 1, "G", _G_BASE), ("B", 1, "U", _U_BASE)))

    def test_missing_wc_atom(self):
        # drop O6 from guanine -> the O6-N4 h-bond can't be built
        g = [a for a in _G_BASE if a != "O6"]
        with pytest.raises(ValueError, match="atom O6 missing"):
            _resolve(_adapter(("A", 1, "G", g), ("B", 1, "C", _C_BASE)))

    def test_resname_none_no_pair(self):
        with pytest.raises(ValueError, match="could not identify"):
            _resolve(_adapter(("A", 1, None, _G_BASE), ("B", 1, None, _C_BASE)))

    def test_selection_matches_nothing(self):
        with pytest.raises(ValueError, match="matched no atoms"):
            _resolve(
                _adapter(("A", 1, "G", _G_BASE), ("B", 1, "C", _C_BASE)),
                residue1="chain Z and resid 9",
            )


# ----------------------------------------------------------------------- config wiring
class TestConfigWiring:
    def test_top_level_key_accepted(self):
        cfg = RestraintsConfig.from_dict({"base_pair_restraints_config": [_entry()]})
        assert len(cfg.base_pair_data) == 1
        assert cfg.base_pair_data[0].start_sigma == float("inf")  # None -> +inf

    def test_setup_builds_distance_and_plane_spec(self):
        cr = CombinedRestraints()
        cr.setup(
            _adapter(("A", 1, "G", _G_BASE), ("B", 1, "C", _C_BASE)),
            1,
            {"gpu": False, "base_pair_restraints_config": [_entry()]},
        )
        assert int(cr.spec.distance.mask.sum()) == 3
        # the coplanarity plane is the STANDALONE plane term (per-entry gate), not the
        # conformer `plane` sub-term it used to be injected into
        assert int(cr.spec.group_plane.mask.sum()) == 1
        assert cr.spec.plane is None

    def test_multi_pair_duplex(self):
        # three stacked pairs (G-C, A-U, C-G): distances 3+2+3=8, one plane group each.
        # Exercises the active-union / multi-plane-group merge for a real duplex.
        cr = CombinedRestraints()
        cr.setup(
            _adapter(
                ("A", 1, "G", _G_BASE),
                ("A", 2, "A", _A_BASE),
                ("A", 3, "C", _C_BASE),
                ("B", 1, "C", _C_BASE),
                ("B", 2, "U", _U_BASE),
                ("B", 3, "G", _G_BASE),
            ),
            1,
            {
                "gpu": False,
                "base_pair_restraints_config": [
                    {
                        "residue1": "chain A and resid 1",
                        "residue2": "chain B and resid 1",
                    },
                    {
                        "residue1": "chain A and resid 2",
                        "residue2": "chain B and resid 2",
                    },
                    {
                        "residue1": "chain A and resid 3",
                        "residue2": "chain B and resid 3",
                    },
                ],
            },
        )
        assert int(cr.spec.distance.mask.sum()) == 8
        assert int(cr.spec.group_plane.mask.sum()) == 3

    def test_entry_gate_and_move_reach_the_coplanarity_plane(self):
        # The migration off `extra_plane_groups`: the plane now carries the entry's own
        # sigma window and move mask. It used to ride the shared conformer gate, so a
        # stop_sigma released the H-bonds but left the coplanarity pulling.
        cr = CombinedRestraints()
        cr.setup(
            _adapter(("A", 1, "G", _G_BASE), ("B", 1, "C", _C_BASE)),
            1,
            {
                "gpu": False,
                "base_pair_restraints_config": [
                    _entry(start_sigma=4.0, stop_sigma=1.0, move=1)
                ],
            },
        )
        gp = cr.spec.group_plane
        assert float(gp.start_sigma[0]) == 4.0 and float(gp.stop_sigma[0]) == 1.0
        # move: 1 -> residue1's atoms free, residue2's pinned in the plane fit
        n1, n2 = len(_G_BASE), len(_C_BASE)
        assert gp.free[0, :n1].sum() == n1
        assert gp.free[0, n1 : n1 + n2].sum() == 0.0

    def test_base_pair_does_not_mutate_config(self):
        # a config-less re-setup() must not duplicate the generated distances (the
        # local-merge contract mirroring custom_data)
        cr = CombinedRestraints()
        adapter = _adapter(("A", 1, "G", _G_BASE), ("B", 1, "C", _C_BASE))
        config = {"gpu": False, "base_pair_restraints_config": [_entry()]}
        cr.setup(adapter, 1, config)
        cr.setup(adapter, 1)  # re-setup, reuse the same parsed config
        assert int(cr.spec.distance.mask.sum()) == 3  # still 3, not 6


# ----------------------------------------------------------------------- convergence
class TestConverge:
    def test_torch_hbonds_and_plane_converge(self):
        torch = pytest.importorskip("torch")
        # minimal G (N1,N2,O6 + 2 ring) / C (N3,O2,N4 + 2 ring): WC atoms first
        g = ["N1", "N2", "O6", "C2", "C6"]
        c = ["N3", "O2", "N4", "C4", "C5"]
        recs = [AtomRecord("A", 1, i, nm, "rna", "G") for i, nm in enumerate(g)]
        recs += [AtomRecord("B", 1, 5 + j, nm, "rna", "C") for j, nm in enumerate(c)]

        rng = np.random.default_rng(0)
        coords = np.zeros((10, 3))
        coords[:5, 0] = [0.0, 0.5, -0.5, 1.0, -1.0]
        coords[:5, 1] = [0.0, 1.4, -1.4, 2.0, -2.0]
        coords[5:, 0] = [8.0, 8.5, 7.5, 9.0, 7.0]
        coords[5:, 1] = [0.2, 1.5, -1.3, 2.1, -1.9]
        coords[:, 2] = rng.uniform(-1.0, 1.0, 10)  # pucker -> plane has work

        cr = CombinedRestraints()
        cr.setup(
            MockAdapter(recs),
            1,
            {
                "gpu": False,
                "max_iter": 400,
                "base_pair_restraints_config": [_entry()],
            },
        )

        def _plane_dev(x):
            c0 = x.mean(0)
            _u, _s, vt = np.linalg.svd(x - c0)
            return float(np.sqrt(np.mean(((x - c0) @ vt[-1]) ** 2)))

        wc = [(0, 5), (1, 6), (2, 7)]
        dev0 = _plane_dev(coords)
        out = cr.minimize(torch.tensor(coords, dtype=torch.float64), 0, 1.0)
        xo = out.detach().cpu().numpy()
        for i, j in wc:
            d = float(np.linalg.norm(xo[i] - xo[j]))
            assert 2.6 <= d <= 3.2, f"h-bond {i}-{j} = {d:.2f} out of window"
        assert _plane_dev(xo) < 0.5 * dev0  # flattened


# ---------------------------------------------------------------------------- triple
class TestTriple:
    """``residue3`` widens the coplanarity plane to a base triple.

    A triple is real geometry the pair form cannot express: the third base docks on a
    groove edge (non-WC, so its identity must not be validated) and sits measurably out
    of the pair's plane (so the pair's slack-0 target would flatten it).
    """

    def _triple_adapter(self):
        return _adapter(
            ("A", 1, "G", _G_BASE), ("B", 1, "C", _C_BASE), ("A", 2, "A", _A_BASE)
        )

    def test_third_base_joins_the_plane_only(self):
        adapter = self._triple_adapter()
        dists, plane = _resolve(adapter, residue3="chain A and resid 2")
        # H-bonds stay the three G-C ones: the third base contributes none, because
        # which of its atoms bond depends on the Leontis-Westhof family.
        assert len(dists) == 3
        atoms = _plane_atoms(plane)
        assert len(atoms) == len(_G_BASE) + len(_C_BASE) + len(_A_BASE)
        bb_indices = {a.index for a in adapter._atoms if a.name in _BACKBONE}
        assert not (set(atoms) & bb_indices)

    def test_triple_gets_slack_and_pair_does_not(self):
        _, pair_plane = _resolve(
            _adapter(("A", 1, "G", _G_BASE), ("B", 1, "C", _C_BASE))
        )
        _, triple_plane = _resolve(
            self._triple_adapter(), residue3="chain A and resid 2"
        )
        # slack 0 -> harmonic toward 0 (unchanged for every pre-existing config)
        assert _plane_slack(pair_plane) == 0.0
        assert pair_plane.geom_type == "harmonic"
        # a positive slack -> the upper-bound-only flat-bottomed2, same max(0, rms-slack)
        assert _plane_slack(triple_plane) > 0.0
        assert triple_plane.geom_type == "flat-bottomed2"

    def test_explicit_slack_overrides_both(self):
        _, pair_plane = _resolve(
            _adapter(("A", 1, "G", _G_BASE), ("B", 1, "C", _C_BASE)), coplanar_slack=0.2
        )
        _, triple_plane = _resolve(
            self._triple_adapter(), residue3="chain A and resid 2", coplanar_slack=0.2
        )
        assert _plane_slack(pair_plane) == pytest.approx(0.2)
        assert _plane_slack(triple_plane) == pytest.approx(0.2)

    def test_negative_slack_raises(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            BasePairData().set_config(_entry(coplanar_slack=-0.1))

    def test_third_base_identity_is_not_validated(self):
        # G-G would be rejected as a PAIR; as the third base of a triple it is fine.
        _, plane = _resolve(
            _adapter(
                ("A", 1, "G", _G_BASE), ("B", 1, "C", _C_BASE), ("A", 2, "G", _G_BASE)
            ),
            residue3="chain A and resid 2",
        )
        assert len(_plane_atoms(plane)) == len(_G_BASE) * 2 + len(_C_BASE)

    def test_residue3_with_coplanar_off_raises(self):
        with pytest.raises(ValueError, match="no-op with coplanar: false"):
            BasePairData().set_config(
                _entry(residue3="chain A and resid 2", coplanar=False)
            )

    def test_residue3_must_match_one_residue(self):
        with pytest.raises(
            ValueError, match="residue3 selection must match exactly one"
        ):
            _resolve(self._triple_adapter(), residue3="chain A")

    def test_slack_admits_a_tilted_third_base(self):
        """The slack must be wide enough for a real triple and narrow enough to bite.

        Out-of-plane RMS measured over the reference triples runs 0.15-0.44 A, so a
        third base at that tilt must cost nothing while a grossly non-planar one does.
        """
        torch = pytest.importorskip("torch")
        g = ["N1", "N2", "O6", "C2", "C6"]
        c = ["N3", "O2", "N4", "C4", "C5"]
        a = ["N1", "N6", "N7", "C2", "C8"]
        recs = [AtomRecord("A", 1, i, nm, "rna", "G") for i, nm in enumerate(g)]
        recs += [AtomRecord("B", 1, 5 + j, nm, "rna", "C") for j, nm in enumerate(c)]
        recs += [AtomRecord("A", 2, 10 + k, nm, "rna", "A") for k, nm in enumerate(a)]

        coords = np.zeros((15, 3))
        coords[:5, 0] = [0.0, 0.5, -0.5, 1.0, -1.0]
        coords[:5, 1] = [0.0, 1.4, -1.4, 2.0, -2.0]
        coords[5:10, 0] = [8.0, 8.5, 7.5, 9.0, 7.0]
        coords[5:10, 1] = [0.2, 1.5, -1.3, 2.1, -1.9]
        coords[10:, 0] = [-6.0, -5.5, -6.5, -5.0, -7.0]
        coords[10:, 1] = [0.1, 1.4, -1.3, 2.0, -1.9]
        coords[10:, 2] = [0.0, 0.35, -0.35, 0.5, -0.5]  # ~0.35 A RMS tilt: allowed

        def _run(**kw):
            cr = CombinedRestraints()
            cr.setup(
                MockAdapter(recs),
                1,
                {
                    "gpu": False,
                    "max_iter": 200,
                    "base_pair_restraints_config": [
                        _entry(residue3="chain A and resid 2", **kw)
                    ],
                },
            )
            out = cr.minimize(
                torch.tensor(coords.copy(), dtype=torch.float64), 0, 1.0
            ).numpy()
            return float(np.abs(out[10:, 2] - coords[10:, 2]).max())

        # inside the default slack -> the plane term must not drag the third base flat
        assert _run() < 0.1
        # ...and the same tilt IS pulled flat once the slack is taken away, so the
        # assertion above cannot pass vacuously
        assert _run(coplanar_slack=0.0) > 0.2


# ------------------------------------------------------------------------ plane only
class TestPlaneOnly:
    """``hbonds: false`` keeps the plane and drops the Watson-Crick lookup.

    Without it a non-WC arrangement is unreachable: the table either has no entry (G-A
    raises) or has the WRONG one (a reverse-Hoogsteen U-A bonds O2-N6/N3-N7, but the
    table would restrain the WC N1-N3/N6-O4 and quietly build a different pair).
    """

    def test_no_hbonds_but_a_plane(self):
        dists, plane = _resolve(
            _adapter(("A", 1, "G", _G_BASE), ("B", 1, "C", _C_BASE)), hbonds=False
        )
        assert dists == []
        assert len(_plane_atoms(plane)) == len(_G_BASE) + len(_C_BASE)

    def test_non_wc_pair_is_allowed(self):
        # G-G raises as a normal entry; with hbonds: false it is just two bases.
        with pytest.raises(ValueError, match="not a canonical"):
            _resolve(_adapter(("A", 1, "G", _G_BASE), ("B", 1, "G", _G_BASE)))
        dists, plane = _resolve(
            _adapter(("A", 1, "G", _G_BASE), ("B", 1, "G", _G_BASE)), hbonds=False
        )
        assert dists == [] and len(_plane_atoms(plane)) == 2 * len(_G_BASE)

    def test_plane_only_triple(self):
        _, plane = _resolve(
            _adapter(
                ("A", 1, "G", _G_BASE), ("B", 1, "A", _A_BASE), ("A", 2, "A", _A_BASE)
            ),
            hbonds=False,
            residue3="chain A and resid 2",
        )
        assert len(_plane_atoms(plane)) == len(_G_BASE) + 2 * len(_A_BASE)

    def test_both_off_raises(self):
        with pytest.raises(ValueError, match="generates nothing"):
            BasePairData().set_config(_entry(hbonds=False, coplanar=False))

    def test_slack_default_still_depends_on_the_third_base(self):
        _, pair = _resolve(
            _adapter(("A", 1, "G", _G_BASE), ("B", 1, "G", _G_BASE)), hbonds=False
        )
        _, triple = _resolve(
            _adapter(
                ("A", 1, "G", _G_BASE), ("B", 1, "G", _G_BASE), ("A", 2, "A", _A_BASE)
            ),
            hbonds=False,
            residue3="chain A and resid 2",
        )
        assert _plane_slack(pair) == 0.0 and _plane_slack(triple) > 0.0
