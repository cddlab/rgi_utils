"""Config-parsing validation + adapter resid-convention regression tests.

Covers the silent-config-drop failure class surfaced by the audit: a typo'd section
name or entry key that used to be dropped with no error, plus the bool() coercion trap
and the per-chain resid convention. Pure parsing / lightweight numpy fixtures — no GPU.

NOTE: a full cross-adapter (same structure -> all six adapters -> diff AtomRecords)
runtime parity test is intentionally NOT here: faithful fixtures need each tool's real
feats format, and a hand-built one risks not reproducing the very divergence the
invariant guards. The esmfold2 case below is the one adapter constructible from plain
numpy; the others are exercised by their tools' E2E runs.
"""

import logging

import numpy as np
import pytest

from rgi_utils._config_util import coerce_bool
from rgi_utils.config import RestraintsConfig
from rgi_utils.rmsd_restr_data import RmsdData


# --- top-level section whitelist (F1) ---------------------------------------------
def test_unknown_top_level_key_raises():
    """A misspelled SECTION name must raise, not silently drop the whole block."""
    with pytest.raises(ValueError, match="unknown top-level key"):
        RestraintsConfig.from_dict(
            {"distance_restraint_config": [{"atom_selection1": "chain A"}]}
        )


def test_known_top_level_keys_ok():
    """The full set of valid top-level keys parses without error."""
    cfg = RestraintsConfig.from_dict(
        {
            "verbose": False,
            "gpu": False,
            "backend": "torch",
            "method": "CG",
            "max_iter": 50,
            "distance_restraints_config": [],
            "rmsd_restraints_config": [],
            "angle_restraints_config": [],
            "dihedral_restraints_config": [],
            "conformer_restraints_config": {},
        }
    )
    assert cfg.backend == "torch"


def test_empty_config_is_vanilla():
    """None / {} is a valid no-restraint run (must NOT trip the whitelist)."""
    assert RestraintsConfig.from_dict(None).distance_data == []
    assert RestraintsConfig.from_dict({}).rmsd_data == []


# --- per-entry unknown-key warnings (F1) ------------------------------------------
def test_unknown_distance_entry_key_warns(caplog):
    """A key distance does not read (e.g. 'weight') is warned, not silently dropped."""
    with caplog.at_level(logging.WARNING):
        RestraintsConfig.from_dict(
            {
                "distance_restraints_config": [
                    {
                        "atom_selection1": "chain A",
                        "atom_selection2": "chain B",
                        "harmonic": {"target_distance": 5.0},
                        "weight": 2.0,  # distance is closed-form / unweighted -> warn
                    }
                ]
            }
        )
    assert any(
        "unknown config key(s) ['weight']" in r.getMessage() for r in caplog.records
    )


def test_bare_atom_selection_on_rmsd_raises():
    """The documented footgun: a bare 'atom_selection' (only the _ref/_target shorthand
    and the _fit/_calc keys are read) would be silently dropped, broadening the
    superposition to the whole structure -- so it is now rejected loudly, like a
    misspelled section name or a top-level start_sigma."""
    rr = RmsdData()
    with pytest.raises(ValueError, match="atom_selection"):
        rr.set_config(
            {
                "ref_pdb": "x.pdb",
                "harmonic": {"target_rmsd": 1.0},
                "atom_selection": "chain A",  # footgun: not a real key
            }
        )


# --- bool coercion trap (F4) ------------------------------------------------------
def test_best_effort_string_false_disables():
    """best_effort: "false" (quoted) must disable it — plain bool("false") is True."""
    rr = RmsdData()
    rr.set_config(
        {"ref_pdb": "x.pdb", "harmonic": {"target_rmsd": 1.0}, "best_effort": "false"}
    )
    assert rr.best_effort is False
    rr2 = RmsdData()
    rr2.set_config({"ref_pdb": "x.pdb", "harmonic": {"target_rmsd": 1.0}})
    assert rr2.best_effort is True  # default when omitted


def test_coerce_bool():
    assert coerce_bool(None, True) is True
    assert coerce_bool(None, False) is False
    assert coerce_bool(True) is True and coerce_bool(False) is False
    for falsey in ("false", "False", "no", "off", "0"):
        assert coerce_bool(falsey) is False
    for truthy in ("true", "yes", "on", "1"):
        assert coerce_bool(truthy) is True
    assert coerce_bool(1) is True and coerce_bool(0) is False


# --- esmfold2 adapter: per-chain resid convention + token-pad guard (F10) ---------
def _min_esm_features(asym_ids):
    """Minimal ESMFold2 features dict (1 atom / token) for the convention checks."""
    n = len(asym_ids)
    return {
        "asym_id": np.array([asym_ids], dtype=np.int64),
        "mol_type": np.zeros((1, n), dtype=np.int64),
        "atom_to_token": np.arange(n, dtype=np.int64).reshape(1, n),
        "atom_attention_mask": np.ones((1, n), dtype=bool),
        "ref_pos": np.zeros((1, n, 3), dtype=np.float64),
        "ref_element": np.full((1, n), 6, dtype=np.int64),
        "token_bonds": np.zeros((1, n, n), dtype=np.int64),
    }


def test_esmfold2_resid_resets_per_chain():
    """resid is a per-chain 1-based ordinal that resets at each chain boundary."""
    from rgi_utils.esmfold2.adapter import ESMFold2Adapter

    ad = ESMFold2Adapter(_min_esm_features([0, 0, 0, 1, 1]))
    assert ad._tok_ordinal == {0: 1, 1: 2, 2: 3, 3: 1, 4: 2}


def test_esmfold2_token_padding_guard_raises():
    """A padded token mask would shift the resid ordinals -> the guard must raise."""
    from rgi_utils.esmfold2.adapter import ESMFold2Adapter

    feats = _min_esm_features([0, 0, 1])
    feats["token_attention_mask"] = np.array([[1, 1, 0]], dtype=np.int64)  # a pad token
    with pytest.raises(ValueError, match="token_attention_mask has padding"):
        ESMFold2Adapter(feats)
