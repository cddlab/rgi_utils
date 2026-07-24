"""Shared config parsing for named external structure references.

Distance, angle, dihedral, and custom restraints use the same entry-local
``refs`` map. Reference names are deliberately restricted to ``ref1``,
``ref2``, ... and a reference-backed atom selection is written as
``refN and <selection DSL>``.
"""

from __future__ import annotations

import logging
import re

from rgi_utils._config_util import coerce_bool, warn_unknown_keys

logger = logging.getLogger(__name__)

_REF_NAME_RE = re.compile(r"ref[1-9][0-9]*\Z")
_REF_SELECTION_RE = re.compile(r"\s*(ref[1-9][0-9]*)\s+and\s+(.+?)\s*\Z", re.DOTALL)
_REF_LIKE_PREFIX_RE = re.compile(r"\s*ref[0-9]+\b")

KNOWN_REF_KEYS = {
    "ref_pdb",
    "ref_cif",
    "atom_selection_ref_fit",
    "atom_selection_target_fit",
    "pairing",
    "best_effort",
}


def is_ref_name(value: object) -> bool:
    """Return whether ``value`` is a valid reserved reference name."""
    return isinstance(value, str) and _REF_NAME_RE.fullmatch(value) is not None


def split_ref_selection(selection: str, label: str) -> tuple[str, str] | None:
    """Split ``refN and <selection>`` into its reference name and selection.

    A normal prediction-side selection returns ``None``. A string that starts
    with a reference-like token but does not use the exact syntax raises rather
    than falling through to the ordinary atom-selection parser.
    """
    if not isinstance(selection, str):
        raise ValueError(f"{label}: atom selection must be a string")
    match = _REF_SELECTION_RE.fullmatch(selection)
    if match is not None:
        return match.group(1), match.group(2)
    if _REF_LIKE_PREFIX_RE.match(selection) is not None:
        raise ValueError(
            f"{label}: malformed reference selection {selection!r}; expected "
            "'refN and <atom selection>' with N >= 1"
        )
    return None


def parse_ref_defs(refs: object, label: str) -> dict[str, dict]:
    """Parse one restraint entry's shared ``refs`` map."""
    if refs is None:
        return {}
    if not isinstance(refs, dict):
        raise ValueError(f"{label}: 'refs' must be a mapping")

    out: dict[str, dict] = {}
    for raw_name, raw_config in refs.items():
        name = str(raw_name)
        if not is_ref_name(name):
            raise ValueError(
                f"{label}: invalid reference name {name!r}; expected ref1, ref2, ..."
            )
        if not isinstance(raw_config, dict):
            raise ValueError(f"{label} refs[{name!r}]: definition must be a mapping")
        config = dict(raw_config)
        warn_unknown_keys(config, KNOWN_REF_KEYS, f"{label} refs[{name!r}]", logger)
        ref_pdb = config.get("ref_pdb")
        ref_cif = config.get("ref_cif")
        if (ref_pdb is None) == (ref_cif is None):
            raise ValueError(
                f"{label} refs[{name!r}]: give exactly one of ref_pdb / ref_cif"
            )
        pairing = config.get("pairing") or "align"
        if pairing not in ("identity", "align"):
            raise ValueError(
                f"{label} refs[{name!r}]: pairing must be 'identity' or 'align', "
                f"got {pairing!r}"
            )
        out[name] = {
            "ref_cif": ref_cif,
            "ref_path": ref_pdb if ref_pdb is not None else ref_cif,
            "sel_ref_fit": config.get("atom_selection_ref_fit"),
            "sel_target_fit": config.get("atom_selection_target_fit"),
            "pairing": pairing,
            "best_effort": coerce_bool(config.get("best_effort"), True),
        }
    return out


def validate_ref_usage(
    ref_defs: dict[str, dict],
    used_refs: set[str],
    label: str,
    *,
    max_refs: int | None = None,
) -> None:
    """Require every named reference to be defined and used exactly by the entry."""
    undefined = used_refs - set(ref_defs)
    if undefined:
        raise ValueError(
            f"{label}: undefined reference name(s) {sorted(undefined)}; define them "
            "in the entry's 'refs' map"
        )
    unused = set(ref_defs) - used_refs
    if unused:
        raise ValueError(f"{label}: unused reference definition(s) {sorted(unused)}")
    if max_refs is not None and len(used_refs) > max_refs:
        raise ValueError(
            f"{label}: at most {max_refs} distinct reference structure(s) are allowed; "
            f"got {len(used_refs)}"
        )
