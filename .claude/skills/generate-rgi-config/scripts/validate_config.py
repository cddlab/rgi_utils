#!/usr/bin/env python3
"""Validate an RGI ``restraints_config`` BEFORE you burn a GPU run on it.

What this checks (everything resolvable without a real structure):

  1. **Schema** — runs ``rgi_utils.config.RestraintsConfig.from_dict`` on every
     ``restraints_config`` found in the file. That raises on the silent-no-op traps:
     an unknown / misspelled top-level section (would drop the whole block), a
     top-level ``start_sigma``, a leftover ``backend`` key (now inferred, not
     configured), the old conformer ``dihedral`` key, mixing the sigma
     and step windows, an empty window, etc.
  2. **Selection DSL syntax** — feeds every ``atom_selection*`` string through
     ``AtomSelector`` (which parses eagerly), so a malformed selection (bad keyword,
     unbalanced parens) fails here instead of at run time.
  3. **Conformer opt-in** — if a ``conformer_restraints_config`` block is present, warns
     when no ligand opts in (``conformer_restraints: true`` on the ligand, or a chai
     sidecar ``conformer_restraints: {B: true}`` map). Without the opt-in the conformer
     block is a SILENT no-op.

What this CANNOT check (needs the real predicted structure):

  * whether a selection actually matches any atoms. A syntactically valid
    ``chain A and resid 5 to 84`` that resolves to ZERO atoms (wrong range, forgotten
    ``chain`` qualifier, ligand swept in) passes here but does nothing at run time.
    "Passes validation" is NOT "selects the atoms you meant." The only way to confirm
    the real selection is to run the tool with ``verbose: true`` and read the setup
    ``built spec: ... distances=N ...`` counts — a count of 0 means it selected nothing.

Usage:
    uv run --project <rgi-utils-dir> --frozen --with pyyaml \\
        python validate_config.py <input-file ...>  # .yaml / .yml / .json

Handles every tool's layout: boltz YAML (``restraints_config:`` nested), protenix JSON
(a list of jobs), AF3 fold-input JSON, openfold ``queries.<name>.restraints_config``,
the chai top-level sidecar, or a bare ``restraints_config`` dict.

Needs only numpy and pyyaml. Run it through the rgi_utils uv project as shown above. If
rgi_utils is not installed, this script adds the repo ``src/`` to ``sys.path``
automatically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# --- locate rgi_utils (installed, or the repo src/ found by walking up) --------------
# Prefer an installed rgi_utils; otherwise search ancestor dirs for a `src/rgi_utils`
# package. Walking up (rather than a fixed parents[N]) keeps this working no matter how
# deep the skill is placed (e.g. skills/... vs .claude/skills/...).
try:
    import rgi_utils  # noqa: F401
except ImportError:
    for _p in Path(__file__).resolve().parents:
        if (_p / "src" / "rgi_utils").is_dir():
            sys.path.insert(0, str(_p / "src"))
            break

try:
    from rgi_utils.config import RestraintsConfig
    from rgi_utils.ref_config import split_ref_selection
    from rgi_utils.selection import AtomSelector
except ImportError as exc:  # pragma: no cover
    sys.exit(
        f"cannot import rgi_utils ({exc}). Run with a Python that has rgi_utils on its "
        f"path, e.g. uv run --project <rgi-utils-dir> --frozen --with pyyaml "
        f"python validate_config.py <file>."
    )


# selection-string keys per restraint type (read off the raw config dict)
_SELECTION_KEYS = (
    "atom_selection1",
    "atom_selection2",
    "atom_selection3",
    "atom_selection4",
    "atom_selection_ref",
    "atom_selection_target",
    "atom_selection_ref_fit",
    "atom_selection_target_fit",
    "atom_selection_ref_calc",
    "atom_selection_target_calc",
)


def _load(path: Path):
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            sys.exit(
                "pyyaml is needed to read a YAML file. Either `pip install pyyaml` into "
                "the venv, or convert the file to JSON. (chai/boltz use YAML.)"
            )
        return yaml.safe_load(text)
    return json.loads(text)


def _find_configs(obj, path="<root>"):
    """Yield ``(location, restraints_config_dict, enclosing_object)`` for every config.

    ``enclosing_object`` is the object the ``restraints_config`` sits inside (a fold job
    / query / the whole file), used to look for the ligand opt-in flag. For a chai
    sidecar the config IS the top-level object, so enclosing == config.
    """
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from _find_configs(item, f"{path}[{i}]")
        return
    if not isinstance(obj, dict):
        return
    if "restraints_config" in obj and isinstance(obj["restraints_config"], dict):
        # boltz / protenix job / AF3 / a single openfold query
        yield (f"{path}.restraints_config", obj["restraints_config"], obj)
    # openfold: queries.<name>.restraints_config
    if "queries" in obj and isinstance(obj["queries"], dict):
        for name, q in obj["queries"].items():
            yield from _find_configs(q, f"{path}.queries.{name}")
    # chai sidecar / bare dict: the object itself is the restraints_config
    _restraint_keys = {
        "distance_restraints_config",
        "angle_restraints_config",
        "dihedral_restraints_config",
        "base_pair_restraints_config",
        "conformer_restraints_config",
        "rmsd_restraints_config",
        "custom_restraints_config",
    }
    if "restraints_config" not in obj and (_restraint_keys & set(obj)):
        yield (path, obj, obj)


def _collect_selection_strings(cfg: dict):
    """Walk a restraints_config and yield (key, selection_string) pairs to syntax-check."""
    for section in (
        "distance_restraints_config",
        "angle_restraints_config",
        "dihedral_restraints_config",
        "rmsd_restraints_config",
    ):
        for i, entry in enumerate(cfg.get(section, []) or []):
            if not isinstance(entry, dict):
                continue
            for key in _SELECTION_KEYS:
                if key in entry and isinstance(entry[key], str):
                    yield (f"{section}[{i}].{key}", entry[key])
            for ref_name, ref_def in (entry.get("refs", {}) or {}).items():
                if not isinstance(ref_def, dict):
                    continue
                for key in (
                    "atom_selection_ref_fit",
                    "atom_selection_target_fit",
                ):
                    if key in ref_def and isinstance(ref_def[key], str):
                        yield (
                            f"{section}[{i}].refs.{ref_name}.{key}",
                            ref_def[key],
                        )
    for i, entry in enumerate(cfg.get("base_pair_restraints_config", []) or []):
        if not isinstance(entry, dict):
            continue
        for key in ("residue1", "residue2"):
            if key in entry and isinstance(entry[key], str):
                yield (f"base_pair_restraints_config[{i}].{key}", entry[key])

    # custom: the named selections map
    for i, entry in enumerate(cfg.get("custom_restraints_config", []) or []):
        if isinstance(entry, dict):
            for name, sel in (entry.get("selections", {}) or {}).items():
                if isinstance(sel, str):
                    yield (f"custom_restraints_config[{i}].selections.{name}", sel)
            for ref_name, ref_def in (entry.get("refs", {}) or {}).items():
                if not isinstance(ref_def, dict):
                    continue
                for key in (
                    "atom_selection_ref_fit",
                    "atom_selection_target_fit",
                ):
                    if key in ref_def and isinstance(ref_def[key], str):
                        yield (
                            f"custom_restraints_config[{i}].refs.{ref_name}.{key}",
                            ref_def[key],
                        )


def _has_conformer_optin(enclosing: dict, cfg: dict) -> bool:
    """True if any ligand opts into conformer restraints (best-effort across formats)."""
    # chai sidecar: a {chain_id: bool} map lives in the config itself
    cmap = cfg.get("conformer_restraints")
    if isinstance(cmap, dict) and any(bool(v) for v in cmap.values()):
        return True
    # boltz/protenix/AF3/openfold: conformer_restraints:true on a ligand object somewhere
    found = [False]

    def _walk(o):
        if found[0]:
            return
        if isinstance(o, dict):
            if o.get("conformer_restraints") is True:
                found[0] = True
                return
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)

    _walk(enclosing)
    return found[0]


def _validate_one(location: str, cfg: dict, enclosing: dict) -> int:
    print(f"\n=== {location} ===")
    errors = 0

    # chai sidecar carries the opt-in map as a top-level key; chai strips it before
    # from_dict, so strip it here too or from_dict rejects the unknown key.
    cfg_for_schema = {k: v for k, v in cfg.items() if k != "conformer_restraints"}

    # 1. schema
    try:
        rc = RestraintsConfig.from_dict(cfg_for_schema)
    except Exception as exc:  # noqa: BLE001 — we want to print any parse failure
        print(f"  ✗ SCHEMA ERROR: {exc}")
        return 1
    print(
        f"  ✓ schema ok — distance={len(rc.distance_data)} "
        f"angle={len(rc.angle_data)} dihedral={len(rc.dihedral_data)} "
        f"base_pair={len(rc.base_pair_data)} "
        f"rmsd={len(rc.rmsd_data)} custom={len(rc.custom_data)}"
    )
    conf = cfg_for_schema.get("conformer_restraints_config") or {}
    conf_terms = [
        t for t in ("bond", "angle", "chiral", "plane", "cistrans", "vdw") if t in conf
    ]
    if conf_terms:
        print(f"    conformer terms: {', '.join(conf_terms)}")

    # 2. selection-DSL syntax
    for key, sel in _collect_selection_strings(cfg):
        try:
            if key.startswith("base_pair_restraints_config") or ".refs." in key:
                AtomSelector(sel)
            else:
                ref_selection = split_ref_selection(sel, key)
                AtomSelector(ref_selection[1] if ref_selection is not None else sel)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ SELECTION SYNTAX ERROR in {key}: {sel!r} — {exc}")
            errors += 1

    # 3. conformer opt-in (the #1 silent no-op)
    if conf_terms and not _has_conformer_optin(enclosing, cfg):
        print(
            "  ⚠ conformer_restraints_config is present but NO ligand opts in "
            "(no `conformer_restraints: true` on a ligand, no chai `conformer_restraints` "
            "map). The conformer block will be a SILENT no-op — add the opt-in flag."
        )

    if errors == 0:
        print("  ✓ all selection strings parse")
    return 1 if errors else 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    rc = 0
    n_configs = 0
    for arg in argv:
        path = Path(arg)
        if not path.is_file():
            print(f"✗ not a file: {arg}")
            rc = 1
            continue
        print(f"\n########## {path} ##########")
        try:
            data = _load(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ could not parse file: {exc}")
            rc = 1
            continue
        found = list(_find_configs(data))
        if not found:
            print("  ⚠ no restraints_config found in this file.")
            continue
        for location, cfg, enclosing in found:
            n_configs += 1
            rc |= _validate_one(location, cfg, enclosing)

    print(
        "\n"
        + ("─" * 70)
        + f"\nchecked {n_configs} restraints_config block(s). "
        + ("ALL OK." if rc == 0 else "SEE ERRORS ABOVE.")
        + "\nReminder: a clean result means the SCHEMA and selection SYNTAX are valid. "
        "\nIt does NOT prove a selection matches any atoms — run the tool with "
        "verbose: true\nand check the `built spec: ... distances=N ...` counts are "
        "non-zero for what you asked for."
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
