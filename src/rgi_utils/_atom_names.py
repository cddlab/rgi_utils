"""Canonical atom-name spelling shared by selection and geometry readers."""


def normalise_atom_name(name: str | None) -> str:
    """Unify whitespace, case, and the PDB prime and double-prime spellings."""
    return (name or "").strip().upper().replace('"', "''").replace("*", "'")
