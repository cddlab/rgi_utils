"""Normalized molecule-type tables shared by adapters + the PDB reference reader.

The atom-selection DSL exposes ``protein`` / ``dna`` / ``rna`` selectors. Every
atom carries a NORMALIZED molecule-type string ("protein"/"dna"/"rna"/"ligand" or
None) on ``AtomRecord.mol_type`` / ``PdbAtom.mol_type``. Adapters that expose a
framework molecule-type enum normalize it at the source; the reference-PDB reader
(and any adapter lacking the annotation) classifies by residue name via
``moltype_from_resname`` here.

Only STANDARD polymer residues are classified. A modified/unknown residue or any
ligand / HETATM maps to None, so ``protein`` deliberately EXCLUDES it rather than
guessing (the cross-tool parity rule: never silently widen "protein"). Extend the
tables if a project needs a modified residue (e.g. MSE) to count as protein.
"""

from __future__ import annotations

# Standard amino acids (3-letter PDB codes). Modified residues (e.g. MSE) are
# intentionally absent -> None (no silent guessing); adapters that expose an
# is_protein/entity annotation still classify them correctly upstream of this.
PROTEIN_RESIDUES = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }
)
# Standard nucleotides: DNA is D-prefixed, RNA is the bare base.
DNA_RESIDUES = frozenset({"DA", "DC", "DG", "DT"})
RNA_RESIDUES = frozenset({"A", "C", "G", "U"})


def moltype_from_resname(resname: str | None) -> str | None:
    """Map a PDB residue name to "protein"/"dna"/"rna", else None (ligand / water /
    modified / unknown). Used by the reference-PDB reader and as the fallback for an
    adapter whose framework exposes no molecule-type annotation."""
    if resname is None:
        return None
    r = resname.strip().upper()
    if r in PROTEIN_RESIDUES:
        return "protein"
    if r in DNA_RESIDUES:
        return "dna"
    if r in RNA_RESIDUES:
        return "rna"
    return None
