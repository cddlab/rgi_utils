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

POLYMER_TYPES = ("protein", "dna", "rna")

# Framework molecule-type enum -> normalized string, for adapters whose enum uses
# this ordering: boltz ``const.chain_types`` and esmfold2 ``constants`` both number
# PROTEIN=0, DNA=1, RNA=2, NONPOLYMER=3. NOTE chai/openfold use RNA=1/DNA=2 and do
# NOT use this table (they leave ``mol_type=None``), so this is the one shared order.
MOLTYPE_BY_ID = {0: "protein", 1: "dna", 2: "rna", 3: "ligand"}


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


def polymer_type(mol_type: str | None, resname: str | None) -> str | None:
    """Effective polymer type of an atom, "protein"/"dna"/"rna" or None. Shared by the
    backbone/sidechain selectors (selection.py) and RMSD align pairing
    (rmsd_restr_data.py) so they classify polymers IDENTICALLY across tools.

    Three-way, order matters:
    1. an explicit polymer ``mol_type`` (boltz/esm/AF3 set it) is trusted as-is;
    2. any OTHER explicitly-set ``mol_type`` (e.g. "ligand") returns None -- a typed
       non-polymer is never re-derived from its residue name;
    3. only when ``mol_type`` is absent (chai/of3/protenix don't set it) is the type
       derived from the residue name via ``moltype_from_resname``.

    Consequence (an accepted cross-tool divergence, NOT a bug): a MODIFIED residue
    such as MSE gets mol_type="protein" from the framework enum in boltz/esm/AF3 but
    is unannotated + resname="MSE" -> None in chai/of3/protenix. This mirrors the
    `protein`/`dna`/`rna` selectors' deliberate "never silently widen protein" rule."""
    if mol_type in POLYMER_TYPES:
        return mol_type
    if mol_type is not None:
        return None
    return moltype_from_resname(resname) if resname else None
