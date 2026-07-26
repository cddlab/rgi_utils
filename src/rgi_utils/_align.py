"""Residue-sequence alignment for RMSD-restraint correspondence (PyMOL ``align``-like).

Pairs the residues of a moving (target) chain with a reference chain by a semi-global
(free end-gap) affine-gap alignment, so a homolog reference with substitutions and
insertions/deletions still maps onto the prediction. Protein chains score with BLOSUM62;
nucleic-acid (and any other) chains score by residue-name identity (match/mismatch) --
this also dodges the 1-letter collision between nucleotides (A/C/G/T/U) and amino acids
(Ala/Cys/Gly/Thr).

The alignment itself is delegated to Biopython's :class:`Bio.Align.PairwiseAligner`
(global mode, free end gaps, affine gaps ``open=-11`` / ``extend=-1``, protein = BLOSUM62,
NA/other = ``+5`` / ``-4`` identity). ``Bio`` is imported **lazily** inside
:func:`pair_residues` so ``import rgi_utils`` stays numpy-only.

The public entry point is :func:`pair_residues`. It takes the two chains' ordered
``(resid, resname)`` lists plus the polymer ``mol_type`` and returns the aligned
``(target_resid, ref_resid)`` pairs (gap columns dropped), which the RMSD restraint then
uses to pair atoms by name within each corresponding residue.
"""

from __future__ import annotations

# 3-letter -> 1-letter for the 20 standard amino acids; anything else -> "X".
THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

# affine gap (BLAST BLOSUM62 defaults) for proteins; identity match/mismatch for NA.
# Biopython's gap model matches the old hand-rolled one: a gap of length L costs
# ``open + (L-1)*extend``, so these values transfer verbatim.
GAP_OPEN = -11
GAP_EXTEND = -1
NA_MATCH = 5
NA_MISMATCH = -4


def _configure(aligner):
    """Shared affine-gap + free-end-gap (semi-global) settings for both scoring schemes."""
    aligner.mode = "global"
    aligner.open_gap_score = GAP_OPEN
    aligner.extend_gap_score = GAP_EXTEND
    # free leading/trailing gaps on both sequences (skip a target/ref overhang at no cost).
    # ``end_gap_score`` sets all four end-gap scores at once and is stable across the
    # supported Biopython range (1.84-1.87); the granular ``*_end_gap_score`` names are
    # deprecated in 1.87.
    aligner.end_gap_score = 0.0
    return aligner


def _protein_seq(residues: list[tuple[int, str]]) -> str:
    """(resid, resname) list -> 1-letter protein sequence (unknown residue -> 'X')."""
    return "".join(THREE_TO_ONE.get((rn or "").upper(), "X") for _, rn in residues)


def _encode_identity(target_residues, ref_residues):
    """Encode both chains' resnames as a SHARED single-char alphabet so identity scoring
    works through PairwiseAligner's per-character comparison -- a multi-letter resname
    (e.g. 'DA') would otherwise be compared letter-by-letter. Same resname -> same char on
    both sides. Uses private-use BMP code points to avoid clashing with anything."""
    encoding: dict[str, str] = {}

    def enc(residues):
        out = []
        for _, rn in residues:
            key = (rn or "").upper()
            if key not in encoding:
                encoding[key] = chr(0xE000 + len(encoding))  # private-use area
            out.append(encoding[key])
        return "".join(out)

    return enc(target_residues), enc(ref_residues)


def _match_pairs(alignment) -> list[tuple[int, int]]:
    """Aligned index pairs (i, j) for the true match columns (both sides non-gap), 0-based
    into the target / query sequences. ``alignment.aligned`` gives the co-advancing blocks
    (no gap within a block), so every position inside a block is a match column."""
    target_blocks, query_blocks = alignment.aligned
    pairs = []
    for (t0, t1), (r0, r1) in zip(target_blocks, query_blocks):
        for k in range(t1 - t0):
            pairs.append((t0 + k, r0 + k))
    return pairs


def pair_residues(
    target_residues: list[tuple[int, str]],
    ref_residues: list[tuple[int, str]],
    mol_type: str | None,
) -> list[tuple[int, int]]:
    """Align one chain's target residues to the reference and return the matched
    ``(target_resid, ref_resid)`` pairs (gap columns dropped). ``*_residues`` are the
    chain's residues in sequence order as ``(resid, resname)``; ``resid`` is the per-chain
    ordinal used everywhere else in rgi_utils."""
    if not target_residues or not ref_residues:
        return []
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = _configure(PairwiseAligner())
    if mol_type == "protein":
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        t_seq = _protein_seq(target_residues)
        r_seq = _protein_seq(ref_residues)
    else:
        aligner.match_score = NA_MATCH
        aligner.mismatch_score = NA_MISMATCH
        t_seq, r_seq = _encode_identity(target_residues, ref_residues)

    t_ids = [r for r, _ in target_residues]
    r_ids = [r for r, _ in ref_residues]
    alignment = aligner.align(t_seq, r_seq)[0]
    return [(t_ids[i], r_ids[j]) for i, j in _match_pairs(alignment)]
