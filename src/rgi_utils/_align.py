"""Residue-sequence alignment for RMSD-restraint correspondence (PyMOL ``align``-like).

Pairs the residues of a moving (target) chain with a reference chain by a semi-global
(free end-gap) affine-gap Needleman-Wunsch alignment, so a homolog reference with
substitutions and insertions/deletions still maps onto the prediction. Protein chains
score with BLOSUM62; nucleic-acid (and any other) chains score by residue-name identity
(match/mismatch) -- this also dodges the 1-letter collision between nucleotides
(A/C/G/T/U) and amino acids (Ala/Cys/Gly/Thr). numpy only -- no biopython.

The public entry point is :func:`pair_residues`. It takes the two chains' ordered
``(resid, resname)`` lists plus the polymer ``mol_type`` and returns the aligned
``(target_resid, ref_resid)`` pairs (gap columns dropped), which the RMSD restraint
then uses to pair atoms by name within each corresponding residue.
"""

from __future__ import annotations

import numpy as np

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

# BLOSUM62, 20 standard residues; parsed into a {(a,b): score} dict at import.
_BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_RAW = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
_BLOSUM62: dict[tuple[str, str], int] = {}
_rows = [r.split() for r in _BLOSUM62_RAW.strip().splitlines()]
for _i, _a in enumerate(_BLOSUM62_ORDER):
    for _j, _b in enumerate(_BLOSUM62_ORDER):
        _BLOSUM62[(_a, _b)] = int(_rows[_i][_j])

# affine gap (BLAST BLOSUM62 defaults) for proteins; identity match/mismatch for NA.
GAP_OPEN = -11
GAP_EXTEND = -1
NA_MATCH = 5
NA_MISMATCH = -4
_NEG = -1.0e9


def _blosum_score(a: str, b: str) -> int:
    """BLOSUM62 score; unknown ('X') residues score as match(+1)/mismatch(-1)."""
    s = _BLOSUM62.get((a, b))
    if s is not None:
        return s
    return 1 if a == b else -1


def _identity_score(a: str, b: str) -> int:
    return NA_MATCH if a == b else NA_MISMATCH


def _tokens(residues: list[tuple[int, str]], mol_type: str | None):
    """(resid, resname) list -> (tokens, resids, score_fn) for the given polymer type.
    Proteins use 1-letter codes + BLOSUM62; everything else compares resnames by
    identity (so nucleotide single letters never collide with amino-acid ones)."""
    resids = [r for r, _ in residues]
    if mol_type == "protein":
        toks = [THREE_TO_ONE.get((rn or "").upper(), "X") for _, rn in residues]
        return toks, resids, _blosum_score
    toks = [(rn or "").upper() for _, rn in residues]
    return toks, resids, _identity_score


def _align_tokens(s1, s2, score_fn):
    """Semi-global (free end-gap) affine-gap Needleman-Wunsch over two token lists.
    Returns the list of aligned index pairs (i, j) for true match columns (both
    sides non-gap), 0-based into s1 / s2."""
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return []
    # M = end in match/mismatch; X = end in gap in s2 (consume s1); Y = gap in s1.
    M = np.full((n + 1, m + 1), _NEG)
    X = np.full((n + 1, m + 1), _NEG)
    Y = np.full((n + 1, m + 1), _NEG)
    M[0, 0] = 0.0
    X[1:, 0] = 0.0  # free leading gap in s2 (skip a target prefix at no cost)
    Y[0, 1:] = 0.0  # free leading gap in s1 (skip a ref prefix at no cost)
    for i in range(1, n + 1):
        si = s1[i - 1]
        for j in range(1, m + 1):
            diag = max(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1])
            M[i, j] = diag + score_fn(si, s2[j - 1])
            X[i, j] = max(M[i - 1, j] + GAP_OPEN, X[i - 1, j] + GAP_EXTEND)
            Y[i, j] = max(M[i, j - 1] + GAP_OPEN, Y[i, j - 1] + GAP_EXTEND)

    # free trailing gaps: best score sits anywhere in the last row or last column.
    best, bi, bj, bmat = _NEG, n, m, "M"
    for j in range(m + 1):
        for mat, name in ((M, "M"), (X, "X"), (Y, "Y")):
            if mat[n, j] > best:
                best, bi, bj, bmat = mat[n, j], n, j, name
    for i in range(n + 1):
        for mat, name in ((M, "M"), (X, "X"), (Y, "Y")):
            if mat[i, m] > best:
                best, bi, bj, bmat = mat[i, m], i, m, name

    pairs = []
    i, j, mat = bi, bj, bmat
    while i > 0 and j > 0:
        if mat == "M":
            pairs.append((i - 1, j - 1))
            diag = max(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1])
            i, j = i - 1, j - 1
            mat = "M" if diag == M[i, j] else ("X" if diag == X[i, j] else "Y")
        elif mat == "X":  # gap in s2: came from M(open) or X(extend), step i
            mat = "M" if X[i, j] == M[i - 1, j] + GAP_OPEN else "X"
            i -= 1
        else:  # Y: gap in s1, step j
            mat = "M" if Y[i, j] == M[i, j - 1] + GAP_OPEN else "Y"
            j -= 1
    pairs.reverse()
    return pairs


def pair_residues(
    target_residues: list[tuple[int, str]],
    ref_residues: list[tuple[int, str]],
    mol_type: str | None,
) -> list[tuple[int, int]]:
    """Align one chain's target residues to the reference and return the matched
    ``(target_resid, ref_resid)`` pairs (gap columns dropped). ``*_residues`` are the
    chain's residues in sequence order as ``(resid, resname)``; ``resid`` is the
    per-chain ordinal used everywhere else in rgi_utils."""
    if not target_residues or not ref_residues:
        return []
    t_tok, t_ids, score_fn = _tokens(target_residues, mol_type)
    r_tok, r_ids, _ = _tokens(ref_residues, mol_type)
    return [(t_ids[i], r_ids[j]) for i, j in _align_tokens(t_tok, r_tok, score_fn)]
