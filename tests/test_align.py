"""Unit tests for the residue-sequence aligner (rgi_utils._align)."""

from rgi_utils._align import (
    THREE_TO_ONE,
    pair_residues,
)

_ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}


def _prot(seq, start=1):
    """1-letter protein string -> [(resid, 3-letter resname)] in order."""
    return [(start + i, _ONE_TO_THREE[c]) for i, c in enumerate(seq)]


# (The old BLOSUM62-transcription spot check is gone: the matrix is no longer hand-
# transcribed -- _align now loads Biopython's canonical BLOSUM62 -- so there is nothing
# to guard against a bad copy. The protein-vs-nucleotide scoring behaviour is covered by
# the pair_residues cases below.)


def test_identical_sequence_pairs_one_to_one():
    res = _prot("MKLAV")
    assert pair_residues(res, res, "protein") == [(i, i) for i in range(1, 6)]


def test_substitutions_same_length_still_one_to_one():
    # homolog: same length, a few substitutions -> register unchanged
    tgt = _prot("MKLAV")
    ref = _prot("MRLAI")  # K2R, V5I substitutions
    assert pair_residues(tgt, ref, "protein") == [(i, i) for i in range(1, 6)]


# Indel tests use full-length-like sequences: an affine gap (-11) only wins over a
# register shift when enough downstream matches penalize the shift, which is always
# true for real chains but not for 5-residue toys (where a free end-gap shift is
# cheaper than one internal gap).
_BASE20 = "MKLAVDEFGHIKLMNPQRST"


def test_insertion_in_ref_recovers_register():
    # ref has an extra residue (W) inserted after position 10 -> gap column; target
    # residues 11..20 map across it to ref 12..21.
    tgt = _prot(_BASE20)  # resids 1..20
    ref = _prot(_BASE20[:10] + "W" + _BASE20[10:])  # resids 1..21, W at ref-resid 11
    expected = [(i, i) for i in range(1, 11)] + [(i, i + 1) for i in range(11, 21)]
    assert pair_residues(tgt, ref, "protein") == expected


def test_deletion_in_ref_skips_unmatched_target():
    # target has an extra residue (W) the ref lacks -> that target residue is dropped.
    tgt = _prot(_BASE20[:10] + "W" + _BASE20[10:])  # resids 1..21, W at target-resid 11
    ref = _prot(_BASE20)  # resids 1..20
    expected = [(i, i) for i in range(1, 11)] + [(i, i - 1) for i in range(12, 22)]
    assert pair_residues(tgt, ref, "protein") == expected


def test_free_end_gaps_fragment_reference():
    # a short reference aligns to the interior of a longer target at no end-gap cost
    tgt = _prot("GGMKLAVGG")  # resids 1..9
    ref = _prot("MKLAV")  # resids 1..5, should land on target 3..7
    assert pair_residues(tgt, ref, "protein") == [(i + 2, i) for i in range(1, 6)]


def test_nucleotide_identity_no_aa_collision():
    # DNA chain: scored by resname identity, not BLOSUM (DA must not score as Ala).
    # resid 4 (DT) is deleted in the ref; 4 downstream matches outweigh the gap.
    tnames = ["DA", "DC", "DG", "DT", "DA", "DG", "DT", "DC"]
    rnames = ["DA", "DC", "DG", "DA", "DG", "DT", "DC"]
    tgt = [(i + 1, n) for i, n in enumerate(tnames)]
    ref = [(i + 1, n) for i, n in enumerate(rnames)]
    expected = [(1, 1), (2, 2), (3, 3), (5, 4), (6, 5), (7, 6), (8, 7)]
    assert pair_residues(tgt, ref, "dna") == expected


def test_empty_inputs():
    assert pair_residues([], _prot("MK"), "protein") == []
    assert pair_residues(_prot("MK"), [], "protein") == []
