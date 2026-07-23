"""Minimal, dependency-free PDB / mmCIF reader for RMSD reference structures.

rgi_utils core stays numpy-only (no gemmi/biotite at import), so the RMSD
restraint parses the reference's ``ATOM``/``HETATM`` records directly — from a PDB
(``read_pdb_atoms``, ``ref_pdb``) or an mmCIF (``read_cif_atoms``, ``ref_cif``). Both
emit the SAME ``PdbAtom`` list for the same structure (so ``ref_pdb`` / ``ref_cif`` are
interchangeable downstream): the format-specific parse normalises each record to a
``(group_pdb, chain, res_key, name, res_name, element, x, y, z)`` row, and the shared
``_build_atoms`` applies the per-chain ordinal / index convention once. We expose just
what the atom-selection DSL (``selection.py``) needs:

- ``chain``  — chain id (PDB column 22; mmCIF ``auth_asym_id``, label fallback)
- ``resid``  — **per-chain 1-based ordinal** (resets at each chain), matching
  ``AtomRecord.resid`` (NOT the author ``resSeq``): a polymer (ATOM) residue's atoms
  share one ordinal, while a ligand (HETATM) atom gets its own ordinal (the adapters
  tokenise a ligand one atom per token). So one ``atom_selection`` string selects the
  same atoms in the reference and in the diffusion structure, for polymers and
  ligands alike (a ligand reference must list atoms in the tool's ligand-atom order).
- ``index``  — 0-based row in the parsed atom list (for ``index ...`` selections).
- ``element`` + ``x/y/z`` coordinates.

PDB ATOM/HETATM columns are fixed-width (PDB 3.30): record(1-6), serial(7-11),
name(13-16), altLoc(17), resName(18-20), chainID(22), resSeq(23-26), iCode(27),
x(31-38), y(39-46), z(47-54), element(77-78) — here in 0-based Python slices.
mmCIF ``_atom_site`` columns are self-describing (order varies per file), so the reader
maps each ``_atom_site.<field>`` tag to its position and pulls fields by name. Data rows are
tokenised whitespace-first but QUOTE-AWARE (``_split_cif_row``): a ``'...'`` / ``"..."`` value
with an embedded space (possible in any column, even ones we never read) stays ONE token, so
it doesn't inflate the token count and truncate the parse. Each token then has at most ONE
outer quote pair removed: a gemmi-written name like ``"O5'"`` (gemmi double-quotes a value
containing ``'``) reads back as ``O5'``, while an already-unquoted ``O5'`` is left intact. A
blanket ``str.strip("'\"")`` would corrupt the unquoted form (``O5'`` → ``O5``) and ``shlex``
mis-tokenises the bare apostrophe — hence the precise single-outer-pair rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from rgi_utils._moltype import moltype_from_resname


@dataclass
class PdbAtom:
    chain: str
    resid: int  # per-chain 1-based ordinal (resets per chain)
    index: int  # 0-based row in the parsed atom list
    name: str  # atom name (cols 13-16), for identity-based RMSD pairing
    element: str
    x: float
    y: float
    z: float
    mol_type: str | None = None  # "protein"/"dna"/"rna"/None from res_name (cols
    # 18-20); powers the protein/dna/rna selectors on the RMSD reference side.
    res_name: str | None = None  # 3-letter residue/CCD code (cols 18-20); feeds the
    # sequence aligner for PyMOL-align-like (pairing="align") RMSD correspondence.


def _build_atoms(rows, *, source_label: str, path: str) -> list[PdbAtom]:
    """Assemble ``PdbAtom`` records from normalised ``(group_pdb, chain, res_key, name,
    res_name, element, x, y, z)`` rows, applying the per-chain 1-based ordinal + 0-based
    index convention ONCE for both readers (so a PDB and an mmCIF of the same structure
    yield identical lists). ``group_pdb`` is ``"ATOM"`` / ``"HETATM"``; ``res_key`` is an
    opaque hashable (``(resSeq, iCode)``) used only for residue-boundary detection.

    The ordinal matches the ``AtomRecord.resid`` convention every adapter uses: a polymer
    (ATOM) residue's atoms share one ordinal, while a ligand/non-polymer (HETATM) atom
    gets its OWN ordinal (the adapters tokenise a ligand one atom per token). Aligning the
    readers here is what lets RMSD identity-pair ligand atoms. Raises ValueError if no
    rows survived (the RMSD restraint must fail loudly).

    LIMITATION (context-free reader — no entity info): a HETATM that sits INSIDE a polymer
    chain (a modified residue such as MSE, or a bound ion/water) is per-atom-incremented
    like a ligand, so it shifts the per-chain ordinal of every following ATOM residue. This
    is harmless under the default ``pairing: align`` (sequence-position pairing absorbs the
    shift), but breaks ``pairing: identity`` and can drop the modified residue from the
    aligned sequence. Distinguishing a polymer-internal HETATM from a standalone ligand
    needs entity typing this reader deliberately does not have; use ``pairing: align`` (the
    default) for references with in-chain HETATM."""
    atoms: list[PdbAtom] = []
    chain_ord: dict[str, int] = {}
    last_key: dict[str, tuple | None] = {}
    for group_pdb, chain, res_key, name, res_name, element, x, y, z in rows:
        if chain not in chain_ord:
            chain_ord[chain] = 0
            last_key[chain] = None
        if group_pdb == "HETATM" or last_key[chain] != res_key:
            chain_ord[chain] += 1
            last_key[chain] = res_key
        atoms.append(
            PdbAtom(
                chain=chain,
                resid=chain_ord[chain],
                index=len(atoms),
                name=name,
                element=element,
                x=x,
                y=y,
                z=z,
                mol_type=moltype_from_resname(res_name),
                res_name=res_name,
            )
        )
    if not atoms:
        raise ValueError(f"rmsd {source_label} has no ATOM/HETATM records: {path!r}")
    return atoms


def _iter_pdb_rows(lines):
    """Yield one normalised row per ATOM/HETATM line, keeping only the primary alternate
    location and skipping records with malformed coordinate columns."""
    for ln in lines:
        rec = ln[0:6]
        if rec not in ("ATOM  ", "HETATM"):
            continue
        altloc = ln[16] if len(ln) > 16 else " "
        if altloc not in (" ", "A"):
            continue
        chain = (ln[21].strip() if len(ln) > 21 else "") or " "
        res_seq = ln[22:26].strip()
        icode = ln[26] if len(ln) > 26 else " "
        name = ln[12:16].strip() if len(ln) > 15 else ""
        res_name = ln[17:20].strip() if len(ln) > 19 else ""
        try:
            x = float(ln[30:38])
            y = float(ln[38:46])
            z = float(ln[46:54])
        except ValueError:
            continue  # malformed coordinate columns -> skip the record
        element = ln[76:78].strip() if len(ln) >= 78 else ""
        group_pdb = "HETATM" if rec == "HETATM" else "ATOM"
        yield (group_pdb, chain, (res_seq, icode), name, res_name, element, x, y, z)


def read_pdb_atoms(path: str) -> list[PdbAtom]:
    """Parse ATOM/HETATM records from a PDB ``path``. Raises ValueError on a missing /
    unreadable file or one with no atoms (the RMSD restraint must fail loudly)."""
    try:
        with open(path) as fh:
            lines = fh.readlines()
    except OSError as exc:
        raise ValueError(f"rmsd ref_pdb could not be read: {path!r} ({exc})") from exc
    return _build_atoms(_iter_pdb_rows(lines), source_label="ref_pdb", path=path)


# mmCIF _atom_site fields, preferring the auth_* columns (what gemmi writes into PDB
# columns, so an mmCIF read matches a PDB of the same structure) with a label_* fallback.
_CIF_FIELD_PREFERENCE = {
    "chain": ("auth_asym_id", "label_asym_id"),
    "res_seq": ("auth_seq_id", "label_seq_id"),
    "name": ("auth_atom_id", "label_atom_id"),
    "res_name": ("auth_comp_id", "label_comp_id"),
}


def _cif_unquote(v: str) -> str:
    """Strip ONE outer mmCIF quote pair (matching ``'`` or ``"``) if present. gemmi
    double-quotes a value containing an apostrophe, so a primed atom name lands as
    ``"O5'"`` — this returns ``O5'`` while leaving an unquoted ``O5'`` (or any number)
    untouched (a blanket strip of all quotes would corrupt the unquoted form)."""
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def _split_cif_row(raw: str) -> list[str]:
    """Whitespace-tokenise one mmCIF ``_atom_site`` data row, honouring ``'...'`` / ``"..."``
    quoting so a value with an EMBEDDED SPACE (in any column, even ones we never read) stays
    a single token instead of splitting into several — which used to inflate the token count
    and make the strict ``len(toks) == ncol`` check treat the row as the block terminator,
    silently truncating the parse. mmCIF rule: a quote opens a value only at a token start,
    and the matching quote closes it only when followed by whitespace / end-of-line (so a
    mid-value apostrophe such as ``O5'`` is literal, not a close). The quote characters are
    KEPT in the returned token so the downstream ``_cif_unquote`` strips them exactly as
    before. Multiline (``;``-delimited) values are not handled — ``_atom_site`` rows never
    use them (and such a line is already caught as a terminator by the ``;`` prefix check)."""
    toks: list[str] = []
    i, n = 0, len(raw)
    while i < n:
        c = raw[i]
        if c.isspace():
            i += 1
            continue
        if c in ("'", '"'):
            start = i  # keep the opening quote
            i += 1
            while i < n and not (raw[i] == c and (i + 1 >= n or raw[i + 1].isspace())):
                i += 1
            i = min(i + 1, n)  # include the closing quote
            toks.append(raw[start:i])
        else:
            start = i
            while i < n and not raw[i].isspace():
                i += 1
            toks.append(raw[start:i])
    return toks


def _iter_cif_rows(text: str, path: str):
    """Yield one normalised row per ``_atom_site`` data line of the FIRST ``_atom_site``
    loop. Columns are resolved by name from the loop header (their order varies per file),
    data rows are whitespace-split, and ``.`` / ``?`` (mmCIF null/unknown) read as empty.
    Only the first ``pdbx_PDB_model_num`` is kept (drop trailing NMR models). Raises
    ValueError if the file has no ``_atom_site`` loop."""
    lines = text.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        if lines[i].strip() == "loop_":
            cols, j = [], i + 1
            while j < n and lines[j].lstrip().startswith("_"):
                cols.append(lines[j].strip())
                j += 1
            if cols and cols[0].startswith("_atom_site."):
                prefix = "_atom_site."
                colmap = {c[len(prefix) :]: k for k, c in enumerate(cols)}
                yield from _iter_cif_atom_site(lines, j, n, colmap, len(cols))
                return
            i = j  # a non-atom_site loop -> skip past its header and continue scanning
            continue
        i += 1
    raise ValueError(f"rmsd ref_cif has no _atom_site loop: {path!r}")


def _iter_cif_atom_site(lines, start, n, colmap, ncol):
    """Yield normalised rows from the ``_atom_site`` data block beginning at ``start``."""

    def fld(toks, name, default=""):
        k = colmap.get(name)
        if k is None or k >= len(toks):
            return default
        v = _cif_unquote(toks[k])
        return default if v in (".", "?") else v  # mmCIF null / unknown

    def first(toks, key):  # auth_* preferred, label_* fallback
        for name in _CIF_FIELD_PREFERENCE[key]:
            v = fld(toks, name)
            if v:
                return v
        return ""

    first_model = None
    for raw in lines[start:n]:
        s = raw.strip()
        # the data block ends at the next category / loop / comment (#) / save / blank.
        if not s or s == "loop_" or s.startswith(("_", "#", ";", "data_", "save_")):
            break
        toks = _split_cif_row(raw)
        # quote-aware tokeniser above keeps a quoted embedded-space value as ONE token, so a
        # token-count mismatch now genuinely means the block ended (the real terminators —
        # blank / loop_ / _category / # / ; / data_ / save_ — are already caught above) ->
        # stop rather than mis-parse.
        if len(toks) != ncol:
            break
        model = fld(toks, "pdbx_PDB_model_num")
        if model:
            if first_model is None:
                first_model = model
            elif model != first_model:
                continue  # a trailing model (e.g. NMR) -> ignore
        altloc = fld(toks, "label_alt_id")
        if altloc not in ("", "A"):  # keep only the primary alternate location
            continue
        group_pdb = "HETATM" if fld(toks, "group_PDB").upper() == "HETATM" else "ATOM"
        chain = first(toks, "chain") or " "
        name = first(toks, "name")
        res_name = first(toks, "res_name")
        res_key = (first(toks, "res_seq"), fld(toks, "pdbx_PDB_ins_code"))
        element = fld(toks, "type_symbol")
        try:
            x = float(fld(toks, "Cartn_x"))
            y = float(fld(toks, "Cartn_y"))
            z = float(fld(toks, "Cartn_z"))
        except ValueError:
            continue  # malformed coordinate columns -> skip the record
        yield (group_pdb, chain, res_key, name, res_name, element, x, y, z)


def read_cif_atoms(path: str) -> list[PdbAtom]:
    """Parse ``_atom_site`` records from an mmCIF ``path`` into the SAME ``PdbAtom`` list a
    PDB of the same structure yields (so ``ref_cif`` and ``ref_pdb`` are interchangeable).
    Dependency-free — no gemmi/biotite. Raises ValueError on a missing / unreadable file,
    a file with no ``_atom_site`` loop, or one with no atoms (fail loudly)."""
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError as exc:
        raise ValueError(f"rmsd ref_cif could not be read: {path!r} ({exc})") from exc
    return _build_atoms(_iter_cif_rows(text, path), source_label="ref_cif", path=path)
