"""PDB / mmCIF reader for RMSD reference structures, backed by gemmi.

rgi_utils core stays numpy-only at import (``import rgi_utils`` needs numpy only), so
gemmi is imported **lazily** inside the reader functions -- only the RMSD restraint, when
it actually resolves a ``ref_pdb`` / ``ref_cif``, pulls it in. Both readers emit the SAME
``PdbAtom`` list for the same structure (so ``ref_pdb`` / ``ref_cif`` are interchangeable
downstream): the format-specific extraction normalises each atom to a
``(group_pdb, chain, res_key, name, res_name, element, x, y, z)`` row, and the shared
``_build_atoms`` applies the per-chain ordinal / index convention once.

Why two gemmi entry points (not just ``gemmi.read_structure`` for both):
- **PDB** goes through ``gemmi.read_structure`` -- fixed-column parsing + ``het_flag``
  (ATOM/HETATM) is exactly what we need and comes for free.
- **mmCIF** reads the ``_atom_site`` loop **directly via ``gemmi.cif``**. ``read_structure``
  builds its model hierarchy from ``auth_asym_id`` and yields **zero atoms** for a
  label-only coordinate block (no ``auth_*`` columns), silently dropping the RMSD
  reference. Reading the loop ourselves keeps gemmi's robust, spec-correct CIF tokeniser
  (quoting, embedded spaces, primed atom names like ``O5'``, multiline) while preserving
  the auth-preferred / label-fallback column selection the downstream matching relies on.

Exposed on ``PdbAtom`` is just what the atom-selection DSL (``selection.py``) needs:

- ``chain``  -- chain id (PDB chain / mmCIF ``auth_asym_id``, label fallback).
- ``resid``  -- **per-chain 1-based ordinal** (resets at each chain), matching
  ``AtomRecord.resid`` (NOT the author ``resSeq``): a polymer (ATOM) residue's atoms
  share one ordinal, while a ligand (HETATM) atom gets its own ordinal (the adapters
  tokenise a ligand one atom per token). So one ``atom_selection`` string selects the
  same atoms in the reference and in the diffusion structure, for polymers and ligands
  alike (a ligand reference must list atoms in the tool's ligand-atom order).
- ``index``  -- 0-based row in the parsed atom list (for ``index ...`` selections).
- ``element`` + ``x/y/z`` coordinates.

Only the first model is kept (both readers): ``st[0]`` for PDB, the first
``pdbx_PDB_model_num`` value for mmCIF. (The previous hand-written PDB reader concatenated
ALL models; first-model-only is the intended, safer behaviour -- multiple models would
otherwise collide into duplicate ``(chain, resid, name)`` keys downstream.)
"""

from __future__ import annotations

from dataclasses import dataclass

from rgi_utils._moltype import moltype_from_resname


@dataclass
class PdbAtom:
    chain: str
    resid: int  # per-chain 1-based ordinal (resets per chain)
    index: int  # 0-based row in the parsed atom list
    name: str  # atom name, for identity-based RMSD pairing
    element: str
    x: float
    y: float
    z: float
    mol_type: str | None = None  # "protein"/"dna"/"rna"/None from res_name; powers the
    # protein/dna/rna selectors on the RMSD reference side.
    res_name: str | None = None  # 3-letter residue/CCD code; feeds the sequence aligner
    # for PyMOL-align-like (pairing="align") RMSD correspondence.


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

    LIMITATION (context-free numbering -- no entity info used): a HETATM that sits INSIDE a
    polymer chain (a modified residue such as MSE, or a bound ion/water) is per-atom-
    incremented like a ligand, so it shifts the per-chain ordinal of every following ATOM
    residue. This is harmless under the default ``pairing: align`` (sequence-position
    pairing absorbs the shift), but breaks ``pairing: identity`` and can drop the modified
    residue from the aligned sequence. We deliberately derive ``mol_type`` and the ordinal
    from the record type + resname only (not gemmi's ``entity_type``) to keep the reference
    numbering identical to the tools' adapters; use ``pairing: align`` (the default) for
    references with in-chain HETATM."""
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


def _iter_pdb_rows(model):
    """Yield one normalised row per atom of the first PDB model, keeping only the primary
    alternate location (blank / ``A``). ``chain.name`` is the chain id; ``res.het_flag`` is
    ``'A'`` (ATOM) / ``'H'`` (HETATM); ``res.seqid`` supplies the residue-boundary key."""
    for chain in model:
        for res in chain:
            group_pdb = "HETATM" if res.het_flag == "H" else "ATOM"
            res_key = (res.seqid.num, res.seqid.icode)
            for atom in res:
                # gemmi uses '\x00' for a blank altloc; keep blank / "A" only.
                if atom.altloc not in ("\x00", "", "A"):
                    continue
                yield (
                    group_pdb,
                    (chain.name or " "),
                    res_key,
                    atom.name,
                    res.name,
                    atom.element.name,
                    atom.pos.x,
                    atom.pos.y,
                    atom.pos.z,
                )


# mmCIF _atom_site fields, preferring the auth_* columns (what a PDB read exposes, so the
# two formats agree and match the tools' adapters) with a label_* fallback.
_CIF_FIELD_PREFERENCE = {
    "chain": ("auth_asym_id", "label_asym_id"),
    "seq": ("auth_seq_id", "label_seq_id"),
    "name": ("auth_atom_id", "label_atom_id"),
    "comp": ("auth_comp_id", "label_comp_id"),
}


def _iter_cif_rows(block, path: str):
    """Yield one normalised row per ``_atom_site`` data row of an mmCIF block, read through
    gemmi's CIF tokeniser (so quoting / embedded spaces / primed names are handled
    correctly). Columns are selected by name with the auth_* -> label_* preference; ``.`` /
    ``?`` (mmCIF null/unknown) read as empty; only the first ``pdbx_PDB_model_num`` and the
    primary altloc (blank / ``A``) are kept. Raises ValueError if the block has no
    ``_atom_site`` coordinate loop."""
    import gemmi
    from gemmi import cif

    def column(tag):
        col = block.find_loop("_atom_site." + tag)
        return list(col) if len(col) else None

    def value(raw):
        if raw is None:
            return ""
        s = cif.as_string(raw)  # strips CIF quoting
        return "" if s in (".", "?") else s

    def preferred(field):
        columns = [
            col
            for tag in _CIF_FIELD_PREFERENCE[field]
            if (col := column(tag)) is not None
        ]
        if not columns:
            return None
        # Some writers provide auth_* columns but leave individual values unknown.
        return [
            next((raw for raw in row if value(raw)), row[0]) for row in zip(*columns)
        ]

    x = column("Cartn_x")
    y = column("Cartn_y")
    z = column("Cartn_z")
    chain = preferred("chain")
    seq = preferred("seq")
    name = preferred("name")
    comp = preferred("comp")
    # a coordinate loop must at least have positions + a chain id; else there is no
    # _atom_site loop to read (fail loudly -- a missing RMSD reference is the worst
    # silent failure).
    if x is None or y is None or z is None or chain is None:
        raise ValueError(f"rmsd ref_cif has no _atom_site loop: {path!r}")
    missing = [
        field
        for field, col in (("seq", seq), ("name", name), ("comp", comp))
        if col is None
    ]
    if missing:
        tags = ["/".join(_CIF_FIELD_PREFERENCE[field]) for field in missing]
        raise ValueError(
            f"rmsd ref_cif missing required _atom_site column(s) {tags}: {path!r}"
        )
    group = column("group_PDB")
    element = column("type_symbol")
    altloc = column("label_alt_id")
    icode = column("pdbx_PDB_ins_code")
    model = column("pdbx_PDB_model_num")

    first_model = None
    for i in range(len(x)):
        if model is not None:
            m = value(model[i])
            if m:
                if first_model is None:
                    first_model = m
                elif m != first_model:
                    continue  # a trailing model (e.g. NMR) -> ignore
        alt = value(altloc[i]) if altloc is not None else ""
        if alt not in ("", "A"):  # keep only the primary alternate location
            continue
        grp = (
            "HETATM"
            if group is not None and value(group[i]).upper() == "HETATM"
            else "ATOM"
        )
        el = gemmi.Element(value(element[i])).name if element is not None else ""
        try:
            xi = float(value(x[i]))
            yi = float(value(y[i]))
            zi = float(value(z[i]))
        except ValueError:
            continue  # malformed coordinate columns -> skip the record
        yield (
            grp,
            value(chain[i]) or " ",
            (value(seq[i]), value(icode[i]) if icode is not None else ""),
            value(name[i]),
            value(comp[i]),
            el,
            xi,
            yi,
            zi,
        )


def read_pdb_atoms(path: str) -> list[PdbAtom]:
    """Parse ATOM/HETATM records from a PDB ``path`` (first model only). Raises ValueError
    on a missing / unreadable file or one with no atoms (the RMSD restraint must fail
    loudly)."""
    import gemmi

    try:
        st = gemmi.read_structure(path, format=gemmi.CoorFormat.Pdb)
    except Exception as exc:  # gemmi raises RuntimeError/ValueError/IOError variants
        raise ValueError(f"rmsd ref_pdb could not be read: {path!r} ({exc})") from exc
    if len(st) == 0:
        raise ValueError(f"rmsd ref_pdb has no ATOM/HETATM records: {path!r}")
    return _build_atoms(_iter_pdb_rows(st[0]), source_label="ref_pdb", path=path)


def read_cif_atoms(path: str) -> list[PdbAtom]:
    """Parse ``_atom_site`` records from an mmCIF ``path`` into the SAME ``PdbAtom`` list a
    PDB of the same structure yields (so ``ref_cif`` and ``ref_pdb`` are interchangeable).
    Reads the loop through gemmi's CIF tokeniser. Raises ValueError on a missing /
    unreadable file, a file with no ``_atom_site`` loop, or one with no atoms (fail
    loudly)."""
    from gemmi import cif

    try:
        block = cif.read(path).sole_block()
    except Exception as exc:
        raise ValueError(f"rmsd ref_cif could not be read: {path!r} ({exc})") from exc
    return _build_atoms(_iter_cif_rows(block, path), source_label="ref_cif", path=path)
