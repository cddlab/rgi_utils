import logging
from collections.abc import Iterable
from typing import Dict, List, Protocol, Union, runtime_checkable

import numpy as np

from rgi_utils._atom_names import normalise_atom_name
from rgi_utils._moltype import polymer_type

logger = logging.getLogger(__name__)


@runtime_checkable
class SelectionNode(Protocol):
    def eval(self, mol: Dict[str, Union[str, int]]) -> bool: ...

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__


class Chain(SelectionNode):
    def __init__(self, names: List[str]):
        self.names = names

    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        chain = mol.get("chain")
        return isinstance(chain, str) and chain in self.names


class ResId(SelectionNode):
    def __init__(self, ids: Iterable[int]):
        self.ids = ids if isinstance(ids, range) else frozenset(ids)

    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        resid = mol.get("resid")
        # accept numpy ints too: isinstance(np.int64(5), int) is False, which would
        # otherwise make every atom fail to match if an adapter yields a numpy scalar
        return isinstance(resid, (int, np.integer)) and int(resid) in self.ids


class Index(SelectionNode):
    def __init__(self, indices: Iterable[int]):
        self.indices = indices if isinstance(indices, range) else frozenset(indices)

    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        index = mol.get("index")
        return isinstance(index, (int, np.integer)) and int(index) in self.indices


class MolType(SelectionNode):
    """Matches atoms by NORMALIZED molecule type ("protein"/"dna"/"rna").

    ``mol["mol_type"]`` is the adapter-normalized string. A None (water / unknown /
    ligand, or an adapter that supplies no type) never matches, so the selector
    cleanly EXCLUDES non-polymer and untyped atoms instead of guessing."""

    def __init__(self, kind: str):
        self.kind = kind

    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        return mol.get("mol_type") == self.kind


class Name(SelectionNode):
    """Matches atoms by atom name (e.g. "CA"), case-insensitively.

    Enables backbone/CA-only RMSD superposition (``name CA``) and picking out
    individual nucleic-acid atoms (``name C1'``, ``name O2*``) for e.g. a hydrogen-bond
    distance restraint. ``mol["name"]`` is the atom name supplied by the adapter
    (target side) or parsed from the PDB (reference side); a None (adapter that
    supplies no name) never matches. Both sides go through ``normalise_atom_name``, so
    prime spelling never decides a match."""

    def __init__(self, names: List[str]):
        self.names = [normalise_atom_name(n) for n in names]

    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        name = mol.get("name")
        return isinstance(name, str) and normalise_atom_name(name) in self.names


# MDTraj-like backbone atom names, by polymer type, matched case-folded against
# ``mol["name"]``. Protein = the peptide unit; nucleic = the full sugar-phosphate
# backbone, listing BOTH modern (OP1/OP2/OP3) and legacy (O1P/O2P/O3P) phosphate-oxygen
# names so a reference PDB written either way classifies the same. Primes ("O5'") are
# fine here: these sets match the raw atom name, NOT the alnum-only ``name`` parser.
_PROTEIN_BACKBONE = frozenset({"N", "CA", "C", "O", "OXT"})
_NUCLEIC_BACKBONE = frozenset(
    {
        "P",
        "OP1",
        "OP2",
        "OP3",
        "O1P",
        "O2P",
        "O3P",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "O2'",
    }
)


def _backbone_names(kind: str) -> frozenset:
    """Backbone atom-name set for a polymer ``kind`` ("protein"/"dna"/"rna")."""
    return _PROTEIN_BACKBONE if kind == "protein" else _NUCLEIC_BACKBONE


class Backbone(SelectionNode):
    """Matches POLYMER backbone atoms (MDTraj ``backbone``): protein N/CA/C/O(/OXT) or
    the nucleic sugar-phosphate. GATED on polymer type, so a ligand atom merely named
    "C"/"N"/"O"/"P" never matches (organic ligands are full of those). Polymer type is
    ``mol_type`` where the adapter sets it (boltz/esm/AF3), else derived from
    ``resname`` (chai/of3/protenix) -- so the candidate dict must carry ``resname``.
    A modified residue (e.g. MSE) counts as polymer only where the framework typed it;
    see ``polymer_type`` for that accepted cross-tool divergence."""

    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        kind = polymer_type(mol.get("mol_type"), mol.get("resname"))
        if kind is None:
            return False
        name = mol.get("name")
        return isinstance(name, str) and normalise_atom_name(name) in _backbone_names(
            kind
        )


class Sidechain(SelectionNode):
    """Matches POLYMER non-backbone atoms (MDTraj ``sidechain``): the complement of
    ``backbone`` WITHIN a polymer residue. Glycine therefore has no sidechain heavy
    atom, and ligand/water atoms never match (same polymer-type gating + ``resname``
    requirement as ``Backbone``). The backbone sets are heavy-atom, so a present
    backbone hydrogen would fall here -- harmless for the heavy-atom diffusion outputs
    this serves."""

    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        kind = polymer_type(mol.get("mol_type"), mol.get("resname"))
        if kind is None:
            return False
        name = mol.get("name")
        return isinstance(name, str) and normalise_atom_name(
            name
        ) not in _backbone_names(kind)


class Not(SelectionNode):
    def __init__(self, selection: SelectionNode):
        self.selection = selection

    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        return not self.selection.eval(mol)


class And(SelectionNode):
    def __init__(self, selections: List[SelectionNode]):
        self.selections = selections

    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        if not self.selections:
            return True
        return all(s.eval(mol) for s in self.selections)


class Or(SelectionNode):
    def __init__(self, selections: List[SelectionNode]):
        self.selections = selections

    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        if not self.selections:
            return False
        return any(s.eval(mol) for s in self.selections)


class Bracket(SelectionNode):
    def __init__(self, selection: SelectionNode):
        self.selection = selection

    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        return self.selection.eval(mol)


# --- Parser Error ---
class ParseError(ValueError):
    pass


class SelectionError(ValueError):
    """A hard, non-recoverable selection error (e.g. a descending ``resid`` range).

    Deliberately NOT a subclass of ``ParseError``: the recursive-descent parser uses
    ``except ParseError`` for backtracking (try the next alternative), so a ``ParseError``
    raised deep in ``_parse_numbers`` gets swallowed and reported as a misleading generic
    "Expected an atomic selection" at position 0. ``SelectionError`` propagates past every
    ``except ParseError`` to the top, where ``parse_selection`` turns it into the real
    message."""


# --- Parser Class ---
RESERVED_KEYWORDS = {
    "and",
    "or",
    "not",
    "to",
    "resid",
    "index",
    "chain",
    "name",
    "protein",
    "dna",
    "rna",
    "backbone",
    "sidechain",
}


class SelectionParser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def _peek(self) -> Union[str, None]:
        return self.text[self.pos] if self.pos < len(self.text) else None

    def _consume_char(self, char: str):
        if self._peek() == char:
            self.pos += 1
            return char
        raise ParseError(
            f"Expected '{char}' at position {self.pos}, got '{self._peek()}'"
        )

    def _consume_tag(self, tag: str):
        if self.text.startswith(tag, self.pos):
            # Refuse to match an operator/keyword that is merely the prefix of a
            # longer glued token (e.g. "not" in "notchain", "and" in "andresid"):
            # in a valid selection a tag is always followed by space / "(" / ")" /
            # end-of-string, never another alnum char. Without this the tag was
            # silently stripped, yielding a wrong selection (e.g. "notchain A" ->
            # Not(Chain(A))). The parser's backtracking recovers from this error.
            if tag.isalpha() and (
                self.pos + len(tag) < len(self.text)
                and self.text[self.pos + len(tag)].isalnum()
            ):
                raise ParseError(
                    f"'{tag}' is a prefix of a longer token at position {self.pos}"
                )
            self.pos += len(tag)
            return tag
        raise ParseError(f"Expected '{tag}' at position {self.pos}")

    def _skip_space0(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _skip_space1(self):
        start_pos = self.pos
        self._skip_space0()
        if self.pos == start_pos:
            raise ParseError(f"Expected one or more spaces at position {self.pos}")

    def _parse_alphanumeric1(self) -> str:
        start_pos = self.pos
        if self.pos < len(self.text) and self.text[self.pos].isalnum():
            self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isalnum():
                self.pos += 1
            return self.text[start_pos : self.pos]
        raise ParseError(f"Expected alphanumeric characters at position {self.pos}")

    def _parse_digit1(self) -> str:
        start_pos = self.pos
        if self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
            return self.text[start_pos : self.pos]
        raise ParseError(f"Expected digits at position {self.pos}")

    def _parse_usize(self) -> int:
        s = self._parse_digit1()
        try:
            return int(s)
        except ValueError:
            raise ParseError(f"Invalid unsigned integer: {s}")

    # Characters that may appear in an atom name on top of alphanumerics: the three
    # prime spellings (see normalise_atom_name). Only the `name` clause accepts them —
    # chain ids and the other identifier clauses stay alphanumeric.
    _ATOM_NAME_EXTRA = "'\"*"

    def _parse_atom_name1(self) -> str:
        start_pos = self.pos

        def allowed(char: str) -> bool:
            return char.isalnum() or char in self._ATOM_NAME_EXTRA

        if self.pos < len(self.text) and allowed(self.text[self.pos]):
            self.pos += 1
            while self.pos < len(self.text) and allowed(self.text[self.pos]):
                self.pos += 1
            return self.text[start_pos : self.pos]
        raise ParseError(f"Expected an atom name at position {self.pos}")

    def _parse_atom_name(self) -> str:
        """One atom name, with the same operator guards as ``_parse_identifier``.

        The guards are what stop the name LIST: `and`/`or`/`not` are ordinary
        alphanumeric words, so without rejecting them here `name CA and resid 5` would
        swallow the operator as a third atom name.
        """
        start = self.pos
        name = self._parse_atom_name1()
        if name in {"and", "or", "not", "to"}:
            raise ParseError(
                f"Identifier cannot be a reserved keyword: '{name}' at position {start}"
            )
        for op in ("and", "or", "not", "to"):
            if name.startswith(op) and len(name) > len(op):
                raise ParseError(
                    f"'{op}' is glued to a longer token '{name}' at position"
                    f" {start} (add a space, e.g. '{op} {name[len(op) :]}')"
                )
        return name

    def _parse_list_of_atom_names(self) -> List[str]:
        names = [self._parse_atom_name()]
        while True:
            saved_pos = self.pos
            try:
                self._skip_space1()
                names.append(self._parse_atom_name())
            except ParseError:
                self.pos = saved_pos
                break
        return names

    def _parse_identifier(self) -> str:
        start = self.pos
        identifier = self._parse_alphanumeric1()
        if identifier in RESERVED_KEYWORDS - {"resid", "index"}:
            if identifier in {"and", "or", "not", "to"}:
                raise ParseError(
                    f"Identifier cannot be a reserved keyword:"
                    f" '{identifier}' at position {start}"
                )
        # Reject an identifier whose prefix is an operator glued to a longer token
        # (a missing space, e.g. "andresid", "orchain"): without this it is silently
        # swallowed as a chain name and the operator/clause is dropped, yielding a
        # WRONG selection with no error. Mirrors the glued-token guard in
        # _consume_tag so the parser backtracks and the operator is parsed correctly
        # (or the whole selection is rejected loudly). Chain ids are short tokens, so
        # this never rejects a real name in practice.
        for op in ("and", "or", "not", "to"):
            if identifier.startswith(op) and len(identifier) > len(op):
                raise ParseError(
                    f"'{op}' is glued to a longer token '{identifier}' at position"
                    f" {start} (add a space, e.g. '{op} {identifier[len(op) :]}')"
                )
        return identifier

    def _parse_list_of_identifiers(self) -> List[str]:
        identifiers = [self._parse_identifier()]
        while True:
            saved_pos = self.pos
            try:
                self._skip_space1()
                identifiers.append(self._parse_identifier())
            except ParseError:
                self.pos = saved_pos
                break
        return identifiers

    def _parse_numbers(self) -> list[int] | range:
        first = self._parse_usize()

        saved_pos_for_to = self.pos
        try:
            self._skip_space1()
            self._consume_tag("to")
            self._skip_space1()
            last = self._parse_usize()
        except ParseError:
            # not a range ("resid 5 6 7" list form) -> backtrack and parse a number list.
            # The range-order check is deliberately NOT inside this try: a descending range
            # must be a hard error, not something this `except` swallows into list form.
            self.pos = saved_pos_for_to
            numbers = [first]
            while True:
                saved_pos_loop = self.pos
                try:
                    self._skip_space1()
                    numbers.append(self._parse_usize())
                except ParseError:
                    self.pos = saved_pos_loop
                    break
            return numbers
        # committed to a range ("first to last"): a descending range is a hard error.
        # Raise SelectionError (NOT ParseError) so the `except ParseError` backtracking in
        # _parse_atom/_parse_primary can't swallow it into the list form and report a
        # misleading "Expected an atomic selection at position 0".
        if last < first:
            raise SelectionError(f"Range end {last} is less than start {first}")
        return range(first, last + 1)

    def _parse_resid(self) -> SelectionNode:
        self._consume_tag("resid")
        self._skip_space1()
        return ResId(self._parse_numbers())

    def _parse_index(self) -> SelectionNode:
        self._consume_tag("index")
        self._skip_space1()
        return Index(self._parse_numbers())

    def _parse_chain(self) -> SelectionNode:
        self._consume_tag("chain")
        self._skip_space1()
        return Chain(self._parse_list_of_identifiers())

    def _parse_name(self) -> SelectionNode:
        # Atom-name selector, e.g. "name CA" / "name CA CB CG" / "name C1'". Uses its
        # OWN token parser rather than the identifier one: atom names legally carry a
        # prime (C1' / C1* / H2"), which the alphanumeric identifier parser rejects.
        # It stops at and/or/not the same way, so "name C1' and resid 5" needs no
        # special-casing.
        self._consume_tag("name")
        self._skip_space1()
        return Name(self._parse_list_of_atom_names())

    # Bare-keyword (no-argument) selectors -> node factory. Molecule type
    # (protein/dna/rna) and polymer sub-structure (backbone/sidechain).
    _KEYWORD_SELECTORS = {
        "protein": lambda: MolType("protein"),
        "dna": lambda: MolType("dna"),
        "rna": lambda: MolType("rna"),
        "backbone": Backbone,
        "sidechain": Sidechain,
    }

    def _parse_keyword_selector(self) -> SelectionNode:
        # Bare-keyword selectors take no argument. _consume_tag's glued-token guard
        # rejects a longer token (e.g. "proteinase" / "backbones"), so these never
        # swallow a chain name that merely starts with the keyword.
        for kw, make in self._KEYWORD_SELECTORS.items():
            saved_pos = self.pos
            try:
                self._consume_tag(kw)
                return make()
            except ParseError:
                self.pos = saved_pos
        raise ParseError(
            f"Expected a keyword selection (protein/dna/rna/backbone/sidechain)"
            f" at position {self.pos}"
        )

    # --- Grammar hierarchy ---
    def _parse_atom(self) -> SelectionNode:
        atom_parsers = [
            self._parse_keyword_selector,
            self._parse_chain,
            self._parse_name,
            self._parse_resid,
            self._parse_index,
        ]
        for parser_func in atom_parsers:
            saved_pos = self.pos
            try:
                return parser_func()
            except ParseError:
                self.pos = saved_pos
        raise ParseError(
            f"Expected an atomic selection (e.g., 'chain A', 'resid 1', 'name CA',"
            f" 'protein') at position {self.pos}"
        )

    def _parse_bracket(self) -> SelectionNode:
        self._consume_char("(")
        expr = self.parse_expr()
        self._consume_char(")")
        return Bracket(expr)

    def _parse_primary(self) -> SelectionNode:
        self._skip_space0()
        saved_pos = self.pos
        try:
            return self._parse_bracket()
        except ParseError:
            self.pos = saved_pos
            try:
                return self._parse_atom()
            except ParseError as e_atom:
                if self.text[saved_pos:].startswith("("):
                    raise ParseError(
                        f"Syntax error in parenthesized expression or"
                        f" mismatched parentheses near position {saved_pos}"
                    ) from e_atom
                raise

    def _parse_not(self) -> SelectionNode:
        num_nots = 0
        while True:
            saved_pos = self.pos
            self._skip_space0()
            try:
                self._consume_tag("not")
                num_nots += 1
            except ParseError:
                self.pos = saved_pos
                break

        selection = self._parse_primary()

        for _ in range(num_nots):
            selection = Not(selection)
        return selection

    def _parse_and(self) -> SelectionNode:
        operands = [self._parse_not()]
        while True:
            saved_pos = self.pos
            try:
                self._skip_space1()
                self._consume_tag("and")
                self._skip_space1()
                operands.append(self._parse_not())
            except ParseError:
                self.pos = saved_pos
                break
        return operands[0] if len(operands) == 1 else And(operands)

    def _parse_or(self) -> SelectionNode:
        operands = [self._parse_and()]
        while True:
            saved_pos = self.pos
            try:
                self._skip_space1()
                self._consume_tag("or")
                self._skip_space1()
                operands.append(self._parse_and())
            except ParseError:
                self.pos = saved_pos
                break
        return operands[0] if len(operands) == 1 else Or(operands)

    def parse_expr(self) -> SelectionNode:
        expr = self._parse_or()
        self._skip_space0()
        return expr

    def parse(self) -> SelectionNode:
        parsed_node = self.parse_expr()
        if self.pos < len(self.text):
            raise ParseError(
                f"Unexpected trailing characters: '{self.text[self.pos :]}'"
                f" at position {self.pos}"
            )
        return parsed_node


# --- Public API ---
def parse_selection(selection_string: str) -> Union[SelectionNode, str]:
    try:
        parser = SelectionParser(selection_string)
        return parser.parse()
    except (ParseError, SelectionError) as e:
        # SelectionError (hard errors like a descending range) reaches here past the
        # per-alternative `except ParseError` backtracking; surface its real message.
        return str(e)


# --- AtomSelector Class ---
class AtomSelector:
    def __init__(self, selection_string: str) -> None:
        self.selection_string = selection_string
        self.parsed_selection: Union[SelectionNode, None] = None
        self._error: Union[str, None] = None

        result = parse_selection(selection_string)
        if isinstance(result, SelectionNode):
            self.parsed_selection = result
        else:
            self._error = result
            raise ValueError(f"Failed to parse selection string: {self._error}")

    def matches(self, mol: Dict[str, Union[str, int]]) -> bool:
        """Selection-match alias; some callers (e.g. AF3) call ``matches``."""
        return getattr(self, "eval")(mol)

    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        """
        mol: Dict[str, str | int]
        e.g. mol = { "chain": "A", "resid": 1, "index": 0 }
        """
        if self.parsed_selection is None:
            raise RuntimeError("Selection was not successfully parsed.")

        return self.parsed_selection.eval(mol)
