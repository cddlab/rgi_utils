"""CCP4 monomer-library targets for the polymer conformer restraints.

By default the polymer bond / angle / plane targets are MEASURED from each
predictor's per-residue reference conformer (``polymer.build_polymer_geometry`` ->
``featurizer._extract_conformer``, ``relax=False``). Those conformers are not
refinement geometry. AF3 builds ``ref_pos`` by RDKit **ETKDG-embedding the free CCD
component**, and comparing that embedding with the monomer library shows how far off
its nucleotide targets are (A, library value on the right):

    exocyclic C6-N6   1.421-1.423 -> 1.330    an amine single bond, not the
                                              conjugated base value
    phosphate P-OP2   1.674-1.709 -> 1.517    P-OH, because the free component is a
                                              monophosphate
    glycosidic C1'-N1 1.404-1.453 -> 1.476

The embed also takes a random seed, so those targets shift by ~0.02-0.03 A between
runs. Restraining toward them pulls a residue AWAY from crystallographic geometry.

This module reads the library Refmac/servalcat refine against -- the CCP4 monomer
library, through gemmi -- and returns bond / angle / plane targets for the residues
it covers, plus the peptide / phosphodiester LINK geometry. Residues the library
does not know (modified bases, ligands) keep their reference-conformer targets, so
enabling it is safe for any structure.

**Scope**: bonds, angles, planes, links. The `chiral` term keeps its
reference-conformer volumes -- only the SIGN matters for stereochemistry, and the
library's ``ChiralityType`` convention would have to be reconciled with the local
``_chiral_vol`` atom ordering before it could be adopted without risking a silent
stereochemistry flip.

gemmi is imported lazily so ``import rgi_utils`` stays numpy-only.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 3 atoms are trivially coplanar and exert no force -- the same floor
# `featurizer._extract_conformer` applies to conformer-perceived plane groups.
_MIN_PLANE_ATOMS = 4

# Library link ids for the canonical polymer connections. CIS and TRANS carry
# IDENTICAL bond+angle values (they differ only in omega, which no conformer term
# restrains), so the trans entry serves both peptide states.
_LINK_ID = {"protein": "TRANS", "dna": "p", "rna": "p"}

_ON_MISSING = ("fallback", "error")


def _normalise_name(name: str | None) -> str:
    """Atom-name spelling shared with polymer.py (upper case, PDB v2 ``*`` -> ``'``)."""
    return (name or "").strip().upper().replace("*", "'")


@dataclass(frozen=True)
class LibraryTargets:
    """Library-derived restraints in GLOBAL atom indices.

    ``atoms`` is every atom of every residue the library covered; the featurizer drops
    each reference-conformer bond/angle/plane whose atoms all lie inside it, so the two
    sources never both restrain the same residue.
    """

    bonds: list[tuple[int, int, float]] = field(default_factory=list)
    angles: list[tuple[int, int, int, float]] = field(default_factory=list)
    planes: list[tuple[int, ...]] = field(default_factory=list)
    atoms: frozenset[int] = frozenset()
    covered: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()


def parse_config(conformer_config: dict | None) -> tuple[str, str] | None:
    """``(library path, on_missing)`` from the conformer config, or None when off.

    Accepts either the shorthand ``monomer_library: "<path>"`` or the full form
    ``monomer_library: {path: "<path>", on_missing: "fallback"|"error"}``. The path
    goes through ``expandvars``/``expanduser``, so ``"$CLIBD_MON"`` works verbatim.
    """
    spec = (conformer_config or {}).get("monomer_library")
    if spec is None:
        return None
    on_missing = "fallback"
    if isinstance(spec, str):
        path = spec
    elif isinstance(spec, dict):
        path = spec.get("path")
        on_missing = str(spec.get("on_missing", on_missing))
        unknown = set(spec) - {"path", "on_missing"}
        if unknown:
            raise ValueError(
                f"conformer_restraints_config.monomer_library: unknown key(s) "
                f"{sorted(unknown)}. Known keys: ['on_missing', 'path']"
            )
    else:
        raise ValueError(
            "conformer_restraints_config.monomer_library must be a path string or a "
            f"dict with 'path'/'on_missing', got {type(spec).__name__}"
        )
    if not path:
        raise ValueError(
            "conformer_restraints_config.monomer_library: 'path' is required (point it "
            'at a CCP4 monomer library directory, e.g. "$CLIBD_MON")'
        )
    if on_missing not in _ON_MISSING:
        raise ValueError(
            f"conformer_restraints_config.monomer_library.on_missing: unknown value "
            f"{on_missing!r}, expected one of {list(_ON_MISSING)}"
        )
    return os.path.expanduser(os.path.expandvars(str(path))), on_missing


class MonomerLibrary:
    """A loaded CCP4 monomer library, resolved against one structure's atom names."""

    def __init__(self, monlib, path: str) -> None:
        self._monlib = monlib
        self.path = path

    @classmethod
    def load(cls, path: str, resnames) -> MonomerLibrary:
        """Read the components named in ``resnames`` from the library at ``path``.

        Fails loudly on a bad path: a silently empty library would look exactly like
        "the restraint ran with library targets" while every residue quietly fell back
        to the reference conformer.
        """
        import gemmi  # lazy: keeps `import rgi_utils` numpy-only

        if not os.path.isdir(path):
            raise ValueError(
                f"conformer_restraints_config.monomer_library: {path!r} is not a "
                "directory (expected a CCP4 monomer library, i.e. the directory holding "
                "list/mon_lib_list.cif and the a/ b/ c/ ... component subdirectories)"
            )
        monlib = gemmi.MonLib()
        # gemmi resolves <dir>/<lowercase first char>/<NAME>.cif, so the directory must
        # be handed over with a trailing separator.
        monlib.read_monomer_lib(os.path.join(path, ""), sorted(set(resnames)))
        return cls(monlib, path)

    def covers(self, resname: str | None) -> bool:
        return bool(resname) and resname in self._monlib.monomers

    def link_mods(self, mol_type: str, side1: bool, side2: bool) -> list:
        """The ``_chem_mod`` entries a polymer link applies to a residue on its side(s).

        CCP4's TRANS link names ``DEL-OXT`` for side 1 and ``DEL-HN1`` for side 2. Both do
        far more than delete an atom: they REWRITE the targets the free residue carries
        (``CA-C`` 1.531 -> 1.526, ``C-O`` 1.251 -> 1.229, ``CA-C-O`` 117.191 -> 120.614,
        ``CA-N`` 1.483 -> 1.453). Without them a peptide-bonded residue is restrained to
        zwitterion geometry.
        """
        link_id = _LINK_ID.get(mol_type)
        if link_id is None or link_id not in self._monlib.links:
            return []
        link = self._monlib.links[link_id]
        wanted = []
        if side1:
            wanted.append(getattr(link.side1, "mod", ""))
        if side2:
            wanted.append(getattr(link.side2, "mod", ""))
        return [
            self._monlib.modifications[m]
            for m in wanted
            if m in self._monlib.modifications
        ]

    def residue_restraints(self, resname: str, names: dict[str, int], mods=()):
        """Library bonds / angles / planes for one residue, in global atom indices.

        ``names`` maps this residue's normalised atom names to global indices; any
        library restraint naming an atom the structure does not model is skipped. That
        is what drops the hydrogens (predictors model heavy atoms only) and the 5'/3'
        terminal atoms of an internal residue, without a special case for either.

        ``mods`` are the link modifications to fold in (see ``link_mods``). gemmi exposes
        a mod's changes as a plain ``Restraints`` block, so a DELETED restraint arrives as
        a NaN value and a CHANGED one as a finite value; deletions need no handling here
        because the atoms they name (OXT, H2, H3) are hydrogens or terminal atoms the
        predictor does not model, so those restraints are already skipped.
        """
        chem_comp = self._monlib.monomers[resname]
        restraints = chem_comp.rt
        bond_overrides: dict[frozenset, float] = {}
        angle_overrides: dict[tuple, float] = {}
        for mod in mods:
            for bond in mod.rt.bonds:
                value = float(bond.value)
                if math.isfinite(value):
                    key = frozenset(
                        (_normalise_name(bond.id1.atom), _normalise_name(bond.id2.atom))
                    )
                    bond_overrides[key] = value
            for angle in mod.rt.angles:
                value = float(angle.value)
                if math.isfinite(value):
                    ends = frozenset(
                        (
                            _normalise_name(angle.id1.atom),
                            _normalise_name(angle.id3.atom),
                        )
                    )
                    angle_overrides[(_normalise_name(angle.id2.atom), ends)] = value

        bonds = []
        for bond in restraints.bonds:
            n1 = _normalise_name(bond.id1.atom)
            n2 = _normalise_name(bond.id2.atom)
            i, j = names.get(n1), names.get(n2)
            if i is not None and j is not None:
                target = bond_overrides.get(frozenset((n1, n2)), float(bond.value))
                bonds.append((i, j, target))

        angles = []
        for angle in restraints.angles:
            trio = tuple(
                _normalise_name(a.atom) for a in (angle.id1, angle.id2, angle.id3)
            )
            idx = tuple(names.get(n) for n in trio)
            if all(k is not None for k in idx):
                target = angle_overrides.get(
                    (trio[1], frozenset((trio[0], trio[2]))), float(angle.value)
                )
                # The library stores angles in DEGREES; the energy layer wants radians.
                angles.append((*idx, math.radians(target)))

        planes = []
        for plane in restraints.planes:
            group = {
                names[key]
                for key in (_normalise_name(a.atom) for a in plane.ids)
                if key in names
            }
            if len(group) >= _MIN_PLANE_ATOMS:
                planes.append(tuple(sorted(group)))
        return bonds, angles, planes

    def link_restraints(
        self, mol_type: str, prev_names: dict[str, int], curr_names: dict[str, int]
    ):
        """Inter-residue link bonds / angles / planes for one adjacent pair.

        Returns ``None`` when the library has no entry for this polymer type, so the
        caller can keep its built-in link geometry.

        The PLANES matter as much as the bonds here. Refmac/servalcat split the peptide
        link into two 4-atom sp2 groups — ``plan-1`` {CA(1), C(1), O(1), N(2)} at the
        carbonyl carbon and ``plan-2`` {CA(2), C(1), H(2), N(2)} at the amide nitrogen —
        and neither contains BOTH CA atoms, so the plane restraints deliberately leave
        omega free; omega is a separate ``_chem_link_tor`` (180 deg, esd 5 deg). ``plan-2``
        drops to 3 atoms once the hydrogens are gone (predictors model heavy atoms only)
        and is filtered out by the ``_MIN_PLANE_ATOMS`` check, which is correct: a 3-atom
        group is trivially planar.
        """
        link_id = _LINK_ID.get(mol_type)
        if link_id is None or link_id not in self._monlib.links:
            return None
        restraints = self._monlib.links[link_id].rt
        sides = {1: prev_names, 2: curr_names}

        def resolve(atom_id):
            side = sides.get(int(atom_id.comp))
            return None if side is None else side.get(_normalise_name(atom_id.atom))

        bonds = []
        for bond in restraints.bonds:
            i, j = resolve(bond.id1), resolve(bond.id2)
            if i is not None and j is not None:
                bonds.append((i, j, float(bond.value)))

        angles = []
        for angle in restraints.angles:
            idx = tuple(resolve(a) for a in (angle.id1, angle.id2, angle.id3))
            if all(k is not None for k in idx):
                angles.append((*idx, math.radians(float(angle.value))))

        planes = []
        for plane in restraints.planes:
            group = {k for k in (resolve(a) for a in plane.ids) if k is not None}
            if len(group) >= _MIN_PLANE_ATOMS:
                planes.append(tuple(sorted(group)))
        return bonds, angles, planes


def collect(library: MonomerLibrary, residues, on_missing: str) -> LibraryTargets:
    """Library targets for every residue the library covers.

    ``residues`` are the per-residue metadata dicts built by ``polymer.py``
    (``resname`` plus ``names``: normalised atom name -> global index, plus the
    ``link_side1`` / ``link_side2`` markers saying which side(s) of a polymer link this
    residue sits on — see ``link_mods`` for why the targets depend on that).
    """
    bonds: list[tuple[int, int, float]] = []
    angles: list[tuple[int, int, int, float]] = []
    planes: list[tuple[int, ...]] = []
    atoms: set[int] = set()
    covered: set[str] = set()
    missing: set[str] = set()

    for meta in residues:
        resname = meta.get("resname")
        if not library.covers(resname):
            missing.add(str(resname))
            continue
        names = meta["names"]
        side1 = meta.get("link_side1")
        side2 = meta.get("link_side2")
        mods = library.link_mods(side1 or side2 or "", bool(side1), bool(side2))
        rb, ra, rp = library.residue_restraints(resname, names, mods)
        bonds.extend(rb)
        angles.extend(ra)
        planes.extend(rp)
        atoms.update(names.values())
        covered.add(resname)

    if missing and on_missing == "error":
        raise ValueError(
            f"conformer_restraints_config.monomer_library: no library entry for "
            f"residue(s) {sorted(missing)} in {library.path!r}. Add the component to "
            "the library, or set on_missing: fallback to keep the reference-conformer "
            "targets for them."
        )
    return LibraryTargets(
        bonds=bonds,
        angles=angles,
        planes=planes,
        atoms=frozenset(atoms),
        covered=tuple(sorted(covered)),
        missing=tuple(sorted(missing)),
    )
