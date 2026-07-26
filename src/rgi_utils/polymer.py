"""Build local polymer geometry restraints from per-residue reference coordinates.

Reference conformers in the supported predictors are residue-local: each
``ref_space_uid`` identifies one independently positioned CCD component. That makes
them suitable targets for intra-residue bonds, angles, chirality and planar groups
(aromatic side chains / nucleic-acid bases), but not for measuring inter-residue link
geometry. Canonical peptide and phosphodiester links (and the peptide plane) are
therefore supplied explicitly below.

Those reference conformers are approximate chemistry, not refinement geometry (AF3
ETKDG-embeds the free CCD component). Set
``conformer_restraints_config.monomer_library`` to take the bond / angle / plane /
link targets from the CCP4 monomer library instead -- the same values Refmac and
servalcat refine against; see ``monlib_geom``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from rgi_utils import monlib_geom
from rgi_utils._mol_build import build_ligand_mol
from rgi_utils._moltype import polymer_type
from rgi_utils.atom_context import LigandConf

logger = logging.getLogger(__name__)


@dataclass
class PolymerGeometry:
    """Reference residue conformers plus canonical inter-residue geometry."""

    residue_confs: list[LigandConf]
    atom_indices: np.ndarray
    link_bonds: list[tuple[int, int, float]]
    link_angles: list[tuple[int, int, int, float]]
    # Canonical inter-residue planar groups (e.g. the peptide plane): each a tuple of
    # global atom indices scored by the `plane` term (best-fit-plane flatness).
    link_planes: list[tuple[int, ...]]
    # Monomer-library targets, empty unless `monomer_library` is configured. These
    # REPLACE (never supplement) the reference-conformer bond/angle/plane targets for
    # the residues the library covered: `featurizer` drops every conformer-derived
    # tuple whose atoms all lie in `library_atoms`, so no residue is restrained twice.
    library_bonds: list[tuple[int, int, float]] = field(default_factory=list)
    library_angles: list[tuple[int, int, int, float]] = field(default_factory=list)
    library_planes: list[tuple[int, ...]] = field(default_factory=list)
    library_atoms: frozenset[int] = frozenset()


# Side selectors: each link atom names its residue explicitly (previous / current)
# rather than being inferred from the atom-name tuple, so the asymmetric peptide and
# phosphodiester assignments read declaratively.
_PREV, _CURR = 0, 1


@dataclass(frozen=True)
class _LinkGeometry:
    """Canonical inter-residue link targets as (atom_name, side) references.

    ``bond`` is ``((name, side), (name, side), target_angstrom)``; each ``angles`` entry
    is three ``(name, side)`` atoms plus a target in DEGREES; each ``planes`` entry is an
    N-tuple of ``(name, side)`` atoms forming one planar group restrained to zero
    out-of-plane deviation (peptide-plane flatness, carried by the `plane` energy leaf
    so polymer chemistry stays bond/angle/chiral/plane/VdW).
    """

    bond: tuple
    angles: tuple
    planes: tuple = ()


# Engh-Huber peptide-link targets and conventional phosphodiester targets.
_PEPTIDE_BOND = 1.329
_PHOSPHODIESTER_BOND = 1.607
_PROTEIN_LINK = _LinkGeometry(
    bond=(("C", _PREV), ("N", _CURR), _PEPTIDE_BOND),
    angles=(
        (("CA", _PREV), ("C", _PREV), ("N", _CURR), 116.2),
        (("O", _PREV), ("C", _PREV), ("N", _CURR), 122.7),
        (("C", _PREV), ("N", _CURR), ("CA", _CURR), 121.7),
    ),
    # Peptide plane: the union of the two classic omega-plane impropers is one 5-atom
    # planar group {C, CA, O of the previous residue; N, CA of the current}.
    planes=((("C", _PREV), ("CA", _PREV), ("O", _PREV), ("N", _CURR), ("CA", _CURR)),),
)
_NUCLEIC_LINK = _LinkGeometry(
    bond=(("O3'", _PREV), ("P", _CURR), _PHOSPHODIESTER_BOND),
    angles=(
        (("C3'", _PREV), ("O3'", _PREV), ("P", _CURR), 119.7),
        (("O3'", _PREV), ("P", _CURR), ("O5'", _CURR), 104.0),
    ),
)


def _normalise_name(name: str | None) -> str:
    return (name or "").strip().upper().replace("*", "'")


def _is_enabled_polymer(record) -> bool:
    """A polymer atom record whose chain opted into conformer restraints."""
    return polymer_type(record.mol_type, record.resname) is not None and bool(
        getattr(record, "conformer_restraints", False)
    )


def _link_geometry(previous, current, mol_type: str, library=None):
    """Return canonical link bond/angles/planes for two adjacent residue atom maps.

    With a monomer ``library`` the bond and angle targets come from its link entry
    (``TRANS`` / ``p``) instead of the built-in table; the planes stay built-in either
    way, because the library's peptide plane is the 4-atom ``CA-C-N-O`` group while the
    5-atom omega group used here (adding the next CA) is the stronger restraint.
    """

    names = (previous["names"], current["names"])  # index by _PREV / _CURR
    link = _PROTEIN_LINK if mol_type == "protein" else _NUCLEIC_LINK

    def resolve(atom):
        name, side = atom
        return names[side].get(name)

    from_library = (
        None
        if library is None
        else library.link_restraints(mol_type, previous["names"], current["names"])
    )
    if from_library is not None:
        bonds, angles = from_library
    else:
        b0, b1, bond_target = link.bond
        g0, g1 = resolve(b0), resolve(b1)
        bonds = [] if g0 is None or g1 is None else [(g0, g1, bond_target)]

        angles = []
        for a0, a1, a2, degrees in link.angles:
            idx = tuple(resolve(a) for a in (a0, a1, a2))
            if all(i is not None for i in idx):
                angles.append((*idx, float(np.deg2rad(degrees))))

    planes = []
    for group in link.planes:
        idx = tuple(resolve(a) for a in group)
        if all(i is not None for i in idx):
            planes.append(idx)
    return bonds, angles, planes


def build_polymer_geometry(
    adapter, conformer_config: dict | None, elements=None
) -> PolymerGeometry | None:
    """Build polymer-local reference conformers through the framework adapter.

    Requested adapters must expose reference positions in addition to ordinary atom
    records and elements. ``elements`` may be passed in when the caller already resolved
    it (avoids a second ``get_elements`` call); otherwise it is read from the adapter.
    Reference-space UIDs are used when available; otherwise the grouping falls back to
    chain/residue/type records. Failing loudly is important: silently omitting polymer
    restraints would otherwise look like a successful run.
    """

    cfg_present = any(not str(key).startswith("_") for key in (conformer_config or {}))
    if not cfg_present:
        return None
    if not hasattr(adapter, "iter_atoms"):
        return None
    records = list(adapter.iter_atoms())
    if not any(_is_enabled_polymer(record) for record in records):
        return None

    required = ["get_reference_positions"]
    if elements is None:
        required.insert(0, "get_elements")
    missing = [name for name in required if not hasattr(adapter, name)]
    if missing:
        raise TypeError(
            "polymer conformer restraints require adapter method(s): "
            + ", ".join(missing)
        )

    if elements is None:
        elements = adapter.get_elements()
    elements = np.asarray(elements)
    ref_pos = np.asarray(adapter.get_reference_positions(), dtype=np.float64)
    n = min(len(elements), len(ref_pos))
    if ref_pos.ndim != 2 or ref_pos.shape[-1] != 3:
        raise ValueError(
            f"adapter reference positions must have shape (n_atom, 3), got {ref_pos.shape}"
        )

    ref_uid = None
    if hasattr(adapter, "get_reference_space_uid"):
        try:
            ref_uid = np.asarray(adapter.get_reference_space_uid()).reshape(-1)
        except (AttributeError, KeyError):
            ref_uid = None
        if ref_uid is not None and len(ref_uid) < n:
            raise ValueError(
                "adapter reference-space UID array is shorter than reference positions"
            )
    if ref_uid is None:
        # Some framework adapters expose residue-local reference positions but not the
        # framework's UID feature at their existing construction site.  Derive an
        # equivalent stable grouping from normalized chain/residue/type records.
        ref_uid = np.full(n, -1, dtype=np.int64)
        uid_for_key = {}
        for record in records:
            g = int(record.index)
            if _is_enabled_polymer(record) and 0 <= g < n:
                ptype = polymer_type(record.mol_type, record.resname)
                key = (record.chain, int(record.resid), ptype)
                ref_uid[g] = uid_for_key.setdefault(key, len(uid_for_key))

    by_uid: dict[int, list] = {}
    for record in records:
        g = int(record.index)
        if not _is_enabled_polymer(record) or g < 0 or g >= n:
            continue
        if int(elements[g]) <= 0 or int(ref_uid[g]) < 0:
            continue
        by_uid.setdefault(int(ref_uid[g]), []).append(record)

    residue_confs: list[LigandConf] = []
    residue_meta = []
    atom_indices: set[int] = set()
    for uid, group in by_uid.items():
        group = sorted(group, key=lambda r: int(r.index))
        ptypes = {polymer_type(r.mol_type, r.resname) for r in group}
        chains = {r.chain for r in group}
        if len(ptypes) != 1 or len(chains) != 1:
            raise ValueError(
                f"ref_space_uid {uid} spans multiple polymer residues: "
                f"types={ptypes}, chains={chains}"
            )
        gidx = np.asarray([int(r.index) for r in group], dtype=np.int64)
        coords = ref_pos[gidx]
        mol = build_ligand_mol(elements[gidx], coords, [], perceive_bonds=True)
        residue_confs.append(
            LigandConf(
                mol=mol,
                conf_coords=coords,
                global_indices=gidx,
                conformer_restraints=True,
            )
        )
        names = {
            _normalise_name(record.name): int(record.index)
            for record in group
            if _normalise_name(record.name)
        }
        residue_meta.append(
            {
                "uid": uid,
                "chain": group[0].chain,
                "order": int(gidx.min()),
                "mol_type": next(iter(ptypes)),
                "resname": group[0].resname,
                "names": names,
            }
        )
        atom_indices.update(int(g) for g in gidx)

    library, targets = _load_library(conformer_config, residue_meta)

    link_bonds = []
    link_angles = []
    link_planes = []
    by_chain: dict[str, list[dict]] = {}
    for meta in residue_meta:
        by_chain.setdefault(meta["chain"], []).append(meta)
    for residues in by_chain.values():
        # Global atom order is the reliable component order. Modified residues are
        # atom-tokenized in some adapters, so their AtomRecord.resid values differ even
        # though ref_space_uid correctly groups them into one residue.
        residues.sort(key=lambda x: (x["order"], x["uid"]))
        for previous, current in zip(residues, residues[1:]):
            if current["mol_type"] != previous["mol_type"]:
                continue
            bonds, angles, planes = _link_geometry(
                previous, current, current["mol_type"], library
            )
            link_bonds.extend(bonds)
            link_angles.extend(angles)
            link_planes.extend(planes)

    return PolymerGeometry(
        residue_confs=residue_confs,
        atom_indices=np.asarray(sorted(atom_indices), dtype=np.int64),
        link_bonds=link_bonds,
        link_angles=link_angles,
        link_planes=link_planes,
        library_bonds=targets.bonds,
        library_angles=targets.angles,
        library_planes=targets.planes,
        library_atoms=targets.atoms,
    )


def _load_library(conformer_config: dict | None, residue_meta: list[dict]):
    """``(MonomerLibrary | None, LibraryTargets)`` for the configured library.

    Logs a SEPARATE coverage line (the base_pair macro's convention) rather than
    folding into the setup summary: "the library loaded but covered nothing" and "no
    library configured" produce identical restraint counts, so the distinction has to
    be visible.
    """
    spec = monlib_geom.parse_config(conformer_config)
    if spec is None:
        return None, monlib_geom.LibraryTargets()
    path, on_missing = spec
    library = monlib_geom.MonomerLibrary.load(
        path, {m["resname"] for m in residue_meta if m["resname"]}
    )
    targets = monlib_geom.collect(library, residue_meta, on_missing)
    n_covered = len(residue_meta) - sum(
        1 for m in residue_meta if not library.covers(m["resname"])
    )
    msg = (
        f"[rgi_utils] monomer library: {n_covered}/{len(residue_meta)} residues from "
        f"{path} (components {list(targets.covered)}; bonds={len(targets.bonds)} "
        f"angles={len(targets.angles)} planes={len(targets.planes)})"
    )
    if targets.missing:
        msg += f"; NOT in library, kept reference conformer: {list(targets.missing)}"
    logger.info(msg)
    return library, targets
