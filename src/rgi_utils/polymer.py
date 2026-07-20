"""Build local polymer geometry restraints from per-residue reference coordinates.

Reference conformers in the supported predictors are residue-local: each
``ref_space_uid`` identifies one independently positioned CCD component. That makes
them suitable targets for intra-residue bonds, angles and chirality, but not for
measuring inter-residue link geometry. Canonical peptide and phosphodiester links are
therefore supplied explicitly below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rgi_utils._mol_build import build_ligand_mol
from rgi_utils._moltype import POLYMER_TYPES, polymer_type
from rgi_utils.atom_context import LigandConf


@dataclass
class PolymerGeometry:
    """Reference residue conformers plus canonical inter-residue geometry."""

    residue_confs: list[LigandConf]
    atom_indices: np.ndarray
    link_bonds: list[tuple[int, int, float]]
    link_angles: list[tuple[int, int, int, float]]
    link_chirals: list[tuple[int, int, int, int, float]]


# Engh-Huber peptide-link targets and conventional phosphodiester targets.
# Angles are stored in degrees here and converted once while building the result.
_PEPTIDE_BOND = 1.329
_PEPTIDE_ANGLES = (
    ("CA", "C", "N", 116.2),
    ("O", "C", "N", 122.7),
    ("C", "N", "CA", 121.7),
)
_PHOSPHODIESTER_BOND = 1.607
_PHOSPHODIESTER_ANGLES = (
    ("C3'", "O3'", "P", 119.7),
    ("O3'", "P", "O5'", 104.0),
)


def parse_polymer_types(conformer_config: dict | None) -> tuple[str, ...]:
    """Return validated polymer types requested by ``polymer_types``."""

    raw = (conformer_config or {}).get("polymer_types", ())
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)):
        raise ValueError(
            "conformer_restraints_config.polymer_types must be a string or list"
        )
    values = tuple(dict.fromkeys(str(v).strip().lower() for v in raw))
    unknown = sorted(set(values) - set(POLYMER_TYPES))
    if unknown:
        raise ValueError(
            "conformer_restraints_config.polymer_types accepts only "
            f"{list(POLYMER_TYPES)}, got {unknown}"
        )
    return values


def _normalise_name(name: str | None) -> str:
    return (name or "").strip().upper().replace("*", "'")


def _link_geometry(previous, current, mol_type: str):
    """Return canonical link bond/angles for two adjacent residue atom maps."""

    if mol_type == "protein":
        bond_names = ("C", "N")
        angle_defs = _PEPTIDE_ANGLES
    else:
        bond_names = ("O3'", "P")
        angle_defs = _PHOSPHODIESTER_ANGLES

    prev_names = previous["names"]
    curr_names = current["names"]
    g0 = prev_names.get(bond_names[0])
    g1 = curr_names.get(bond_names[1])
    bond_target = _PEPTIDE_BOND if mol_type == "protein" else _PHOSPHODIESTER_BOND
    bonds = [] if g0 is None or g1 is None else [(g0, g1, bond_target)]

    angles = []
    for n0, n1, n2, degrees in angle_defs:
        # The first two atoms belong to the previous residue for CA-C-N and
        # C3'-O3'-P. C-N-CA and O3'-P-O5' cross into the current residue.
        if mol_type == "protein":
            maps = (
                (prev_names, prev_names, curr_names)
                if (n0, n1, n2) != ("C", "N", "CA")
                else (prev_names, curr_names, curr_names)
            )
        elif (n0, n1, n2) == ("C3'", "O3'", "P"):
            maps = (prev_names, prev_names, curr_names)
        else:
            maps = (prev_names, curr_names, curr_names)
        idx = tuple(m.get(n) for m, n in zip(maps, (n0, n1, n2)))
        if all(i is not None for i in idx):
            angles.append((*idx, float(np.deg2rad(degrees))))
    chirals = []
    if mol_type == "protein":
        # Zero-volume impropers keep the five peptide-plane atoms coplanar.  They use
        # the chiral energy leaf deliberately, so polymer chemistry still consists of
        # only bond/angle/chiral/VdW terms.
        improper_defs = (
            (prev_names, "C", prev_names, "CA", prev_names, "O", curr_names, "N"),
            (curr_names, "N", prev_names, "C", prev_names, "O", curr_names, "CA"),
        )
        for cm, cn, m1, n1, m2, n2, m3, n3 in improper_defs:
            idx = (cm.get(cn), m1.get(n1), m2.get(n2), m3.get(n3))
            if all(i is not None for i in idx):
                chirals.append((*idx, 0.0))
    return bonds, angles, chirals


def build_polymer_geometry(
    adapter, conformer_config: dict | None
) -> PolymerGeometry | None:
    """Build polymer-local reference conformers through the framework adapter.

    Requested adapters must expose reference positions in addition to ordinary atom
    records and elements. Reference-space UIDs are used when available; otherwise the
    grouping falls back to chain/residue/type records. Failing loudly is important:
    silently omitting polymer restraints would otherwise look like a successful run.
    """

    selected_types = set(parse_polymer_types(conformer_config))
    if not selected_types:
        return None
    required = ("iter_atoms", "get_elements", "get_reference_positions")
    missing = [name for name in required if not hasattr(adapter, name)]
    if missing:
        raise TypeError(
            "polymer conformer restraints require adapter method(s): "
            + ", ".join(missing)
        )

    records = list(adapter.iter_atoms())
    elements = np.asarray(adapter.get_elements())
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
            ptype = polymer_type(record.mol_type, record.resname)
            if ptype in selected_types and 0 <= g < n:
                key = (record.chain, int(record.resid), ptype)
                ref_uid[g] = uid_for_key.setdefault(key, len(uid_for_key))

    by_uid: dict[int, list] = {}
    for record in records:
        g = int(record.index)
        ptype = polymer_type(record.mol_type, record.resname)
        if ptype not in selected_types or g < 0 or g >= n:
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
                "names": names,
            }
        )
        atom_indices.update(int(g) for g in gidx)

    link_bonds = []
    link_angles = []
    link_chirals = []
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
            bonds, angles, chirals = _link_geometry(
                previous, current, current["mol_type"]
            )
            link_bonds.extend(bonds)
            link_angles.extend(angles)
            link_chirals.extend(chirals)

    return PolymerGeometry(
        residue_confs=residue_confs,
        atom_indices=np.asarray(sorted(atom_indices), dtype=np.int64),
        link_bonds=link_bonds,
        link_angles=link_angles,
        link_chirals=link_chirals,
    )
