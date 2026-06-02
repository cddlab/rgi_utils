"""Adapter from an OpenFold-3 batch dict to the rgi_utils adapter protocols.

OpenFold-3 exposes a biotite ``AtomArray`` (per-atom chain/res/element + a
``BondList``), stashed into the features dict as ``batch["atom_array"]``. Its
AtomArray is the same biotite type protenix uses. Two OpenFold-specific facts
shape this adapter (both were real bugs in the first draft):

1. **Ligand identity**: OpenFold sets a per-atom ``molecule_type_id`` annotation
   (MoleculeType.LIGAND == 3). We key ligand detection on that, NOT on biotite's
   ``hetero`` flag — biotite marks *any* non-standard CCD residue hetero=True
   (including non-canonical polymer residues), which would misclassify a modified
   protein/NA residue as a ligand. We fall back to ``hetero`` only if the
   annotation is absent.

2. **Reference geometry**: OpenFold zeroes ``atom_array.coord`` at inference
   (query.py: ``atom_array.coord[:] = 0.0`` "for consistency"). The real ligand
   conformer geometry lives in the per-atom ``ref_pos`` feature (built from the
   ProcessedReferenceMolecule conformers). The build site passes that array in as
   ``ref_coords`` so the bond/angle/chiral restraint TARGETS are built from the
   real conformer, not from a degenerate origin cloud. Intra-ligand bonds DO come
   from ``atom_array.bonds`` (``connect_via_residue_names`` populates them), so
   unlike chai we keep the real connectivity.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from rgi_utils._mol_build import atomic_number as _atomic_number
from rgi_utils._mol_build import build_ligand_mol as _build_ligand_mol
from rgi_utils.atom_context import AtomRecord, LigandConf

logger = logging.getLogger(__name__)

_LIGAND_MOLTYPE = 3  # openfold MoleculeType.LIGAND


class Openfold3Adapter:
    """rgi_utils adapter over an OpenFold-3 biotite ``AtomArray``.

    ``num_atoms`` is the padded coordinate length in the diffusion loop
    (``batch["atom_mask"].shape[-1]``), passed in from the build site because the
    AtomArray itself is un-padded.

    ``ref_coords`` is the real per-atom reference-conformer coordinate array
    (``batch["ref_pos"]`` as a CPU ndarray, aligned 1:1 with the AtomArray atom
    order). When provided it is the source of ligand conformer geometry; when
    omitted the adapter falls back to ``atom_array.coord`` (which OpenFold zeroes,
    so conformer restraints would be degenerate — a warning is logged).
    """

    def __init__(self, atom_array, num_atoms: int, ref_coords=None) -> None:
        self.atom_array = atom_array
        self._n_atom = int(num_atoms)
        self._ref_coords = None if ref_coords is None else np.asarray(ref_coords, dtype=np.float64)

    # --- ligand identity ------------------------------------------------------
    def _ligand_mask(self) -> np.ndarray:
        """Per-atom bool: True where the atom belongs to a LIGAND entity.

        Prefer the ``molecule_type_id`` annotation (entity-level, reliable); fall
        back to biotite ``hetero`` only if that annotation is missing.
        """
        aa = self.atom_array
        cats = aa.get_annotation_categories()
        if "molecule_type_id" in cats:
            return np.asarray(aa.molecule_type_id) == _LIGAND_MOLTYPE
        return np.asarray(aa.hetero, dtype=bool)

    # --- FrameworkAdapter -----------------------------------------------------
    def iter_atoms(self) -> Iterator[AtomRecord]:
        aa = self.atom_array
        if aa is None:
            return
        chains = np.asarray(aa.chain_id)
        resids = np.asarray(aa.res_id)
        is_lig = self._ligand_mask()
        # Per-chain 1-based residue/token ordinal (cross-tool selection
        # convention): count polymer residues by res_id group, but give each
        # ligand atom its own ordinal -> 1..N, matching boltz/protenix/AF3/chai.
        chain_resmap: dict[str, dict[int, int]] = {}
        chain_counter: dict[str, int] = {}
        for i in range(len(aa)):
            ch = str(chains[i])
            if bool(is_lig[i]):
                chain_counter[ch] = chain_counter.get(ch, 0) + 1
                ordinal = chain_counter[ch]
            else:
                rid = int(resids[i])
                seen = chain_resmap.setdefault(ch, {})
                if rid not in seen:
                    chain_counter[ch] = chain_counter.get(ch, 0) + 1
                    seen[rid] = chain_counter[ch]
                ordinal = seen[rid]
            yield AtomRecord(chain=ch, resid=ordinal, index=int(i))

    # --- ConformerAdapter -----------------------------------------------------
    def num_atoms(self) -> int:
        return self._n_atom

    def get_elements(self) -> np.ndarray:
        """(num_atoms,) atomic numbers; padding atoms (beyond the AtomArray) are 0."""
        elements = np.zeros(self._n_atom, dtype=np.int64)
        aa = self.atom_array
        if aa is not None:
            syms = np.asarray(aa.element)
            n = min(len(aa), self._n_atom)
            for i in range(n):
                elements[i] = _atomic_number(syms[i])
        return elements

    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        aa = self.atom_array
        if aa is None:
            return
        is_lig = self._ligand_mask()
        chains = np.asarray(aa.chain_id)
        elements_all = np.asarray(aa.element)
        # Conformer geometry: real reference coords (ref_pos) when supplied; else
        # atom_array.coord (zeroed by OpenFold -> degenerate, warn once).
        if self._ref_coords is not None:
            coords_all = self._ref_coords
        else:
            coords_all = np.asarray(aa.coord, dtype=np.float64)
            logger.warning(
                "openfold3 adapter: no ref_coords passed; conformer targets use "
                "atom_array.coord which OpenFold zeroes — bond/angle/chiral "
                "restraints will be degenerate. Pass batch['ref_pos'] to fix."
            )
        # per-ligand conformer_restraints opt-out (annotation); default on when
        # absent (OpenFold-3 v1 does not set it -> every ligand restrained).
        conf_rest_annot = None
        if "conformer_restraints" in aa.get_annotation_categories():
            conf_rest_annot = np.asarray(aa.conformer_restraints, dtype=bool)
        # bonds may be absent (monatomic ions have no BondList); treat as no bonds
        # so the ion still surfaces as LigandConf for ligand-protein VdW.
        bond_arr = (
            aa.bonds.as_array()  # (n_bond, 3): i, j, order
            if getattr(aa, "bonds", None) is not None
            else np.empty((0, 3), dtype=np.int64)
        )

        for chain_id in np.unique(chains[is_lig]):
            idxs = np.where((chains == chain_id) & is_lig)[0]
            g2l = {int(g): li for li, g in enumerate(idxs)}
            bonds_local = [
                (g2l[int(i)], g2l[int(j)], int(o))
                for i, j, o in bond_arr
                if int(i) in g2l and int(j) in g2l
            ]
            coords = coords_all[idxs]
            mol = _build_ligand_mol(elements_all[idxs], coords, bonds_local)
            conf_rest = True
            if conf_rest_annot is not None:
                conf_rest = bool(conf_rest_annot[idxs].any())
            yield LigandConf(
                mol=mol,
                conf_coords=coords,
                global_indices=idxs.astype(np.int64),
                conformer_restraints=conf_rest,
            )
