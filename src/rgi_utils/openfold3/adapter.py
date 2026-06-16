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

from rgi_utils._biotite_adapter import biotite_get_elements, biotite_ligand_confs
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
        names = np.asarray(aa.atom_name) if hasattr(aa, "atom_name") else None
        resnames = np.asarray(aa.res_name) if hasattr(aa, "res_name") else None
        is_lig = self._ligand_mask()
        # Non-standard residues are biotite hetero=True; a standard polymer residue
        # is not. (molecule_type_id is used only for ligand-CONFORMER detection — see
        # _ligand_mask — because it does not flag modified polymer residues.)
        hetero = np.asarray(aa.hetero, dtype=bool)
        # Per-chain 1-based PER-TOKEN ordinal (the cross-tool convention shared by
        # boltz/protenix/esmfold2): a standard polymer residue gets one ordinal for
        # all its atoms (= residue ordinal); a ligand atom and each atom of a
        # NON-standard residue gets its own ordinal. (Previously a modified polymer
        # residue was grouped by res_id -> ONE ordinal, diverging from the other
        # tools for that edge case; standard residues + ligands are unaffected.)
        chain_resmap: dict[str, dict[int, int]] = {}
        chain_counter: dict[str, int] = {}
        for i in range(len(aa)):
            ch = str(chains[i])
            if bool(is_lig[i]) or bool(hetero[i]):
                chain_counter[ch] = chain_counter.get(ch, 0) + 1
                ordinal = chain_counter[ch]
            else:
                rid = int(resids[i])
                seen = chain_resmap.setdefault(ch, {})
                if rid not in seen:
                    chain_counter[ch] = chain_counter.get(ch, 0) + 1
                    seen[rid] = chain_counter[ch]
                ordinal = seen[rid]
            nm = str(names[i]).strip() if names is not None else None
            rnm = str(resnames[i]).strip() if resnames is not None else None
            yield AtomRecord(
                chain=ch, resid=ordinal, index=int(i), name=nm, resname=rnm
            )

    # --- ConformerAdapter -----------------------------------------------------
    def num_atoms(self) -> int:
        return self._n_atom

    def get_elements(self) -> np.ndarray:
        """(num_atoms,) atomic numbers; padding atoms (beyond the AtomArray) are 0."""
        return biotite_get_elements(self.atom_array, self._n_atom)

    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        aa = self.atom_array
        if aa is None:
            return
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
        # openfold marks ligand atoms via molecule_type_id (see _ligand_mask); chains
        # are chain_id; openfold-3 has no per-ligand conformer_restraints input flag, so
        # opt-in defaults ON (governed by conformer_restraints_config presence) unless
        # the AtomArray carries the annotation.
        yield from biotite_ligand_confs(
            aa,
            ligand_mask=self._ligand_mask(),
            chain_attr="chain_id",
            coords_all=coords_all,
            conf_rest_default=True,
            post_build=None,
        )
