"""RMSD restraint config + atom-site resolution.

One ``rmsd_restraints_config`` entry restrains the **Kabsch-superposed RMSD**
between a group of moving atoms in the diffusion structure
(``atom_selection_target``) and a fixed group from a reference PDB
(``ref_pdb`` + ``atom_selection_ref``) toward ``target_rmsd``. The actual energy /
optimisation lives in the energy + optim layers (RMSD is optimised by the CG
solver like conformer restraints); this module only parses the entry and resolves
the two atom groups + the reference coordinates.

Mirrors ``distance_restr_data.DistanceData``: ``set_config`` parses one entry and
``resolve_sites(adapter)`` turns the selections into a target global-index list and
a reference-coordinate array. The two groups are paired atom-for-atom **by
selection order**, so they must select the same number of corresponding atoms — a
count mismatch raises ``ValueError`` (the user's explicit requirement).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from rgi_utils.atom_context import FrameworkAdapter
from rgi_utils.pdb_ref import select_ref_coords
from rgi_utils.selection import AtomSelector

logger = logging.getLogger(__name__)


@dataclass
class RmsdData:
    ref_pdb: str  # path to the reference PDB
    target_rmsd: float  # target RMSD (A) the superposed group is driven toward
    atom_selection_ref: str  # selection into the reference PDB
    atom_selection_target: str  # selection into the diffusion structure
    weight: float
    start_sigma: float  # apply only when noise level <= start_sigma (optional)
    target_sites: list  # resolved global target atom indices (selection order)
    ref_coords: np.ndarray  # (n_atoms, 3) reference coords (selection order)
    run_restr: bool

    def __init__(self):
        self.ref_pdb = None
        self.target_rmsd = None
        self.atom_selection_ref = None
        self.atom_selection_target = None
        self.weight = None
        self.start_sigma = None  # per-restraint; from_dict defaults None -> +inf
        self.target_sites = None
        self.ref_coords = None
        self.run_restr = None

    def set_config(self, config: dict):
        self.ref_pdb = config.get("ref_pdb", None)
        self.atom_selection_ref = config.get("atom_selection_ref", None)
        self.atom_selection_target = config.get("atom_selection_target", None)
        _tr = config.get("target_rmsd", None)
        self.target_rmsd = float(_tr) if _tr is not None else None
        self.weight = float(config.get("weight", 1.0) or 1.0)
        # per-restraint start_sigma (OPTIONAL; from_dict defaults None -> +inf = every
        # step). Guard float() against an explicit null so `start_sigma: null` is the
        # default, not a crash.
        _ss = config.get("start_sigma")
        if _ss is not None:
            self.start_sigma = float(_ss)
        self.run_restr = (
            self.ref_pdb is not None
            and self.target_rmsd is not None
            and self.atom_selection_ref is not None
            and self.atom_selection_target is not None
        )
        if not self.run_restr:
            raise ValueError(
                "rmsd_restraints_config entry requires ref_pdb, target_rmsd, "
                "atom_selection_ref, and atom_selection_target"
            )
        logger.info("rmsd restraint configured: target_rmsd=%.3f", self.target_rmsd)

    def resolve_sites(self, adapter: FrameworkAdapter) -> None:
        """Resolve the target group (via the adapter) and the reference coordinates
        (from ``ref_pdb``). Raises ValueError if the two groups differ in size or
        either is empty (atoms are paired by selection order)."""
        if not self.run_restr:
            return

        # 1) target atoms in the diffusion structure (global indices, selection order)
        self.target_sites = []
        sel = AtomSelector(self.atom_selection_target)
        for atom in adapter.iter_atoms():
            if sel.matches(
                {"chain": atom.chain, "resid": atom.resid, "index": atom.index}
            ):
                self.target_sites.append(atom.index)

        # 2) reference atoms from the PDB (coords, selection order)
        self.ref_coords = select_ref_coords(self.ref_pdb, self.atom_selection_ref)

        # 3) validate counts (the user's requirement: mismatch -> error)
        n_t = len(self.target_sites)
        n_r = int(self.ref_coords.shape[0])
        if n_t == 0:
            raise ValueError(
                f"rmsd target selection matched no atoms: "
                f"{self.atom_selection_target!r}"
            )
        if n_r == 0:
            raise ValueError(
                f"rmsd ref selection matched no atoms: {self.atom_selection_ref!r} "
                f"in {self.ref_pdb!r}"
            )
        if n_t != n_r:
            raise ValueError(
                f"rmsd restraint atom-count mismatch: target={n_t} atoms "
                f"({self.atom_selection_target!r}) vs ref={n_r} atoms "
                f"({self.atom_selection_ref!r} in {self.ref_pdb!r}). The two "
                f"selections must pick the same number of corresponding atoms."
            )

        logger.info(
            "rmsd restraint resolved: %d atoms, target_rmsd=%.3f", n_t, self.target_rmsd
        )

    def is_valid(self) -> bool:
        return self.run_restr
