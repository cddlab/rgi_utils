"""RMSD restraint config + atom-site resolution.

One ``rmsd_restraints_config`` entry restrains the **Kabsch-superposed RMSD**
between a moving group in the diffusion structure and a fixed group from a
reference PDB toward ``target_rmsd``, optimised by the CG solver. The
superposition ("fit") atoms and the measured ("calc") atoms can differ:

  atom_selection_ref_fit / atom_selection_target_fit   -> Kabsch superposition
  atom_selection_ref_calc / atom_selection_target_calc -> RMSD measured here

Backward-compatible shorthand: ``atom_selection_ref`` / ``atom_selection_target``
set BOTH fit and calc (so a single selection == fit==calc == the original
behaviour).

Selections are OPTIONAL. Omit them (no ``atom_selection*`` at all) to fit + measure
RMSD over the WHOLE structure: the whole diffusion structure is superposed onto the
whole reference and the RMSD is taken over everything, BEST-EFFORT -- atoms matched to
the reference by identity (chain, resid, name) are used and any structure atom missing
from the reference (e.g. the ref has no hydrogens) is skipped, so an incomplete ref
still works (pymol-align-like). Only ``ref_pdb`` and ``target_rmsd`` are required.

Reference and target atoms are paired by IDENTITY (chain, resid, atom-name) when
both sides expose atom names, so the reference PDB's atom order need not match the
tool's internal order. If names are unavailable on either side it falls back to
selection-order pairing (the original behaviour). With an EXPLICIT selection the
pairing is strict: a target atom with no matching (chain, resid, name) in the
reference -- or an atom-count mismatch in the order fallback -- raises ``ValueError``
(best-effort skipping applies only to the no-selection whole-structure default).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from rgi_utils.atom_context import FrameworkAdapter
from rgi_utils.pdb_ref import read_pdb_atoms
from rgi_utils.selection import AtomSelector

logger = logging.getLogger(__name__)


@dataclass
class RmsdData:
    ref_pdb: str = None
    target_rmsd: float = None
    weight: float = None
    start_sigma: float = None  # per-restraint; from_dict defaults None -> +inf
    # selection strings (fit = superposition atoms, calc = measured atoms)
    sel_ref_fit: str = None
    sel_target_fit: str = None
    sel_ref_calc: str = None
    sel_target_calc: str = None
    # resolved: global target atom indices + paired reference coords (n_atoms, 3)
    fit_target_sites: list = field(default=None)
    fit_ref_coords: np.ndarray = field(default=None)
    calc_target_sites: list = field(default=None)
    calc_ref_coords: np.ndarray = field(default=None)
    run_restr: bool = None

    def set_config(self, config: dict):
        self.ref_pdb = config.get("ref_pdb", None)
        _tr = config.get("target_rmsd", None)
        self.target_rmsd = float(_tr) if _tr is not None else None
        # None -> default 1.0; an explicit 0 stays 0 (a zero-weight, no-op restraint),
        # so `or 1.0` truthiness must NOT be used here.
        _w = config.get("weight")
        self.weight = 1.0 if _w is None else float(_w)
        _ss = config.get("start_sigma")
        if _ss is not None:
            self.start_sigma = float(_ss)
        # explicit _fit / _calc override the shared ref/target shorthand. A selection
        # left None means "the whole structure on that side" (resolved best-effort).
        ref = config.get("atom_selection_ref")
        tgt = config.get("atom_selection_target")
        self.sel_ref_fit = config.get("atom_selection_ref_fit", ref)
        self.sel_target_fit = config.get("atom_selection_target_fit", tgt)
        self.sel_ref_calc = config.get("atom_selection_ref_calc", ref)
        self.sel_target_calc = config.get("atom_selection_target_calc", tgt)
        # selections are OPTIONAL (omit -> whole-structure best-effort); only ref_pdb +
        # target_rmsd are required.
        self.run_restr = self.ref_pdb is not None and self.target_rmsd is not None
        if not self.run_restr:
            raise ValueError(
                "rmsd_restraints_config entry requires ref_pdb and target_rmsd (atom "
                "selections are optional: omit them to fit + measure RMSD over the whole "
                "structure, best-effort over atoms matched to the reference)"
            )
        logger.info("rmsd restraint configured: target_rmsd=%.3f", self.target_rmsd)

    def resolve_sites(self, adapter: FrameworkAdapter) -> None:
        if not self.run_restr:
            return
        atoms = list(adapter.iter_atoms())
        ref_atoms = read_pdb_atoms(self.ref_pdb)  # raises ValueError on a bad file
        # no target selection -> whole structure, paired best-effort (skip atoms missing
        # from the reference); an explicit selection stays strict.
        self.fit_target_sites, self.fit_ref_coords = self._pair(
            atoms, ref_atoms, self.sel_target_fit, self.sel_ref_fit, "fit",
            best_effort=self.sel_target_fit is None,
        )
        self.calc_target_sites, self.calc_ref_coords = self._pair(
            atoms, ref_atoms, self.sel_target_calc, self.sel_ref_calc, "calc",
            best_effort=self.sel_target_calc is None,
        )
        logger.info(
            "rmsd restraint resolved: fit=%d calc=%d atoms, target_rmsd=%.3f",
            len(self.fit_target_sites),
            len(self.calc_target_sites),
            self.target_rmsd,
        )

    def _pair(self, atoms, ref_atoms, sel_target, sel_ref, tag, best_effort=False):
        """Resolve one (target, ref) selection pair -> (target_global_indices, ref_coords
        aligned to the target order). A ``None`` selection means the WHOLE structure on
        that side (no filter). Pairing is by IDENTITY (chain, resid, name) when both sides
        expose names; else selection-order (counts must match). With ``best_effort`` (the
        no-selection whole-structure default) a target atom missing from the reference is
        SKIPPED rather than raising, so an incomplete ref still fits 'as much as possible';
        with an explicit selection it stays strict (a missing match raises)."""
        if sel_target is None:  # whole structure (no filter)
            tgt = list(atoms)
        else:
            st = AtomSelector(sel_target)
            tgt = [
                a
                for a in atoms
                if st.matches({"chain": a.chain, "resid": a.resid, "index": a.index})
            ]
        if sel_ref is None:  # whole reference (no filter)
            ref = list(ref_atoms)
        else:
            sr = AtomSelector(sel_ref)
            ref = [
                r
                for r in ref_atoms
                if sr.matches({"chain": r.chain, "resid": r.resid, "index": r.index})
            ]
        if not tgt:
            raise ValueError(
                f"rmsd {tag} target selection matched no atoms: {sel_target!r}"
            )
        if not ref:
            raise ValueError(
                f"rmsd {tag} ref selection matched no atoms: {sel_ref!r} "
                f"in {self.ref_pdb!r}"
            )
        tgt_named = all(a.name for a in tgt)
        ref_named = all(r.name for r in ref)
        logger.debug(
            "rmsd %s pairing=%s target=%d ref=%d; target names[:4]=%s ref names[:4]=%s",
            tag,
            "identity" if (tgt_named and ref_named) else "order",
            len(tgt),
            len(ref),
            [a.name for a in tgt[:4]],
            [r.name for r in ref[:4]],
        )
        if tgt_named and ref_named:
            # duplicate (chain, resid, name) keys would silently collapse last-wins
            # and mispair atoms, so reject an ambiguous reference loudly instead.
            refmap = {}
            for r in ref:
                k = (r.chain, r.resid, r.name)
                if k in refmap:
                    raise ValueError(
                        f"rmsd {tag}: duplicate reference atom {k} in "
                        f"{self.ref_pdb!r} — ambiguous identity pairing; "
                        f"disambiguate the ref selection"
                    )
                refmap[k] = (r.x, r.y, r.z)
            sites, coords, skipped = [], [], 0
            for a in tgt:
                key = (a.chain, a.resid, a.name)
                if key not in refmap:
                    if best_effort:  # whole-structure default: use what matches
                        skipped += 1
                        continue
                    raise ValueError(
                        f"rmsd {tag}: target atom {key} has no matching "
                        f"(chain, resid, name) in ref {self.ref_pdb!r}"
                    )
                sites.append(int(a.index))
                coords.append(refmap[key])
            if not sites:
                raise ValueError(
                    f"rmsd {tag}: no target atom matched the reference by "
                    f"(chain, resid, name) in {self.ref_pdb!r}"
                )
            if skipped:  # transparency: do not silently drop atoms
                logger.info(
                    "rmsd %s (whole-structure best-effort): matched %d / %d atoms "
                    "(%d unmatched in ref skipped)",
                    tag, len(sites), len(tgt), skipped,
                )
            return sites, np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        # order fallback (no atom names): pair by selection order, counts must match
        if len(tgt) != len(ref):
            raise ValueError(
                f"rmsd {tag} atom-count mismatch (order pairing): target={len(tgt)} "
                f"vs ref={len(ref)}; provide atom names or matching selections"
            )
        sites = [int(a.index) for a in tgt]
        coords = np.asarray([(r.x, r.y, r.z) for r in ref], dtype=np.float64)
        return sites, coords.reshape(-1, 3)

    def is_valid(self) -> bool:
        return self.run_restr
