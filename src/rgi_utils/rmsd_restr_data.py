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
selection-order pairing (the original behaviour).

Pairing is BEST-EFFORT by default (PyMOL align/super-like): a target atom with no
matching (chain, resid, name) in the reference is SKIPPED, so a partially-overlapping
reference (missing hydrogens/side chains, an incomplete model) still fits + measures
over whatever overlaps. It still raises if NOTHING overlaps, so a wholly-wrong
selection is not silent. Set ``best_effort: false`` on the entry for STRICT pairing
that raises on the first unmatched atom (catch a mistyped selection loudly). The
order-fallback path (no atom names) always requires equal counts regardless.

``pairing`` is **"align" by DEFAULT** (set ``pairing: "identity"`` for the pure
ordinal pairing above). align matches a **homolog** reference (different sequence,
substitutions and indels): each polymer chain is sequence-aligned (``_align``:
BLOSUM62 for protein, residue-name identity for nucleic acids, semi-global with free
end gaps) so target residues map onto the corresponding ref residues regardless of
numbering or register, then atoms pair by name within each aligned residue pair. align
**only engages when the structure has polymer atoms** -- a ligand-only structure (or
any atom lacking a polymer type) falls back to ordinal identity, so the default is safe
on non-polymer inputs and never demands a sequence where there is none. NOTE align
pairs ALL shared atom names -- backbone (N/CA/C/O) PLUS CB and any side-chain names
that coincide -- so for a substituted residue the prediction's matching side-chain
atoms are pinned onto the REFERENCE's side-chain coordinates. To avoid that pinning,
restrict the selection to the backbone with a ``backbone`` / ``name CA`` atom_selection,
so only those atoms are superposed; this is PyMOL-align without the outlier-rejection
cycles. align defaults to best-effort (gap/unshared atoms skipped), but ``best_effort:
false`` with an EXPLICIT selection is honoured -- a residue aligned to a gap then raises
(no longer a silent no-op). align needs residue names on both sides (the reference
always has them; the target needs an adapter that fills ``AtomRecord.resname`` -- it
raises loudly otherwise). Ligand / non-polymer atoms stay on ordinal (chain, resid,
name) identity even under align.

``start_sigma`` / ``stop_sigma`` bound the NOISE WINDOW in which the restraint acts:
active when ``stop_sigma <= sigma <= start_sigma`` (``start_sigma`` defaults to +inf =
on from the first step; ``stop_sigma`` defaults to -1 = never released). Setting
``stop_sigma > 0`` RELEASES the restraint for the final low-sigma steps so the model's
own denoising re-idealises geometry the restraint would otherwise hold distorted. This
is the fix for a broken peptide bond at the junction between a restrained residue and a
FREE unmodeled tail: with ``target_rmsd=0`` the CG drives the restrained residue exactly
onto the reference every step (a per-atom weight only changes the convergence RATE, not
the fixed point, so down-weighting the terminus does NOT help), while the free tail lags
and the bond snaps (length AND omega planarity). Releasing the restraint below
``stop_sigma`` lets the final steps pull the bond back to ideal; the global reference
bias, established over the earlier (higher-sigma) steps, survives because low-sigma
denoising only refines locally. Pick ``stop_sigma`` in the model's sigma units; boltz2
(sigma_data=16, ~2560 -> ~0.006 over 200 steps) was validated at ``stop_sigma: 1.0``
(bond fully healed, ref CA-RMSD held ~0.3 A); the other tools share sigma_data=16.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from rgi_utils._align import pair_residues
from rgi_utils._config_util import coerce_bool, warn_unknown_keys
from rgi_utils._moltype import polymer_type
from rgi_utils.atom_context import FrameworkAdapter
from rgi_utils.pdb_ref import read_pdb_atoms
from rgi_utils.selection import AtomSelector

logger = logging.getLogger(__name__)

_KNOWN_RMSD_KEYS = {
    "ref_pdb",
    "target_rmsd",
    "weight",
    "start_sigma",
    "stop_sigma",
    "atom_selection_ref",
    "atom_selection_target",
    "atom_selection_ref_fit",
    "atom_selection_target_fit",
    "atom_selection_ref_calc",
    "atom_selection_target_calc",
    "best_effort",
    "pairing",
}


@dataclass
class RmsdData:
    ref_pdb: str = None
    target_rmsd: float = None
    weight: float = None
    start_sigma: float = None  # per-restraint; from_dict defaults None -> +inf
    # per-restraint LOWER noise bound: the restraint is RELEASED for sigma < stop_sigma,
    # so the model's final low-sigma denoising steps re-idealise geometry the restraint
    # would otherwise hold distorted -- notably the peptide bond between a restrained
    # residue and a FREE unmodeled tail (target_rmsd=0 drives the restrained residue
    # onto the reference while the tail lags, snapping the bond; releasing late lets the
    # model repair it). -1 (default) = never released (sigma>=-1 always true), active
    # down to sigma=0 (old behaviour); any value <= 0 is "off".
    stop_sigma: float = -1.0
    # selection strings (fit = superposition atoms, calc = measured atoms)
    sel_ref_fit: str = None
    sel_target_fit: str = None
    sel_ref_calc: str = None
    sel_target_calc: str = None
    # PyMOL align/super-like tolerant matching: skip target atoms with no
    # (chain, resid, name) match in the ref instead of raising. DEFAULT True (set
    # best_effort:false for strict pairing that raises on any unmatched atom). If
    # NOTHING matches it still raises, so a wholly-wrong selection is not silent.
    best_effort: bool = True
    # residue correspondence: "identity" = pair by (chain, resid, name) ordinal;
    # "align" (the DEFAULT, set from config in set_config; None here pre-config) =
    # sequence-align polymer chains first (BLOSUM62/identity) so a homolog reference
    # with substitutions/indels maps on (PyMOL align-like). align engages only when
    # polymer atoms exist (else identity). Populated in resolve_sites.
    pairing: str = None
    resid_map: dict = field(default=None)  # (chain, target_resid) -> ref_resid (align)
    # resolved: global target atom indices + paired reference coords (n_atoms, 3)
    fit_target_sites: list = field(default=None)
    fit_ref_coords: np.ndarray = field(default=None)
    calc_target_sites: list = field(default=None)
    calc_ref_coords: np.ndarray = field(default=None)
    run_restr: bool = None

    def set_config(self, config: dict):
        warn_unknown_keys(
            config, _KNOWN_RMSD_KEYS, "rmsd_restraints_config entry", logger
        )
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
        # release the restraint below this noise level so the model re-idealises the
        # boundary geometry in its final steps (-1, the default, or any value <= 0 ->
        # never released).
        _stop = config.get("stop_sigma")
        self.stop_sigma = -1.0 if _stop is None else float(_stop)
        # explicit _fit / _calc override the shared ref/target shorthand. A selection
        # left None means "the whole structure on that side" (resolved best-effort).
        ref = config.get("atom_selection_ref")
        tgt = config.get("atom_selection_target")
        self.sel_ref_fit = config.get("atom_selection_ref_fit", ref)
        self.sel_target_fit = config.get("atom_selection_target_fit", tgt)
        self.sel_ref_calc = config.get("atom_selection_ref_calc", ref)
        self.sel_target_calc = config.get("atom_selection_target_calc", tgt)
        # tolerate partial topology overlap by default (skip unmatched atoms); set
        # best_effort:false for strict pairing that raises on any unmatched atom.
        self.best_effort = coerce_bool(config.get("best_effort"), True)
        # DEFAULT "align": polymer chains (protein/dna/rna) are sequence-aligned so a
        # homolog reference maps on by residue, not by fragile ordinal numbering;
        # non-polymer atoms (ligands) always stay on ordinal identity. align only
        # ENGAGES when the structure actually has polymer atoms (see resolve_sites), so
        # this default never forces alignment on a ligand-only structure. Set
        # pairing:"identity" to force pure ordinal pairing everywhere.
        self.pairing = config.get("pairing") or "align"
        if self.pairing not in ("identity", "align"):
            raise ValueError(
                f"rmsd pairing must be 'identity' or 'align', got {self.pairing!r}"
            )
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
        # "align": sequence-align polymer chains so a homolog reference (substitutions,
        # indels) maps onto the prediction by residue, not by ordinal. Built once and
        # reused for both fit and calc. Polymer atoms are then keyed by the aligned ref
        # resid; ligands stay on (chain, resid, name) identity. align ENGAGES only when
        # the structure has polymer atoms -- so the "align" DEFAULT degrades to pure
        # identity on a bare/ligand-only structure (no resname, no names) instead of
        # demanding alignment of things that have no sequence. With polymer atoms but no
        # alignable chain (chain-id mismatch), _build_resid_map raises loudly.
        has_polymer = any(
            polymer_type(a.mol_type, a.resname) is not None for a in atoms
        )
        align = self.pairing == "align" and has_polymer
        if align:
            self.resid_map = self._build_resid_map(atoms, ref_atoms)
        # no target selection -> whole structure, paired best-effort (skip atoms missing
        # from the ref); an explicit selection stays strict when best_effort:false.
        # align no longer forces best-effort: a gap/unmatched atom raises under
        # best_effort:false (see _pair), so strict pairing is honoured even with align.
        self.fit_target_sites, self.fit_ref_coords = self._pair(
            atoms,
            ref_atoms,
            self.sel_target_fit,
            self.sel_ref_fit,
            "fit",
            best_effort=self.sel_target_fit is None or self.best_effort,
            align=align,
        )
        self.calc_target_sites, self.calc_ref_coords = self._pair(
            atoms,
            ref_atoms,
            self.sel_target_calc,
            self.sel_ref_calc,
            "calc",
            best_effort=self.sel_target_calc is None or self.best_effort,
            align=align,
        )
        logger.info(
            "rmsd restraint resolved: fit=%d calc=%d atoms, target_rmsd=%.3f",
            len(self.fit_target_sites),
            len(self.calc_target_sites),
            self.target_rmsd,
        )

    def _build_resid_map(self, atoms, ref_atoms) -> dict:
        """Sequence-align each polymer chain present on both sides and return
        ``{(chain, target_resid): ref_resid}``. Polymer typing prefers an explicit
        ``mol_type`` (boltz/esm set it) and otherwise derives it from the residue name
        (protenix/of3/chai don't set mol_type), so only resname must be plumbed."""

        def seqs(records, resname_attr, side):
            by_chain: dict = {}
            mtype: dict = {}
            for a in records:
                rn = getattr(a, resname_attr, None)
                ptype = polymer_type(getattr(a, "mol_type", None), rn)
                if ptype is None:
                    continue  # ligand/water/unknown -> not sequence-aligned
                if not rn:  # a polymer residue with no name can't be aligned
                    raise ValueError(
                        f"rmsd pairing='align' needs residue names on the {side} side, "
                        f"but a polymer atom (chain {a.chain}) has none"
                        + (
                            ""
                            if side == "reference"
                            else " (adapter not plumbed for resname)"
                        )
                    )
                d = by_chain.setdefault(a.chain, {})
                d.setdefault(a.resid, rn)
                mtype.setdefault(a.chain, ptype)
            return {c: sorted(d.items()) for c, d in by_chain.items()}, mtype

        t_seq, t_mt = seqs(atoms, "resname", "target")
        r_seq, _ = seqs(ref_atoms, "res_name", "reference")
        resid_map: dict = {}
        matched_chains = 0
        for ch, t_res in t_seq.items():
            r_res = r_seq.get(ch)
            if not r_res:
                continue
            matched_chains += 1
            for t_rid, r_rid in pair_residues(t_res, r_res, t_mt.get(ch)):
                resid_map[(ch, t_rid)] = r_rid
        if not resid_map:
            raise ValueError(
                "rmsd pairing='align' aligned no residues (no common polymer chain, or "
                "chain ids differ between prediction and ref_pdb "
                f"{self.ref_pdb!r}); check chain naming"
            )
        logger.info(
            "rmsd align: %d chain(s), %d residue pairs", matched_chains, len(resid_map)
        )
        return resid_map

    def _pair(
        self, atoms, ref_atoms, sel_target, sel_ref, tag, best_effort=False, align=False
    ):
        """Resolve one (target, ref) selection pair -> (target_global_indices, ref_coords
        aligned to the target order). A ``None`` selection means the WHOLE structure on
        that side (no filter). Pairing is by IDENTITY (chain, resid, name) when both sides
        expose names; else selection-order (counts must match). With ``best_effort`` (the
        no-selection whole-structure default) a target atom missing from the reference is
        SKIPPED rather than raising, so an incomplete ref still fits 'as much as possible';
        with an explicit selection it stays strict (a missing match raises). With
        ``align`` a polymer target atom's resid is first translated to the aligned ref
        resid via ``self.resid_map`` (ligands stay on ordinal identity)."""
        if sel_target is None:  # whole structure (no filter)
            tgt = list(atoms)
        else:
            st = AtomSelector(sel_target)
            tgt = [
                a
                for a in atoms
                if st.matches(
                    {
                        "chain": a.chain,
                        "resid": a.resid,
                        "index": a.index,
                        "mol_type": a.mol_type,
                        "name": a.name,
                        "resname": a.resname,
                    }
                )
            ]
        if sel_ref is None:  # whole reference (no filter)
            ref = list(ref_atoms)
        else:
            sr = AtomSelector(sel_ref)
            ref = [
                r
                for r in ref_atoms
                if sr.matches(
                    {
                        "chain": r.chain,
                        "resid": r.resid,
                        "index": r.index,
                        "mol_type": r.mol_type,
                        "name": r.name,
                        "resname": r.res_name,
                    }
                )
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
        if align and not (tgt_named and ref_named):
            raise ValueError(
                f"rmsd {tag} pairing='align' needs atom names on both sides to pair "
                "atoms within aligned residues"
            )
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
                if align and polymer_type(a.mol_type, a.resname) is not None:
                    # polymer: translate target resid -> aligned ref resid. No entry =
                    # residue aligned to a gap (the homolog ref lacks it); skip under
                    # best_effort, else raise so best_effort:false is honoured (not a
                    # silent no-op). Ligand atoms fall to ordinal identity below.
                    mapped = self.resid_map.get((a.chain, a.resid))
                    if mapped is None:
                        if best_effort:
                            skipped += 1
                            continue
                        raise ValueError(
                            f"rmsd {tag}: target polymer residue (chain {a.chain}, "
                            f"resid {a.resid}) aligned to a gap in ref "
                            f"{self.ref_pdb!r} (no corresponding residue); set "
                            f"best_effort:true to skip gaps"
                        )
                    key = (a.chain, mapped, a.name)
                else:
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
                    "rmsd %s (best-effort): matched %d / %d atoms "
                    "(%d unmatched in ref skipped)",
                    tag,
                    len(sites),
                    len(tgt),
                    skipped,
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
