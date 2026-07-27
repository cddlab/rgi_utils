"""Parse + resolve nucleic-acid base-pair (and base-triple) restraints.

A base-pair restraint pins two nucleotides into Watson-Crick (WC) geometry. It is a
config-time MACRO, not a new energy term: each entry EXPANDS into the existing
primitives —

  * one harmonic / flat-bottomed **distance** restraint per WC hydrogen bond (the
    donor/acceptor atom pair), and
  * (optionally) one best-fit-**plane** restraint over the two bases' atoms, so the
    pair stays coplanar.

``hbonds: false`` keeps only the plane. The Watson-Crick lookup is then skipped
entirely, so ANY residues may be named -- a non-WC pair the table cannot describe (a
reverse-Hoogsteen U-A, whose real bonds are O2-N6/N3-N7 and not the WC N1-N3/N6-O4), a
wobble that is stacked rather than bonded, or a base docked on a G-A. Without it those
either raise or, worse, silently restrain the wrong atom pair.

An optional ``residue3`` extends the plane group to a BASE TRIPLE — a third base docked
onto the WC pair's groove edge, a recurring tertiary motif. Only the plane grows: no
H-bond distances are generated for the third base,
because which of its atoms bond depends on the base identity and the Leontis-Westhof
family, and no small table covers that. Give those with ``distance_restraints_config``.

A triple is NOT as flat as a WC pair. Measured over the base triples of four reference
structures (1.9-3.5 A), the out-of-plane RMS is
0.02-0.29 A for the WC pair alone but 0.15-0.44 A for the triple, the third base sitting
5-28 deg out of the pair's plane. Driving that to zero would flatten real geometry, so
``coplanar_slack`` defaults to ``_TRIPLE_SLACK`` when a third residue is present (a
one-sided flat bottom: nothing is penalised until the RMS exceeds it) and stays at 0 for
a plain pair.

This mirrors what servalcat/Refmac actually do (no dedicated base-pair energy; the
pairing is imposed as H-bond distance + planarity restraints). Reusing the distance /
plane terms means the base-pair restraint inherits their 3-backend energy, CG solver,
gating and finalize reporting for free — nothing new is added to the energy/optim
layers.

The engine has no monomer library, so the WC donor/acceptor atom pairs and their ideal
H-bond distances are the small hard-coded table below (base letter after stripping the
DNA ``D`` prefix; DNA and RNA share base-atom names). The paired nucleotides are named
by the user (``residue1`` / ``residue2`` selectors); auto-detection of pairs from
coordinates is intentionally NOT done (coordinates are pure noise at high sigma).

Each entry is gated on EITHER a sigma window (``start_sigma`` / ``stop_sigma``) OR a step
window (``start_step`` / ``stop_step``) — mutually exclusive, applied to the H-bond
DISTANCE restraints (the coplanarity plane rides the shared conformer gate; see
``combined.setup``).
"""

from __future__ import annotations

import logging

from rgi_utils._config_util import (
    apply_window_params,
    parse_move_indices,
    warn_unknown_keys,
)
from rgi_utils._moltype import moltype_from_resname
from rgi_utils.atom_context import FrameworkAdapter, candidate_dict
from rgi_utils.distance_restr_data import DistanceData
from rgi_utils.selection import _NUCLEIC_BACKBONE, AtomSelector

logger = logging.getLogger(__name__)

_KNOWN_BASE_PAIR_KEYS = {
    "residue1",
    "residue2",
    "residue3",
    "hbonds",
    "pair",
    "coplanar",
    "coplanar_slack",
    "weight",
    "start_sigma",
    "stop_sigma",
    "start_step",
    "stop_step",
    "move",
    "target",
}

# Watson-Crick donor/acceptor atom pairs, keyed by (base1, base2) in the CANONICAL
# orientation (purine first for GC/GU; the pyrimidine partner second). Each tuple is
# ``(atom_on_base1, atom_on_base2)`` — the distance restraint is symmetric so only the
# atom identities matter. DNA/RNA share these base-atom names (the sugar differs, the
# base does not). Target H-bond distance ~2.85-2.90 Angstrom (N...N / N...O).
_WC_ATOMS = {
    ("G", "C"): [("N1", "N3"), ("N2", "O2"), ("O6", "N4")],
    ("A", "T"): [("N1", "N3"), ("N6", "O4")],
    ("A", "U"): [("N1", "N3"), ("N6", "O4")],
    # G.U wobble (RNA). NOT auto-detected (a G and U near each other need not be a
    # wobble pair): only used when the user sets ``pair: GU`` explicitly.
    ("G", "U"): [("N1", "O2"), ("O6", "N3")],
}
# Pairs that AUTO-detection (resname-derived, no explicit ``pair`` key) may form. Wobble
# (GU/UG) is excluded — it is opt-in via ``pair`` only.
_CANONICAL_AUTO = frozenset(
    {("G", "C"), ("C", "G"), ("A", "T"), ("T", "A"), ("A", "U"), ("U", "A")}
)

# default WC H-bond distance flat-bottomed window (Angstrom)
_DEFAULT_TARGET = (2.7, 3.1)

# Default coplanarity slack (Angstrom out-of-plane RMS). A WC pair keeps 0 -- the value
# every existing config was written against. A triple gets room for the third base's real
# tilt: 0.44 A was the worst observed over the reference triples (see the module
# docstring), so 0.45 penalises only arrangements flatter geometry cannot explain.
_PAIR_SLACK = 0.0
_TRIPLE_SLACK = 0.45


def _normalise_name(name: str | None) -> str:
    # match polymer._normalise_name: upper-case + map the '*' prime spelling to "'".
    return (name or "").strip().upper().replace("*", "'")


def _lookup_wc(base1: str, base2: str):
    """Return the WC atom-pair list for ``(base1, base2)`` (atoms ordered base1, base2),
    or ``None`` if the two bases do not form a listed WC pair."""
    if (base1, base2) in _WC_ATOMS:
        return _WC_ATOMS[(base1, base2)]
    if (base2, base1) in _WC_ATOMS:
        # reverse orientation: swap each atom pair so the first atom is on base1
        return [(a2, a1) for (a1, a2) in _WC_ATOMS[(base2, base1)]]
    return None


def _base_letter(resname: str | None) -> str | None:
    """Standard-nucleotide base letter (A/C/G/T/U) from a resname, or None if the
    residue is not a standard DNA/RNA nucleotide. DA->A, DG->G, bare A/U/... unchanged."""
    if moltype_from_resname(resname) not in ("dna", "rna"):
        return None
    return resname.strip().upper()[-1]


class BasePairData:
    """One Watson-Crick base pair between two user-named nucleotides.

    ``residue1`` / ``residue2`` are selection-DSL strings that must each resolve to
    EXACTLY one residue. The base identities are auto-detected from ``resname`` unless a
    ``pair`` (e.g. ``"GC"``, ``"AU"``, ``"GU"``) overrides them (needed when resname is
    unavailable or for non-standard/wobble pairing). ``coplanar`` (default True) adds the
    inter-base planarity restraint. ``weight`` / ``move`` / the gate windows behave like
    the distance restraint (they flow into every generated H-bond distance entry).

    ``residue3`` (optional) names a third base that joins the plane group, making the
    entry a base TRIPLE. Its base identity is never checked against the WC table -- a
    third base docks on a groove edge and is non-WC by definition -- and it contributes
    no H-bonds. ``move`` still addresses only residues 1 and 2, since it acts on the
    H-bond distances and the third base has none.
    """

    residue1: str
    residue2: str
    residue3: str | None
    hbonds: bool  # False = coplanarity only, no WC lookup and no distances
    pair: (
        str | None
    )  # explicit base pair override, e.g. "GC" (None = auto from resname)
    coplanar: bool
    coplanar_slack: float | None  # None = pick by pair/triple at resolve time
    target: tuple  # WC H-bond distance flat-bottomed window (low, high) Angstrom
    weight: float
    move_mode: int  # 0=both / 1=residue1 only / 2=residue2 only (docking a strand)
    run_restr: bool
    start_sigma: float
    stop_sigma: float
    start_step: float
    stop_step: float

    def __init__(self):
        self.residue1 = None
        self.residue2 = None
        self.residue3 = None
        self.hbonds = True
        self.pair = None
        self.coplanar = True
        self.coplanar_slack = None
        self.target = _DEFAULT_TARGET
        self.weight = 1.0
        self.move_mode = 0
        self.run_restr = None
        self.start_sigma = None  # from_dict defaults None -> +inf (every step)
        self.stop_sigma = -1.0  # never released
        self.start_step = float(
            "-inf"
        )  # step-window (omitted -> always); XOR sigma win
        self.stop_step = float("inf")

    def set_config(self, config: dict):
        warn_unknown_keys(
            config, _KNOWN_BASE_PAIR_KEYS, "base_pair_restraints_config entry", logger
        )
        self.residue1 = config.get("residue1", None)
        self.residue2 = config.get("residue2", None)
        self.residue3 = config.get("residue3", None)
        self.hbonds = bool(config.get("hbonds", True))
        self.pair = config.get("pair", None)
        if self.pair is not None:
            self.pair = str(self.pair).strip().upper()
            if len(self.pair) != 2 or any(b not in "ACGTU" for b in self.pair):
                raise ValueError(
                    f"base_pair 'pair' must be two base letters from ACGTU "
                    f"(e.g. 'GC', 'AU', 'GU'), got {config.get('pair')!r}"
                )
        self.coplanar = bool(config.get("coplanar", True))
        slack = config.get("coplanar_slack")
        if slack is not None:
            self.coplanar_slack = float(slack)
            if self.coplanar_slack < 0.0:
                raise ValueError(
                    "base_pair 'coplanar_slack' is an out-of-plane RMS in Angstrom and "
                    f"cannot be negative (got {slack!r})"
                )
        if not self.hbonds and not self.coplanar:
            raise ValueError(
                "base_pair with hbonds: false and coplanar: false generates nothing. "
                "Turn one of them back on."
            )
        if self.residue3 is not None and not self.coplanar:
            raise ValueError(
                "base_pair residue3 only joins the coplanarity plane, so it is a no-op "
                "with coplanar: false. Drop residue3, or turn coplanar back on and give "
                "the third base's hydrogen bonds via distance_restraints_config."
            )
        # weight + the sigma/step gate windows: shared parse (start_sigma None -> +inf is
        # applied by config.from_dict, matching distance/rmsd/angle/dihedral).
        apply_window_params(self, config, "base_pair_restraints_config entry")
        # `move` (which residue the H-bonds pull): reuse the 2-group distance vocabulary
        # so `both`/`1`/`2` stay in lockstep. 1 -> only residue1 moves (dock it onto a
        # fixed residue2), 2 -> only residue2, both/omitted -> both.
        idx = parse_move_indices(config.get("move"), 2)
        if idx is not None:
            self.move_mode = {
                frozenset({1, 2}): 0,
                frozenset({1}): 1,
                frozenset({2}): 2,
            }[frozenset(idx)]
        # target H-bond distance window (OPTIONAL): a [low, high] pair -> flat-bottomed,
        # a single scalar -> harmonic. Default flat-bottomed (2.7, 3.1).
        tgt = config.get("target")
        if tgt is not None:
            if isinstance(tgt, (list, tuple)):
                if len(tgt) != 2 or float(tgt[0]) >= float(tgt[1]):
                    raise ValueError(
                        "base_pair 'target' list must be [low, high] with low < high "
                        f"(got {tgt!r})"
                    )
                self.target = (float(tgt[0]), float(tgt[1]))
            else:
                self.target = float(tgt)  # scalar -> harmonic
        self.run_restr = self.residue1 is not None and self.residue2 is not None
        if not self.run_restr:
            raise ValueError(
                "base_pair restraint needs both residue1 and residue2 selectors"
            )

    def _resolve_one_residue(
        self, adapter: FrameworkAdapter, selection: str, label: str
    ):
        """Resolve ``selection`` to EXACTLY one residue; return
        ``(name_to_global_index, resname)``. Raises loudly if the selection matches no
        atom or spans more than one residue (a base pair must name a single nucleotide)."""
        selector = AtomSelector(selection)
        name_to_index: dict[str, int] = {}
        residues: set = set()
        resname = None
        for atom in adapter.iter_atoms():
            if selector.eval(candidate_dict(atom)):
                residues.add((atom.chain, atom.resid))
                resname = atom.resname
                nm = _normalise_name(atom.name)
                if nm:
                    name_to_index[nm] = int(atom.index)
        if not residues:
            raise ValueError(
                f"base_pair {label} selection matched no atoms: {selection!r}"
            )
        if len(residues) != 1:
            raise ValueError(
                f"base_pair {label} selection must match exactly one residue, but "
                f"matched {len(residues)} residues ({sorted(residues)}): {selection!r}"
            )
        return name_to_index, resname

    def _base_atoms(self, name_to_index: dict[str, int]) -> list[int]:
        """Global indices of a residue's BASE atoms (heavy, non-sugar/phosphate), for the
        coplanarity plane group. The base is the complement of the nucleic backbone
        (sugar + phosphate); hydrogens are excluded by name."""
        return [
            gi
            for nm, gi in name_to_index.items()
            if nm not in _NUCLEIC_BACKBONE and not nm.startswith("H")
        ]

    def resolve_sites(self, adapter: FrameworkAdapter):
        """Expand this base pair into concrete restraints.

        Returns ``(distances, plane_group)`` where ``distances`` is a list of
        pre-resolved ``DistanceData`` (one per WC H-bond, ``target_sites`` already
        filled) and ``plane_group`` is ``(global_indices, slack, weight)`` for the
        coplanarity plane restraint (or ``None`` when ``coplanar`` is off). The distances
        are pre-resolved, so ``combined.setup`` merges them into the distance list
        WITHOUT a second ``resolve_sites`` pass (which would clobber ``target_sites``)."""
        if not self.run_restr:
            return [], None
        map1, resname1 = self._resolve_one_residue(adapter, self.residue1, "residue1")
        map2, resname2 = self._resolve_one_residue(adapter, self.residue2, "residue2")

        # base identities: explicit `pair` override, else auto-detect from resname.
        # Skipped entirely for a plane-only entry -- there is no table to consult, and
        # demanding a canonical pair is exactly what makes those entries impossible.
        if not self.hbonds:
            base1 = _base_letter(resname1) or "?"
            base2 = _base_letter(resname2) or "?"
            atom_pairs = []
        elif self.pair is not None:
            base1, base2 = self.pair[0], self.pair[1]
            atom_pairs = _lookup_wc(base1, base2)
            if atom_pairs is None:
                raise ValueError(
                    f"base_pair pair={self.pair!r} is not a listed Watson-Crick pair "
                    f"(supported: GC, AT, AU, GU wobble, and reverses)"
                )
        else:
            base1, base2 = _base_letter(resname1), _base_letter(resname2)
            if base1 is None or base2 is None:
                raise ValueError(
                    "base_pair could not identify a nucleotide from resname "
                    f"(residue1 resname={resname1!r}, residue2 resname={resname2!r}). "
                    "Set an explicit `pair:` (e.g. 'GC') on the entry."
                )
            if (base1, base2) not in _CANONICAL_AUTO:
                raise ValueError(
                    f"base_pair auto-detected {base1}-{base2}, not a canonical "
                    "Watson-Crick pair (GC/AT/AU). For a G-U wobble or a non-canonical "
                    "pair set `pair:` explicitly (e.g. 'GU')."
                )
            atom_pairs = _lookup_wc(base1, base2)

        distances = []
        for name_a, name_b in atom_pairs:
            if name_a not in map1:
                raise ValueError(
                    f"base_pair {base1}-{base2}: atom {name_a} missing in residue1 "
                    f"({self.residue1!r})"
                )
            if name_b not in map2:
                raise ValueError(
                    f"base_pair {base1}-{base2}: atom {name_b} missing in residue2 "
                    f"({self.residue2!r})"
                )
            distances.append(
                self._make_hbond(name_a, map1[name_a], name_b, map2[name_b])
            )

        plane_group = None
        base3 = None
        if self.coplanar:
            atoms = self._base_atoms(map1) + self._base_atoms(map2)
            if self.residue3 is not None:
                # The third base joins the plane only. Its identity is NOT checked
                # against the WC table: docked on a groove edge, it is non-WC by
                # definition, and any base can sit there.
                map3, resname3 = self._resolve_one_residue(
                    adapter, self.residue3, "residue3"
                )
                third = self._base_atoms(map3)
                if not third:
                    raise ValueError(
                        f"base_pair residue3 ({self.residue3!r}) contributed no base "
                        f"atoms to the coplanarity restraint (resname={resname3!r}); "
                        "it must name a nucleotide"
                    )
                atoms += third
                base3 = _base_letter(resname3) or "?"
            if len(atoms) < 3:
                raise ValueError(
                    f"base_pair {base1}-{base2}: fewer than 3 base atoms found for the "
                    "coplanarity restraint; set coplanar: false or check the residues"
                )
            # slack: explicit `coplanar_slack` wins, else 0 for a pair (what every
            # existing config was written against) and _TRIPLE_SLACK for a triple, whose
            # third base is genuinely tilted out of the pair's plane. The `target` key
            # tunes only the H-bond distance, not this plane.
            slack = self.coplanar_slack
            if slack is None:
                slack = _PAIR_SLACK if self.residue3 is None else _TRIPLE_SLACK
            plane_group = (atoms, slack, self.weight)

        logger.info(
            "base_pair resolved: %s-%s%s -> %d h-bonds%s",
            base1,
            base2,
            f"-{base3}" if base3 is not None else "",
            len(distances),
            f" + coplanar({len(plane_group[0])} atoms, slack {plane_group[1]:g})"
            if plane_group is not None
            else "",
        )
        return distances, plane_group

    def _make_hbond(
        self, name_a: str, idx_a: int, name_b: str, idx_b: int
    ) -> DistanceData:
        """Build one pre-resolved harmonic/flat-bottomed distance restraint between the
        two named atoms, carrying this entry's weight / move / gate window."""
        dd = DistanceData()
        # descriptive selections (not re-resolved; target_sites are set directly)
        dd.atom_selection1 = f"base_pair {self.residue1} {name_a}"
        dd.atom_selection2 = f"base_pair {self.residue2} {name_b}"
        dd.calc_method = "unfixed-absolute"
        dd.target_sites1 = [idx_a]
        dd.target_sites2 = [idx_b]
        if isinstance(self.target, tuple):
            dd.distance_restraint_type = "flat-bottomed"
            dd.target_distance1, dd.target_distance2 = self.target
        else:
            dd.distance_restraint_type = "harmonic"
            dd.target_distance = self.target
        dd.move_mode = self.move_mode
        dd.weight = self.weight
        dd.start_sigma = self.start_sigma
        dd.stop_sigma = self.stop_sigma
        dd.start_step = self.start_step
        dd.stop_step = self.stop_step
        dd.run_restr = True
        return dd

    def is_valid(self) -> bool:
        return bool(self.run_restr)
