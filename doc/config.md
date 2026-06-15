# `restraints_config` reference

Every RGI tool is driven by one `restraints_config` dict (a YAML/JSON block for boltz / protenix /
chai / AF3 / openfold, or a Python dict for ESMFold2 — see each tool's page for where it lives).
This page documents **every variable**: its type, default, allowed values, and meaning.

The parser reads keys with `.get(key, default)`, so **omitted keys fall back to the default** and
**unknown keys are silently ignored** (a typo neither errors nor takes effect — check spelling
against this page). Source of truth: `rgi_utils/src/rgi_utils/{config,distance_restr_data,
group_geom_restr_data,featurizer,rmsd_restr_data,selection}.py`.

## Shape

```yaml
restraints_config:
  # --- top-level knobs ---
  verbose: ...        # bool
  gpu: ...            # bool
  backend: ...        # "torch" | "jax" | null
  method: ...         # "CG" | "l-bfgs"
  max_iter: ...       # int
  # --- restraints (each block optional) ---
  distance_restraints_config: [ ... ]   # list
  angle_restraints_config:    [ ... ]   # list  (group-centroid angle)
  dihedral_restraints_config: [ ... ]   # list  (group-centroid dihedral)
  conformer_restraints_config: { ... }  # single dict (ligand geometry)
  rmsd_restraints_config:     [ ... ]   # list
```

A restraint type is active only if its block is present (and, for conformer terms, the term's
`weight > 0`).

## Top-level keys

| key | type | default | meaning |
|---|---|---|---|
| `verbose` | bool | `false` | Log the built spec (per-restraint counts) at setup and per-term energies at finalize. Strongly recommended — it is how you confirm a restraint was actually built. |
| `gpu` | bool | `false` | Torch **device**: `true` = accelerator, `false` = CPU. It does **not** change the backend. (Inert for AF3, which always runs the JAX minimizer on the model's device.) Accepts `true/false` and the strings `1/0/yes/no/on/off`. |
| `backend` | str / null | `null` → `torch` | Compute backend: `"torch"` (default) or `"jax"` (the AF3 tool forces `jax`). `"numpy"` is rejected (no numpy optimizer). |
| `method` | str | `"CG"` | Optimizer: `"CG"` (nonlinear conjugate gradient) or `"l-bfgs"` (opt-in). |
| `max_iter` | int | `100` | Max optimizer iterations per denoising step. The examples use 1000 (2000 for AF3). |

**There is no top-level `start_sigma` / `stop_sigma`** — setting one at the top level raises. They
are per-restraint (see below).

## Sigma gating: `start_sigma` & `stop_sigma`

Diffusion runs from a high noise level (`sigma`) down to ~0. Each restraint is gated to a `sigma`
window:

- **`start_sigma`** (float) — the restraint is active only once `sigma <= start_sigma`. **Omitted →
  `+inf`** (active at *every* step). Set e.g. `1.0` to act only late in denoising. The example value
  `99999999` is "always on".
- **`stop_sigma`** (float) — the restraint is **released** once `sigma < stop_sigma`. **Omitted →
  `-1`** (never released; any value `<= 0` means off).
- **Active window:** `stop_sigma <= sigma <= start_sigma`. `stop_sigma > start_sigma` is an empty
  window (warned).

Where they live: **once per `distance` / `angle` / `dihedral` / `rmsd` entry**, and **once for all
conformer terms** (`conformer_restraints_config.start_sigma` / `.stop_sigma`). When `sigma` exceeds
every active restraint's `start_sigma`, the whole minimization step is skipped (cheap at high noise).

The RMSD restraint's `stop_sigma` has a specific use: releasing it late (e.g. `1.0`) lets the model
re-idealise geometry the restraint held distorted — the fix for a peptide bond broken at a
restrained-residue / free unmodeled-tail boundary.

## Atom-selection DSL

Atom groups are chosen with a small boolean language (`selection.py`). Precedence: `not` > `and` >
`or`; parenthesise to override.

| token | matches | example |
|---|---|---|
| `chain <ids>` | atoms in the listed chain IDs | `chain A` · `chain A B` |
| `resid <n…>` / `resid A to B` | residues by **per-chain 1-based ordinal** | `resid 5` · `resid 1 to 84` · `resid 1 3 7` |
| `index <n…>` | atoms by **flat row** in the coordinate tensor | `index 42` |
| `name <names>` | atom name, case-insensitive (alphanumeric only) | `name CA` · `name N CA C O` |
| `protein` / `dna` / `rna` | atoms of that polymer type | `protein` |
| `backbone` / `sidechain` | polymer backbone / sidechain heavy atoms (gated on polymer type; a ligand atom named "C" never matches) | `backbone and chain A` |
| `and` / `or` / `not` / `( )` | boolean composition | `chain A and (resid 5 to 84 or resid 186 to 224)` |

**Critical convention:** `resid` is the **per-chain 1-based ordinal** — it resets at each chain, and
a ligand atom gets its own ordinal. It is **not** 0-based and **not** the author residue number.
Always qualify protein groups with `chain A and (...)`, or a bare `resid` range will also match the
ligand chain's atoms with the same ordinal. This convention is identical across all tools.

## `distance_restraints_config` (list)

Pulls the **centroid distance** between two atom groups toward a target. Applied closed-form (1-DOF),
so it has **no `weight`**.

| key | type | default | meaning |
|---|---|---|---|
| `atom_selection1` | str | — (required) | group 1 (selection DSL) |
| `atom_selection2` | str | — (required) | group 2 |
| `start_sigma` | float | `+inf` | activation upper bound (see Sigma gating) |
| `stop_sigma` | float | `-1` | release lower bound |
| `move` | `"both"`/`1`/`2` | `"both"` | which group the correction moves: `both` = minimal-displacement split; `1` / `2` move only that group and **pin** the other (e.g. move a ligand toward a fixed pocket) |
| one restraint-type block | dict | — (required) | the penalty (below) |

Restraint-type block (exactly one):

| block | params | behaviour |
|---|---|---|
| `harmonic` | `target_distance` | quadratic penalty everywhere toward the target |
| `flat-bottomed` | `target_distance1`, `target_distance2` | no penalty inside `[d1, d2]` (needs `d1 < d2`) |
| `flat-bottomed1` | `target_distance1` | penalise only below `d1` |
| `flat-bottomed2` | `target_distance2` | penalise only above `d2` |

## `angle_restraints_config` / `dihedral_restraints_config` (lists)

The **angle of 3 group centroids** (vertex = group 2) or the **dihedral of 4 group centroids** (axis
= groups 2–3). These are the group-centroid analogues of the distance restraint — distinct from the
per-ligand-atom conformer `angle`/`dihedral` terms. Targets are in **degrees**. CG-solved; rigid
group motion means `weight: 1.0` drives any group size.

| key | type | default | meaning |
|---|---|---|---|
| `atom_selection1..3` (angle) / `..4` (dihedral) | str | — (required) | the groups; group 2 is the angle vertex, groups 2–3 the dihedral axis |
| `start_sigma` / `stop_sigma` | float | `+inf` / `-1` | sigma gating |
| `weight` | float | `1.0` | energy scale |
| `move` | `"all"` / int / list / `"1,3"` | angle: arms (1,3) free, vertex pinned · dihedral: ends (1,4) free, axis pinned | which groups are free; the rest are pinned (stop-gradient). `"all"` frees every group |
| one restraint-type block | dict | — (required) | `harmonic {target_angle\|target_dihedral}` or `flat-bottomed{,1,2}` with `target_angle1/2` / `target_dihedral1/2` (degrees) |

`harmonic` for the dihedral is periodicity-safe (deviation wrapped to ±180°); `flat-bottomed`
windows cannot straddle ±180°.

## `conformer_restraints_config` (single dict)

Holds a **ligand** at ideal RDKit geometry. It is a single dict (not a list); it applies to the
ligand(s) the tool exposes. For boltz / protenix / AF3 a ligand opts in with `conformer_restraints:
true`; chai and openfold apply it to all ligands when this block is present.

Top-level (shared by all terms): `start_sigma` (`+inf`), `stop_sigma` (`-1`).

Each term is a sub-dict; a term with `weight <= 0` is disabled (and not built).

| term | keys (default) | meaning |
|---|---|---|
| `bond` | `weight` (0.05), `slack` (0.0 Å) | bond lengths toward ideal; flat-bottomed by `slack` |
| `angle` | `weight` (0.05), `slack` (0.0 rad) | bond angles toward ideal |
| `chiral` | `weight` (0.1), `slack` (0.05) | chiral volume (stereochemistry) — holds each stereocentre's handedness |
| `dihedral` | `weight` (0.1), `slack` (0.0 rad) | **cis/trans (E/Z)** of acyclic, non-aromatic double bonds (needs real bond orders; detects 0 for ligands with none, e.g. ATP/NAD/GLN) |
| `vdw` | `weight` (**0.0**, off), `mode` (`"both"`), `scale` (0.75), `dmax` (5.0 Å) | non-bonded clash avoidance |

`vdw.mode`: `"intramolecular"` (static ligand-internal pairs, all backends), `"ligand_protein"`
(dynamic ligand-vs-fixed-protein, torch/jax), or `"both"` (default). `scale` = fraction of the
summed VdW radii used as the contact threshold; `dmax` = pairs farther than this are ignored. Note
`vdw.weight` defaults to **0** — set it `> 0` to activate.

## `rmsd_restraints_config` (list)

Drives the **Kabsch-superposed RMSD** of a moving group toward `target_rmsd` versus a reference PDB.
The reference must be generated first (a vanilla prediction → PDB via gemmi); see each tool's page.

| key | type | default | meaning |
|---|---|---|---|
| `ref_pdb` | str | — (required) | path to the reference PDB |
| `target_rmsd` | float | — (required) | target RMSD in Å (`0.0` = match the reference) |
| `weight` | float | `1.0` | energy scale |
| `start_sigma` / `stop_sigma` | float | `+inf` / `-1` | sigma gating (set `stop_sigma`, e.g. `1.0`, to release late and heal a strained terminus) |
| `pairing` | `"align"` / `"identity"` | `"align"` | how reference and prediction residues correspond. `align` = sequence-align polymer chains (BLOSUM62) so a **homolog** ref maps on despite substitutions/indels/renumbering; non-polymer atoms always pair by ordinal. `identity` = strict (chain, resid, name) ordinal pairing. |
| `best_effort` | bool | `true` | skip atoms with no match in the ref (instead of raising); `false` = strict |

**Atom selection** (all optional; omit all → whole structure, best-effort). The superposed ("fit")
and measured ("calc") atom sets are chosen independently:

| key | sets |
|---|---|
| `atom_selection_ref` / `atom_selection_target` | shorthand: **both** fit and calc, on the ref / target side |
| `atom_selection_ref_fit` / `atom_selection_target_fit` | atoms used for the Kabsch superposition |
| `atom_selection_ref_calc` / `atom_selection_target_calc` | atoms over which the RMSD is measured |

Use the shorthand for the common case (fit = calc); use the four `_fit`/`_calc` keys to e.g.
superpose on the backbone but measure over a pocket. Under `pairing: align`, restrict the fit to
`name CA` or `backbone` so a substituted homolog's side chain is not pinned.
