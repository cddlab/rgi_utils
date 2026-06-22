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
  custom_restraints_config:   [ ... ]   # list  (define your OWN restraint — see below)
```

A restraint type is active only if its block is present (and, for conformer terms, the term's
`weight > 0`).

The five blocks above (distance / angle / dihedral / conformer / rmsd) are the **built-ins**.
`custom_restraints_config` lets you define an **original** restraint as a math formula — see the
last section.

## Top-level keys

| key | type | default | meaning |
|---|---|---|---|
| `verbose` | bool | `false` | Log the built spec (per-restraint counts) at setup and per-term energies at finalize. Strongly recommended — it is how you confirm a restraint was actually built. |
| `gpu` | bool | `true` | Torch **device**: `true` = accelerator (default), `false` = CPU. It does **not** change the backend. (Inert for AF3, which always runs the JAX minimizer on the model's device.) Accepts `true/false` and the strings `1/0/yes/no/on/off`. |
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
  window and **raises** (a silent no-op would read as "satisfied").

Where they live: **once per `distance` / `angle` / `dihedral` / `rmsd` / `custom` entry**, and **once
for all conformer terms** (`conformer_restraints_config.start_sigma` / `.stop_sigma`). When `sigma`
exceeds every active restraint's `start_sigma`, the whole minimization step is skipped (cheap at high
noise).

The RMSD restraint's `stop_sigma` has a specific use: releasing it late (e.g. `1.0`) lets the model
re-idealise geometry the restraint held distorted — the fix for a peptide bond broken at a
restrained-residue / free unmodeled-tail boundary.

## Step gating: `start_step` & `stop_step` (alternative to sigma)

Instead of the noise level, a restraint may be gated on the **diffusion step index** (the 0-based
denoising iteration counter). Same shape as the sigma window, on the step axis:

- **`start_step`** (int) — active once `step >= start_step`. **Omitted → `-inf`** (active from the
  first step).
- **`stop_step`** (int) — released once `step > stop_step`. **Omitted → `+inf`** (never released).
- **Active window:** `start_step <= step <= stop_step`. `stop_step < start_step` is an empty window
  and **raises**.

Available on the same entries as the sigma window (`distance` / `angle` / `dihedral` / `rmsd` /
`custom`, and the shared `conformer_restraints_config`).

> **A restraint uses EITHER the sigma window OR the step window — never both.** Setting any of
> `start_sigma` / `stop_sigma` together with any of `start_step` / `stop_step` on one entry **raises**
> (mutually exclusive — 排他選択). The unused axis stays always-on.

> ⚠️ **Step windows are NOT portable across tools.** The number of denoising steps differs per tool
> (e.g. boltz vs AF3 vs protenix), so `start_step: 50` means a *different point in the trajectory* in
> each tool. `sigma` windows ARE portable (all tools share `sigma_data=16`). Prefer `sigma` windows
> unless you specifically need step-index control for one tool.

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

## Penalty shapes (shared)

Every built-in restraint is defined by the same squared penalty,

```
E = Σᵢ wᵢ · δᵢ²
```

where `wᵢ` is the entry `weight` and `δᵢ` is the deviation of a measured quantity `x` (a distance,
angle, volume, …) from its target. Four block names choose how `δ` is shaped:

```
harmonic         δ = x − t                         penalise any deviation from t
flat-bottomed    δ = 0          for t₁ ≤ x ≤ t₂    no penalty inside the window
                 δ = x − t₁     for x < t₁
                 δ = x − t₂     for x > t₂
flat-bottomed1   δ = min(0, x − t₁)                lower bound (penalise only x < t₁)
flat-bottomed2   δ = max(0, x − t₂)                upper bound (penalise only x > t₂)
```

The same four shapes drive the `distance` / `angle` / `dihedral` blocks (only the target key
differs: `target_distance` / `target_angle` / `target_dihedral`, with `…1` / `…2` for the
flat-bottomed bounds). The **conformer** terms use the flat-bottomed shape with a symmetric `slack`:
`δ = 0` within `±slack` of the RDKit-ideal value, quadratic outside (`slack = 0` ⇒ pure harmonic).

`distance` is the one restraint with **no `weight`**: a centroid distance is 1-DOF, so it is solved
closed-form (the `harmonic` minimum `d = target` is reached exactly in a single shift) instead of
being added to the optimiser objective. Every other restraint is CG-minimised and scales with
`weight`.

## `distance_restraints_config` (list)

Pulls the **centroid distance** between two atom groups toward a target. Applied closed-form (1-DOF),
so it has **no `weight`**.

The measured quantity is the distance between the two groups' centroids,

```
d = ‖c₂ − c₁‖,    cₖ = (1 / |Gₖ|) Σ_{a ∈ Gₖ} xₐ    (plain masked-mean centroid)
```

shaped by one of the penalty blocks below (see Penalty shapes); being closed-form, `harmonic`
reaches `d = target_distance` exactly.

| key | type | default | meaning |
|---|---|---|---|
| `atom_selection1` | str | — (required) | group 1 (selection DSL) |
| `atom_selection2` | str | — (required) | group 2 |
| `start_sigma` | float | `+inf` | activation upper bound (see Sigma gating) |
| `stop_sigma` | float | `-1` | release lower bound |
| `start_step` / `stop_step` | int | `-inf` / `+inf` | step gating — the alternative to the sigma window (mutually exclusive; see Step gating) |
| `move` | `"both"`/`1`/`2` | `"both"` | which group the correction moves: `both` = minimal-displacement split; `1` / `2` move only that group and **pin** the other (e.g. move a ligand toward a fixed pocket) |
| one restraint-type block | dict | — (required) | the penalty (below) |

Restraint-type block (exactly one):

| block | params | behaviour |
|---|---|---|
| `harmonic` | `target_distance` | quadratic penalty everywhere toward the target |
| `flat-bottomed` | `target_distance1`, `target_distance2` | no penalty inside `[d1, d2]` (needs `d1 < d2`) |
| `flat-bottomed1` | `target_distance1` | penalise only below `d1` |
| `flat-bottomed2` | `target_distance2` | penalise only above `d2` |

## `angle_restraints_config` (list)

The **angle of 3 group centroids**, with the vertex at group 2 — the group-centroid analogue of the
distance restraint, distinct from the per-ligand-atom conformer `angle` term. CG-solved; rigid group
motion (the centroid-only energy gives every atom in a free group the same gradient, so the group
translates as a unit) means `weight: 1.0` drives any group size.

The measured quantity is the angle at centroid `c₂`,

```
θ = arccos( (c₁ − c₂) · (c₃ − c₂) / (‖c₁ − c₂‖ · ‖c₃ − c₂‖) ),    cₖ = centroid of group k
```

penalised by `E = Σ w · δ²(θ)` with the usual shapes (see Penalty shapes). Targets are in **degrees**
by default — set `unit: radians` on the entry to give them in radians (stored internally as radians
either way).

| key | type | default | meaning |
|---|---|---|---|
| `atom_selection1..3` | str | — (required) | the three groups; group 2 is the vertex |
| `start_sigma` / `stop_sigma` | float | `+inf` / `-1` | sigma gating |
| `start_step` / `stop_step` | int | `-inf` / `+inf` | step gating (mutually exclusive with the sigma window; see Step gating) |
| `unit` | `"degrees"`/`"radians"` | `"degrees"` | unit of the target angle(s) for this entry |
| `weight` | float | `1.0` | energy scale |
| `move` | `"all"` / int / list / `"1,3"` | arms (1,3) free, vertex (2) pinned | which groups are free; the rest are pinned (stop-gradient). `"all"` frees every group |
| one restraint-type block | dict | — (required) | `harmonic {target_angle}` or `flat-bottomed{,1,2}` with `target_angle1` / `target_angle2` (degrees, or radians if `unit: radians`) |

## `dihedral_restraints_config` (list)

The **dihedral of 4 group centroids**, about the axis through groups 2–3 — the group-centroid
analogue of the distance restraint, distinct from the per-ligand-atom conformer `cistrans` term.
CG-solved; `weight: 1.0` drives any group size, as for the angle.

The measured quantity is the torsion about the `c₂`–`c₃` axis,

```
φ = atan2(y, x)   from the four centroids c₁, c₂, c₃, c₄   (standard 4-point torsion)
```

penalised by `E = Σ w · δ²(φ)` (see Penalty shapes). The `harmonic` shape is **periodicity-safe**: the deviation
`φ − target_dihedral` is wrapped to ±180° before squaring, so e.g. +179° and −179° count as a 2°
difference. The `flat-bottomed` shapes use the raw angle and therefore **cannot straddle ±180°**
(`target_dihedral1 < target_dihedral2` is enforced). Targets in **degrees** by default
(`unit: radians` to override).

| key | type | default | meaning |
|---|---|---|---|
| `atom_selection1..4` | str | — (required) | the four groups; groups 2–3 are the axis |
| `start_sigma` / `stop_sigma` | float | `+inf` / `-1` | sigma gating |
| `start_step` / `stop_step` | int | `-inf` / `+inf` | step gating (mutually exclusive with the sigma window; see Step gating) |
| `unit` | `"degrees"`/`"radians"` | `"degrees"` | unit of the target dihedral(s) for this entry |
| `weight` | float | `1.0` | energy scale |
| `move` | `"all"` / int / list / `"1,4"` | ends (1,4) free, axis (2,3) pinned | which groups are free; the rest are pinned (stop-gradient). `"all"` frees every group |
| one restraint-type block | dict | — (required) | `harmonic {target_dihedral}` or `flat-bottomed{,1,2}` with `target_dihedral1` / `target_dihedral2` (degrees, or radians if `unit: radians`) |

## `conformer_restraints_config` (single dict)

Holds a **ligand** at ideal RDKit geometry. It is a single dict (not a list). Conformer restraints
are **per-ligand opt-in in every tool** — a ligand is restrained only when it is flagged, even with
this block present. The flag's placement differs by input format: boltz / protenix / AF3 / openfold
set `conformer_restraints: true` on the ligand object; esmfold2 sets `conformer_restraints=True` on
the `LigandInput`; chai (whose FASTA can't carry it) uses a sidecar `conformer_restraints` map keyed
by ligand chain id (e.g. `{B: true}`).

Top-level (shared by all terms): `start_sigma` (`+inf`), `stop_sigma` (`-1`) — or the step-window
alternative `start_step` (`-inf`) / `stop_step` (`+inf`) (mutually exclusive with the sigma window).

Each term is a sub-dict; a term with `weight <= 0` is disabled (and not built).

| term | keys (default) | meaning |
|---|---|---|
| `bond` | `weight` (0.05), `slack` (0.0 Å) | bond lengths toward ideal; flat-bottomed by `slack` |
| `angle` | `weight` (0.05), `slack` (0.0 rad) | bond angles toward ideal |
| `chiral` | `weight` (0.1), `slack` (0.05) | chiral volume (stereochemistry) — holds each stereocentre's handedness |
| `improper` | `weight` (**0.0**, off), `slack` (0.05) | **planarity** of sp2 double-bond centres — signed volume toward ~0 (same maths as `chiral`). Fires on acyclic, non-aromatic double-bond endpoints with exactly 3 heavy neighbours (carbonyl / amide / ester / carboxyl / trisubstituted alkene); aromatic + in-ring excluded (parity-safe). Off by default — set `weight > 0` to activate |
| `cistrans` | `weight` (0.1), `slack` (0.0 rad) | **cis/trans (E/Z)** of acyclic, non-aromatic double bonds (needs real bond orders; detects 0 for ligands with none, e.g. ATP/NAD/GLN) |
| `vdw` | `weight` (**0.0**, off), `mode` (`"both"`), `scale` (0.75), `dmax` (5.0 Å) | non-bonded clash avoidance |

**Energy.** Each term applies the shared flat-bottomed squared penalty — `δ = 0` within `±slack` of
the RDKit-ideal value, quadratic outside (see Penalty shapes) — to a per-tuple quantity `x`:

```
bond      x = bond length r                                          ideal r₀
angle     x = bond angle θ (radians)                                 ideal θ₀
chiral    x = signed volume V = (a₁ − a₀) · ((a₂ − a₀) × (a₃ − a₀))  ideal V₀ (handedness)
improper  x = the same signed volume V                               target V₀ ≈ 0 (planarity)
cistrans  x = double-bond torsion φ (deviation wrapped to ±180°)     ideal φ₀ (E/Z)
```

`vdw` is one-sided (repulsion only): `E = w · Σ min(0, d − scale·(rᵢ + rⱼ))²` over non-bonded atom
pairs closer than `dmax`, where `d` is the pair distance and `rᵢ`, `rⱼ` are their VdW radii.

`vdw.mode`: `"intramolecular"` (static ligand-internal pairs, all backends), `"ligand_protein"`
(dynamic ligand-vs-fixed-protein, torch/jax), or `"both"` (default). `scale` = fraction of the
summed VdW radii used as the contact threshold; `dmax` = pairs farther than this are ignored. Note
`vdw.weight` defaults to **0** — set it `> 0` to activate.

## `rmsd_restraints_config` (list)

Drives the **Kabsch-superposed RMSD** of a moving group toward `target_rmsd` versus a reference PDB.
The reference must be generated first (a vanilla prediction → PDB via gemmi); see each tool's page.

The energy is

```
E = Σ w · (RMSD − target_rmsd)²,    RMSD = √( (1 / n) Σ_a ‖ Pₐ − R̂ · Qₐ ‖² )
```

where `Pₐ` / `Qₐ` are the prediction / reference **calc** atoms centred on their fit-atom centroids,
`n = n_calc`, and `R̂` is the optimal rotation from a Kabsch SVD on the **fit** atoms. `R̂` (and the
centroids) are treated as **fixed** (stop-gradient), so the gradient pulls the moving atoms, not the
rotation — `target_rmsd = 0` drives the group onto the reference.

| key | type | default | meaning |
|---|---|---|---|
| `ref_pdb` | str | — (required) | path to the reference PDB |
| `target_rmsd` | float | — (required) | target RMSD in Å (`0.0` = match the reference) |
| `weight` | float | `1.0` | energy scale |
| `start_sigma` / `stop_sigma` | float | `+inf` / `-1` | sigma gating (set `stop_sigma`, e.g. `1.0`, to release late and heal a strained terminus) |
| `start_step` / `stop_step` | int | `-inf` / `+inf` | step gating (mutually exclusive with the sigma window; see Step gating) |
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

## `custom_restraints_config` (list)

Define your **own** restraint — not one of the five built-ins — as a differentiable energy. Two
ways, same vocabulary, both run on every backend (torch / jax):

* **config only (expression DSL)**: write the energy as a math **formula** string over named atom
  selections. No Python.
* **code (ctx function)**: write `energy(ctx) -> scalar` in Python and either register it
  (`@custom_restraint("name")`, then reference it here with `use:`) or pass the callable directly
  (`CombinedRestraints.add_custom(fn=...)`, or a `fn:` entry for the Python-dict tools).

Each entry's energy (× `weight`) is added to the CG objective, gated by the usual
`start_sigma` / `stop_sigma` window.

| key | type | default | meaning |
|---|---|---|---|
| `energy` | str | — | the formula (DSL). Exactly one of `energy` / `use` / `fn` is required |
| `use` | str | — | name of a function registered with `@custom_restraint` |
| `fn` | callable | — | a Python `energy(ctx) -> scalar` (Python-dict input only) |
| `selections` | dict | `{}` | `name -> selection string` for the names used in an `energy` formula |
| `weight` | float | `1.0` | scales the whole energy |
| `name` | str | `"custom"` | label shown in the `finalize` per-term log |
| `start_sigma` / `stop_sigma` | float | `+inf` / `-1` | sigma gating (as everywhere) |
| `start_step` / `stop_step` | int | `-inf` / `+inf` | step gating (mutually exclusive with the sigma window; see Step gating) |

**How it works.** Each entry compiles to a closure `energy(active_coords) → scalar`. A selection name
in a formula (`A`) is resolved from the entry's `selections` map to that group's atoms; a string
literal (`"chain A"`) is a raw selection. Selection names resolve to their group **centroids** at
setup (a dry run records which names the energy touches). The formula must reduce to a **scalar** (a
`sum` over any batch dimension). It is parsed safely — no `eval`, and `import` / attribute access /
subscripting / `lambda` all raise — so only the vocabulary below is callable.

The vocabulary has three groups — **geometry** (coordinates → a number), **penalty** (a number → an
energy), and **math** (elementwise / reduction helpers) — plus operators.

**Geometry** — all operate on selection **centroids**; angular results are in **radians** (note: the
built-in `angle` / `dihedral` configs take *degrees*, but a custom formula is in radians). `‖·‖` is
the Euclidean norm:

```
centroid(A)        c_A = mean of A's atoms                                  → vector
distance(A,B)      ‖c_A − c_B‖                                              → scalar
angle(A,B,C)       arccos( (c_A−c_B)·(c_C−c_B) / (‖c_A−c_B‖ · ‖c_C−c_B‖) )  → scalar (rad), vertex B
dihedral(A,B,C,D)  torsion about the B–C centroid axis, range ±π            → scalar (rad)
rg(A)              √( meanᵢ ‖xᵢ − c_A‖² )   radius of gyration              → scalar
norm(v)            ‖v‖   Euclidean norm of a vector quantity                → scalar
dot(u,v)           u · v   dot product of two vector quantities             → scalar
```

**Penalty** — convenience squared penalties (you may also write the algebra directly):

```
harmonic(x, t)          (x − t)²                       quadratic toward t
flat_bottom(x, lo, hi)  min(0, x−lo)² + max(0, x−hi)²  zero inside [lo, hi]
lower(x, lo)            min(0, x−lo)²                  penalise only x < lo
upper(x, hi)            max(0, x−hi)²                  penalise only x > hi
```

`lower` / `upper` are the same maths as the built-in `flat-bottomed1` / `flat-bottomed2` blocks.

**Math** — elementwise and reductions, dispatched to the active backend:

| group | names |
|---|---|
| elementwise | `sqrt` `exp` `log` `abs` `sin` `cos` `clip(x, lo, hi)` |
| reductions | `sum` `minimum` `maximum` |
| branching | `where(cond, a, b)` — there is **no `if`** (keeps the closure jax-traceable, since it must trace inside `lax.scan`) |

**Operators**: `+ - * / ** %`, unary `-`, comparisons (`<` `<=` …), and `&` `|` (combine boolean
masks for `where`).

```yaml
custom_restraints_config:
  # symmetry: keep two inter-domain distances equal
  - name: symmetric
    energy: "(distance(A, B) - distance(C, D))**2"
    selections:
      A: "chain A and resid 10"
      B: "chain B and resid 10"
      C: "chain A and resid 90"
      D: "chain B and resid 90"
    weight: 1.0
  # pull a domain's radius of gyration toward a target compactness
  - name: compact
    energy: "harmonic(rg(dom), 12.0)"            # (rg - 12)**2
    selections: {dom: "chain A and resid 1 to 80"}
```

Code (reusable via `use:`, or passed directly with `add_custom`):

```python
from rgi_utils import custom_restraint, CombinedRestraints

@custom_restraint("symmetric")                 # reusable: config can {use: "symmetric"}
def energy(ctx):
    return (ctx.distance("chain A and resid 10", "chain B and resid 10")
          - ctx.distance("chain A and resid 90", "chain B and resid 90"))**2

restr = CombinedRestraints()
restr.add_custom(fn=energy, weight=1.0)          # throwaway: no registration
restr.setup(adapter, config=restraints_config)   # call add_custom BEFORE setup
```
