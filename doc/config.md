# `restraints_config` reference

[Documentation index](README.md)

Every RGI tool is driven by one `restraints_config` dict (a YAML/JSON block for boltz / protenix /
chai / AF3 / openfold, or a Python dict for ESMFold2 — see each tool's page for where it lives).
This page documents **every variable**: its type, default, allowed values, and meaning.

Omitted keys fall back to their documented defaults. Unknown top-level keys and unknown conformer
keys raise an error. Unknown keys inside individual restraint entries are logged and ignored, so
check their spelling against this page. Source of truth:
`src/rgi_utils/{config,distance_restr_data,group_geom_restr_data,base_pair_restr_data,ref_geom_restr_data,ref_config,featurizer,rmsd_restr_data,selection}.py`.

> **Don't want to hand-write this?** Run the `generate-rgi-config` skill in Claude Code
> (`/generate-rgi-config`) or Codex (`$generate-rgi-config`). It turns a plain-language
> goal into a validated `restraints_config` (correct restraint type, atom-selection DSL,
> target, sigma window) and places it where your tool expects it. Reach for it whenever
> the schema below is more than you need.

## Quick navigation

- [Config shape and global options](#shape)
- [Activation windows](#sigma-gating-start_sigma--stop_sigma)
- [Atom-selection DSL](#atom-selection-dsl) and [penalty shapes](#penalty-shapes-shared)
- [Distance](#distance_restraints_config-list), [group angle](#angle_restraints_config-list), and
  [group dihedral](#dihedral_restraints_config-list)
- [Base pairs](#base_pair_restraints_config-list)
- [Conformer geometry and VdW](#conformer_restraints_config-single-dict)
- [RMSD](#rmsd_restraints_config-list)
- [Custom restraints](#custom_restraints_config-list)

## Shape

```yaml
restraints_config:
  # --- top-level knobs ---
  verbose: ...        # bool
  gpu: ...            # bool
  method: ...         # "CG" | "l-bfgs"
  max_iter: ...       # int
  # --- restraints (each block optional) ---
  distance_restraints_config: [ ... ]   # list
  angle_restraints_config:    [ ... ]   # list  (group-centroid angle)
  dihedral_restraints_config: [ ... ]   # list  (group-centroid dihedral)
  base_pair_restraints_config: [ ... ]  # list  (nucleic-acid Watson-Crick base pairs)
  conformer_restraints_config: { ... }  # single dict (ligand/polymer local geometry)
  rmsd_restraints_config:     [ ... ]   # list
  custom_restraints_config:   [ ... ]   # list  (define your OWN restraint — see below)
```

A restraint type is active only if its block is present (and, for conformer terms, the term's
`weight > 0`).

Distance, angle, dihedral, conformer, and RMSD are direct built-ins. Base-pair entries are macros
that expand into distance and plane terms. `custom_restraints_config` defines an original
restraint as a math formula or Python callable.

## Top-level keys

| key | type | default | meaning |
|---|---|---|---|
| `verbose` | bool | `false` | Log the built spec (per-restraint counts) at setup and per-term energies at finalize. Strongly recommended — it is how you confirm a restraint was actually built. |
| `gpu` | bool | `true` | Torch **device**: `true` = accelerator (default), `false` = CPU. It does **not** change the backend. (Inert for AF3, which always runs the JAX minimizer on the model's device.) Accepts `true/false` and the strings `1/0/yes/no/on/off`. |
| `method` | str | `"CG"` | Optimizer: `"CG"` (nonlinear conjugate gradient) or `"l-bfgs"` (opt-in). |
| `max_iter` | int | `100` | Max optimizer iterations per denoising step. The examples use 1000 (2000 for AF3). |

**There is no `backend` key** — the compute backend (torch / jax) is **inferred from how the
engine is invoked**, not configured: a JAX tool (AF3) grabs the pure minimizer via
`get_minimizer()` → jax; every other tool calls `minimize(coords)`, where a torch/numpy array →
torch. A leftover `backend:` key raises with a migration hint. (`gpu` above still selects the
torch *device*.) There is no numpy optimizer (numpy is the energy reference only).

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
> (mutually exclusive). The unused axis stays always-on.

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

```math
E = \sum_i w_i\,\delta_i^2
```

where $w_i$ is the entry `weight` and $\delta_i$ is the deviation of a measured quantity $x$ (a
distance, angle, volume, …) from its target. Four block names choose how $\delta$ is shaped:

| block | $\delta$ | effect |
|---|---|---|
| `harmonic` | $x - t$ | penalise any deviation from $t$ |
| `flat-bottomed` | $0$ for $t_1 \le x \le t_2$; $x - t_1$ below; $x - t_2$ above | no penalty inside the window |
| `flat-bottomed1` | $\min(0,\, x - t_1)$ | lower bound — penalise only $x \lt t_1$ |
| `flat-bottomed2` | $\max(0,\, x - t_2)$ | upper bound — penalise only $x \gt t_2$ |

The same four shapes drive the `distance` / `angle` / `dihedral` / `rmsd` blocks (only the target
key differs: `target_distance` / `target_angle` / `target_dihedral` / `target_rmsd`, with `…1` /
`…2` for the flat-bottomed bounds). The **conformer** terms use the flat-bottomed shape with a symmetric `slack`:
$\delta = 0$ within $\pm$`slack` of the RDKit-ideal value, quadratic outside (`slack = 0` $\Rightarrow$ pure harmonic).

`distance` is CG-minimised like every other restraint (it used to be a closed-form shift; it is now
part of the optimiser objective). To keep large groups moving as a rigid body under CG — a plain
centroid's per-atom gradient is diluted by `1/N` — its centroid uses the same `_move_centroid`
N×-rescale as the group angle/dihedral terms, with a reduced-mass scale `N1·N2/(N1+N2)` that
reproduces the old **minimal-displacement** split (`s1 : s2 = N2 : N1`) for a single / disjoint
restraint **and the small per-step moves of real diffusion**. (A single large one-shot move can
cross the moving group past the other to the reflected, equal-energy solution — the centroid **gap
always reaches the target**, but the split direction is not guaranteed for big moves; harmless in
the multi-step diffusion loop.) Its `weight` is a **no-op for a single restraint or restraints with disjoint groups** —
CG reaches the target regardless. For **over-constrained coupled** restraints sharing an atom,
`weight` is now the usual least-squares weight (CG jointly minimises `Σ wᵢ·δᵢ²`), which replaces the
old closed-form weighted-average; the single/disjoint behaviour is unchanged.

## `distance_restraints_config` (list)

Pulls the **centroid distance** between two atom groups toward a target. CG-minimised with a
reduced-mass `_move_centroid` rescale so each group translates as a rigid body and reaches the
target. `weight` is a **no-op for a single / disjoint restraint**; it only re-balances atoms shared
by **over-constrained coupled** restraints (see `weight` below).

The measured quantity is the distance between the two groups' centroids,

```math
d = \lVert c_2 - c_1 \rVert, \qquad c_k = \frac{1}{|G_k|}\sum_{a \in G_k} x_a
```

(a plain masked-mean centroid), shaped by one of the penalty blocks below (see Penalty shapes).
`harmonic` drives the target distance ($d = t$) to CG convergence (within `gtol`/`ftol`).

| key | type | default | meaning |
|---|---|---|---|
| `atom_selection1` | str | — (required) | group 1 (selection DSL) |
| `atom_selection2` | str | — (required) | group 2 |
| `start_sigma` | float | `+inf` | activation upper bound (see Sigma gating) |
| `stop_sigma` | float | `-1` | release lower bound |
| `start_step` / `stop_step` | int | `-inf` / `+inf` | step gating — the alternative to the sigma window (mutually exclusive; see Step gating) |
| `move` | `"both"`/`"all"`/`1`/`2`/`[1,2]`/`"1,2"` | `"both"` | which group the correction moves: `both` (= `all` = `[1,2]`) = minimal-displacement split; `1` / `2` (or `[1]` / `[2]`) move only that group and **pin** the other (e.g. move a ligand toward a fixed pocket). Shares the list/comma vocabulary of the angle/dihedral `move`; distance has 2 groups so only indices 1–2 are valid (`[1,3]` raises) |
| `weight` | float | `1.0` | relative strength. **No-op for a single restraint or disjoint groups** (each reaches its exact target regardless). Only bites when an atom is the **sole mover** of two **over-constrained coupled** restraints (each pinning its other group), where the shared atom settles `w₁:w₂` between the two targets (e.g. `2` vs `1` → 2:1). Not a soft "strength" knob in the common case |
| one restraint-type block | dict | — (required) | the penalty (below) |

Restraint-type block (exactly one):

| block | params | behaviour |
|---|---|---|
| `harmonic` | `target_distance` | quadratic penalty everywhere toward the target |
| `flat-bottomed` | `target_distance1`, `target_distance2` | no penalty inside `[d1, d2]` (needs `d1 < d2`) |
| `flat-bottomed1` | `target_distance1` | penalise only below `d1` |
| `flat-bottomed2` | `target_distance2` | penalise only above `d2` |

### Reference groups

A distance entry may use one external structure. Keep both normal group keys; prefix the
reference-side value with `ref1 and`, then define `refs.ref1`. The suffix is evaluated on
the reference structure. At least one group must remain a prediction selection. Reference
groups are fitted independently, held fixed, and do not pull their fit anchors.
`move` controls prediction groups only: omit it or use `all`/`both` to move every prediction
group, or give prediction-side group indices. Selecting a reference group raises.

```yaml
distance_restraints_config:
  - atom_selection1: "chain A and resid 120"
    atom_selection2: "ref1 and chain A and resid 200"
    refs:
      ref1:
        ref_cif: template.cif
        atom_selection_target_fit: "chain A and resid 1 to 80 and backbone"
        atom_selection_ref_fit: "chain A and resid 1 to 80 and backbone"
        pairing: align
        best_effort: true
    harmonic: {target_distance: 5.0}
```

## `angle_restraints_config` (list)

The **angle of 3 group centroids**, with the vertex at group 2 — the group-centroid analogue of the
distance restraint, distinct from the per-ligand-atom conformer `angle` term. CG-solved; rigid group
motion (the centroid-only energy gives every atom in a free group the same gradient, so the group
translates as a unit) means `weight: 1.0` drives any group size.

The measured quantity is the angle at centroid $c_2$ (with $c_k$ the centroid of group $k$),

```math
\theta = \arccos\left( \frac{(c_1 - c_2)\cdot(c_3 - c_2)}{\lVert c_1 - c_2 \rVert\,\lVert c_3 - c_2 \rVert} \right)
```

penalised by $E = \sum w\,\delta^2(\theta)$ with the usual shapes (see Penalty shapes). Targets are in **degrees**
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

### Reference groups

An angle entry may use up to two distinct references. Write each reference group as
`refN and <selection>` and define the corresponding `refs.refN`. Each reference has its own
structure and optional fit configuration, so groups from different fitted structures can be
combined in one angle. At least one of the three groups must use prediction atoms. `move`
accepts prediction-side group indices; omitted/`all`/`both` moves every prediction group, while
selecting a
reference group raises.

```yaml
angle_restraints_config:
  - atom_selection1: "chain A and resid 10"
    atom_selection2: "ref1 and chain A and resid 20"
    atom_selection3: "chain B and resid 30"
    refs:
      ref1: {ref_cif: state1.cif}
    move: 1  # move group 1; group 3 is pinned and group 2 is a fixed reference
    harmonic: {target_angle: 90.0}
```

## `dihedral_restraints_config` (list)

The **dihedral of 4 group centroids**, about the axis through groups 2–3 — the group-centroid
analogue of the distance restraint, distinct from the per-ligand-atom conformer `cistrans` term.
CG-solved; `weight: 1.0` drives any group size, as for the angle.

The measured quantity is the dihedral angle $\phi$ of the four centroids $c_1, c_2, c_3, c_4$ about
the axis through $c_2$ and $c_3$ (with $c_k$ the centroid of group $k$),

```math
\phi = \mathrm{atan2}\big( (n_1 \times \hat{b}_2)\cdot n_2,\; n_1 \cdot n_2 \big)
```

with the bond and normal vectors

```math
b_1 = c_2 - c_1, \quad b_2 = c_3 - c_2, \quad b_3 = c_4 - c_3
```

```math
n_1 = b_1 \times b_2, \quad n_2 = b_2 \times b_3, \quad \hat{b}_2 = b_2 / \lVert b_2 \rVert
```

This is the signed `atan2` convention (range $\pm 180^\circ$); it is penalised by
$E = \sum w\,\delta^2(\phi)$ (see Penalty shapes). The `harmonic` shape is **periodicity-safe**: the
deviation $\phi - t$ is wrapped to $[-180^\circ, 180^\circ]$ before squaring, so e.g. $+179^\circ$
and $-179^\circ$ count as a $2^\circ$ difference. The `flat-bottomed` shapes use the raw angle and
therefore **cannot straddle $\pm 180^\circ$** (`target_dihedral1 < target_dihedral2` is enforced).
Targets in **degrees** by default (`unit: radians` to override).

| key | type | default | meaning |
|---|---|---|---|
| `atom_selection1..4` | str | — (required) | the four groups; groups 2–3 are the axis |
| `start_sigma` / `stop_sigma` | float | `+inf` / `-1` | sigma gating |
| `start_step` / `stop_step` | int | `-inf` / `+inf` | step gating (mutually exclusive with the sigma window; see Step gating) |
| `unit` | `"degrees"`/`"radians"` | `"degrees"` | unit of the target dihedral(s) for this entry |
| `weight` | float | `1.0` | energy scale |
| `move` | `"all"` / int / list / `"1,4"` | ends (1,4) free, axis (2,3) pinned | which groups are free; the rest are pinned (stop-gradient). `"all"` frees every group |
| one restraint-type block | dict | — (required) | `harmonic {target_dihedral}` or `flat-bottomed{,1,2}` with `target_dihedral1` / `target_dihedral2` (degrees, or radians if `unit: radians`) |


### Reference groups

A dihedral entry may use up to three distinct references. Write each reference group as
`refN and <selection>` and define the corresponding `refs.refN`. Each reference is fitted
independently, so one dihedral may combine prediction atoms with groups from up to three
external structures. At least one group must use prediction atoms. `move` accepts only
prediction-side group indices; omitted/`all`/`both` moves all prediction groups. Selecting a
reference
group raises.

```yaml
dihedral_restraints_config:
  - atom_selection1: "chain A and resid 10"
    atom_selection2: "ref1 and chain A and resid 20"
    atom_selection3: "chain B and resid 30"
    atom_selection4: "chain C and resid 40"
    refs:
      ref1: {ref_cif: state1.cif}
    move: [1, 4]  # groups 1 and 4 move; group 3 is pinned
    harmonic: {target_dihedral: 180.0}
```

## `base_pair_restraints_config` (list)

Restrain two nucleotides into **Watson-Crick base-pair geometry**. This is a config-time
**macro**, not a new energy term: each entry EXPANDS into the primitives above — one
[`distance`](#distance_restraints_config-list) restraint per WC hydrogen bond, plus (optionally)
one best-fit [`plane`](#conformer_restraints_config-single-dict) restraint over both bases so the
pair stays coplanar. It mirrors what servalcat/Refmac do (base pairing is imposed as H-bond
distance + planarity restraints, there is no dedicated base-pair potential).

You name the two paired residues; the engine looks up the WC donor/acceptor atoms and their ideal
H-bond distance from a built-in table:

| pair | H-bond atom pairs (residue1, residue2) |
|---|---|
| G·C | `(N1,N3)` `(N2,O2)` `(O6,N4)` |
| A·T / A·U | `(N1,N3)` `(N6,O4)` |
| G·U wobble | `(N1,O2)` `(O6,N3)` — **opt-in via `pair: GU`** (never auto-detected) |

The base identity is auto-detected from each residue's `resname` (DNA `D`-prefix stripped; DNA and
RNA share base-atom names). Pairs are **user-specified only** — auto-detecting which bases pair from
coordinates is intentionally NOT done (coordinates are pure noise at high sigma). Reverse orders
(C·G, T·A, …) are handled automatically.

```yaml
base_pair_restraints_config:
  - residue1: "chain A and resid 5"   # each selector must match EXACTLY one residue
    residue2: "chain B and resid 12"
    # pair: GC          # optional: override auto-detection (needed if resname is
    #                   # unavailable, or for a G-U wobble). Two letters from ACGTU.
    # coplanar: true    # add the inter-base coplanarity plane (default true)
    # target: [2.7, 3.1]  # H-bond distance: [low, high] -> flat-bottomed (default),
    #                   # or a scalar -> harmonic
    # weight: 1.0       # strength of the H-bonds AND the coplanarity plane
    # move: both        # both / 1 / 2 — 1 docks residue1 onto a fixed residue2
    # start_sigma / stop_sigma   # sigma gating (applies to the H-bond distances)
    # start_step / stop_step     # step gating (mutually exclusive with sigma)
```

| key | type | default | meaning |
|---|---|---|---|
| `residue1` / `residue2` | str | — (required) | selection-DSL strings, each matching exactly one nucleotide (else it raises) |
| `pair` | str | auto from `resname` | override the detected bases, e.g. `"GC"`, `"AU"`, `"GU"`. Required when `resname` is unavailable or for a wobble pair |
| `coplanar` | bool | `true` | also add a best-fit-plane restraint over both bases' atoms (keeps the pair coplanar) |
| `target` | `[low, high]` or float | `[2.7, 3.1]` | WC H-bond distance: a `[low, high]` window → flat-bottomed; a scalar → harmonic (Å) |
| `weight` | float | `1.0` | energy scale for the generated H-bond distances and the coplanarity plane |
| `move` | `both` / `1` / `2` | `both` | which residue the H-bonds pull; `1` moves only residue1 (dock a strand onto a fixed template), `2` only residue2 |
| `start_sigma` / `stop_sigma` | float | `+inf` / `-1` | sigma gating of the H-bond distances |
| `start_step` / `stop_step` | int | `-inf` / `+inf` | step gating (mutually exclusive with the sigma window) |

### Gating

The gate window applies to the **H-bond distances**. The coplanarity plane rides the
**shared conformer gate** (`conformer_restraints_config.start_sigma`, default always-on) because it is
built on the same `plane` term as ligand/polymer planarity — so a per-entry `start_sigma` here does not
move the coplanarity release point. Per-base planarity is already maintained by
[`conformer_restraints_config`](#conformer_restraints_config-single-dict) when polymer geometry is
enabled; `coplanar` here additionally makes the **two** bases of the pair share one plane.

### Validation

The parser fails loudly rather than creating a silent no-op: a `residue` selector matching zero or
more than one residue, an
auto-detected non-Watson-Crick pair (e.g. G-G, or a G-U without `pair: GU`), a `resname`-less residue
with no `pair`, or a missing WC atom all raise. Verbose setup logs
`base_pair=P pairs -> H h-bonds + C coplanar groups` (the generated restraints also show up in the
`distances=` / `plane=` counts).

## `conformer_restraints_config` (single dict)

Repairs local geometry in ligands, proteins, DNA, and RNA. It is a single dict (not a
list). Every molecule is **independently opt-in**: set `conformer_restraints: true`
on each boltz / protenix / AF3 / openfold sequence or chain object, or
`conformer_restraints=True` on each esmfold2 input object. Chai uses the equivalent
sidecar map keyed by chain id because FASTA cannot carry the flag. Missing/false chains
remain unrestrained, including when another chain of the same polymer type is enabled.

```yaml
conformer_restraints_config:
  # No start_sigma: all four terms, including VdW, are active from the first step.
  bond: {}
  angle: {}
  chiral: {}
  vdw:
    max_neighbors: 32
```

Intra-residue bond, angle, chiral, and planar-group targets come from each predictor's residue-local
ideal reference conformer. Canonical inter-residue geometry is added explicitly: peptide `C-N` bonds
plus `CA-C-N`, `O-C-N`, and `C-N-CA` angles; and DNA/RNA `O3'-P` phosphodiester bonds plus
`C3'-O3'-P` and `O3'-P-O5'` angles. The adjacent `P-O5'-C5'` angle comes from the current residue's
reference conformer. Together these prevent an RMSD restraint from repairing a selected residue while
breaking the covalent link to its neighbour. Polymer `plane` **is** built (opt-in via the `plane`
sub-block): residue-local aromatic rings — His/Phe/Tyr/Trp side chains and nucleic-acid bases — plus
the protein **peptide plane**, the canonical inter-residue 5-atom group `{C, CA, O}` (previous
residue) `+ {N, CA}` (current), scored by the best-fit-plane `plane` term (this replaces the old
peptide-plane zero-volume impropers that rode the `chiral` term — so a `chiral`-only config no longer
flattens the peptide plane; add a `plane` sub-block). Polymer `cistrans` remains a ligand-only term.

Top-level (shared by all terms): `start_sigma` (`+inf`), `stop_sigma` (`-1`) — or the step-window
alternative `start_step` (`-inf`) / `stop_step` (`+inf`) (mutually exclusive with the sigma window).

Each term is a sub-dict and is **off unless its sub-block is present**: a listed term
defaults to `weight` 1.0 (override it, or set `weight <= 0` to disable a listed term); an
absent term is not built. So a ligand that opts in but lists only `bond:` gets ONLY the
bond term — add each other sub-block to activate it. This is the same "default 1.0, off if
not configured" rule the other restraint types follow.

| term | keys (default) | meaning |
|---|---|---|
| `bond` | `weight` (1.0), `slack` (0.0 Å) | bond lengths toward ideal; flat-bottomed by `slack` |
| `angle` | `weight` (1.0), `slack` (0.0 rad) | bond angles toward ideal |
| `chiral` | `weight` (1.0), `slack` (0.05) | chiral volume (stereochemistry) — holds each stereocentre's handedness |
| `plane` | `weight` (1.0), `slack` (0.0 Å) | **best-fit-plane** flatness of whole planar atom groups ([servalcat](https://github.com/keitaroyam/servalcat)-style) — penalises each group's out-of-plane RMS deviation toward 0. Fires on (a) aromatic/conjugated rings (whole ring) and (b) non-ring sp2 groups (an acyclic double-bond centre + its heavy neighbours: carbonyl / amide / ester / carboxyl / trisubstituted alkene). Group membership is confirmed by the reference conformer being coplanar (not the RDKit aromaticity flag). Add a `plane:` block to activate |
| `cistrans` | `weight` (1.0), `slack` (0.0 rad) | **cis/trans (E/Z)** of acyclic, non-aromatic double bonds (needs real bond orders; detects 0 for ligands with none, e.g. ATP/NAD/GLN) |
| `vdw` | `weight` (1.0), `mode` (`"both"`), `scale` (0.75), `dmax` (5.0 Å), `max_neighbors` (32) | non-bonded clash avoidance |

### Energy terms

Each term applies the shared flat-bottomed squared penalty ($\delta = 0$ within
$\pm$`slack` of the RDKit-ideal value, quadratic outside — see Penalty shapes) to a per-tuple
quantity $x$:

| term | measured quantity $x$ | ideal |
|---|---|---|
| `bond` | bond length $r$ | $r_0$ |
| `angle` | bond angle $\theta$ (radians) | $\theta_0$ |
| `chiral` | signed volume $V = (a_1 - a_0)\cdot\big((a_2 - a_0)\times(a_3 - a_0)\big)$ | $V_0$ (handedness) |
| `plane` | group's out-of-plane RMS deviation $\sqrt{\lambda_{\min}/N}$ ($\lambda_{\min}$ = smallest eigenvalue of the centred covariance) | $0$ (planar) |
| `cistrans` | double-bond torsion $\phi$ (deviation wrapped to $\pm 180^\circ$) | $\phi_0$ (E/Z) |

`vdw` is one-sided (repulsion only),

```math
E = w \sum_{(i,j)} \min\big(0,\; d_{ij} - \text{scale}\cdot(r_i + r_j)\big)^2,
```

over non-bonded atom pairs closer than `dmax`, where $d_{ij}$ is the pair distance and $r_i, r_j$
are their VdW radii.

### Van der Waals modes

`vdw.mode` picks **two categories** (default `"both"` = both):

- `"intramolecular"` — clashes **within** a ligand (static ligand-internal pairs, all backends).
- `"intermolecular"` — clashes between the ligand and **every other molecule**: the **fixed
  background** (every non-padding atom not being optimised — protein, DNA/RNA, any **non-restrained**
  ligand; dynamic, torch/jax) **and** other **restrained** ligands (≥2 ligands that each set
  `conformer_restraints: true` + `vdw` both move, so neither is in the other's background — every
  cross-molecule atom pair gets the same one-sided penalty, scored in the energy layer on all
  backends).

So to make two restrained ligands avoid each other, just keep the default `mode: both` (or set
`intermolecular`) and give both a `vdw` block — no extra key. `scale` = fraction of the summed VdW
radii used as the contact threshold; `dmax` = pairs farther than this are ignored (the inter-ligand
pairs ignore `dmax` — the two ligands' frames are independent, so all cross pairs are listed and the
clamp zeroes non-contacts). Like every conformer term, `vdw` is built only when a `vdw:` block is
present (then `weight` defaults to 1.0); omit the block to leave it off. **The old
`mode: ligand_protein` was removed** (it was only the fixed-background half) — it now raises a
migration error pointing to `intermolecular`.

### Polymer neighbor lists

For selected polymers, a fixed-width active-active neighbour list is rebuilt once from the current
coordinates at each diffusion step and held fixed during CG. Energy evaluation is therefore
`O(N * max_neighbors)` rather than all-pairs on every CG iteration. Covalent 1-2 and 1-3 pairs,
including peptide and phosphodiester links, are excluded. Polymer atoms are also checked against
non-active fixed-background atoms on torch and JAX. Omit `start_sigma` (the default `+inf`) to keep
VdW active from the first denoising step; setting `start_sigma` delays all conformer terms together.

`max_neighbors` (default 32) caps each atom's neighbour list: a buried atom with more than
`max_neighbors` partners within `dmax` keeps only its nearest `max_neighbors`. The nearest are the
most clash-relevant, so this is a deliberate approximation — raise it for very dense cores. Under
JAX the pair codes are int32 (JAX runs with x64 disabled by default), so a **single restrained
polymer selection is limited to ~46340 active atoms on AF3**; exceeding it raises rather than
silently corrupting the covalent-pair exclusion (the torch tools use int64 and are unaffected).

## `rmsd_restraints_config` (list)

Drives the **Kabsch-superposed RMSD** of a moving group versus a reference structure, shaped by one
of the penalty blocks (see Penalty shapes) on the RMSD value. The reference must be generated first
(a vanilla prediction → PDB or mmCIF via gemmi); see each tool's page. Supply it as **either**
`ref_pdb` (legacy PDB) **or** `ref_cif` (mmCIF) — mutually exclusive; both parse via gemmi
(lazy-imported) to the same atom records, so a `.cif` prediction can be used directly without
converting to PDB first.

The energy is

```math
E = \sum w\,\delta^2, \qquad \mathrm{RMSD} = \sqrt{\tfrac{1}{n}\sum_{a} \lVert P_a - \hat{R}\,Q_a \rVert^2}
```

where $\delta$ is the penalty-block deviation of the RMSD (see Penalty shapes; `harmonic`
$\Rightarrow \delta = \mathrm{RMSD} - t$), $P_a$ / $Q_a$ are the prediction / reference **calc**
atoms centred on their fit-atom centroids, $n = n_\text{calc}$, and $\hat{R}$ is the optimal rotation
from a Kabsch SVD on the **fit** atoms. $\hat{R}$ (and the centroids) are treated as **fixed**
(stop-gradient), so the gradient pulls the moving atoms, not the rotation —
`harmonic: {target_rmsd: 0}` drives the group onto the reference, while
`flat-bottomed2: {target_rmsd2: X}` keeps it within $X$ Å of the reference.

| key | type | default | meaning |
|---|---|---|---|
| `ref_pdb` / `ref_cif` | str | — (one required, mutually exclusive) | path to the reference structure — `ref_pdb` = legacy PDB, `ref_cif` = mmCIF; both read via gemmi to the same atoms |
| `harmonic` / `flat-bottomed` / `flat-bottomed1` / `flat-bottomed2` | block | — (one required) | restraint-type block on the RMSD (Å); target keys `target_rmsd` (harmonic) / `target_rmsd1` / `target_rmsd2` (see Penalty shapes). `harmonic: {target_rmsd: 0}` = match the reference; `flat-bottomed2: {target_rmsd2: X}` = stay within $X$ Å |
| `weight` | float | `1.0` | energy scale |
| `start_sigma` / `stop_sigma` | float | `+inf` / `-1` | sigma gating (set `stop_sigma`, e.g. `1.0`, to release late and heal a strained terminus) |
| `start_step` / `stop_step` | int | `-inf` / `+inf` | step gating (mutually exclusive with the sigma window; see Step gating) |
| `pairing` | `"align"` / `"identity"` | `"align"` | how reference and prediction residues correspond. `align` = sequence-align polymer chains (BLOSUM62) so a **homolog** ref maps on despite substitutions/indels/renumbering; non-polymer atoms always pair by ordinal. `identity` = strict (chain, resid, name) ordinal pairing. |
| `best_effort` | bool | `true` | skip atoms with no match in the ref (instead of raising); `false` = strict |

### Atom pairing and selection

All selection keys are optional; omit all of them to use the whole structure in best-effort mode.
The superposed ("fit") and measured ("calc") atom sets are chosen independently:

| key | sets |
|---|---|
| `atom_selection_ref` / `atom_selection_target` | shorthand: **both** fit and calc, on the ref / target side |
| `atom_selection_ref_fit` / `atom_selection_target_fit` | atoms used for the Kabsch superposition |
| `atom_selection_ref_calc` / `atom_selection_target_calc` | atoms over which the RMSD is measured |

Use the shorthand for the common case (fit = calc); use the four `_fit`/`_calc` keys to e.g.
superpose on the backbone but measure over a pocket. Under `pairing: align`, restrict the fit to
`name CA` or `backbone` so a substituted homolog's side chain is not pinned.

## `custom_restraints_config` (list)

### Authoring methods

Define your **own** restraint — not one of the five built-ins — as a differentiable energy. Two
ways, same vocabulary, both run on every backend (torch / jax):

- **config only (expression DSL)**: write the energy as a math **formula** string over named atom
  selections. No Python.
- **code (ctx function)**: write `energy(ctx) -> scalar` in Python and either register it
  (`@custom_restraint("name")`, then reference it here with `use:`) or pass the callable directly
  (`CombinedRestraints.add_custom(fn=...)`, or a `fn:` entry for the Python-dict tools).

Each entry's energy (× `weight`) is added to the CG objective, gated by the usual
`start_sigma` / `stop_sigma` window.

| key | type | default | meaning |
|---|---|---|---|
| `energy` | str | — | the formula (DSL). Exactly one of `energy` / `use` / `fn` is required |
| `use` | str | — | name of a function registered with `@custom_restraint` |
| `fn` | callable | — | a Python `energy(ctx) -> scalar` (Python-dict input only) |
| `selections` | dict | `{}` | `name -> selection string`; use `refN and <selection>` for a group selected from `refs.refN` |
| `refs` | dict | `{}` | `refN -> {ref_pdb`\|`ref_cif, atom_selection_ref_fit, atom_selection_target_fit, pairing, best_effort}`; names are reserved as `ref1`, `ref2`, ... |
| `move` | `"all"` / `"both"` / str / list[str] | `"all"` | prediction selection name(s) that receive this restraint's gradient. Omitted/`all`/`both` moves every prediction selection. Reference-backed selections are always fixed and cannot be named |
| `weight` | float | `1.0` | scales the whole energy |
| `name` | str | `"custom"` | label shown in the `finalize` per-term log |
| `start_sigma` / `stop_sigma` | float | `+inf` / `-1` | sigma gating (as everywhere) |
| `start_step` / `stop_step` | int | `-inf` / `+inf` | step gating (mutually exclusive with the sigma window; see Step gating) |

### Evaluation model

Each entry compiles to a closure `energy(active_coords) → scalar`. A selection name
in a formula (`A`) is resolved from the entry's `selections` map to that group's atoms; a string
literal (`"chain A"`) is a raw selection. Selection names resolve to their group **centroids** at
setup (a dry run records which names the energy touches). The formula must reduce to a **scalar** (a
`sum` over any batch dimension). `move` applies stop-gradient to every unlisted prediction
selection for this custom term; another restraint may still move the same atoms. The formula is parsed safely — no `eval`,
and
`import` / attribute access /
subscripting / `lambda` all raise — so only the vocabulary below is callable.

### Vocabulary

The vocabulary has three groups — **geometry** (coordinates → a number), **penalty** (a number → an
energy), and **math** (elementwise / reduction helpers) — plus operators.

#### Geometry

Geometry functions operate on selection **centroids**; angular results are in **radians** (note: the
built-in `angle` / `dihedral` configs take *degrees*, but a custom formula is in radians).
$\lVert\cdot\rVert$ is the Euclidean norm:

| call | result | definition | use it for |
|---|---|---|---|
| `centroid(A)` | vector | $c_A$ = mean of $A$'s atoms | a building block — subtract two, or feed one to `norm` / `dot` |
| `distance(A,B)` | scalar | $\lVert c_A - c_B \rVert$ | a separation between two groups; a **difference of two distances** encodes symmetry / equidistance |
| `angle(A,B,C)` | scalar (rad) | $\arccos\big( (c_A - c_B)\cdot(c_C - c_B) / (\lVert c_A - c_B \rVert\,\lVert c_C - c_B \rVert) \big)$, vertex $B$ | the bend of three groups about the vertex $B$ |
| `dihedral(A,B,C,D)` | scalar (rad) | torsion about the B–C centroid axis, range $\pm\pi$ | the twist / handedness across four groups — a **periodic** quantity: wrap its deviation, see below |
| `rg(A)` | scalar | $\sqrt{\frac{1}{\lvert A\rvert}\sum_i \lVert x_i - c_A \rVert^2}$ — radius of gyration | the compactness of one group (collapse vs extension) |
| `norm(v)` | scalar | $\lVert v \rVert$ | the length of a vector you built, e.g. `centroid(A) - centroid(B)` |
| `dot(u,v)` | scalar | $u \cdot v$ | projections and cosine-like terms |
| `coords(A)` | block $(k,3)$ | $A$'s atom coordinates | feed a bare selection into arithmetic with `kabsch` output (a bare name alone is a *selection identifier*, not coordinates) |
| `kabsch(A,B)` | block $(k,3)$ | $A$ rigid-body-superposed onto $B$ (Kabsch) | align two moving groups, then measure the leftover per-atom deviation — see **Reference-backed selections** |
| `rmsd(A,B)` | scalar | superposed RMSD of prediction selection `A` vs reference-backed selection `B` | pull a group onto an external reference; `B` must map to `refN and <selection>` — see **Reference-backed selections** |

Most geometry consumes selection **centroids**; `coords` / `kabsch` instead flow a whole
**$(k,3)$ coordinate block**, so they compose: `centroid(kabsch(A,B))`, `norm(kabsch(A,B) - coords(B))`.

#### Reference-backed selections and superposition

A reference-backed selection is declared in the ordinary `selections` map:

```yaml
selections:
  moving: "chain A and resid 1 to 80"
  state1: "ref1 and chain A and resid 1 to 80"
refs:
  ref1:
    ref_cif: state1.cif
    atom_selection_target_fit: "chain A and backbone"
    atom_selection_ref_fit: "chain A and backbone"
    pairing: align
    best_effort: true
```

`ref1`, `ref2`, ... are reserved entry-local names. Each definition requires exactly one of
`ref_pdb` / `ref_cif`. Its optional target/ref fit selections place all selections from that
reference into the current prediction frame. The fitted reference coordinates and transform are
stop-gradient values: they act as fixed landmarks and do not pull the fit anchor. If both fit
selections are omitted, the reference is used in its own coordinate frame.

Reference-backed selections work with the normal geometry vocabulary: `distance(A,B)`,
`angle(A,B,C)`, `dihedral(A,B,C,D)`, `centroid(A)`, `coords(A)`, and `kabsch(A,B)`. At least one
selection in the custom entry must come from the prediction; an all-reference expression is a
constant and raises.

* **`kabsch(A, B)`** returns A after rigid-body superposition onto B. Both arguments are bare
  selection identifiers and must contain the same number of atoms. Either may be reference-backed.
* **`rmsd(A, B)`** returns Kabsch-superposed RMSD from prediction selection A to reference-backed
  selection B. Atom correspondence is resolved at setup with `pairing: align` (default) or
  `identity`; `best_effort: true` skips atoms missing from the reference. A must be prediction-backed
  and B must start with `refN and`; both must be bare selection identifiers.

The frozen rotation makes `kabsch` / `rmsd` gradients torch/jax-consistent but different from a
numpy finite-difference through the SVD, matching the built-in RMSD restraint.

#### Degenerate geometry

`angle` and `dihedral` are ill-defined when the centroids collapse —
coincident centroids, a collinear A–B–C for `angle`, or a `dihedral` whose central axis runs parallel
to an arm. The value stays finite but its gradient is near-zero, so the term cannot push the
structure; choose groups whose centroids are distinct and non-collinear. (`distance` / `rg` have no
such caveat.)

#### Dihedral periodicity

Wrap every `dihedral` deviation. A dihedral is periodic ($\phi$ and $\phi + 2\pi$
are the same geometry), so a penalty on its **deviation** must fold that deviation into
$[-\pi, \pi]$ first. Write `wrap(dihedral(A,B,C,D) - t)**2` (harmonic), **not** the naïve
`harmonic(dihedral(A,B,C,D), t)`: the naïve form counts $\phi = +179^\circ$ against $t = -179^\circ$
as a $358^\circ$ deviation (huge energy, and a gradient pointing the *long way* round) instead of the
correct $2^\circ$. `wrap(x)` $= \mathrm{atan2}(\sin x, \cos x)$ is exactly the fold the
built-in `dihedral_restraints_config` / conformer `cistrans` apply internally — see the Math table.
For a window, wrap relative to the centre: `flat_bottomed(wrap(dihedral(...) - centre), -w, w)` — and
because the deviation is wrapped, this window **can straddle $\pm 180^\circ$** (the built-in
flat-bottomed dihedral cannot). Note `t` / `centre` are in **radians** (a custom formula does no degree
conversion, unlike the built-in `dihedral_restraints_config`). (`angle` is bounded to $[0, \pi]$ by
`arccos`, so it needs no wrap.)

#### Penalties

These are convenience squared penalties; you may also write the algebra directly. Use
`harmonic` to drive a quantity **to** a value, and the `flat_bottomed` family to **bound** it — leave
it free inside a window, above a floor, or below a ceiling:

| call | definition | effect — use when |
|---|---|---|
| `harmonic(x, t)` | $(x - t)^2$ | quadratic toward $t$ — pin $x$ at a target |
| `flat_bottomed(x, lo, hi)` | $\min(0,\, x - \text{lo})^2 + \max(0,\, x - \text{hi})^2$ | zero inside $[\text{lo}, \text{hi}]$ — keep $x$ within a band |
| `flat_bottomed1(x, lo)` | $\min(0,\, x - \text{lo})^2$ | lower bound — enforce $x \ge \text{lo}$ only |
| `flat_bottomed2(x, hi)` | $\max(0,\, x - \text{hi})^2$ | upper bound — enforce $x \le \text{hi}$ only |

`flat_bottomed` / `flat_bottomed1` / `flat_bottomed2` are the same maths (and names) as the built-in
`flat-bottomed` / `flat-bottomed1` / `flat-bottomed2` blocks.

#### Math and operators

Math functions are dispatched to the active backend:

| group | names |
|---|---|
| elementwise | `sqrt` `exp` `log` `abs` `sin` `cos` `clip(x, lo, hi)` `wrap(x)` = $\mathrm{atan2}(\sin x, \cos x)$, folds an angle/deviation into $[-\pi, \pi]$ (use on `dihedral` deviations — see Periodicity above) |
| reductions | `sum` `minimum` `maximum` |
| branching | `where(cond, a, b)` — there is **no `if`** (keeps the closure jax-traceable, since it must trace inside `lax.scan`) |

Supported operators are `+ - * / ** %`, unary `-`, and comparisons (`<` `<=` …). Use `&` and `|`
to combine boolean masks for `where`.

### Examples

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
    move: [A, C]  # B and D are pinned for this custom term
    weight: 1.0
  # pull a domain's radius of gyration toward a target compactness
  - name: compact
    energy: "harmonic(rg(dom), 12.0)"            # (rg - 12)**2
    selections: {dom: "chain A and resid 1 to 80"}
  # periodicity-safe dihedral toward 180deg (pi rad): wrap the deviation, NOT harmonic(dihedral, t)
  - name: planar_dihedral
    energy: "wrap(dihedral(A, B, C, D) - 3.14159)**2"
    selections: {A: "resid 10", B: "resid 11", C: "resid 12", D: "resid 13"}
  # NCS symmetry: drive two chains to the same shape via Kabsch superposition (|A| == |B|)
  - name: ncs
    energy: "norm(kabsch(A, B) - coords(B))"
    selections: {A: "chain A and name CA", B: "chain B and name CA"}
  # composable RMSD: compare one moving domain with two reference states
  - name: rmsd_compose
    energy: "rmsd(dom, state1) - rmsd(dom, state2)"
    selections:
      dom: "chain A and resid 1 to 80"
      state1: "ref1 and chain A and resid 1 to 80"
      state2: "ref2 and chain A and resid 1 to 80"
    refs:
      ref1: {ref_cif: "state1.cif", pairing: align}
      ref2: {ref_pdb: "state2.pdb", pairing: align}
```

#### Registered Python function

The function can be reused via `use:` or passed directly with `add_custom`:

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
