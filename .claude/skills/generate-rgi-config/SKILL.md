---
name: generate-rgi-config
description: >-
  Create and validate a ready-to-run restraints_config for Restraint-Guided
  Inference (RGI), placing it correctly for boltz, protenix, chai-lab,
  alphafold3, openfold-3, or esmfold2. Use when a user wants to run RGI, write
  a restraint file, constrain a distance, angle, dihedral, ligand conformer,
  RMSD, or custom energy, or translate a plain-language structural goal into
  the selection DSL, target, and activation window. Use only for an
  already-integrated tool; use implement-rgi when adding RGI support to a new
  predictor.
---

# Generate an RGI restraints_config

## What you are producing

Every RGI tool is driven by **one `restraints_config` dict** (YAML for boltz/chai, JSON
for protenix/AF3/openfold, a Python dict for esmfold2). Your job is to turn a user's
plain-language goal into that dict, place it where the tool reads it, and **validate it
before they spend a GPU run on it**. The engine (`rgi_utils`) does all the maths — you
only write config.

RGI nudges the atoms during diffusion sampling so the final structure satisfies the
restraints. There are **five direct built-in restraint types**, a base-pair macro, and a
custom one:

| the user wants to… | restraint type | block |
|---|---|---|
| keep two parts of the structure at a set distance | **distance** | `distance_restraints_config` |
| set the angle / twist between three / four parts | **angle** / **dihedral** | `angle_` / `dihedral_restraints_config` |
| keep a ligand at a chemically sensible shape | **conformer** | `conformer_restraints_config` |
| pull a region onto a reference structure (PDB/mmCIF) | **RMSD** | `rmsd_restraints_config` |
| pair two nucleotides in Watson–Crick geometry | **base-pair macro** | `base_pair_restraints_config` |
| anything else, as a math formula | **custom** | `custom_restraints_config` |

Full mapping + worked phrasings: **`references/restraint-recipes.md`** (read it when you
are unsure which type a goal needs). Full schema (every key, default, allowed value):
the repo's **`doc/config.md`** — it is the source of truth; do not guess defaults.

## The audience is a beginner — interview, then translate

Assume the user is new to RGI *and* to structure prediction. **Do not** ask them for a
selection DSL string or a penalty type by name. Ask in plain language, then translate and
**show your translation back** so they can sanity-check it. For example:

- "Which two parts should be held apart, and how far?" → a `distance` restraint, harmonic,
  `target_distance`.
- "Should the distance be *exactly* X, or just *at least / at most* X?" → harmonic vs
  flat-bottomed (a band / floor / ceiling). See recipes.
- "Should it apply the whole time, or only once the fold has roughly formed?" → the
  `start_sigma` window (omit = always on; set late only if they want it late).

Explain the *why* as you go (this is the point of the skill): e.g. "I'm qualifying the
selection with `chain A` because `resid` numbering restarts on every chain, so a bare
`resid 90 to 180` would also grab the ligand."

## Workflow

### 1. Which tool? (decides the file format + where the config goes)

If the user hasn't said, ask. The tool decides three things that are **easy to get wrong**
— file format, where the `restraints_config` sits, and how a ligand opts into conformer
restraints. The per-tool table is in **`references/tools.md`**; the essentials:

- **boltz / protenix / alphafold3 / openfold-3** — the config is **nested under a
  `restraints_config` key** inside the tool's own input file (the same file that lists the
  sequences). protenix's input is a JSON *list* of jobs; openfold nests it under
  `queries.<name>`.
- **chai-lab** — the **odd one out**: a **separate sidecar YAML** whose top level **IS**
  the `restraints_config` (do **not** nest it). Sequences live in a separate FASTA.
- **esmfold2** — a **Python dict** passed to `ESMFold2InputBuilder().fold(...,
  restraints_config=...)`.

### 2. What does the user want? → restraint type + penalty shape

Map the goal to one or more restraint types using `references/restraint-recipes.md`. The
two decisions that recur:

- **Pin vs bound.** `harmonic` drives a quantity *to* a target. The `flat-bottomed`
  family leaves it free inside a band (`flat-bottomed`), above a floor (`flat-bottomed1`),
  or below a ceiling (`flat-bottomed2`). "exactly 25 Å" → harmonic; "no closer than 25 Å"
  → flat-bottomed1.
- **Angles are in degrees** in the config (not radians) — `target_angle: 90`,
  `target_dihedral: 180`.

### 3. Which atoms? → selection DSL

Translate the user's description of the parts into selection strings. The grammar +
beginner walk-through is **`references/selection-dsl.md`**. The one rule that causes most
silent failures:

> **`resid` is the per-chain 1-based ordinal** (it restarts at each chain; the ligand gets
> its own ordinal). It is NOT the author residue number and NOT global. **Always qualify a
> protein group with `chain A and (...)`** or a bare `resid` range also sweeps in the
> ligand chain.

### 4. Targets, gating, weight

- **Target**: the distance (Å), angle/dihedral (degrees), or `target_rmsd` (Å).
- **Sigma window** (`start_sigma` / `stop_sigma`): omit for "active every step" (the usual
  case). There is **no top-level `start_sigma`** — it goes on each entry (and once for all
  conformer terms). Setting one at the top level is an error.
- **Weight**: default `1.0` is right for almost everything. For a single distance restraint
  `weight` is a *no-op* (it reaches the target exactly regardless) — don't present it as a
  strength knob there. See `doc/config.md` for the exact semantics.

### 5. Write the config in the right place

- If the user **already has an input file**, inject the `restraints_config` (and the
  per-ligand opt-in flag if conformer is used) into it.
- If they **don't**, scaffold a minimal runnable input from the closest example in the
  repo root (`bench_in_<tool>_*` / the `doc/<tool>.md` "Full config" example) and fill in
  their sequences/ligand. Tell them which fields are theirs to replace.
- For **chai**, write the sidecar YAML *and* remind them it pairs with a FASTA.
- For **esmfold2**, write the Python dict + the `.fold(..., restraints_config=...)` call.

### 6. CONFORMER OPT-IN — the #1 silent no-op

A `conformer_restraints_config` block does **nothing** unless the ligand is *also* flagged
to opt in. This flag lives **outside** the config block and its placement differs per tool:

| tool | how the ligand opts in |
|---|---|
| boltz / protenix / alphafold3 / openfold-3 | `conformer_restraints: true` on the ligand object |
| chai-lab | a `conformer_restraints: {<chain_id>: true}` map in the sidecar |
| esmfold2 | `conformer_restraints=True` on the `LigandInput` |

If you write a conformer block, you **must** also write the matching opt-in, or the run
looks fine but applies no ligand restraint.

### 7. Validate before running

Resolve `SKILL_DIR` to the directory containing this `SKILL.md`, then run the
bundled validator on the file you produced:

```bash
uv run --project <rgi-utils-dir> --frozen --with pyyaml \
  python "$SKILL_DIR/scripts/validate_config.py" <file>
```

It runs the real `RestraintsConfig.from_dict` (catching unknown/misspelled section names,
a top-level `start_sigma`, a leftover `backend` key, mixed sigma+step windows, empty windows, …),
syntax-checks every selection string, and warns when a conformer block has no opt-in.

### 8. Hand off with the run command AND the validation ceiling

Give the user the exact run command for their tool (in `references/tools.md`). Then state
plainly what validation does **not** prove:

> The validator confirms the config is well-formed and the selections *parse*. It **cannot**
> confirm a selection matches the atoms you meant — a syntactically valid
> `chain A and resid 5 to 84` that resolves to **zero atoms** (wrong range, forgotten chain
> qualifier) passes validation but does nothing. To confirm the real run, set
> **`verbose: true`** and read the setup log line `built spec: ... distances=N angles=N
> bonds=N ...`: the count must be **non-zero** for every restraint you asked for. A
> `finalize` energy of `0.00000` with a count of 0 is a silent no-op, not "satisfied".

## Sanity checklist before you hand it over

- [ ] Tool identified; config placed correctly (nested vs chai sidecar vs esmfold2 dict).
- [ ] Every protein selection is qualified with `chain ...`.
- [ ] Angles/dihedrals in **degrees**; distances/RMSD in **Å**.
- [ ] If a conformer block exists, the per-ligand opt-in flag exists too.
- [ ] `verbose: true` is set (so the user can confirm the spec counts at run time).
- [ ] The validator passes.
- [ ] You told the user the run command and the "validation ≠ correct selection" caveat.

## Reference files

- `references/restraint-recipes.md` — goal → restraint type + penalty + example block. The
  interview's hardest step. **Read this first** when the goal is vague.
- `references/selection-dsl.md` — the atom-selection language, beginner-first, with the
  `resid` / `chain` gotcha and ready-made patterns.
- `references/tools.md` — per-tool placement, file format, conformer opt-in, run command.
- the repo's `doc/config.md` — the full, authoritative schema (every key/default/value).
  Always defer to it for anything not spelled out above.
