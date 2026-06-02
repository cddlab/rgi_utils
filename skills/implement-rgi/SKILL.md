---
name: implement-rgi
description: >-
  Guide for adding restraint-guided inference (RGI) — distance restraints
  (COM distance between atom groups) and ligand conformer restraints
  (bond/angle/chiral/intramolecular-VdW toward an ideal geometry) — to a
  diffusion-based structure-prediction tool (boltz / protenix / AlphaFold3 and
  similar samplers). The shared engine `rgi_utils` does all the heavy lifting;
  the tool only needs a small adapter, a few hook lines in its sampling loop,
  and to pass one `restraints_config` dict through. Use this skill whenever
  someone wants to add restraints / 拘束 / RGI / guided sampling to a structure
  predictor, inject distance or conformer constraints into a diffusion model,
  port restraints from one predictor to another, or asks "how do I constrain
  ligand geometry / inter-domain distance during sampling". Do NOT reinvent
  energy/optim/selection code or add new CLI flags — keep the tool side minimal
  and reach feature parity with the other tools.
---

# Implement Restraint-Guided Inference (RGI) in a structure-prediction tool

## What RGI does

Diffusion structure predictors denoise atom coordinates over many steps. RGI
injects a short gradient-based minimization at each step — right after the model
produces `positions_denoised`, before the Euler update — that nudges atoms to
satisfy user restraints:

- **distance restraint**: the center-of-mass distance between two atom groups is
  pulled toward a target (harmonic / flat-bottomed / lower- / upper-bound).
- **conformer restraint**: a ligand's bond lengths, bond angles, chiral volumes,
  and intramolecular non-bonded clashes are pulled toward an ideal RDKit
  geometry (keeps the ligand chemically sensible while the pocket forms).

Both are flat-bottomed squared penalties minimized on a GPU; the same maths runs
under PyTorch, JAX, or NumPy.

## Core principle: rgi_utils does the heavy lifting

**Everything reusable already lives in `rgi_utils`** — the restraint spec
(`spec.py`), the differentiable energies (`energy/{numpy,torch,jax}_energy.py`),
the GPU optimizers (`optim/*`), the config parser (`config.py`), the atom
selection DSL (`selection.py`), and the RDKit→restraint featurizer
(`featurizer.py`). The single entry point is `combined.CombinedRestraints`.

So adding RGI to a new tool means writing **only** three small things on the
tool side:

1. an **adapter** that exposes the tool's data through the rgi_utils protocols,
2. a few **hook lines** in the sampling loop (`setup → minimize → finalize`),
3. **passing one `restraints_config` dict** through to rgi_utils.

This is what makes every tool reach the *same* feature set with the *same*
YAML/JSON. Resist the urge to re-implement restraint maths, re-parse config, or
add tool-specific flags — that work is already done and duplicating it is how
tools drift out of parity. If you find yourself writing a restraint dataclass,
an energy term, or an atom-selection parser in the tool, stop and use the
rgi_utils one instead.

## Implementation — 3 steps

### Step 1 — Install and let rgi_utils pick the backend

Install editable into the tool's venv: `uv pip install -e <path>/rgi_utils[torch]`
for a PyTorch tool, `[jax]` for a JAX tool. You do **not** choose the backend in
code — `rgi_utils.config` reads `gpu` / `backend` from the restraints_config and
picks `torch` (gpu) / `jax` (explicit) / `numpy` (cpu fallback).

### Step 2 — Write an adapter

The adapter is the *only* place the tool's internal data structures meet
rgi_utils. Implement up to four methods (distance-only needs just `iter_atoms`;
add the rest for conformer/VdW). They yield rgi_utils' framework-agnostic
records:

- `iter_atoms() -> Iterator[AtomRecord(chain, resid, index)]` — for distance
  selection. `resid` is the **1-based residue/token ordinal within the chain**
  (resets per chain, so "chain B and resid 5" means the same atom in every tool);
  `index` is the **global flat atom index** in the coordinate tensor.
- `num_atoms() -> int` — padded atom count (the coordinate tensor length).
- `get_elements() -> np.ndarray` — per-atom atomic numbers (padding → 0); only
  needed for the dynamic ligand-protein VdW term.
- `iter_ligand_confs() -> Iterator[LigandConf(mol, conf_coords, global_indices)]`
  — one per ligand: an RDKit mol (heavy atoms), its reference coordinates, and
  the global flat indices of those atoms. The featurizer derives bond/angle/
  chiral restraints from the mol.

Read `references/adapter-protocol.md` for the exact contracts and three worked
adapters (boltz from a feats dict, protenix from a biotite AtomArray, AF3 from a
CCD-based batch).

### Step 3 — Hook the sampling loop

Use the **instance-scoped lifecycle** — construct one `CombinedRestraints` per
structure so batch runs never share state:

```python
from rgi_utils.combined import CombinedRestraints

restr = CombinedRestraints()
restr.setup(YourAdapter(feats), nbatch=multiplicity, config=restraints_config)
# ... inside the denoising loop, right after positions_denoised, before Euler:
coords = restr.minimize(coords, step, sigma)   # torch/numpy: mutates+returns
# ... after sampling (optional per-term energy log):
restr.finalize(coords, step)
```

`setup` clears any prior state and rebuilds the spec, so it is also safe to
reuse an instance. `minimize(coords, step, sigma)` runs the gated optimization
(a restraint is active only when `sigma <= its start_sigma`). For a **JAX/JIT**
tool whose loop is inside `lax.scan` (no Python callbacks allowed), don't call
`minimize` per step — build the spec outside the scan and grab the pure-JAX
`(coords, sigma) -> coords` closure with `restr.get_minimizer()`, then call it
inside the scan. See `references/lifecycle-and-hooks.md`.

### Step 4 — Pass the config (do not add flags)

The tool's input (YAML/JSON) already carries a `restraints_config` dict; route
it unchanged into `setup(config=...)`. Its schema (distance + conformer +
start_sigma + gpu/method/max_iter) is parsed by
`rgi_utils.config.RestraintsConfig.from_dict`, shared across all tools. **Do not
define restraint types, parse the config, resolve atoms, or add a new CLI flag
in the tool** — one dict is enough, and it keeps the tool at parity.

## Minimality check (do this before you finish)

The tool side should contain **none** of these (if it does, move it to
rgi_utils): a restraint dataclass/type, distance/conformer construction logic, an
energy term, an atom-selection parser, a new restraint CLI flag. What *may*
legitimately stay on the tool side: the adapter (framework-specific data
extraction), the loop hooks, passing the config dict, and exposing a
framework-specific object the adapter needs (e.g. pulling out a ligand RDKit
mol). When in doubt, compare against an already-integrated tool — they are the
reference for "how small this should be."

## Framework selection

- **PyTorch (eager loop)** — boltz, protenix, chai-lab, openfold-3. Adapter can
  live in `rgi_utils/<tool>/adapter.py` (it receives a plain dict/array and never
  imports the framework). Call `minimize` each step. Watch the
  autograd-under-inference_mode gotcha. **What the tool exposes for the ligand
  varies, and is the main integration risk**: real bonds but zeroed coords
  (openfold-3 → take geometry from the reference-conformer feature), or real coords
  but no bonds (chai → perceive connectivity), or an over-broad ligand flag (use the
  entity type, not biotite `hetero`). See pitfalls 10–12.
- **JAX (JIT / lax.scan)** — alphafold3. Adapter lives *in the tool* (it usually
  needs framework-specific CCD/atom-name machinery). Build the spec outside the scan;
  inject the pure `get_minimizer()` closure inside.

Details, code, and the non-obvious traps (torch `inference_mode`, jax line
search collapsing atoms, coordinate reshape) are in
`references/framework-notes.md`.

## Pitfalls — read before integrating

`references/pitfalls.md` covers the ones that actually bit during the existing
integrations: per-structure instances (avoid a singleton — it cross-contaminates
batches), leaving atoms (a CCD ligand may model fewer atoms than the mol, e.g.
glucose O1 — subset the mol), the flat-index convention, the per-chain `resid`
definition, the JAX optimizer's line-search choice, and batch/retry behaviour
(instance-scoped lifecycle makes both safe with no extra flags). The chai/openfold
integrations added: zeroed structure coords → use the reference conformer (10);
no intra-ligand bonds → perceive connectivity, then re-enable implicit-H or chiral
restraints silently vanish (11); identify ligands by entity type not `hetero` (12);
a `finalize` energy of 0.0 can mean "0 restraints built", not "satisfied" — check
the setup spec counts (13); undeclared deps / CPU torch builds (14).

## Verify

- **CPU**: run rgi_utils' `tests/test_backend_parity.py` — confirms numpy/torch/
  jax agree on energy and gradient, so whichever backend the tool uses is sound.
- **GPU (real device, usually via the tool's batch/sbatch harness)**:
  - **first, read the `setup` spec counts** (`built spec: bonds=.. angles=..
    chirals=.. distances=.. vdw=..`) and confirm the count is non-zero for every
    restraint type you requested — a type that built 0 restraints reports a perfect
    `finalize` energy of `0.00000`, so near-zero energy alone does NOT prove a
    restraint is working (pitfall 13);
  - distance: the predicted structure's COM distance reaches the target;
  - conformer: spec counts non-zero AND `finalize` bond/angle/chiral energies are
    small, or the ligand RMSD differs between restraint-on and restraint-off runs;
  - batch: put two structures with *different* configs in one run and confirm
    each uses its own (the decisive test that there is no cross-contamination).

Pick a ligand without a leaving atom (ATP, NAD, caffeine) when comparing tools —
CCD sugars/amino acids drop a leaving atom and change the restraint count.
