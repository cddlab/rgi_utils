# Lifecycle and hooks

## Instance-scoped lifecycle (the supported pattern)

```python
from rgi_utils.combined import CombinedRestraints

restr = CombinedRestraints()                        # ONE per structure
restr.setup(adapter, nbatch=N, config=cfg_dict)     # build spec (auto-resets state)
coords = restr.minimize(coords, step, sigma)        # each denoise step
restr.finalize(coords, step)                        # optional per-term energy log
```

Construct a **fresh instance per structure**. `setup()` clears derived state
first, so reusing an instance is also safe, but a per-structure instance is what
makes batch runs and retries correct with zero extra bookkeeping (no shared
state to leak the previous structure's config). `get_instance()` / `reset()`
exist only as a back-compat singleton shim — do not build new code on them.

## `setup(adapter, nbatch=1, config=None)`
- `config=`: pass the tool's `restraints_config` dict; setup calls `set_config`
  internally, folding the old `set_config` + `setup` into one call.
- It resolves distance selections via `adapter.iter_atoms`, builds the conformer
  spec via `adapter.iter_ligand_confs` + the featurizer, and picks the backend
  from the config (`gpu` → torch, `backend: jax` → jax, else numpy).
- Clears `spec`/optimizers up front, so a reused instance never carries a stale
  spec.

## `minimize(coords, step, sigma)`
- torch / numpy: optimizes in place and returns `coords`.
- **Gated**: each restraint contributes only when `sigma <= start_sigma`; when
  `sigma` exceeds every restraint's `start_sigma`, the whole step is skipped
  (cheap at high noise, so you can hook it unconditionally).

## `finalize(coords, step)`
- Runs only when `verbose`. Logs the per-term energy
  `bond=.. angle=.. chiral=.. vdw=.. distance=.. total=..`, so you can see how
  well each restraint type is satisfied in the final structure. The sum equals
  the total restraint energy (checked against `total_energy`).

## `get_minimizer()` — for JAX/JIT loops
Returns a pure `(coords, sigma) -> coords` closure with no Python side effects,
safe to call inside `lax.scan` / `hk.scan`. Build the spec outside the scan (it
is numpy work done at build time), grab the closure once, then call it per step
inside the compiled loop.

## PyTorch hook (boltz / protenix): where in the loop
Right after the network's denoised prediction, before the Euler/integrator step:

```python
restr = CombinedRestraints()
restr.setup(ToolAdapter(feats), nbatch=multiplicity,
            config=feats["record"][0].restraints_config)   # per-structure config
for step, sigma in schedule:
    x_denoised = net(x_noisy, sigma)
    x_denoised = restr.minimize(x_denoised, step, sigma)    # <-- the hook
    x = integrator_update(x_noisy, x_denoised, sigma, ...)
restr.finalize(x, step)
```

Note the config comes from a **per-structure source** (here the boltz Record), so
each structure in a batch builds its own spec. Do not read it from a global.

## JAX hook (AF3): closure inside the scan
```python
# build time (outside scan, host/numpy):
restr = CombinedRestraints()
restr.setup(adapter, config=fold_input.restraints_config)   # backend=jax in config
minimizer = restr.get_minimizer()       # pure (flat_coords, sigma) -> flat_coords

# inside the compiled apply_denoising_step (hk.scan):
shape = positions_denoised.shape        # (num_tokens, max_atoms, 3)
flat = positions_denoised.reshape(-1, 3)
positions_denoised = minimizer(flat, sigma).reshape(shape)  # <-- the hook
```
The `(num_tokens, max_atoms, 3) <-> (-1, 3)` reshape is the only tool-specific
glue; the minimizer itself is rgi_utils.

## Config dict schema (shared across all tools)

```yaml
restraints_config:
  gpu: true                 # -> torch backend; or set  backend: jax  /  backend: numpy
  verbose: true             # print setup + finalize stats
  max_iter: 100             # optimizer iterations per step
  method: "CG"              # jaxopt solver: CG (NonlinearCG) or LBFGS
  # NOTE: there is NO top-level/global start_sigma (a top-level one raises ValueError).
  # Set it per distance entry and once in conformer_restraints_config; it is OPTIONAL —
  # omit it for "active at every step", or set e.g. 1.0 to act only late in denoising.
  distance_restraints_config:
    - atom_selection1: "resid 5 to 84"
      atom_selection2: "chain B and resid 90 to 180"
      # start_sigma:  optional; omitted -> active at every step (set e.g. 1.0 for late-only)
      harmonic: {target_distance: 25.0}
      # alternatives: flat-bottomed {target_distance1, target_distance2},
      #               flat-bottomed1 {target_distance1}, flat-bottomed2 {target_distance2}
  conformer_restraints_config:
    # start_sigma:  optional; one value for ALL conformer terms (omitted -> every step)
    bond:     {weight: 1.0}
    angle:    {weight: 1.0}
    chiral:   {weight: 1.0}
    dihedral: {weight: 1.0}                          # cis/trans (E/Z): holds acyclic, non-aromatic double bonds at their reference dihedral. ON by default; weight<=0 disables. optional slack (radians).
    vdw:      {weight: 1.0, mode: "intramolecular"}  # intramolecular = static, works in all backends
```

The `dihedral` term needs the ligand mol to carry real bond ORDERS (it keys on
`BondType.DOUBLE`). boltz (CCD mol), protenix/openfold (biotite BondList) and AF3
(SMILES/CCD) all supply them. chai's tokenized context exposes only heavy atoms
with no bonds, so its perceived topology is all-single and `dihedral` finds
nothing there (graceful: `dihedrals=0`).

The selection DSL (`selection.py`) supports `chain`, `resid N`, `resid A to B`,
`index`, and `and`/`or`/`not`/`( )`. `resid` is the per-chain 1-based ordinal the
adapter yields in `AtomRecord.resid`.
