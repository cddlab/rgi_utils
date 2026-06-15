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
      # move:  optional; both (default, split) | 1 (only atom_selection1) | 2 (only
      #        atom_selection2). 1/2 pin the OTHER group (e.g. move only a ligand to a pocket).
      harmonic: {target_distance: 25.0}
      # alternatives: flat-bottomed {target_distance1, target_distance2},
      #               flat-bottomed1 {target_distance1}, flat-bottomed2 {target_distance2}
  angle_restraints_config:        # angle of THREE group centroids (vertex = group 2), DEGREES
    - atom_selection1: "chain A and resid 10 and name CA"
      atom_selection2: "chain A and resid 20 and name CA"   # vertex (group 2)
      atom_selection3: "chain B and resid 5 and name CA"
      harmonic: {target_angle: 90.0}   # types mirror distance: or flat-bottomed
      #   {target_angle1, target_angle2} | flat-bottomed1 {target_angle1} | flat-bottomed2 {target_angle2}
      # move: 1,3                       # DEFAULT: arms (1,3) free, vertex (2) pinned. or 'all' | index | list
      # weight / start_sigma / stop_sigma: optional. The whole group moves RIGIDLY (the
      #   1/N centroid gradient is un-suppressed in _move_centroid), so the default weight: 1.0 drives
      #   any group size — no need to scale weight by group size.
  dihedral_restraints_config:     # dihedral of FOUR group centroids (axis = group2-3), DEGREES
    - atom_selection1: "chain A and resid 1 and name CA"
      atom_selection2: "chain A and resid 2 and name CA"
      atom_selection3: "chain A and resid 3 and name CA"
      atom_selection4: "chain A and resid 4 and name CA"
      harmonic: {target_dihedral: 180.0}   # periodicity-safe; or flat-bottomed
      #   {target_dihedral1, target_dihedral2} (cannot straddle +-180) | flat-bottomed1 | flat-bottomed2
      # move: 1,4                           # DEFAULT: ends (1,4) free, axis (2,3) pinned. or 'all' | index | list
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
`index`, `name` (atom name, e.g. `name CA` / `name N CA C O`), the molecule-type
keywords `protein` / `dna` / `rna`, the polymer-substructure keywords `backbone` /
`sidechain`, and `and`/`or`/`not`/`( )`. `resid` is the
per-chain 1-based ordinal the adapter yields in `AtomRecord.resid`. `name` matches
`AtomRecord.name` case-insensitively (alphanumeric only — a nucleic-acid `C1'` is not
selectable but fails loudly, never silently); it is one way to restrict an RMSD
superposition to backbone (`name CA`) so a substituted homolog side chain is not
pinned. `protein`/`dna`/`rna` match on `AtomRecord.mol_type`, the
adapter-normalized molecule type: every adapter MUST map its framework's
molecule-type enum to the shared strings `"protein"/"dna"/"rna"/"ligand"` (raw enum
ints differ across tools — boltz/esm DNA=1/RNA=2 vs chai/openfold RNA=1/DNA=2 — so
the string is the only safe cross-tool currency). A ligand / water / untyped atom
(`mol_type=None`) matches none of the three.

`backbone`/`sidechain` are PyMOL-like polymer selectors, matched by atom name but
GATED on polymer type: `backbone` = protein N/CA/C/O(/OXT) or the nucleic
sugar-phosphate; `sidechain` = the polymer complement (so glycine has no sidechain
heavy atom, and a ligand atom merely named "C"/"N"/"O"/"P" never matches either). The
polymer type is `AtomRecord.mol_type` where the adapter sets it (boltz/esm/AF3) else
derived from `AtomRecord.resname` (chai/of3/protenix don't set `mol_type`) — so BOTH
`mol_type` and `resname` flow through the selection candidate dict, and a modified
residue (e.g. MSE) counts as polymer only where the framework typed it (the same
accepted cross-tool divergence as `protein`). `backbone` is the register-stable way to
limit an RMSD fit to the main chain; `name CA` is the narrower CA-only variant. Note
`not backbone` is NOT `sidechain` — `not backbone` also matches non-polymer atoms,
`sidechain` is polymer-gated. Compose freely, e.g.
`protein and chain A`, `not protein`, `(protein or rna) and resid 5 to 84`,
`name CA and resid 5 to 84`, `backbone and chain A`, `sidechain and resid 5 to 84`.
