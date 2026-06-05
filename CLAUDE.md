# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
task lint        # ruff check + format validation
task format      # ruff format + auto-fix lint
task test        # run all tests (uses /venv Python; in docker images)
task test-ci     # run non-GPU tests only (-m "not gpu")
```

Local dev (this checkout) uses `.venv`:
```bash
.venv/bin/python -m pytest -m "not gpu" -q          # full non-GPU suite
.venv/bin/python -m pytest tests/test_optim.py -v   # single file
uvx ruff check src tests && uvx ruff format --check src tests
```
GPU paths (real CUDA torch / jax devices) are exercised by the host tools via
`sbatch`, not on the login node.

## Architecture

**rgi_utils** — Restraint-Guided Inference (RGI): inject distance + ligand
conformer restraints into a structure-prediction diffusion loop via gradient
optimization. Shared by boltz / protenix / chai-lab / openfold-3 / esmfold2 (torch)
and alphafold3 (jax). The end-to-end guide for integrating a new tool is the skill at
`skills/implement-rgi/` (SKILL.md + references/).

Design = **3 layers + autodiff + static shapes + GPU-complete optimization**:

1. **Spec layer** (`spec.py`, backend-agnostic): `RestraintSpec` holds every
   restraint as padded NumPy arrays. All indices are *local* indices into
   `active_sites` (the subset of atoms that participate in any restraint);
   `active_sites` itself stores *global* flat atom indices. Multiple ligands are
   collision-free because each supplies disjoint global indices.

2. **Energy layer** (`energy/{numpy,torch,jax}_energy.py`, differentiable pure
   functions): identical flat-bottomed maths in all three backends —
   `bond/angle/chiral/dihedral/vdw/distance` (dihedral = periodicity-safe torsion
   for cis/trans). `prepare_spec(spec)` → backend arrays;
   `total_energy(positions, prepared)` → scalar. Gradients come from autodiff
   (no hand-written grad). `numpy_energy` is the reference;
   `tests/test_backend_parity.py` checks energy+grad agreement across backends.

3. **Optim layer** (`optim/{numpy,torch,jax}_optim.py`, GPU-complete): optimize
   only `active_sites` coords, scatter back. torch = `LBFGS` autograd (runs on the
   coords' device); jax = `fori_loop`+`value_and_grad` JIT-able inside `lax.scan` (no
   `pure_callback`, no scipy). There is **no numpy optimizer backend** (the old
   scipy path was removed); `numpy_energy` remains only as the pure-numpy energy
   reference for `tests/test_backend_parity.py`. Optimization requires torch or jax.

Supporting modules:
- **`featurizer.py`**: `build_spec(ligand_confs, distance_restraints,
  conformer_config, elements)` — the single place RDKit mols become bond/angle/
  chiral/dihedral restraints (global indices, multi-ligand) and the dynamic
  ligand-protein `VdwConfig` is assembled. Dihedral (cis/trans) detection keys on
  acyclic, non-aromatic `BondType.DOUBLE` bonds and targets the reference-conformer
  torsion; it needs real bond orders (all tools but chai supply them).
- **`config.py`**: `RestraintsConfig.from_dict()` parses the shared
  `restraints_config` (one source of truth for boltz YAML / protenix JSON / AF3).
- **`combined.py`**: `CombinedRestraints` entry point — **instance-scoped, one per
  structure** (NOT a singleton): `CombinedRestraints()` →
  `setup(adapter, nbatch, config=dict)` (folds in the old `set_config`; clears any
  prior spec/optimizer up front so a reused instance is safe) →
  `minimize(coords, step, sigma)` → `finalize(coords, step)`. `get_instance()` /
  `reset()` remain only as back-compat shims — do not build new code on them. Picks
  the backend from config — **default torch**; `gpu` selects the *device* (CPU when
  `gpu:false`, the accelerator when `gpu:true`), NOT the backend, so `gpu:false` runs
  the torch optimizer on CPU (moving GPU coords to CPU and back). The numpy/scipy
  optimizer backend was removed, so `backend` must be torch (default) or jax — an
  explicit `backend: numpy` raises. torch/jax imported lazily. JAX
  tools inside `lax.scan` grab the pure minimizer via `get_minimizer()` instead of
  calling `minimize` per step (AF3 forces `backend: jax`, so for AF3 the `gpu` flag is
  inert — to run AF3 restraints on CPU, run the whole process on the JAX CPU platform).
- **Framework adapters** (`{boltz,protenix,chai,openfold3,esmfold2}/adapter.py`; AF3's lives
  in-tool because it needs CCD machinery): implement `iter_atoms()` (→
  `AtomRecord(chain, resid, index)` for distance selection) and optionally
  `num_atoms()`, `get_elements()`, `iter_ligand_confs()` (→
  `LigandConf(mol, conf_coords, global_indices)` for conformer + VdW).
- **`_mol_build.py`**: `build_ligand_mol(elements, coords, bonds_local,
  perceive_bonds=False)` — the shared RDKit builder. protenix/openfold pass real
  `bonds_local`. chai exposes NO intra-ligand bonds, so its adapter passes
  `perceive_bonds=True`: connectivity is derived from the reference conformer via
  `DetermineConnectivity`, which leaves atoms `noImplicit=True` — so the branch then
  runs `SetNoImplicit(False)` + `UpdatePropertyCache` **before**
  `AssignStereochemistryFrom3D`, else stereocentres read as 3-coordinate and every
  chiral restraint silently vanishes.
- **Atom selection DSL** (`selection.py`): `AtomSelector` parses
  `"(chain A or chain B) and resid 1 to 10"`; used by `DistanceData`
  (`distance_restr_data.py`) to resolve COM-based distance groups.

### VdW (two flavours)

- **Intramolecular** (`conformer vdw: {mode: intramolecular}`): static non-bonded
  ligand-internal pairs (topo distance > 2, within `dmax`), built in `featurizer.py`
  and carried in `spec.vdw` (`VdwArrays`). Works in **all backends** — prefer this.
- **Dynamic ligand-protein** (torch only): lives in `optim/torch_optim.py`. The
  ligand moves (it is in `active_sites`); the protein is a **fixed background** read
  from the full coordinate tensor (`VdwConfig.protein_global`), so only the ligand is
  pushed out of contacts. Penalty `weight * clamp(d - scale*(r_i+r_j), max=0)**2`,
  all-pairs (zero gradient beyond contact) — same maths as boltz's radius search.

The static `vdw_energy` (idx pairs) lives in the energy layer for parity across backends.

### Key design points

- `CombinedRestraints` is **instance-scoped — a fresh `CombinedRestraints()` per
  structure** (this is what makes batch runs / retries correct without leaking the
  previous structure's config). `get_instance()`/`reset()` are back-compat shims only.
- `AtomRecord.index` is the atom's **row in the coordinate tensor handed to
  `minimize`** (global flat index, after any reshape); `resid` is the **per-chain
  1-based residue/token ordinal** (resets at each chain) — not the author residue
  number, not a cumulative token index. These two conventions must match across tools.
- `minimize` is gated on `sigma <= start_sigma`. `start_sigma` is set in exactly two
  places — **per distance entry** (each `distance_restraints_config` entry) and **once
  for all conformer terms** (`conformer_restraints_config.start_sigma` ->
  `RestraintSpec.conf_start_sigma`). There is **no top-level/global `start_sigma`** (a
  top-level one raises `ValueError`); per restraint it is **optional** and defaults to
  `+inf` when omitted — i.e. active at **every** step (set it, e.g. `1.0`, to act only
  late). When `sigma` exceeds every restraint's `start_sigma` the whole step is skipped.
- Distance restraints: `harmonic`, `flat-bottomed`, `flat-bottomed1`,
  `flat-bottomed2`; only `calc_method=unfixed-absolute` (COM-based).
- Top-level `import rgi_utils` must work with numpy only (no torch/jax) — keep
  heavy imports lazy inside the backend modules.
- GPU tests are marked `@pytest.mark.gpu` and excluded in CI.
