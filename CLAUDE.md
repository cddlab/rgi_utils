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
optimization. Shared by boltz/protenix (torch) and alphafold3 (jax).

Design = **3 layers + autodiff + static shapes + GPU-complete optimization**:

1. **Spec layer** (`spec.py`, backend-agnostic): `RestraintSpec` holds every
   restraint as padded NumPy arrays. All indices are *local* indices into
   `active_sites` (the subset of atoms that participate in any restraint);
   `active_sites` itself stores *global* flat atom indices. Multiple ligands are
   collision-free because each supplies disjoint global indices.

2. **Energy layer** (`energy/{numpy,torch,jax}_energy.py`, differentiable pure
   functions): identical flat-bottomed maths in all three backends —
   `bond/angle/chiral/vdw/distance`. `prepare_spec(spec)` → backend arrays;
   `total_energy(positions, prepared)` → scalar. Gradients come from autodiff
   (no hand-written grad). `numpy_energy` is the reference;
   `tests/test_backend_parity.py` checks energy+grad agreement across backends.

3. **Optim layer** (`optim/{numpy,torch,jax}_optim.py`, GPU-complete): optimize
   only `active_sites` coords, scatter back. torch = `LBFGS` autograd on GPU
   tensors; jax = `fori_loop`+`value_and_grad` JIT-able inside `lax.scan` (no
   `pure_callback`, no scipy); numpy = `scipy.optimize.minimize` (CPU fallback).

Supporting modules:
- **`featurizer.py`**: `build_spec(ligand_confs, distance_restraints,
  conformer_config, elements)` — the single place RDKit mols become bond/angle/
  chiral restraints (global indices, multi-ligand) and the dynamic ligand-protein
  `VdwConfig` is assembled.
- **`config.py`**: `RestraintsConfig.from_dict()` parses the shared
  `restraints_config` (one source of truth for boltz YAML / protenix JSON / AF3).
- **`combined.py`**: `CombinedRestraints` singleton entry point —
  `set_config(dict)` → `setup(adapter, nbatch)` → `minimize(coords, step, sigma)`
  → `finalize(coords, step)`. Picks the backend from config; torch/jax imported
  lazily. JAX tools that run inside `lax.scan` grab the pure minimizer via
  `get_minimizer()` instead of calling `minimize` per step.
- **Framework adapters** (`boltz/adapter.py`, more per tool): implement
  `iter_atoms()` (→ `AtomRecord(chain, resid, index)` for distance selection) and
  optionally `iter_ligand_confs()` + `get_elements()` (conformer + VdW).
- **Atom selection DSL** (`selection.py`): `AtomSelector` parses
  `"(chain A or chain B) and resid 1 to 10"`; used by `DistanceData`
  (`distance_restr_data.py`) to resolve COM-based distance groups.

### VdW (ligand-protein)

The static `vdw_energy` (idx pairs) lives in the energy layer for parity. The
*dynamic* ligand-protein clash term lives in `optim/torch_optim.py`: the ligand
moves (it is in `active_sites`), the protein is a **fixed background** read from
the full coordinate tensor (`VdwConfig.protein_global`), so only the ligand is
pushed out of contacts. Penalty `weight * clamp(d - scale*(r_i+r_j), max=0)**2`,
all-pairs (zero gradient beyond contact) — same maths as boltz's radius search.

### Key design points

- `CombinedRestraints` is a singleton; call `reset()` between uses in tests.
- `AtomRecord.index` is a **padded** index (raw tensor position); `resid` is
  1-based residue id.
- Distance restraints: `harmonic`, `flat-bottomed`, `flat-bottomed1`,
  `flat-bottomed2`; only `calc_method=unfixed-absolute` (COM-based).
- Top-level `import rgi_utils` must work with numpy only (no torch/jax) — keep
  heavy imports lazy inside the backend modules.
- GPU tests are marked `@pytest.mark.gpu` and excluded in CI.
