# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
task lint        # ruff check + format validation
task format      # ruff format + auto-fix lint
task test        # run all tests (uses /venv Python to preserve torch ecosystem)
task test-ci     # run non-GPU tests only (-m "not gpu")
```

Single test:
```bash
/venv/bin/python -m pytest tests/test_selection.py::TestBooleanOperations -v
```

## Architecture

**rgi_utils** — Restraint-Guided Inference utilities for PyTorch-based structure prediction models.

### Core flow

1. **`CombinedRestraints`** (`combined_restraints.py`) is the singleton entry point. Callers:
   - call `set_config()` with a dict of restraint parameters
   - call `setup()` with a `FrameworkAdapter` + coordinate/feature tensors to build restraint data
   - call `set_feats()` to resolve distance restraint atom selections
   - call `minimize()` to run CPU (scipy) or GPU (torchmin) optimization

2. **Framework adapters** (`boltz/adapter.py`) implement the `FrameworkAdapter` protocol from `atom_context.py`, providing an `iter_atoms()` generator that yields `AtomRecord(chain, resid, index)` for each non-padded atom.

3. **Restraint data classes** (`bond_restr_data.py`, `angle_restr_data.py`, `chiral_data.py`, `distance_restr_data.py`) each implement `calc()` / `grad()` / `print()` / `calc_sd()`. They operate on flat coordinate arrays (CPU) or tensors (GPU via `torch_restr_impl.py`).

4. **`RestrTorchImpl`** (`torch_restr_impl.py`) packs all restraint indices and parameters into batched tensors for GPU execution, including dynamic VdW contact detection via `torch_cluster` radius search.

5. **Atom selection DSL** (`selection.py`): `AtomSelector` parses strings like `"(chain A or chain B) and resid 1 to 10"` into a node tree and evaluates it against `AtomRecord` lists. Used by `DistanceData` to resolve COM-based distance restraint atom sets.

### Key design points

- `CombinedRestraints` is a singleton; call `reset()` between uses in tests.
- `AtomRecord.index` is a **padded** index (raw tensor position); `resid` is 1-based residue id.
- Distance restraints support four types: `harmonic`, `flat-bottomed`, `flat-bottomed1`, `flat-bottomed2`. The only supported `calc_method` is `unfixed-absolute` (COM-based).
- GPU tests are marked `@pytest.mark.gpu` and excluded in CI.
