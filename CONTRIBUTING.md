# Contributing to rgi-utils

`rgi_utils` is the shared **Restraint-Guided Inference (RGI)** engine — it injects
differentiable distance / angle / dihedral / ligand-conformer / RMSD / custom restraints into
the denoising loop of diffusion structure predictors, on both the **torch** and **jax**
backends. One engine is integrated into eight models across six predictors (boltz, protenix,
chai-lab, openfold-3, esmfold2, alphafold3).

This guide is for people **hacking on the engine itself**. Start here, then:

- [`README.md`](README.md) — user-facing usage and the restraint catalogue.
- [`AGENTS.md`](AGENTS.md) (= `CLAUDE.md`) — the deep architecture reference and full list of
  design invariants.
- [`doc/config.md`](doc/config.md) — the complete `restraints_config` schema and selection DSL.
- [`doc/`](doc/) — one per-tool integration write-up (`boltz_restr.md`, `alphafold3_restr.md`, …).

Adding RGI support to a **new** predictor is a different task — use the `implement-rgi` skill
(under `.claude/skills/` / `.agents/skills/`), not this doc.

## Development environment

Python `>=3.12`. From this checkout:

```bash
uv sync          # creates .venv with the dev dependencies (pytest, ruff)
```

The optional `torch` / `jax` extras (see [`pyproject.toml`](pyproject.toml)) pull the backend
you want to exercise; `uv sync` installs the CPU-only torch by default to keep the environment
small. `import rgi_utils` on its own needs **only numpy** (see the invariants below).

## Running tests and lint

Use the local `.venv` directly — these are the commands a contributor runs:

```bash
.venv/bin/python -m pytest -m "not gpu" -q                 # full non-GPU suite
.venv/bin/python -m pytest tests/test_backend_parity.py -q # backend energy+grad agreement
uvx ruff check src tests && uvx ruff format --check src tests
```

Notes:

- **Do not use `task test` / `task test-ci` for local work.** Those targets invoke the docker
  image's `/venv` (the CI environment), not your checkout's `.venv`, and will fail locally with
  a missing interpreter. The `task lint` / `task format` targets are fine (they use `uv run`).
- **GPU tests** are marked `@pytest.mark.gpu`, need a real CUDA device, and are **excluded from
  CI** (`-m "not gpu"`). Run them in your own GPU environment. Maintainer-facing cluster
  run details live in [`AGENTS.md`](AGENTS.md).

## Code style

- **ruff**, `line-length = 88` — code is wrapped by the formatter. `E501` is deliberately
  relaxed to 120 so the codebase's dense long-form **comment** style is not rewrapped; keep
  *code* within 88.
- Comments, documentation, and commit messages are **English**.

## Invariants you must not break

These are the properties a change can silently violate. A divergence here means one
`restraints_config` selects different atoms or fires at a different noise level in different
tools. The full list is in [`AGENTS.md`](AGENTS.md) ("Key design points"); the essentials:

- **`import rgi_utils` stays numpy-only.** torch and jax are imported **lazily** inside the
  backend modules — never at the top level.
- **Three-backend parity.** Any energy change lands in **all three** of
  `src/rgi_utils/energy/{numpy,torch,jax}_energy.py`. `numpy_energy` is the reference; the torch
  and jax versions must match it in **energy and gradient** — guarded by
  `tests/test_backend_parity.py`. There is **no numpy optimizer** (numpy is the energy reference
  only); optimization requires torch or jax.
- **`AtomRecord` conventions.** `resid` = per-chain **1-based ordinal** (resets at each chain),
  *not* the author residue number or a global token index. `index` = the atom's **row in the
  coordinate tensor handed to `minimize`** (after any reshape). These must match across tools.
- **`CombinedRestraints` is instance-scoped** — one fresh instance per structure, never a
  singleton. This is what keeps batch runs from leaking the previous structure's config.
- **Gating.** `minimize` is gated on the **pre-step schedule sigma** in every tool. A restraint
  uses a **sigma window** (`start_sigma`/`stop_sigma`) **XOR** a **step window**
  (`start_step`/`stop_step`) — mutually exclusive, enforced at config time. Prefer sigma windows;
  step counts differ per tool, so step windows are not portable.

## Adding or changing a restraint

The data flow is three layers plus autodiff (details in [`AGENTS.md`](AGENTS.md)):

- `src/rgi_utils/spec.py` — `RestraintSpec`, padded NumPy arrays, local indices into
  `active_sites`.
- `src/rgi_utils/energy/*` — the differentiable maths, one file per backend.
- `src/rgi_utils/optim/{torch,jax}_optim.py` — the CG solver that minimises the active coords.
- `src/rgi_utils/featurizer.py` — turns RDKit mols into bond/angle/chiral/cistrans/plane/VdW
  restraints.
- `src/rgi_utils/config.py` — parses the shared `restraints_config`.
- `src/rgi_utils/selection.py` — the atom-selection DSL.
- `src/rgi_utils/custom/` — the extension point for user-defined (formula/`ctx`-fn) restraints.

The rule: a **new energy term** ⇒ implement it in all three backends **and** add a
`tests/test_backend_parity.py` case. A new **per-entry gated** term additionally needs its gate
key in the torch GPU pre-gate (`optim/torch_optim.py` `_gated_prepared`) — CPU CI cannot catch a
term that silently goes ungated on the compiled GPU path. Then:

- Config-surface changes ⇒ update [`doc/config.md`](doc/config.md).
- User-facing changes ⇒ update [`README.md`](README.md).

## Adding a new predictor

Use the `implement-rgi` skill. A tool adds only three small things — a framework **adapter**
(`src/rgi_utils/<tool>/adapter.py`, thin: `iter_atoms()` and optionally `iter_ligand_confs()`),
the **loop hooks** around `minimize`, and one `restraints_config` pass-through. **Never**
reimplement restraint maths, config parsing, or atom selection inside a tool — that is how tools
drift out of parity.

## Submitting changes

- Base your work on and open pull requests against **`rgi-integration`** (the active development
  branch). `main` is the CI/release-tracking branch.
- Run the non-GPU suite and ruff locally before opening a PR — CI runs exactly these
  (`uv sync --frozen`, then `ruff check` / `ruff format --check` / `pytest -m "not gpu"`).
- Keep commits scoped, with English messages.

## Docs to keep in sync

| Doc                                    | Covers                                          |
| -------------------------------------- | ----------------------------------------------- |
| [`README.md`](README.md)               | user-facing usage, restraint catalogue          |
| [`doc/config.md`](doc/config.md)       | full `restraints_config` schema + selection DSL |
| [`doc/`](doc/) `*_restr.md`            | per-tool integration notes                      |
| [`AGENTS.md`](AGENTS.md) / `CLAUDE.md` | deep architecture + invariants                  |
