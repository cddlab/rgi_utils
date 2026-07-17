# Conflict playbook

## The one principle

RGI is an **additive** patch: it never replaces the tool's logic, it inserts hooks around
it. So a conflict between upstream and `rgi-integration` is almost never a genuine
either/or. The correct resolution is nearly always:

> **Take upstream's change in full, then re-install the RGI hook on top of the new code.**

"Pick ours" silently reverts an upstream fix. "Pick theirs" silently deletes the restraint.
Both leave a tree that compiles and a merge that looks clean, which is why both are so easy
to do by accident. When you catch yourself reaching for `--ours` or `--theirs` on a source
file, that is the moment to slow down — those flags are appropriate for lockfiles and for
AF3's reformatted `run_alphafold.py`, and almost nowhere else.

## A worked example (real, from `transformers_restr`)

Upstream added MSA inference-diversity knobs to the same signature RGI threads its
`restraints` kwarg through, and git had no way to order the two additions:

```python
<<<<<<< HEAD
        restraints=None,
=======
        lm_mask_pct: float | None = None,
        msa_max_depth: int = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
>>>>>>> main
        **kwargs,
```

Neither side is wrong; they are disjoint additions to one parameter list. `--ours` deletes
upstream's new feature, `--theirs` deletes the restraint plumbing, and **both resolutions
compile and leave a merge that looks clean**. The resolution is to keep everything:

```python
        lm_mask_pct: float | None = None,
        msa_max_depth: int = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
        restraints=None,
        **kwargs,
```

Most conflicts in this sync look exactly like this. Notice also what the RGI side is here:
just a kwarg being threaded down a call chain. There is no `rgi_utils` import, no
`CombinedRestraints` — so if you resolved this by taking `--theirs`, every obvious RGI
marker in the repo would still grep clean while the hook downstream silently never fires.
That is what `rgi_probe.sh` is watching for.

## What "correctly re-installed" means

There is a spec; you do not have to judge by eye:

- **Hook shape and placement** — `rgi_utils/.claude/skills/implement-rgi/references/lifecycle-and-hooks.md`.
  The hook goes right after the network's denoised x0 and before the integrator step:
  `restr.setup(adapter, nbatch, config)` once, `coords = restr.minimize(coords, step, sigma)`
  per step, `restr.finalize(coords, step)` at the end.
- **Known traps** — `.../implement-rgi/references/pitfalls.md`.
- **Cross-tool invariants** — the workspace `CLAUDE.md`, section "Non-obvious invariants
  that must hold across all six tools". These are the acceptance criteria for a resolution.

The invariant most often broken by a resolution — because it is invisible in a diff — is the
**sigma gating**: `minimize` must be gated on the *pre-step / pre-churn* schedule sigma
(boltz `sigma_tm`, protenix `c_tau_last`, chai `sigma_curr`, openfold `noise_schedule[tau]`),
not the churn-inflated level. When upstream reshuffles a sampling loop, the variable you
were reading may still exist under the same name while now holding the churned value. Read
the surrounding loop, not just the conflict hunk.

`CombinedRestraints` is **instance-scoped, one per structure** — never hoist it to a module
global while untangling a merge, or batch runs leak the previous structure's config.

## By hunk type

### Diffusion loop hook (`diffusion.py`, `generator.py`, `chai1.py`, `diffusion_module.py`, `diffusion_head.py`, `modeling_esmfold2_common.py`)

The highest-value conflict and the one worth the most care. Read upstream's new loop
end-to-end first, then place the hook by its contract (after x0, before the step, gated on
pre-step sigma) rather than by matching the old line numbers. If upstream renamed the
denoised tensor or restructured the step, the hook moves — that is expected and fine.

For JAX (AF3): the spec is built **outside** the scan and only the pure
`restr.get_minimizer()` closure goes inside. If a merge tempts you to construct restraints
inside the scanned function, that is a trace-time error waiting to happen.

### Config plumbing (`schema.py`, `types.py`, `json_parser.py`, `query.py`, `folding_input.py`, `prepare_input.py`)

Threading `restraints_config` from the tool's input file to `setup(config=...)`, plus
per-ligand opt-in flags. Upstream refactors of input schemas hit this often.

This is the plumbing that gets dropped without conflicting. After resolving, trace the
config from the input file to `setup()` by reading, not by assuming — a break anywhere in
the chain yields `n_active=0`, which is a silent no-op, not an error.

### Dependency declarations (`pyproject.toml`, `requirements.in/txt`, `pixi.toml`)

One added line declaring `rgi_utils`. Keep both sides. Then check the corresponding lockfile.

### Lockfiles (`pixi.lock`, `uv.lock`)

Never resolve by hand. Take upstream's, regenerate with the resolver, commit the result —
see the lockfile section of `tools.md`. Regeneration downloads packages: run it under
`sbatch`, not on the login node.

### `README.md` / `.gitignore`

Pure additions on the RGI side. Keep upstream's changes and the RGI block; resolve by
concatenating, not by choosing.

### AF3 `run_alphafold.py`

A ~1500-line reformat. Do not line-merge — see the AF3 section of `tools.md`.

## Useful commands while resolving

```bash
git -C <tool> status                                  # what conflicted
git -C <tool> log --oneline -p main..rgi-integration -- <file>   # what RGI added to this file, originally
git -C <tool> log --oneline -p HEAD..main -- <file>   # what upstream just changed here
git -C <tool> diff --check                            # leftover markers
git -C <tool> merge --abort                           # bail out cleanly and rethink
```

`git log -p main..rgi-integration -- <file>` is the most useful of these: it shows the RGI
patch for one file in isolation, which is exactly what you need to re-install onto
upstream's new version.

## When to stop and ask

- A conflict on **`main`** in any tool other than boltz — that means `main` is not a
  pristine upstream mirror, which is a broken invariant and a separate decision.
- Upstream **removed** the function RGI hooks into. Re-deriving the hook is then an
  implement-rgi task, not a merge resolution; loop in the user.
- The probe shows plumbing dropped and `git log -p` doesn't make it obvious what was lost.

Reporting an unresolved sync is a good outcome. Reporting a resolved sync whose restraints
silently do nothing is the bad one — the tool still emits confident, plausible structures,
so nothing downstream will catch it.
