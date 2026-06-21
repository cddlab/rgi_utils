# Adding a restraint type

The engine ships five **frozen** built-in restraints (conformer / distance / angle /
dihedral / RMSD), each hand-wired across spec / featurizer / config / energy / optim. Do
**not** add a sixth that way. New restraints go through the **registry** — one
`RestraintType` descriptor + a per-backend leaf energy function — and the engine's generic
branches pick them up with no core edits. Two authoring routes share the one mechanism:

- **Config-only (pattern B)** — no Python. The built-in `custom` restraint
  (`custom_restraint.py`) reads a declarative `custom_restraints_config` (a vocabulary of
  geometric `measure`s + standard penalty `form`s). Add an original restraint by editing
  the config. Use this when your restraint is a combination of existing primitives
  (distance / angle / dihedral / radius_of_gyration of atom-group centroids).
- **Code-level (pattern A)** — register your own `RestraintType` with new maths. Use this
  when no `measure` fits. This page is about pattern A.

## The one rule: parity is structural

A custom restraint is a **per-entry, CG-solved energy term** that must run on **every**
backend — numpy (the reference that keeps the parity test a 3-way check), torch (5 tools),
and jax (AF3). `register_restraint` therefore **requires a leaf fn for all three
backends**. They may be lazy `"module:function"` dotted-path strings (resolved only when
that backend runs), so declaring a jax leaf costs no jax import for a torch-only run.

## Recipe

```python
from rgi_utils import RestraintType, register_restraint

register_restraint(RestraintType(
    name="myterm",                              # unique; not a built-in term name
    config_section="myterm_restraints_config",  # top-level config key (auto-whitelisted)
    data_class=MyData,                          # per-entry parse + site resolution (below)
    data_builder=build_my_arrays,               # (resolved_items, g2l) -> arrays dataclass
    spec_schema=(("grp1_idx","i"), ("grp1_mask","f"), ("target","f"),
                 ("weight","f"), ("mask","f"), ("start_sigma","f"), ("stop_sigma","f")),
    term_args=("grp1_idx","grp1_mask","target","weight"),  # leaf args (mask appended last)
    leaf_fns={"numpy": myterm_numpy, "torch": "mypkg.energy:myterm_torch",
              "jax": "mypkg.energy:myterm_jax"},
    gate="myterm",                              # any label != "conf"/"dist" -> per-entry CG
))
```

The engine then, with **zero** further edits, routes it through: config parse
(`config.py`, the section is added to the top-level whitelist so it is accepted, not
silently dropped), site resolution + `active_sites` union + array build
(`featurizer.py`), the energy dispatch (`_terms.py` `term_energies` / `pack_spec`, all
three backends), the per-entry sigma gate (including the torch GPU pre-gate compile cache,
so it can't go ungated on CUDA), the solver run-condition (torch + jax), and the
`finalize` energy breakdown.

### `data_class` contract (mirrors `group_geom_restr_data.AngleRestraintData`)

- `set_config(entry: dict) -> None` — parse one config entry; set `run_restr`. Call
  `warn_unknown_keys(entry, KNOWN, label, logger)` so a typo'd key warns.
- `resolve_sites(adapter) -> None` — resolve selections to **global** atom indices
  (reuse `group_geom_restr_data._resolve_group_sites`).
- `run_restr: bool`, `start_sigma: float | None` (None -> active every step),
  `stop_sigma: float` (-1 -> never released).
- `iter_global_sites() -> Iterable[int]` — every global atom referenced (so the featurizer
  can union them into `active_sites` generically).

### `data_builder(items, g2l)` and the arrays dataclass

Returns a dataclass whose fields match `spec_schema`. Every `*_idx` field holds **local**
indices (`g2l[global_index]`); it must carry `mask`, `start_sigma`, `stop_sigma`. Reuse
the featurizer's padding helpers for groups.

### Leaf fn contract

Signature `fn(positions, *term_args_values, mask)` (mask is appended last by the
dispatch). `positions` is `(..., n_active, 3)`; return a scalar. Index with the local
`*_idx`, zero padding with `*_mask`. Hard constraints:

- **numpy** = the value reference (no autodiff needed). **torch** = autograd-diff'able.
  **jax** = pure `jnp`, static shapes, no python branching on data (it runs inside
  `lax.scan` for AF3). Use `jnp.where` / `.clip`, not `if`.
- All three must agree on **value** to ~1e-6 and on **gradient** (torch vs jax to ~1e-6;
  vs numpy finite-difference to ~1e-4 unless you intentionally rescale the gradient — see
  `_move_centroid`, whose rigid-translation rescale makes group grads torch-vs-jax only).
- Keep it finite at degenerate geometry (`+ _EPS` inside sqrt, nudge `atan2(0,0)`), or the
  optimizer NaNs.

## Verify (mandatory — this is where custom restraints break)

Add your restraint to `tests/test_registry.py` (the harness) and run it through all three
hard paths — a restraint can pass numpy yet NaN on AF3 or go ungated on the compiled GPU
path:

```bash
# non-GPU (3-backend parity, jax lax.scan closure, torch eager, the GPU pre-gate fold):
rgi_utils/.venv/bin/python -m pytest tests/test_registry.py -m "not gpu" -q   # via sbatch
# GPU (the inductor-fused compiled CG must stay NaN-free + converge):
#   run tests/test_registry.py without -m on a GPU node (sbatch_rgi_registry.sh)
```

Mirror `test_custom_*`: energy parity (3-way), grad parity (torch-vs-jax, + numpy-FD where
the gradient isn't rescaled), `test_gated_prepared_folds_*` (the per-entry gate is folded
on the compiled path), a torch + a jax minimize-to-target, and the `@pytest.mark.gpu`
CUDA NaN-free run.
