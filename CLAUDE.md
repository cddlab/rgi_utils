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
conformer + RMSD restraints into a structure-prediction diffusion loop via gradient
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
   `bond/angle/chiral/cistrans/vdw/distance/rmsd/group_angle/group_dihedral` (cistrans =
   periodicity-safe torsion for cis/trans; rmsd = Kabsch-superposed RMSD toward a target,
   fit/calc separable; `group_angle`/`group_dihedral` = the angle/dihedral of 3/4 atom
   GROUPS' centroids — the angular analogue of the centroid-distance restraint, distinct from the
   per-atom `angle`/`cistrans` conformer terms). `prepare_spec(spec)` → backend arrays;
   `total_energy(positions, prepared)` → scalar. Gradients come from autodiff
   (no hand-written grad). `numpy_energy` is the reference;
   `tests/test_backend_parity.py` checks energy+grad agreement across backends.

3. **Optim layer** (`optim/{torch,jax}_optim.py` + `optim/distance_shift.py`,
   GPU-complete): optimize only `active_sites` coords, scatter back. Default
   `method='CG'`: torch = a hand-rolled nonlinear CG (Polak-Ribiere+, backtracking
   Armijo); jax = a pure-jax port of it (`lax.while_loop`, JIT-able inside `lax.scan`).
   `method='l-bfgs'` is opt-in (torch `LBFGS` strong-Wolfe / `jaxopt.LBFGS`, lazily
   imported). Distance restraints skip the solver — a centroid-distance is 1-DOF and applied
   closed-form in `distance_shift.py`. No `pure_callback`, no scipy. On CUDA the torch CG
   runs through `optim/_torch_cg_gpu.py` — the same early-exit CG but with a `torch.compile`
   (inductor-fused, NOT cudagraph) energy+grad, so conformer/RMSD optimization is GPU-faster
   than eager. RMSD needs the hand-rolled CG on BOTH backends (jaxopt NonlinearCG stalls on
   RMSD's fixed-rotation `stop_gradient` gradient). There is **no numpy
   optimizer backend** (the old scipy path was removed); `numpy_energy` remains only as
   the pure-numpy energy reference for `tests/test_backend_parity.py`. Optimization
   requires torch or jax.

Supporting modules:
- **`featurizer.py`**: `build_spec(ligand_confs, distance_restraints, conformer_config,
  elements, conf_start_sigma, rmsd_restraints)` — the single place RDKit mols become bond/angle/
  chiral/cistrans restraints (global indices, multi-ligand) and the dynamic
  ligand-protein `VdwConfig` is assembled. Cis/trans detection keys on
  acyclic, non-aromatic `BondType.DOUBLE` bonds and targets the reference-conformer
  torsion; it needs real bond orders, which every tool supplies — chai via its adapter's
  source-SMILES path (`chai/adapter.py` `_mol_from_smiles`, Kekulized orders), the
  geometry-perceived fallback (no SMILES) being all-single so `cistrans=0`.
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
- **Framework adapters** (`{boltz,protenix,chai,openfold3,esmfold2,alphafold3}/adapter.py` —
  framework-free EXCEPT boltz, whose feats arrive as native torch tensors so its adapter
  imports torch (read at batch 0); the others import no framework. AF3's CCD/SMILES mol
  resolution lives in a thin in-tool shim
  (`alphafold3_restr` `build_af3_adapter`) that feeds `rgi_utils/alphafold3/adapter.py` plain
  data): implement `iter_atoms()` (→
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
  `"(chain A or chain B) and resid 1 to 10"` (an MDTraj-like keyword/range/boolean
  vocabulary); tokens are
  `chain`/`resid`/`index`/`name`/`protein`/`dna`/`rna`/`backbone`/`sidechain` +
  `and`/`or`/`not`/`()`. Used by `DistanceData` (`distance_restr_data.py`) for centroid
  groups and `RmsdData` for fit/calc; `name CA` (atom-name, case-insensitive,
  alnum-only — a nucleic-acid `C1'` is not selectable but fails loudly) restricts an
  RMSD superposition to backbone. `backbone`/`sidechain` are MDTraj-like POLYMER
  selectors (name-based but gated on polymer type via `_moltype.polymer_type`, which
  prefers `AtomRecord.mol_type` and falls back to `resname` — so both flow through the
  candidate dict); a ligand atom named "C"/"N"/"O" never matches them.
- **RMSD restraint** (`rmsd_restr_data.py` + `pdb_ref.py`): `RmsdData` resolves a moving
  group against a reference PDB (parsed by the dependency-free `read_pdb_atoms`), driving the
  Kabsch-superposed RMSD toward `target_rmsd`. The superposition ("fit") and measured ("calc")
  atoms are selected INDEPENDENTLY by four keys — `atom_selection_target_fit`,
  `atom_selection_ref_fit`, `atom_selection_target_calc`, `atom_selection_ref_calc`
  (`atom_selection_target` / `atom_selection_ref` are a both-sides shorthand for the
  fit+calc pair). **There is NO bare `atom_selection` key** — passing one is silently
  ignored (a footgun), so use the suffixed keys. All four omitted ⇒ fit+measure over the
  WHOLE structure best-effort (atoms missing from the ref are skipped). `pairing`
  **defaults to `align`** (sequence-align polymer chains so a homolog ref maps on by
  residue; ligands + structures with no polymer fall back to ordinal identity, so the
  default is safe everywhere) — set `pairing: identity` to force pure (chain, resid, name)
  ordinal pairing. A `name CA` or `backbone` selection superposes on backbone only, which
  under align keeps a substituted homolog's side chain from being pinned.
  Each RMSD entry takes a `start_sigma` (active once `sigma <= start_sigma`) **and** a
  `stop_sigma` (released once `sigma < stop_sigma`; default **-1** = never released): the
  active window is `stop_sigma <= sigma <= start_sigma`. `stop_sigma > 0` releases the
  restraint for the final low-sigma steps so the model re-idealises geometry the restraint
  held distorted — the fix for a **broken peptide bond between a restrained residue and a
  FREE unmodeled tail** (`target_rmsd=0` drives the restrained residue onto the ref every
  step while the free tail lags and the bond snaps in length AND omega; releasing late lets
  the model repair it without losing the global ref bias set over the earlier steps). The
  CG fully converges each step, so a per-atom RMSD weight only changes convergence RATE not
  the fixed point — it can NOT keep the terminus off the ref; releasing (stop_sigma) is what
  works. **Validated on boltz2: `stop_sigma: 1.0` fully heals the bond (ref Cα-RMSD ~0.3 Å);
  the same knob is available on distance + conformer restraints** (see the start_sigma note
  below). All tools share sigma_data=16, so the value transfers.

### Conformer: VdW (two flavours)

VdW is **not a sixth restraint type** — it is the non-bonded term of the **conformer**
restraint, configured under `conformer_restraints_config.vdw` (one of bond/angle/chiral/
cistrans/vdw). It has two flavours:

- **Intramolecular** (`mode: intramolecular`): static non-bonded ligand-internal pairs
  (topo distance > 2, within `dmax`), built in `featurizer.py` and carried in `spec.vdw`
  (`VdwArrays`). Scored in the **energy layer → all backends**.
- **Dynamic ligand-protein** (`mode: ligand_protein`): lives in the optimizers
  (`optim/torch_optim.py` AND `optim/jax_optim.py` — `_vdw_pair_energy` is the shared
  formula, ported to jnp). The ligand moves (it is in `active_sites`); the protein is a
  **fixed background** read from the full coordinate tensor (`VdwConfig.protein_global`)
  at minimize time, so only the ligand is pushed out of contacts. Penalty
  `weight * clamp(d - scale*(r_i+r_j), max=0)**2`, all-pairs (zero gradient beyond
  contact) — same maths as boltz's radius search. Works on **torch + jax** (numpy is the
  energy reference only, so it does not run this optimizer term).

`mode` defaults to **`both`** (`ligand_protein` / `intramolecular` pick one). "both"
builds `spec.vdw` AND `spec.vdw_config` from the one `vdw` block (shared weight/scale/
dmax); they sit in separate spec fields scored independently, so they compose. An
unknown mode raises. Both halves run on torch and jax (so AF3 gets the full VdW — it no
longer force-downgrades to intramolecular). VdW `weight` defaults to **0** (off) — set
it > 0 to activate. The static `vdw_energy` (idx pairs) lives in the energy layer for
parity across backends.

### Key design points

- `CombinedRestraints` is **instance-scoped — a fresh `CombinedRestraints()` per
  structure** (this is what makes batch runs / retries correct without leaking the
  previous structure's config). `get_instance()`/`reset()` are back-compat shims only.
- `AtomRecord.index` is the atom's **row in the coordinate tensor handed to
  `minimize`** (global flat index, after any reshape); `resid` is the **per-chain
  1-based residue/token ordinal** (resets at each chain) — not the author residue
  number, not a cumulative token index. These two conventions must match across tools.
- `minimize` is gated on `sigma <= start_sigma`. `start_sigma` is set **per distance/RMSD
  entry** (each `distance_restraints_config` / `rmsd_restraints_config` entry) and **once
  for all conformer terms** (`conformer_restraints_config.start_sigma` ->
  `RestraintSpec.conf_start_sigma`). There is **no top-level/global `start_sigma`** (a
  top-level one raises `ValueError`); per restraint it is **optional** and defaults to
  `+inf` when omitted — i.e. active at **every** step (set it, e.g. `1.0`, to act only
  late). When `sigma` exceeds every restraint's `start_sigma` the whole step is skipped.
  **Every** restraint type additionally takes a `stop_sigma` LOWER bound (default **-1** =
  never released, any value `<= 0` is off): the restraint is released for `sigma <
  stop_sigma`, so its active window is `stop_sigma <= sigma <= start_sigma`. It is
  per-distance / per-rmsd entry, and **once for all conformer terms**
  (`conformer_restraints_config.stop_sigma` -> `RestraintSpec.conf_stop_sigma`). The gate
  lives in the shared `_terms.sigma_gate` (per-restraint distance/rmsd) and the conformer
  `cg` in each backend's `_gates` (eager), plus `torch_optim._gated_prepared` /
  `distance_shift` (compiled GPU / closed-form). `stop_sigma > start_sigma` (empty window)
  is flagged by `_warn_never_active`.
- Distance restraints: `harmonic`, `flat-bottomed`, `flat-bottomed1`,
  `flat-bottomed2`; only `calc_method=unfixed-absolute` (centroid-based). The per-entry `move`
  key picks which group the closed-form centroid shift moves: `both` (default = minimal-
  displacement split, both move) / `1` (only `atom_selection1`'s group) / `2` (only
  `atom_selection2`'s) — `1`/`2` PIN the other group (e.g. move only a ligand toward a
  fixed pocket). It is a per-restraint `move_mode` int (0/1/2) parallel to `dist_type`,
  wired ONLY in `optim/distance_shift.py` (`_split`); the energy layer ignores it
  (centroid-distance is move-agnostic). All modes change the centroid separation by the same `delta`,
  so for a single restraint (or disjoint groups) convergence is identical — only the
  distribution differs (coupled restraints moving a SHARED atom can reach a different fixed
  point under `1`/`2` vs `both`).
- Group angle/dihedral restraints (`angle_restraints_config` 3 groups / vertex=group2;
  `dihedral_restraints_config` 4 groups / axis=group2-3): restrain the angle/dihedral of
  the groups' centroids. The config surface MIRRORS the distance restraint — the four types
  `harmonic{target_angle}` / `flat-bottomed{target_angle1,target_angle2}` / `flat-bottomed1`
  / `flat-bottomed2` (dihedral uses `target_dihedral*`), plus the `move` key. Targets are in
  **DEGREES** (→ radians in `group_geom_restr_data.py`); the spec carries
  `target1/target2/geom_type` (reusing the distance `DIST_TYPE_CODES`) + `move_free` (a
  per-group `(n, n_groups)` {0,1} mask). `weight` defaults 1.0; per-restraint
  `start_sigma`/`stop_sigma` like distance/rmsd. Unlike distance these are **CG-solved
  energy terms** (not closed-form): a centroid angle/dihedral is not 1-DOF. The energy depends
  only on the centroids, so every atom in a free group gets the same gradient → the CG translates
  it rigidly (verified in `test_backend_parity`). The dihedral `harmonic` wraps the deviation
  to +-180 (periodicity-safe); flat-bottomed enforces `target1<target2` so a window can't
  straddle +-180. **`move`** selects which groups are free (the rest pinned); it can free
  SEVERAL at once (`move: [1,4]` / `"1,4"`). The DEFAULT (omitted) moves the arms and pins
  the anchor — angle frees groups 1+3 (vertex 2 pinned), dihedral frees 1+4 (axis 2+3
  pinned); `move: all` frees every group. It is implemented IN THE ENERGY: pinned groups'
  centroids are `stop_gradient`/`.detach()`'d (the rmsd `_kabsch_R` pattern), so the value is
  unchanged (all-backend parity) but the CG doesn't move them — so `move_free` flows through
  `_TERMS` to the leaf fn (numpy ignores it, value-only). Their `_TERMS` gate is `"group"` — any gate other than `conf`/`dist` means
  per-restraint sigma gate + ALWAYS in the solver (like `rmsd`); such keys are collected in
  `_terms.PER_ENTRY_KEYS`, which the torch GPU pre-gate (`torch_optim._gated_prepared`) folds
  + keys its compile cache on (so a new per-entry term can't silently go ungated on the
  compiled GPU path — a bug CPU CI can't catch). Solver-run condition in both optimizers ORs
  in `has_group_angle()/has_group_dihedral()`. `move!=both` stop-gradients pinned groups, so
  its grad parity is torch-vs-jax (not numpy-FD) — the rmsd carve-out (`test_optim`).
  Caveat: a degenerate geometry — coincident centroids, or centroid1-centroid2-centroid3 collinear for the
  dihedral — gives a near-zero / ill-defined gradient (same failure mode as the conformer
  cistrans; the clip/atan2 guards keep it finite but it won't move), so pick groups whose
  centroids are non-collinear. **Rigid group motion / weight independence** (`_move_centroid`
  `centroid_eff`): the centroid gradient is naturally `1/N` per atom (dcentroid/datom = 1/N), so a large
  group would barely move per CG step (needing weight ~ N). `_move_centroid` cancels the `1/N`
  with `centroid_eff = centroid_d + N*(centroid - centroid_d)` (value == centroid, gradient N×), so the whole group
  translates RIGIDLY by the full step and **`weight: 1` (the default) drives ANY group
  size** — the analogue of the distance restraint's rigid closed-form shift, as the user
  requested. Cost: the group gradient is intentionally N×-rescaled, so it does NOT match a
  numpy finite-difference of the true energy — group grad parity is therefore torch-vs-jax
  (not numpy-FD), the same carve-out as rmsd's stop-gradient. Verified E2E on boltz: the
  qbp 3-region angle (624/690/314 atoms) reaches 90.0° and the 4-region dihedral ±180° at
  the default `weight: 1`.
- Top-level `import rgi_utils` must work with numpy only (no torch/jax) — keep
  heavy imports lazy inside the backend modules.
- GPU tests are marked `@pytest.mark.gpu` and excluded in CI.
