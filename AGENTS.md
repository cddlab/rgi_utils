# AGENTS.md

This file provides guidance to coding agents, including Codex and Claude Code, when
working with code in this repository. `CLAUDE.md` is a symlink to this file.

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
and alphafold3 (jax). The end-to-end guide for integrating a new tool is the
`implement-rgi` skill, shared from `.claude/skills/` to `.agents/skills/`. Per-tool
as-built integration write-ups live in
`doc/<tool>.md` (one per tool); the shared config + selection-DSL surface is `doc/config.md`.

Design = **3 layers + autodiff + static shapes + GPU-complete optimization**:

1. **Spec layer** (`spec.py`, backend-agnostic): `RestraintSpec` holds every
   restraint as padded NumPy arrays. All indices are *local* indices into
   `active_sites` (the subset of atoms that participate in any restraint);
   `active_sites` itself stores *global* flat atom indices. Multiple ligands are
   collision-free because each supplies disjoint global indices.

2. **Energy layer** (`energy/{numpy,torch,jax}_energy.py`, differentiable pure
   functions): identical flat-bottomed maths in all three backends —
   `bond/angle/chiral/plane/cistrans/vdw/distance/rmsd/group_angle/group_dihedral/group_plane`
   (cistrans = periodicity-safe torsion for cis/trans; plane = [servalcat](https://github.com/keitaroyam/servalcat)-style best-fit
   plane over whole planar atom GROUPS (aromatic/conjugated rings + non-ring sp2 groups),
   penalising each group's out-of-plane RMS deviation via the smallest-eigenvalue plane
   normal (stop-gradient like `rmsd`'s Kabsch rotation), opt-in/off by
   default; rmsd = Kabsch-superposed RMSD toward a target,
   fit/calc separable; `group_angle`/`group_dihedral` = the angle/dihedral of 3/4 atom
   GROUPS' centroids — the angular analogue of the centroid-distance restraint, distinct from the
   per-atom `angle`/`cistrans` conformer terms; `group_plane` = the SAME best-fit-plane quantity as
   `plane` but over selection-resolved groups (`plane_restraints_config`), with the four
   distance-style types and a PER-ENTRY gate instead of the shared conformer one — the two share
   one `_plane_rms` helper per backend so the eigh maths is never duplicated).
   `prepare_spec(spec)` → backend arrays;
   `total_energy(positions, prepared)` → scalar. Gradients come from autodiff
   (no hand-written grad). `numpy_energy` is the reference;
   `tests/test_backend_parity.py` checks energy+grad agreement across backends.

3. **Optim layer** (`optim/{torch,jax}_optim.py`,
   GPU-complete): optimize only `active_sites` coords, scatter back. Default
   `method='CG'`: torch = a hand-rolled nonlinear CG (Polak-Ribiere+, backtracking
   Armijo); jax = a pure-jax port of it (`lax.while_loop`, JIT-able inside `lax.scan`).
   `method='l-bfgs'` is opt-in (torch `LBFGS` strong-Wolfe / `jaxopt.LBFGS`, lazily
   imported). Distance is CG-minimised like every other restraint (the old closed-form
   `distance_shift.py` was removed); to stop the `1/N` centroid-gradient dilution from
   freezing large groups, its centroid uses the same `_move_centroid` N×-rescale as the
   group terms, with a reduced-mass scale `N1·N2/(N1+N2)` reproducing the old
   minimal-displacement split. No `pure_callback`, no scipy. On CUDA the torch CG
   runs through `optim/_torch_cg_gpu.py` — the same early-exit CG but with a `torch.compile`
   (inductor-fused, NOT cudagraph) energy+grad, so conformer/RMSD optimization is GPU-faster
   than eager. RMSD needs the hand-rolled CG on BOTH backends (jaxopt NonlinearCG stalls on
   RMSD's fixed-rotation `stop_gradient` gradient). There is **no numpy
   optimizer backend** (the old scipy path was removed); `numpy_energy` remains only as
   the pure-numpy energy reference for `tests/test_backend_parity.py`. Optimization
   requires torch or jax.

### Supporting modules

#### `featurizer.py`

`build_spec(ligand_confs, distance_restraints, conformer_config,
elements, conf_start_sigma, rmsd_restraints)` — the single place RDKit mols become bond/angle/
chiral/cistrans restraints (global indices, multi-ligand) and the dynamic
fixed-background `VdwConfig` is assembled. Cis/trans detection keys on
acyclic, non-aromatic `BondType.DOUBLE` bonds and targets the reference-conformer
torsion; it needs real bond orders, which every tool supplies — chai via its adapter's
source-SMILES path (`chai/adapter.py` `_mol_from_smiles`, Kekulized orders), the
geometry-perceived fallback (no SMILES) being all-single so `cistrans=0`.

#### `monlib_geom.py`

Polymer bond/angle/plane/link TARGETS from a **CCP4 monomer library** (gemmi, lazy-imported)
instead of the predictor's reference conformer — opt in with
`conformer_restraints_config.monomer_library` (`"<path>"` or `{path, on_missing}`). Motivation:
the reference conformer is NOT refinement geometry — AF3 fills `ref_pos` by ETKDG-embedding the
free CCD component, giving an unconjugated exocyclic C-N (1.42 Å vs 1.33) and a P-OH phosphate
(P-OP2 1.69 Å vs 1.52), and the embed's random seed moves those targets ±0.02-0.03 Å per run, so
`bond`/`angle` measurably degrade nucleotide geometry. `polymer.py` loads the library and collects targets;
`featurizer.py` DROPS every conformer-derived tuple whose atoms all lie in a covered residue
(`PolymerGeometry.library_atoms`), so the two sources replace rather than stack, per residue.
Library planes are named groups — a whole nucleobase (ring + exocyclic + `C1'`) in ONE group where
SSSR perception splits a purine in two. Links come from the `TRANS` / `p` entries (the peptide
PLANE stays built-in: its 5-atom omega group beats the library's 4-atom one). `chiral` stays
conformer-derived — only the sign matters, and the library's `ChiralityType` convention would have
to be reconciled with `_chiral_vol`'s atom ordering first. Uncovered residues fall back
(`on_missing: error` to refuse instead); a bad path raises. Tests: `tests/test_monlib_geom.py`
(self-contained fixture library in tmp_path — no CCP4 install needed).

#### `config.py`

`RestraintsConfig.from_dict()` parses the shared
`restraints_config` (one source of truth for boltz YAML / protenix JSON / AF3).

#### `combined.py`

`CombinedRestraints` entry point — **instance-scoped, one per
structure** (NOT a singleton): `CombinedRestraints()` →
`setup(adapter, nbatch, config=dict)` (folds in the old `set_config`; clears any
prior spec/optimizer up front so a reused instance is safe) →
`minimize(coords, step, sigma)` → `finalize(coords, step)`. Always construct a fresh
instance per structure (the old `get_instance()`/`reset()` singleton shim was removed). The
backend is **inferred from invocation, NOT a config key**: `get_minimizer()` →
jax (only AF3 calls it); `minimize(coords)` → jax if `coords` is a jax array, else
torch (numpy/torch coords both take the torch path). Resolution is lazy — `setup`
leaves `self._backend = None` and builds the optimizer on the first
`minimize`/`get_minimizer` (`_ensure_backend` raises if one instance is invoked under
two backends). `gpu` selects the torch *device* (CPU when `gpu:false`, the accelerator
when `gpu:true`), NOT the backend, so `gpu:false` runs the torch optimizer on CPU
(moving GPU coords to CPU and back). There is no numpy optimizer (numpy is the energy
reference only); a leftover `backend:` config key raises with a migration hint.
torch/jax imported lazily. JAX tools inside `lax.scan` grab the pure minimizer via
`get_minimizer()` instead of calling `minimize` per step (so for AF3 the `gpu` flag is
inert — to run AF3 restraints on CPU, run the whole process on the JAX CPU platform).

#### Framework adapters

(`{boltz,protenix,chai,openfold3,esmfold2,alphafold3}/adapter.py` —
framework-free EXCEPT boltz, whose feats arrive as native torch tensors so its adapter
imports torch (read at batch 0); the others import no framework. AF3's CCD/SMILES mol
resolution lives in a thin in-tool shim
(`alphafold3_restr` `build_af3_adapter`) that feeds `rgi_utils/alphafold3/adapter.py` plain
data): implement `iter_atoms()` (→
`AtomRecord(chain, resid, index)` for distance selection) and optionally
`num_atoms()`, `get_elements()`, `iter_ligand_confs()` (→
`LigandConf(mol, conf_coords, global_indices)` for conformer + VdW).

**AF3 residue names carry a gap-token hazard.** AF3 encodes `aatype` with the vocabulary that
has a GAP entry right after `UNK` (`… 20:UNK, 21:'-', 22:A, 23:G, 24:C, 25:U, 26:DA …`), while
the shim historically passed the gap-less `POLYMER_TYPES` (`… 20:UNK, 21:A, 22:G …`). Indexing
one with the other leaves PROTEINS right (they sit below the gap) and shifts every NUCLEIC name
by one — an adenine token reads as `G`, a uridine as `DA`. That silently mis-identifies bases for
the base-pair macro and for monomer-library lookups. `AF3RestraintAdapter._resolve_name_shift`
settles it from evidence instead of trusting the vocabulary: it scores both readings by how many
nucleic tokens land on a name their own `is_rna`/`is_dna` flag allows, and warns when it has to
shift. Protein/ligand-only batches keep shift 0 (nothing below the gap moves).

#### `_mol_build.py`

`build_ligand_mol(elements, coords, bonds_local,
perceive_bonds=False)` — the shared RDKit builder. protenix/openfold pass real
`bonds_local`. chai exposes NO intra-ligand bonds, so its adapter passes
`perceive_bonds=True`: connectivity is derived from the reference conformer via
`DetermineConnectivity`, which leaves atoms `noImplicit=True` — so the branch then
runs `SetNoImplicit(False)` + `UpdatePropertyCache` **before**
`AssignStereochemistryFrom3D`, else stereocentres read as 3-coordinate and every
chiral restraint silently vanishes.

#### Standalone plane restraints (`plane_restr_data.py`)

`plane_restraints_config` is the selection-driven form of the conformer `plane` term: the user names
atoms with the DSL instead of relying on RDKit perception / the monomer library, so ANY group
(nucleobase, peptide plane, aromatic side chain) can be held flat. Same measured quantity
(out-of-plane RMS from the group's own best-fit plane) — so it inherits the
[servalcat](https://github.com/keitaroyam/servalcat)/Refmac plane-restraint formulation the conformer
term was modelled on (whole planar GROUP, not per-centre impropers; citations in `README.md`
under *References*) — but the four distance-style types
(`target_plane`, in ANGSTROM — no `unit` key) and a PER-ENTRY gate. Non-obvious points:

- **The type block is OPTIONAL** (angle/dihedral raise without one): a plane's target is always 0, so
  an omitted block means `harmonic{target_plane: 0}`. This is also what lets the base-pair macro
  express `coplanar_slack: 0` — `harmonic: {}` alone would raise in `parse_geom_type`.
- **1..4 groups per entry are POOLED into ONE plane** (that is the "keep these coplanar" idiom).
  Because the group count varies per entry, the spec's `move` mask is per-ATOM (`GroupPlaneArrays.free`)
  rather than the per-group `move_free` that `group_angle`/`group_dihedral` use. Numbering must be
  contiguous from 1 — a gap raises (it would shift what `move` indices name).
- **`move` defaults to every group free** (a plane has no anchor group to pin, unlike the angle vertex
  / dihedral axis). Pinned atoms still shape the fit but get no gradient (value-preserving, so numpy
  parity holds).
- **No N-rescale.** The centroid terms cancel their `1/N` gradient dilution via `_move_centroid`;
  plane deliberately does NOT, because the plane RMS is a genuine least-squares fit rather than a
  rigid-body translation (and matching the pre-migration base-pair convergence requires it). Cost:
  a very large group is weak *relative to other restraints* — raise its `weight`.
- **A `refN and <selection>` entry is a DIFFERENT energy** and is routed to `RefGeomData("plane")`
  (a `ref_geom` closure, `_GEOM_SPEC["plane"] = (None, "target_plane")` = caller-supplied group
  count) instead of the array path: the plane is fitted to the REFERENCE atoms alone and held fixed,
  and the value is the RMS distance of the prediction atoms from it — the prediction is pulled ONTO
  the reference's plane. `config.py` counts the entry's groups (`count_plane_groups`) before
  constructing `RefGeomData` so `n_groups` stays a plain int in all five places it is read.

#### Atom selection DSL

(`selection.py`): `AtomSelector` parses
`"(chain A or chain B) and resid 1 to 10"` (an MDTraj-like keyword/range/boolean
vocabulary); tokens are
`chain`/`resid`/`index`/`name`/`protein`/`dna`/`rna`/`backbone`/`sidechain` +
`and`/`or`/`not`/`()`. Used by `DistanceData` (`distance_restr_data.py`) for centroid
groups and `RmsdData` for fit/calc; `name CA` (atom-name, case-insensitive) restricts an
RMSD superposition to backbone, and `name C1'` picks a single nucleic-acid atom for e.g.
an H-bond distance restraint. `name` has its OWN token parser (`_parse_atom_name1`), not
the alphanumeric identifier one, because atom names legally carry a prime; `'` (PDB v3 /
mmCIF), `*` (PDB v2) and `"` (double prime) are folded onto one form by
`normalise_atom_name` on BOTH sides, so prime spelling never decides a match. The
operator guards are duplicated there — `and`/`or`/`not` are ordinary alphanumeric words,
so without rejecting them the name list would swallow the operator.
`backbone`/`sidechain` are MDTraj-like POLYMER
selectors (name-based but gated on polymer type via `_moltype.polymer_type`, which
prefers `AtomRecord.mol_type` and falls back to `resname` — so both flow through the
candidate dict); a ligand atom named "C"/"N"/"O" never matches them.

#### RMSD restraint

(`rmsd_restr_data.py` + `pdb_ref.py`): `RmsdData` resolves a moving
group against a reference structure — `ref_pdb` (PDB) or `ref_cif` (mmCIF), **mutually
exclusive**, both **coordinate-parsed via gemmi** (lazy-imported, so `import rgi_utils` stays numpy-only)
by `read_pdb_atoms` / `read_cif_atoms` into the same `PdbAtom` list (a shared `_build_atoms`
applies the per-chain ordinal once, so the two are interchangeable; PDB goes through
`gemmi.read_structure`, mmCIF reads the `_atom_site` loop via `gemmi.cif` preferring the
`auth_*` columns — this keeps the label-only fallback `read_structure` drops) — driving the
Kabsch-superposed RMSD, shaped by a restraint-type block (`harmonic` / `flat-bottomed` /
`flat-bottomed1` / `flat-bottomed2`, the same four types as distance/angle/dihedral). The
superposition ("fit") and measured ("calc")
atoms are selected INDEPENDENTLY by four keys — `atom_selection_target_fit`,
`atom_selection_ref_fit`, `atom_selection_target_calc`, `atom_selection_ref_calc`
(`atom_selection_target` / `atom_selection_ref` are a both-sides shorthand for the
fit+calc pair). **There is NO bare `atom_selection` key** — passing one now RAISES
(it used to be silently dropped, a footgun), so use the suffixed keys. All four omitted ⇒ fit+measure over the
WHOLE structure best-effort (atoms missing from the ref are skipped). `pairing`
**defaults to `align`** (sequence-align polymer chains via **biopython** — `_align.py`
→ `Bio.Align.PairwiseAligner` + BLOSUM62, also lazy-imported — so a homolog ref maps on by
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

### Conformer: VdW (two categories — intramolecular / intermolecular)

VdW is **not a sixth restraint type** — it is the non-bonded term of the **conformer**
restraint, configured under `conformer_restraints_config.vdw` (one of bond/angle/chiral/
plane/cistrans/vdw). `mode` picks **two categories** (default `both` = both):

- **Intramolecular** (`mode: intramolecular`): clashes WITHIN one ligand. Static
  non-bonded ligand-internal pairs (topo distance > 2, within `dmax`), built in
  `featurizer.py` (`_build_intramolecular_vdw`) and carried in `spec.vdw` (`VdwArrays`).
  Scored in the **energy layer → all backends**.
- **Intermolecular** (`mode: intermolecular`): clashes between that ligand and **every
  other molecule** — protein, DNA/RNA, and any other ligand (restrained or not). One
  category, **two implementation halves** depending on whether the other molecule is fixed
  or moving:
  - *Fixed background* (`_build_vdw_config` → `VdwConfig` → optimizers): the partner is
    every non-padding atom NOT in `active_sites` (protein / DNA/RNA / **non-restrained** ligand),
    read from the full coordinate tensor at minimize time and held fixed (it needs no
    gradient). Lives in `optim/torch_optim.py` AND `optim/jax_optim.py` (`_vdw_pair_energy`
    is the shared formula, ported to jnp). **torch + jax** (numpy is the energy reference
    only, so it does not run this optimizer term).
  - *Other restrained ligands* (`_build_interligand_vdw` → `VdwArrays`): two ligands that
    each opted into conformer restraints both sit in `active_sites`, so neither is in the
    other's fixed background — this half repels A↔B. Both endpoints move, so autodiff
    drives BOTH apart. Carried in `spec.vdw` **concatenated with the intramolecular rows**
    (same `vdw_energy` term, same conformer gate → all backends + parity for free). Unlike
    intramolecular there is **no topological skip and no `dmax` cutoff** (two ligands'
    `conf_coords` live in independent frames, so a build-time distance is meaningless); all
    cross pairs are listed and the clamp contributes zero beyond contact. Only built when
    ≥2 ligands opted in.

Both halves share `weight * clamp(d - scale*(r_i+r_j), max=0)**2` (all-pairs, zero gradient
beyond contact — same maths as boltz's radius search). `mode` defaults to **`both`**
(intramolecular + intermolecular); the explicit values pick one category. **The old
`mode: ligand_protein` is REMOVED** — it was only the fixed-background half; it now raises a
migration hint pointing to `intermolecular` (which additionally repels other restrained
ligands), mirroring the rejected `backend:` key. An unknown mode raises. VdW is **off unless
a `vdw:` block is present** (then `weight` defaults to 1.0, like every conformer term — see
`featurizer._conf_weight`); omit the block to leave it off. The `built spec: ...
vdw=Iintra+Jinter+Llig/Mbg` log breaks the counts down: `intra` = intramolecular, `inter`
+ `lig/bg` together = intermolecular (`inter` = restrained-ligand pairs, `lig/bg` =
fixed background) — confirm `Jinter>0` when you expect ligand-ligand repulsion.

### Custom restraints (the extension point — `rgi_utils/custom/`)

Beyond the six built-ins, a user can define an **original** restraint as a backend-agnostic
energy `energy(ctx) -> scalar`. Two authoring paths, ONE mechanism:
- **config (expression DSL)**: a `custom_restraints_config` entry with an `energy` formula string
  over a shared vocabulary + named `selections` (e.g. `"(distance(A,B) - distance(C,D))**2"`).
  A selection value may be reference-backed as `refN and <selection>` with an entry-local
  `refs.refN` definition; the same geometry vocabulary consumes it (`distance`/`angle`/`dihedral`/
  `centroid`/`rg`/`norm`/`dot`/`coords`/`kabsch`/`rmsd`/`plane` + penalties + math incl.
  periodicity-safe `wrap`; full table in `doc/config.md`). External-reference RMSD is `rmsd(A,B)`
  (prediction A, reference-backed B); rigid superposition is `kabsch(A,B)`; best-fit-plane flatness is
  `plane(A)` (own plane) / `plane(A,B)` (A into B's plane, either argument reference-backable).
  (There is no `ref(sel,r)` function — reference-backing is the `refN and <selection>` string form.)
  `move` is a prediction selection name or list of names; unlisted prediction selections are
  stop-gradient pinned for that custom term, and reference-backed selections are always fixed.
- **code (ctx fn)**: a Python `energy(ctx)` — passed directly (`CombinedRestraints.add_custom(fn=…)`
  / config `{"fn": …}`) or registered (`@custom_restraint("name")`, config `{"use": "name"}`).

Both compile to a **closure** `(active_coords) -> scalar` that the optimizers ADD to the CG
objective (`torch_optim` `energy_fn` / `jax_optim` `_descend.energy_fn`, like the dynamic VdW) with a
per-entry sigma gate — they are NOT array-dispatch terms (`_terms`/`total_energy` are untouched), so
arbitrary formulas with variable selections fit. Package layout: `custom/backends.py` (the `ops`
facade — geometry written ONCE per the `axis=`/`dim=` split, so parity is structural),
`vocabulary.py` (geometry+penalty), `context.py` (`RestraintContext` + the setup-time
`ResolveContext` that records selections via shaped-numpy dummies), `dsl.py` (safe `ast` parse — node
whitelist, no `eval`), `registry.py` (`@custom_restraint`), `data.py` (`CustomData`/`CustomSpec`),
`closure.py` (`build_terms`). Storage: `RestraintSpec.custom: list[CustomSpec]` (Python AST/fn + local
index arrays — NOT numpy term arrays, hence a separate field); `has_custom()` joins `is_active()` /
the solver-run condition / `max_start_sigma()`.

Non-obvious invariants: custom energies use **plain centroids** (no `_move_centroid` rigid-translation
trick), so autodiff grad == numpy-FD while every prediction selection is free. `move` pins unlisted
selection blocks with stop-gradient, so pinned cases use torch-vs-jax grad parity instead. The same
exception applies to the three primitives that stop-gradient part of their maths — `kabsch`/`rmsd`
(rotation) and `plane` (normal, via `ops.svd`: the covariance is symmetric PSD so `Vt`'s last row is
the smallest eigenvector, which is why no `eigh` was added to the ops facade): a formula using them is
compared torch-vs-jax, not against a numpy FD. In `plane(A,B)` only the normal is fixed — the plane's
CENTRE still carries gradient, so a free `B` is pulled toward `A` unless `move` pins it.
The closure must reduce to a **scalar** (`ops.sum` over batch dims) or `jax.value_and_grad` rejects it.
Selections resolve at setup via a **resolve pass** (run the energy with `ResolveContext`). On CUDA the
torch CG runs **eager** when any custom is present (the fused `gpu_cg` `_energy` bypasses `energy_fn`,
so it can't see closures) — correct, just unfused; the built-ins keep the fused path. `import
rgi_utils` stays numpy-only (torch/jax pulled lazily per backend by `get_ops`). Harness:
`tests/test_custom.py` + `tests/test_custom_move.py` (both paths × 3-backend energy/grad
parity + move pinning + jax-scan + torch minimize + DSL safety). Full config surface: `doc/config.md`.

### Base-pair restraints (nucleic-acid Watson-Crick — a config-time macro)

`base_pair_restraints_config` (`base_pair_restr_data.py`, `BasePairData`) is **NOT a new
energy term** — it is a **config-time macro** (mirroring servalcat/Refmac) that EXPANDS each
named nucleotide pair into the existing primitives: one **distance** restraint per WC hydrogen
bond (donor/acceptor atoms + ideal length looked up from a built-in `_WC_ATOMS` table:
G·C / A·T / A·U, plus a G·U wobble that is **opt-in via `pair: GU`**, never auto-detected), plus
(optionally, `coplanar: true` default) one best-fit **plane** restraint over both bases so the
pair stays coplanar. So it reuses the distance + plane energy terms → all backends + parity for
free. Base identity auto-detects from each residue's `resname` (DNA `D`-prefix stripped); pairs
are **user-specified only** (coordinates are noise at high sigma). Expansion happens in
`combined.py` (`bp.resolve_sites(adapter)` → a pre-resolved `DistanceData` list merged into the
distance list + a pre-resolved `PlaneRestraintData` merged into the plane list; both are LOCAL
merges, never in-place on the config, or a config-less re-`setup()` would duplicate them).
Fields: `residue1`/`residue2` (each must resolve to
EXACTLY one residue, else raises), `pair`, `coplanar`, `target` (scalar→harmonic, `[low,high]`→
flat-bottomed, default `(2.7, 3.1)` Å), `weight`, `move` (0/1/2 — `1` docks residue1 onto a fixed
residue2), and the gate window `start_sigma`/`stop_sigma` XOR `start_step`/`stop_step` (applies to
BOTH the H-bond distances AND the coplanarity plane, since the plane is now a standalone
`plane_restraints_config` restraint with its own per-entry gate — it used to ride the shared
conformer gate, so `stop_sigma` released the H-bonds but not the coplanarity; `move` likewise now
pins the other base in the plane fit). `coplanar_slack` maps onto the shared four types:
`0` → `harmonic{target_plane: 0}`, `>0` → `flat-bottomed2{target_plane2: slack}` — numerically
identical to the old one-sided `max(0, rms - slack)`. Verbose setup logs a
SEPARATE line `base_pair=P pairs -> H h-bonds + C coplanar groups` (the generated restraints also
show up in the `distances=` / `n_group_plane=` counts). Full field surface: `doc/config.md`.

### Key design points

- `CombinedRestraints` is **instance-scoped — a fresh `CombinedRestraints()` per
  structure** (this is what makes batch runs / retries correct without leaking the
  previous structure's config). There is no singleton accessor (the old
  `get_instance()`/`reset()` shim was removed).
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
  `cg` in each backend's `_gates` (eager), plus `torch_optim._gated_prepared`
  (compiled GPU). `stop_sigma > start_sigma` (empty window)
  is flagged by `_warn_never_active`.
- **Step window (alternative to sigma):** every windowed entry (distance/rmsd/angle/dihedral/
  custom + the conformer block) also accepts `start_step`/`stop_step` — the same window on the
  **diffusion step index** (`minimize(coords, istep, sigma)`'s `istep`, threaded to the optimizer
  and ANDed into every gate alongside sigma; defaults `-inf`/`+inf` = always-on). A restraint uses
  EITHER the sigma window OR the step window — **mutually exclusive**, enforced at config by the
  shared `_config_util.check_window_exclusive` (raises if both). `stop_step < start_step` (empty)
  raises in `_warn_never_active`. **Step counts differ per tool**, so a step window is NOT portable
  across tools the way a sigma window is (all tools share `sigma_data=16`) — prefer sigma. Plumbing
  mirrors `stop_sigma` exactly: spec arrays `start_step`/`stop_step`, `pack_spec` conf scalars,
  `_terms.sigma_gate` (now `(start_sigma, stop_sigma, start_step, stop_step, mask)`), the VdW gate
  in both optimizers (hand-maintained — not a `_TERMS` entry), and `torch_optim._gated_prepared`'s
  cache key. AF3 threads `istep` via a `jnp.arange(steps)` added to the diffusion `hk.scan` xs.
- Distance restraints: `harmonic`, `flat-bottomed`, `flat-bottomed1`,
  `flat-bottomed2`; only `calc_method=unfixed-absolute` (centroid-based). A distance entry may
  replace at most one group with `ref1 and <selection>` and define it under `refs.ref1`; the
  normal `atom_selectionN` keys are retained. Ref groups are permanently fixed; omitted/`all`/`both`
  moves every prediction group, and explicit `move` indices may select prediction groups only.
  **CG-minimised** like
  the group terms (no longer closed-form): `distance_energy` builds each group's centroid via
  `_move_centroid` with a **reduced-mass scale `N1·N2/(N1+N2)`**, so the per-atom gradient is
  `O(1)` (no `1/N` dilution → rigid translation) AND the two groups' gradient magnitudes are in
  ratio `N2:N1` — exactly the old **minimal-displacement** split. The per-entry `move` key picks
  which group moves: `both` (default = minimal-displacement, both move) / `1` (only
  `atom_selection1`'s group) / `2` (only `atom_selection2`'s) — `1`/`2` PIN the other group via
  `_move_centroid(free=0)` (`stop_gradient`, the same mechanism as the group-restraint `move`),
  e.g. move only a ligand toward a fixed pocket. It shares the angle/dihedral `move` VOCABULARY
  (parsed once by `_config_util.parse_move_indices`), so `both`/`all`/`[1,2]`/`"1,2"` all mean
  both, and `[1]`/`[2]` alias `1`/`2`; distance has only 2 groups, so `[1,3]` (index out of
  range) raises. It is a per-restraint `move_mode` int (0/1/2) threaded into the distance leaf
  via `_TERMS`. For a single / disjoint restraint every mode
  reaches the exact target; minimal-displacement is **exactly reproduced only for single /
  disjoint** restraints (a coupled/shared atom settles at the CG joint minimum, not the closed-form
  split) **and only for the small per-step moves of real diffusion**: a single LARGE one-shot move
  can let the moving group cross past the other to the reflected, equal-energy solution (e.g. `move:2`
  landing the group at `-target` instead of `+target`) — the centroid **gap always reaches target**,
  but the split *direction* is not guaranteed for big moves. Harmless in the multi-step diffusion loop
  (each step's move is small, staying in the minimal-displacement basin); the small-move split is
  verified in `test_optim`. Each entry also takes a per-entry `weight` (default 1.0): now the usual **least-squares
  weight** (CG jointly minimises `Σ wᵢ·δᵢ²`), replacing the old closed-form Jacobi weighted-average.
  So `weight` is a **NO-OP for a single / disjoint restraint** (target reached regardless), and only
  re-balances an atom shared by **over-constrained coupled** restraints, where it settles
  `B = (t1*w1 + t2*w2)/(w1+w2)` — same as angle/dihedral `weight` (a no-op at full CG convergence
  for a lone restraint). Distance is a per-entry `_TERMS` gate (like rmsd/group), so it is in the
  CG objective and the GPU pre-gate; its finalize energy comes from the same `energy_breakdown`.
- Group angle/dihedral restraints (`angle_restraints_config` 3 groups / vertex=group2;
  `dihedral_restraints_config` 4 groups / axis=group2-3): restrain the angle/dihedral of
  the groups' centroids. Reference groups use the same `refN and <selection>` values: angle
  entries allow up to two distinct refs and dihedral entries up to three, each fitted
  independently. On these ref-geometry closures, omitted/`all`/`both` moves all prediction groups;
  explicit `move` indices pin unlisted prediction groups and cannot select a reference group.
  The config surface MIRRORS the distance restraint — the four types
  `harmonic{target_angle}` / `flat-bottomed{target_angle1,target_angle2}` / `flat-bottomed1`
  / `flat-bottomed2` (dihedral uses `target_dihedral*`), plus the `move` key. Targets are in
  **DEGREES** by default — set `unit: radians` on the entry to give them in radians instead
  (single conversion point in `group_geom_restr_data._parse_geom_type`); stored internally as
  radians either way. The spec carries
  `target1/target2/geom_type` (reusing the distance `DIST_TYPE_CODES`) + `move_free` (a
  per-group `(n, n_groups)` {0,1} mask). `weight` defaults 1.0; per-restraint
  `start_sigma`/`stop_sigma` like distance/rmsd. Like distance these are **CG-solved
  energy terms** using the shared `_move_centroid` rescale. The energy depends
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
  size** — the same rigid-translation mechanism the distance restraint now uses (with a
  reduced-mass scale). Cost: the group gradient is intentionally N×-rescaled, so it does NOT match a
  numpy finite-difference of the true energy — group grad parity is therefore torch-vs-jax
  (not numpy-FD), the same carve-out as rmsd's stop-gradient. Verified E2E on boltz: the
  qbp 3-region angle (624/690/314 atoms) reaches 90.0° and the 4-region dihedral ±180° at
  the default `weight: 1`.
- Top-level `import rgi_utils` must work with numpy only (no torch/jax) — keep
  heavy imports lazy inside the backend modules.
- GPU tests are marked `@pytest.mark.gpu` and excluded in CI.
