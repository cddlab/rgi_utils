# rgi-utils

Restraint-Guided Inference (RGI) utilities for diffusion-based structure predictors
(PyTorch and JAX).

**Implemented and available in the following 8 models** (across 6 predictor integrations):

| Model | Integration | Backend | Details |
|-------|-------------|---------|---------|
| **Boltz-1** | boltz | torch | [`doc/boltz_restr.md`](doc/boltz_restr.md) |
| **Boltz-2** | boltz | torch | [`doc/boltz_restr.md`](doc/boltz_restr.md) |
| **AlphaFold3** | alphafold3 | jax | [`doc/alphafold3_restr.md`](doc/alphafold3_restr.md) |
| **Protenix v1** | protenix | torch | [`doc/protenix_restr.md`](doc/protenix_restr.md) |
| **Protenix v2** | protenix | torch | [`doc/protenix_restr.md`](doc/protenix_restr.md) |
| **ESMFold2** | esmfold2 | torch | [`doc/esmfold2_restr.md`](doc/esmfold2_restr.md) |
| **OpenFold-3** | openfold-3 | torch | [`doc/openfold-3_restr.md`](doc/openfold-3_restr.md) |
| **Chai-1** | chai-lab | torch | [`doc/chai-lab_restr.md`](doc/chai-lab_restr.md) |

See each tool's guide in [`doc/`](doc/) for install / run details, and
[`doc/config.md`](doc/config.md) for the full `restraints_config` schema.

Five **built-in** restraint types, all minimized during the denoising loop to guide coordinate optimization:

- **conformer** — ligand bond / angle / chiral-volume / cistrans (E/Z) / improper
  (sp2 double-bond planarity, opt-in) toward an ideal RDKit geometry, plus **VdW**
  non-bonded clash avoidance (intramolecular and/or dynamic ligand-protein; `mode`
  defaults to `both`).
- **RMSD** — Kabsch-superposed RMSD of a group toward a reference PDB.
- **distance** — centroid distance between two atom groups (applied closed-form).
- **angle** — the angle of three atom groups' centroids (vertex = group 2), in degrees;
  the angular analogue of the distance restraint.
- **dihedral** — the dihedral of four atom groups' centroids (axis = groups 2–3), in degrees.

These five are the built-ins; you can add an **original restraint** without editing the
engine — config-only (the `custom` restraint) or in code (the registry). See
[Custom restraints](#custom-restraints-extending-the-restraint-set) below.

The default `method='CG'` solver (a nonlinear conjugate gradient with autodiff gradients)
runs on GPU or CPU via the same torch/jax backend (`gpu: false` runs it on CPU); distance
restraints skip the solver and are applied closed-form. Each restraint is gated by an
optional `start_sigma` (active once `sigma <= start_sigma`) and `stop_sigma` (released
once `sigma < stop_sigma`).

## Installation

```bash
uv sync                                  # this checkout (dev)
uv pip install -e <path>/rgi_utils[torch]   # into a PyTorch tool's env
uv pip install -e <path>/rgi_utils[jax]     # into a JAX tool's env
```

## Usage

```python
from rgi_utils.combined import CombinedRestraints

restraints_config = {
    "gpu": True,                 # device: True=GPU, False=CPU. backend: torch (default) / jax
    "method": "CG",
    "max_iter": 200,
    "verbose": True,
    # NOTE: start_sigma / stop_sigma are NOT top-level keys (a top-level start_sigma
    # raises). They are set per distance/rmsd/group entry and once inside
    # conformer_restraints_config. start_sigma omitted -> active at every step (set it,
    # e.g. 1.0, to act only late); stop_sigma omitted -> never released.
    "distance_restraints_config": [          # a LIST of entries
        {
            "atom_selection1": "chain A and resid 10",
            "atom_selection2": "chain B and resid 20",
            "harmonic": {"target_distance": 5.0},
            "start_sigma": 1.0,              # optional; active when sigma <= start_sigma
            # "move": "both",               # which group the centroid shift moves: both / 1 / 2
        }
    ],
    "angle_restraints_config": [             # group-centroid angle: 3 groups, vertex = group 2
        {
            "atom_selection1": "chain A and resid 1 to 10",
            "atom_selection2": "chain A and resid 40 to 50",
            "atom_selection3": "chain A and resid 80 to 90",
            "harmonic": {"target_angle": 90.0},   # DEGREES
        }
    ],
    "conformer_restraints_config": {
        "start_sigma": 1.0,                  # one value for all conformer terms (optional)
        "bond": {"weight": 1.0},
        "angle": {"weight": 1.0},
        "chiral": {"weight": 1.0},
        "cistrans": {"weight": 1.0},         # ligand cis/trans (acyclic C=C only)
        "improper": {"weight": 1.0},         # sp2 double-bond planarity (off by default)
        "vdw": {"weight": 1.0},              # mode defaults to "both"
    },
    # "rmsd_restraints_config": [{"ref_pdb": "ref.pdb", "target_rmsd": 0.0}],
    # "dihedral_restraints_config": [...],   # group-centroid dihedral: 4 groups, axis = 2-3
    # "custom_restraints_config": [          # config-only custom restraints (registry; see below)
    #     {"measure": "radius_of_gyration", "atom_selection": "chain A",
    #      "form": "harmonic", "target": 12.0}],
}

# ONE instance per structure (not a singleton). setup() takes the config dict.
restr = CombinedRestraints()
restr.setup(adapter, nbatch=multiplicity, config=restraints_config)

# Inside the denoising loop, right after the network's denoised x0 prediction:
coords = restr.minimize(coords, step, sigma)   # torch/numpy: mutates in place + returns
# After sampling (optional per-term energy log when verbose):
restr.finalize(coords, step)
```

For a **JAX** tool whose loop runs inside `lax.scan` (no Python callbacks), build the
spec outside the scan and grab the pure closure with `restr.get_minimizer()`
(`(flat_coords, sigma) -> flat_coords`), then call it inside the compiled loop instead
of `minimize`.

### Atom selection syntax

Distance restraints use a selection DSL to specify atom groups. Its keyword / range /
boolean vocabulary is **MDTraj-like** (`resid 1 to 5`, `and` / `or` / `not`,
`protein` / `backbone` / …), though `chain` takes letter ids and `resid` is the
per-chain 1-based ordinal:

| Example | Meaning |
|---------|---------|
| `chain A` | all atoms in chain A |
| `resid 10` | residue 10 (1-based, per-chain ordinal) |
| `resid 1 to 5` | residues 1–5 |
| `resid 1 3 7` | residues 1, 3, 7 |
| `index 42` | atom at padded index 42 |
| `name CA` | atoms named CA (case-insensitive) |
| `protein` / `dna` / `rna` | polymer-type selectors |
| `backbone` / `sidechain` | MDTraj-like polymer selectors (gated on polymer type) |
| `chain A and resid 1 to 5` | boolean AND |
| `chain A or chain B` | boolean OR |
| `not chain A` | negation |
| `(chain A or chain B) and resid 1 to 10` | parenthesized expressions |

### Distance restraint types

| Type | Parameters | Behavior |
|------|-----------|---------|
| `harmonic` | `target_distance` | Quadratic penalty at all distances |
| `flat-bottomed` | `target_distance1`, `target_distance2` | No penalty between d1–d2 |
| `flat-bottomed1` | `target_distance1` | No penalty below d1 |
| `flat-bottomed2` | `target_distance2` | No penalty above d2 |

Distance is calculated between the centroids (unweighted geometric centers) of the two selected atom groups (`calc_method: "unfixed-absolute"`).

The per-entry `move` key picks which group the closed-form centroid shift moves: `both`
(default, both move) / `1` / `2` (pin the other group — e.g. move only a ligand toward a
fixed pocket).

### Angle / dihedral restraints

`angle_restraints_config` (3 groups, vertex = group 2) and `dihedral_restraints_config`
(4 groups, axis = group 2–3) restrain the angle/dihedral of the groups' centroids — distinct
from the per-atom `angle` / `cistrans` *conformer* terms (internally these are the
`group_angle` / `group_dihedral` energy terms). Same four
types as distance (`harmonic` / `flat-bottomed` / `flat-bottomed1` / `flat-bottomed2`),
but targets are in **degrees** (`target_angle` / `target_dihedral`). `weight` defaults to
1.0 and translates any group size rigidly. `move` selects which groups are free (default:
the arms move, the anchor group is pinned).

### Custom restraints (extending the restraint set)

The five types above are the **built-ins**. To add an *original* restraint, use the
**registry** — no hand-editing of the engine's spec / energy / optim layers:

- **Config-only** (no Python): the built-in `custom` restraint reads `custom_restraints_config`,
  a vocabulary of geometric **measures** (`distance` / `angle` / `dihedral` /
  `radius_of_gyration` of atom-group centroids) × the distance-style penalty **forms**
  (`harmonic` / `flat-bottomed{,1,2}`). E.g. compact a domain:

  ```python
  "custom_restraints_config": [
      {"measure": "radius_of_gyration", "atom_selection": "chain A and resid 1 to 80",
       "form": "harmonic", "target": 12.0, "weight": 1.0},
  ]
  ```

- **Code-level** (new maths): `register_restraint(RestraintType(...))` with your own
  per-backend (numpy / torch / jax) leaf energy. Registered restraints run on **every**
  backend like the built-ins (parity is required at registration).

Full config schema: [`doc/config.md`](doc/config.md) (the `custom_restraints_config` section).
Full code recipe + contracts:
[`skills/implement-rgi/references/adding-a-restraint.md`](skills/implement-rgi/references/adding-a-restraint.md).

### Implementing a framework adapter

```python
from rgi_utils.atom_context import AtomRecord, LigandConf
from typing import Iterator

class MyAdapter:
    # Required for distance restraints:
    def iter_atoms(self) -> Iterator[AtomRecord]:
        for atom in self.real_atoms:                 # skip padding
            yield AtomRecord(
                chain=atom.chain_id,
                resid=atom.per_chain_ordinal,        # 1-based, resets at each chain
                index=atom.row_in_coord_tensor,      # global flat index into the coord tensor
            )

    # Optional — add these for conformer / VdW restraints:
    def num_atoms(self) -> int: ...                  # padded coord-tensor length
    def get_elements(self): ...                      # (num_atoms,) atomic numbers, 0 = padding
    def iter_ligand_confs(self) -> Iterator[LigandConf]:
        for lig in self.ligands:                     # one LigandConf per ligand
            yield LigandConf(mol, conf_coords, global_indices)
```

Tool-side adapters are tiny — see `src/rgi_utils/{boltz,protenix,chai,openfold3}/adapter.py`
for worked examples, and the `skills/implement-rgi/` guide for the full integration recipe.

## Development

```bash
task lint       # check style
task format     # auto-fix style
task test       # run all tests
task test-ci    # run non-GPU tests only
```

GPU tests are marked `@pytest.mark.gpu` and excluded in CI.
