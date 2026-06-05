# rgi-utils

Restraint-Guided Inference (RGI) utilities for diffusion-based structure predictors
(PyTorch and JAX). Integrated into boltz, protenix, chai-lab, openfold-3 (torch) and
alphafold3 (jax).

Provides distance restraints (center-of-mass between atom groups) and ligand conformer
restraints (bond / angle / chiral volume / dihedral / intramolecular VdW toward an ideal
RDKit geometry) plus a Kabsch-superposed RMSD restraint, minimized during the denoising
loop to guide coordinate optimization. The default `method='CG'` solver (a nonlinear
conjugate gradient with autodiff gradients) runs on GPU or CPU via the same torch/jax
backend (`gpu: false` runs it on CPU); distance restraints are applied closed-form.

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
    # NOTE: start_sigma is NOT a top-level key (a top-level one raises). It is set
    # per distance entry and once inside conformer_restraints_config; omitted -> the
    # restraint is active at every step (set it, e.g. 1.0, to act only late).
    "distance_restraints_config": [          # a LIST of entries
        {
            "atom_selection1": "chain A and resid 10",
            "atom_selection2": "chain B and resid 20",
            "harmonic": {"target_distance": 5.0},
            "start_sigma": 1.0,              # optional; active when sigma <= start_sigma
        }
    ],
    "conformer_restraints_config": {
        "bond": {"weight": 1.0},
        "angle": {"weight": 1.0},
        "chiral": {"weight": 1.0},
        "vdw": {"weight": 1.0, "mode": "intramolecular"},
    },
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

Distance restraints use a selection DSL to specify atom groups:

| Example | Meaning |
|---------|---------|
| `chain A` | all atoms in chain A |
| `resid 10` | residue 10 (1-based) |
| `resid 1 to 5` | residues 1–5 |
| `resid 1 3 7` | residues 1, 3, 7 |
| `index 42` | atom at padded index 42 |
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

Distance is calculated between the centers of mass (COM) of the two selected atom groups (`calc_method: "unfixed-absolute"`).

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
