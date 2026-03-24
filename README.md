# rgi-utils

Restraint-Guided Inference utilities for PyTorch-based protein structure prediction models.

Provides bond, angle, chiral volume, and distance restraints that can be applied during inference to guide coordinate optimization — on CPU (scipy) or GPU (torchmin).

## Installation

```bash
uv sync
```

## Usage

```python
from rgi_utils import CombinedRestraints, FrameworkAdapter

restr = CombinedRestraints.get_instance()

# Configure restraints
restr.set_config({
    "gpu": True,
    "method": "CG",
    "max_iter": 100,
    "start_sigma": 1.0,
    "distance_restraints_config": {
        "restraints": [
            {
                "atom_selection1": "chain A and resid 10",
                "atom_selection2": "chain B and resid 20",
                "calc_method": "unfixed-absolute",
                "harmonic": {"target_distance": 5.0},
            }
        ]
    },
})

# Set up with a framework adapter and coordinate tensors
restr.setup(adapter, conformer_restraint, atom_pad_mask, ref_element)

# Optimize coordinates
coords = restr.minimize(coords)
```

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
from rgi_utils import AtomRecord, FrameworkAdapter
from typing import Iterator

class MyAdapter:
    def iter_atoms(self) -> Iterator[AtomRecord]:
        for atom in self.atoms:
            if not atom.is_padding:
                yield AtomRecord(
                    chain=atom.chain_id,
                    resid=atom.residue_index,  # 1-based
                    index=atom.padded_index,
                )
```

## Development

```bash
task lint       # check style
task format     # auto-fix style
task test       # run all tests
task test-ci    # run non-GPU tests only
```

GPU tests are marked `@pytest.mark.gpu` and excluded in CI.
