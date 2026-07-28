# rgi-utils

Restraint-Guided Inference (RGI) utilities for diffusion-based structure predictors
(PyTorch and JAX).

**Implemented and available in the following 8 models** (across 6 predictor integrations):

| Model           | Integration | Backend | Details                                              |
| --------------- | ----------- | ------- | ---------------------------------------------------- |
| **Boltz-1**     | boltz       | torch   | [`doc/boltz_restr.md`](doc/boltz_restr.md)           |
| **Boltz-2**     | boltz       | torch   | [`doc/boltz_restr.md`](doc/boltz_restr.md)           |
| **AlphaFold3**  | alphafold3  | jax     | [`doc/alphafold3_restr.md`](doc/alphafold3_restr.md) |
| **Protenix v1** | protenix    | torch   | [`doc/protenix_restr.md`](doc/protenix_restr.md)     |
| **Protenix v2** | protenix    | torch   | [`doc/protenix_restr.md`](doc/protenix_restr.md)     |
| **ESMFold2**    | esmfold2    | torch   | [`doc/esmfold2_restr.md`](doc/esmfold2_restr.md)     |
| **OpenFold-3**  | openfold-3  | torch   | [`doc/openfold-3_restr.md`](doc/openfold-3_restr.md) |
| **Chai-1**      | chai-lab    | torch   | [`doc/chai-lab_restr.md`](doc/chai-lab_restr.md)     |

See each tool's guide in [`doc/`](doc/) for install / run details, and
[`doc/config.md`](doc/config.md) for the full `restraints_config` schema.

> **Stuck writing a config?** Run the `generate-rgi-config` skill in Claude Code
> (`/generate-rgi-config`) or Codex (`$generate-rgi-config`). It interviews you about the
> goal, picks the right restraint type / atom selection / target / sigma window, validates
> the result, and writes the `restraints_config` to the correct place for your tool. Use it
> instead of hand-writing from this README when you're unsure. (For adding RGI support to a
> *new* tool's code, use the separate `implement-rgi` skill.)

Seven **built-in** restraint types, all minimized during the denoising loop to guide coordinate optimization:

- **conformer** — ligand and polymer-local bond / angle / chiral-volume / VdW;
  ligand-only cistrans (E/Z) / plane
  ([servalcat](https://github.com/keitaroyam/servalcat)-style best-fit-plane flatness of aromatic rings + sp2 groups, opt-in)
  toward an ideal RDKit geometry, plus **VdW**
  non-bonded clash avoidance (intramolecular and/or intermolecular; `mode`
  defaults to `both`). For polymers the targets can instead come from a **CCP4 monomer
  library** (`monomer_library`) — the values Refmac/servalcat refine against. Prefer that
  for nucleic acids: the predictor's own reference conformer is an ETKDG embedding of the
  free CCD component, so restraining toward it *worsens* base geometry.
- **RMSD** — Kabsch-superposed RMSD of a group toward a reference PDB.
- **distance** — centroid distance between two atom groups (CG-minimised like every other restraint).
- **angle** — the angle of three atom groups' centroids (vertex = group 2), in degrees;
  the angular analogue of the distance restraint.
- **dihedral** — the dihedral of four atom groups' centroids (axis = groups 2–3), in degrees.
- **plane** — best-fit-plane flatness of any atom group you select (out-of-plane RMS, Angstrom):
  hold a nucleobase or aromatic side chain flat, make two groups share one plane, or pull a group
  onto a plane taken from a reference structure. The selection-driven form of the conformer `plane`
  term (both follow the plane restraints of
  [servalcat](https://github.com/keitaroyam/servalcat) / Refmac — see
  [References](#references)), with its own per-entry weight / tolerance / activation window.
- **base-pair** — a named Watson–Crick nucleotide pair expanded into H-bond distance
  restraints and an optional base-coplanarity restraint.

Beyond these seven built-ins you can define your **own** restraint — see
[Custom restraints](#custom-restraints) below.

The default `method='CG'` solver (a nonlinear conjugate gradient with autodiff gradients)
runs on GPU or CPU via the same torch/jax backend (`gpu: false` runs it on CPU); all
restraints — distance included — are minimised by this solver (distance uses a
reduced-mass-rescaled centroid gradient so each group translates rigidly toward the
target). Each restraint is gated by an optional `start_sigma` (active once
`sigma <= start_sigma`) and `stop_sigma` (released once `sigma < stop_sigma`).

## Installation

`rgi_utils` is the shared engine; each integrated tool **declares it as a dependency**, so installing
a tool (`uv pip install -e .` / `pixi install`) pulls it automatically — see the tool's guide in
[`doc/`](doc/). To hack on the engine itself, in this checkout:

```bash
uv sync          # dev environment for this repo
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
        # Applied only to sequence/chain objects with conformer_restraints: true.
        # Omit start_sigma to enable it from step 0.
        "bond": {"weight": 1.0},
        "angle": {"weight": 1.0},
        "chiral": {"weight": 1.0},
        "cistrans": {"weight": 1.0},         # ligand cis/trans (acyclic C=C only)
        "plane": {"weight": 1.0},            # best-fit plane: rings + sp2 groups (off by default)
        "vdw": {"weight": 1.0, "max_neighbors": 32},  # mode defaults to "both"
    },
    "custom_restraints_config": [            # define your OWN restraint as a formula (DSL)
        {"name": "symmetric",               # keep two inter-domain distances equal
         "energy": "(distance(A, B) - distance(C, D))**2",
         "selections": {"A": "chain A and resid 10", "B": "chain B and resid 10",
                        "C": "chain A and resid 90", "D": "chain B and resid 90"}},
    ],
    # "rmsd_restraints_config": [{"ref_pdb": "ref.pdb", "harmonic": {"target_rmsd": 0.0}}],
    # "dihedral_restraints_config": [...],   # group-centroid dihedral: 4 groups, axis = 2-3
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

| Example                                  | Meaning                                               |
| ---------------------------------------- | ----------------------------------------------------- |
| `chain A`                                | all atoms in chain A                                  |
| `resid 10`                               | residue 10 (1-based, per-chain ordinal)               |
| `resid 1 to 5`                           | residues 1–5                                          |
| `resid 1 3 7`                            | residues 1, 3, 7                                      |
| `index 42`                               | atom at padded index 42                               |
| `name CA`                                | atoms named CA (case-insensitive)                     |
| `protein` / `dna` / `rna`                | polymer-type selectors                                |
| `backbone` / `sidechain`                 | MDTraj-like polymer selectors (gated on polymer type) |
| `chain A and resid 1 to 5`               | boolean AND                                           |
| `chain A or chain B`                     | boolean OR                                            |
| `not chain A`                            | negation                                              |
| `(chain A or chain B) and resid 1 to 10` | parenthesized expressions                             |

### Distance restraint types

| Type             | Parameters                             | Behavior                           |
| ---------------- | -------------------------------------- | ---------------------------------- |
| `harmonic`       | `target_distance`                      | Quadratic penalty at all distances |
| `flat-bottomed`  | `target_distance1`, `target_distance2` | No penalty between d1–d2           |
| `flat-bottomed1` | `target_distance1`                     | No penalty below d1                |
| `flat-bottomed2` | `target_distance2`                     | No penalty above d2                |

Distance is calculated between the centroids (unweighted geometric centers) of the two selected atom groups (`calc_method: "unfixed-absolute"`).

The per-entry `move` key picks which group the CG moves toward the target: `both`
(default, both move — minimal-displacement split) / `1` / `2` (pin the other group via
`stop_gradient` — e.g. move only a ligand toward a fixed pocket).

A distance entry may use one external reference group: keep `atom_selection1/2`, write the
reference-side value as `ref1 and <selection>`, and define `refs.ref1` with `ref_pdb` or
`ref_cif`. The ref group stays fixed; `move` may name only prediction-side group indices
(`all`/omitted = all prediction groups).

### Angle / dihedral restraints

`angle_restraints_config` (3 groups, vertex = group 2) and `dihedral_restraints_config`
(4 groups, axis = group 2–3) restrain the angle/dihedral of the groups' centroids — distinct
from the per-atom `angle` / `cistrans` *conformer* terms (internally these are the
`group_angle` / `group_dihedral` energy terms). Same four
types as distance (`harmonic` / `flat-bottomed` / `flat-bottomed1` / `flat-bottomed2`),
but targets are in **degrees** (`target_angle` / `target_dihedral`). `weight` defaults to
1.0 and translates any group size rigidly. `move` selects which groups are free (default:
the arms move, the anchor group is pinned). With ref groups, references stay fixed and
`move` selects prediction-side group indices; omitted/`all`/`both` moves every prediction group.

### Plane restraints

`plane_restraints_config` restrains the **out-of-plane RMS deviation** of a group you select
(internally the `group_plane` energy term). Several `atom_selectionN` in one entry are **pooled into
a single plane** — that is how you say "keep these two groups coplanar":

```yaml
plane_restraints_config:
  # hold one nucleobase flat, tolerating 0.1 A of pucker, only below sigma 2
  - atom_selection1: "chain A and resid 5 and not backbone"
    start_sigma: 2.0
    flat-bottomed2: {target_plane2: 0.1}

  # two stacked bases share ONE plane; only the first moves
  - atom_selection1: "chain A and resid 10 and not backbone"
    atom_selection2: "chain B and resid 24 and not backbone"
    move: 1
```

The restraint-type block is **optional** (omitted ⇒ `harmonic` toward 0, since a plane's target is
always 0); targets are in **Angstrom** (`target_plane` / `target_plane1` / `target_plane2`). `move`
defaults to every group free — a plane has no anchor to pin. Writing one group as
`refN and <selection>` switches the meaning: the plane is taken from the **reference** structure and
held fixed, so the prediction group is pulled *onto* it. See
[`doc/config.md`](doc/config.md#plane_restraints_config-list).

### Base-pair restraints

`base_pair_restraints_config` restrains two user-selected nucleotides to Watson–Crick geometry.
Each entry expands into the appropriate donor/acceptor distance restraints and, by default, a
best-fit plane over both bases. Standard GC/CG, AT/TA, and AU/UA orientations are detected from
`resname`;
set `pair: GU` explicitly for a wobble pair or when residue names are unavailable.

```yaml
base_pair_restraints_config:
  - residue1: "chain A and resid 5"   # exactly one nucleotide
    residue2: "chain B and resid 12"  # exactly one nucleotide
    # pair: GC          # optional override; GU must be explicit
    # target: [2.7, 3.1]  # H-bond distance window; scalar means harmonic
    # coplanar: true    # add inter-base coplanarity (default true)
    # move: both        # both / 1 / 2; choose which residue the H-bonds move
```

The sigma/step window and `move` apply to the generated H-bond distances **and** to the coplanarity
plane (which is emitted as a `plane_restraints_config` restraint, so `stop_sigma` releases both
together). See [`doc/config.md`](doc/config.md#base_pair_restraints_config-list) for atom pairs,
validation rules, and gating details.

### Custom restraints

Define your **own** restraint — not one of the seven built-ins — as a differentiable energy,
two ways (same vocabulary, both run on every backend):

**Config only** — write the energy as a math **formula** over named selections, no Python:

```yaml
custom_restraints_config:
  - name: symmetric                       # keep two inter-domain distances equal
    energy: "(distance(A, B) - distance(C, D))**2"
    selections: {A: "chain A and resid 10", B: "chain B and resid 10",
                 C: "chain A and resid 90", D: "chain B and resid 90"}
    move: [A, C]                              # B and D are pinned for this term
    weight: 1.0
```

**Code** — write `energy(ctx) -> scalar` and pass it directly (or register it for config reuse):

```python
from rgi_utils import CombinedRestraints, custom_restraint

restr = CombinedRestraints()
restr.add_custom(                         # throwaway: a callable, no registration
    fn=lambda ctx: (ctx.distance("chain A and resid 10", "chain B and resid 10")
                  - ctx.distance("chain A and resid 90", "chain B and resid 90"))**2)
restr.setup(adapter, config=restraints_config)   # add_custom BEFORE setup

@custom_restraint("symmetric")            # reusable: config can {use: "symmetric"}
def energy(ctx): ...
```

`ctx` / the formula expose one **vocabulary**: geometry (`distance` `angle` `dihedral` `centroid`
`rg` `norm` `dot`), penalty (`harmonic` `flat_bottomed` `flat_bottomed1` `flat_bottomed2`), and math
(`sqrt` `exp` `log` `abs` `sin` `cos` `clip` `minimum` `maximum` `where` `sum` + arithmetic). Use `where(cond, a, b)`,
not `if` (keeps it jax-traceable). The energy (× `weight`) is added to the CG objective with the
usual `start_sigma` / `stop_sigma` gating. `move` accepts a prediction selection name or list
of names; omitted/`all`/`both` moves every prediction selection, while ref-backed selections stay fixed. Formulas are parsed safely (no
`eval`). A custom selection can use
`refN and <selection>`; all geometry functions accept it, and `rmsd(A,B)` requires prediction
selection A and reference-backed selection B. Full reference:
[`doc/config.md`](doc/config.md) (the `custom_restraints_config` section).

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
for worked examples, and the shared `implement-rgi` skill under `.claude/skills/` and
`.agents/skills/` for the full integration recipe.

## Development

```bash
task lint       # check style
task format     # auto-fix style
task test       # run all tests
task test-ci    # run non-GPU tests only
```

GPU tests are marked `@pytest.mark.gpu` and excluded in CI.

## References

The **plane** restraints — both the conformer `plane` term and the standalone
`plane_restraints_config` (`group_plane`) — follow the plane-restraint formulation of
[**servalcat**](https://github.com/keitaroyam/servalcat) / Refmac: a whole planar *group* of atoms
is restrained by the RMS deviation of its atoms from their own best-fit plane, instead of the
per-centre improper (signed-volume) terms this project used before. The base-pair macro follows the
same tools (a Watson–Crick pair is imposed as H-bond distances plus base coplanarity, not as a
dedicated base-pair energy), and `monomer_library` reads the CCP4 monomer library that
Refmac/servalcat refine against. The implementation here is independent — autodiff gradients through
a stop-gradient plane normal, the four flat-bottomed penalty shapes, and sigma/step gating inside a
diffusion sampler.

- Yamashita, K., Palmer, C. M., Burnley, T. & Murshudov, G. N. (2021). *Cryo-EM single-particle
  structure refinement and map calculation using Servalcat.* Acta Cryst. **D77**, 1282–1291.
  <https://doi.org/10.1107/S2059798321009475>
- Yamashita, K., Wojdyr, M., Long, F., Nicholls, R. A. & Murshudov, G. N. (2023).
  *GEMMI and Servalcat restrain REFMAC5.* Acta Cryst. **D79**, 368–373.
  <https://doi.org/10.1107/S2059798323002413>
- CCP4 monomer library: <https://github.com/MonomerLibrary/monomers>
