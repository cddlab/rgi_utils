# OpenDDE — Restraint-Guided Inference (RGI)

[Documentation index](README.md) · [Configuration reference](config.md)

The `rgi-integration` branch of the `cddlab/OpenDDE_restr` fork applies
[`rgi_utils`](https://github.com/cddlab/rgi_utils) restraints to OpenDDE's denoised coordinate
estimate during diffusion. It supports the complete shared schema, including conformer/VdW,
distance, angle, dihedral, improper, plane, RMSD, base-pair, and custom restraints.

## Installation

```bash
git clone git@github.com:cddlab/OpenDDE_restr.git
cd OpenDDE_restr
git switch rgi-integration
uv venv --python 3.12
uv pip install --python .venv --torch-backend cu126 -e ".[gpu]"
```

The fork declares `rgi_utils` as a dependency. For co-development, install a sibling checkout
afterward with `uv pip install --python .venv -e ../rgi_utils`.

## Configuration

Put `restraints_config` beside `name` and `sequences` in each OpenDDE job. Add
`conformer_restraints: true` to each entity whose local geometry should be restrained. The default
is false. Explicit `id` values are recommended because the selection DSL uses them as chain names.
`resid` is the per-chain, 1-based token ordinal.

```json
[
  {
    "name": "opendde_rgi",
    "modelSeeds": [0],
    "sequences": [
      {"proteinChain": {"sequence": "ACDEFGHIKLMNPQRSTVWY", "count": 1, "id": ["A"]}},
      {"ligand": {"ligand": "C/C=C\\C", "count": 1, "id": ["L"], "conformer_restraints": true}}
    ],
    "restraints_config": {
      "gpu": true,
      "verbose": true,
      "max_iter": 100,
      "distance_restraints_config": [
        {
          "atom_selection1": "chain A and resid 1",
          "atom_selection2": "chain L",
          "harmonic": {"target_distance": 8.0},
          "start_sigma": 1.0
        }
      ],
      "conformer_restraints_config": {
        "start_sigma": 1.0,
        "bond": {}, "angle": {}, "chiral": {}, "cistrans": {}, "plane": {}, "vdw": {}
      }
    }
  }
]
```

See [`config.md`](config.md) for every field and the atom-selection DSL.

## Run

```bash
uv run --no-project --python .venv opendde pred \
  -i rgi_input.json -o output_rgi -n opendde_v1 \
  --use_msa false --use_template false --use_rna_msa false \
  --sample 1 --step 200 --cycle 10
```

No RGI CLI flag is required. A job without `restraints_config` follows the upstream path. Native
TFG can be enabled independently with `--use_tfg_guidance true`; when both are active, the sampler
applies TFG refinement first and RGI second.

With `verbose: true`, verify that `built spec:` reports non-zero counts for every requested term.

## Integration details

- `rgi_utils.opendde.OpenDDEAdapter` reads atom metadata and real bond orders from the stashed
  Biotite `AtomArray`, reference geometry from `ref_pos`, and the pre-expansion
  `residue_level_atom_to_token_idx` mapping when structural tokens are enabled.
- OpenDDE constructs a fresh `CombinedRestraints` for each structure and passes it through chunked
  sampling. The hook runs after the denoised x0 prediction and before the Euler update.
- After RGI changes x0, the noisy anchor is rigidly aligned to it so the sampler's step scale does
  not extrapolate a restraint correction into a geometry distortion.
- SMILES bond orders are Kekulized and preserved, enabling cis/trans and planar conformer terms.
