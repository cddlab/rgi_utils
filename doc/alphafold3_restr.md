# alphafold3_restr — Restraint-Guided Inference (RGI)

AlphaFold 3 + [`rgi_utils`](https://github.com/cddlab/rgi_utils) restraint-guided inference. Full
`restraints_config` schema & atom-selection DSL: [`config.md`](config.md).

AF3 is the **JAX** tool: the restraint spec is built outside the `hk.scan` sampler and the pure
JIT-able minimizer closure (`get_minimizer()`) runs inside the compiled loop on each x0 prediction.

## Install

The RGI code lives in the `cddlab/alphafold3_restr` fork (branch `rgi-integration`). The base AF3
env is involved (it compiles C++ components via scikit-build-core, and the model parameters must be
obtained from Google) — follow the upstream `docs/installation.md` for the full setup; the RGI
delta is just the editable fork install plus the `[jax]` engine extra:

```bash
git clone -b rgi-integration https://github.com/cddlab/alphafold3_restr.git
cd alphafold3_restr
uv venv && source .venv/bin/activate           # Python 3.12+
uv pip install -e .                            # scikit-build-core compiles the C++ chem components (needs cmake/ninja)
uv pip install "rgi_utils[jax] @ git+https://github.com/cddlab/rgi_utils.git@rgi-integration"
```

- **Model parameters** are not redistributable: request them via Google's form (see the repo's
  `WEIGHTS_TERMS_OF_USE.md`) and point `--model_dir` at them.
- Run on a CUDA GPU.

## Configuring restraints

AF3 reads RGI from a **`restraints_config` key inside the fold-input JSON** (beside
`sequences`/`modelSeeds`). Turn restraints on with:

1. **Per ligand** — `"conformer_restraints": true` on the ligand object.
2. **The `restraints_config` object** — the distance / angle / dihedral / conformer /
   RMSD restraints, plus config-only `custom` restraints (define your own — see config.md). The example below writes **every usable variable** with a concrete value; see
   [`config.md`](config.md) for the alternatives (restraint types, RMSD `atom_selection`
   shorthand).

AF3-specific notes:
- **MSA**: AF3 has no ColabFold-style MSA *server*; it builds MSAs with a local genetic-search data
  pipeline. The run command passes `--run_data_pipeline=True` with `--db_dir` pointing at the
  sequence databases, so the JSON below carries **no `unpairedMsa`/`pairedMsa`/`templates`
  fields** — the pipeline builds them. (To skip the search, inline an `unpairedMsa` and pass
  `--run_data_pipeline=False`.)
- The backend is forced to **jax**; the `gpu` flag is **inert** (the minimizer always runs on the
  model's device — to compute on CPU, run the whole process on the JAX CPU platform).
- AF3's minimizer converges near-target (~24–25 Å for the distance example); `max_iter: 2000`.

`resid` is the **per-chain 1-based ordinal**; there is **no top-level `start_sigma`**.

## Full config (input file)

Save this as `restr_example.json`. The genetic-search data pipeline builds the MSA, so the JSON has
no MSA/template fields. It sets a centroid distance, group angle, group dihedral, GLN conformer, and
whole-structure RMSD restraint.

```json
{
  "dialect": "alphafold3",
  "version": 4,
  "name": "qbp_rgi_example",
  "modelSeeds": [0],
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "ADKKLVVATDTAFVPFEFKQGDKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPALQTKNVDLALAGITITDERKKAIDFSDGYYKSGLLVMVKANNNDVKSVKDLDGKVVAVKSGTGSVDYAKANIKTKDLRQFPNIDNAYMELGTNRADAVLHDTPNILYFIKTAGNGQFKAVGDSLEAQQYGIAFPKGSDELRDKVNGALKTLRENGTYNEIYKKWFGTEPK",
        "modifications": []
      }
    },
    {
      "ligand": {
        "id": "B",
        "ccdCodes": ["GLN"],
        "conformer_restraints": true
      }
    }
  ],
  "restraints_config": {
    "verbose": true,
    "gpu": true,
    "backend": "jax",
    "method": "CG",
    "max_iter": 2000,
    "distance_restraints_config": [
      {
        "atom_selection1": "chain A and ((resid 5 to 84) or (resid 186 to 224))",
        "atom_selection2": "chain A and (resid 90 to 180)",
        "start_sigma": 99999999,
        "stop_sigma": -1,
        "move": "both",
        "harmonic": { "target_distance": 25.0 }
      }
    ],
    "angle_restraints_config": [
      {
        "atom_selection1": "chain A and (resid 5 to 84)",
        "atom_selection2": "chain A and (resid 90 to 180)",
        "atom_selection3": "chain A and (resid 186 to 224)",
        "start_sigma": 99999999,
        "stop_sigma": -1,
        "move": "1,3",
        "weight": 1.0,
        "harmonic": { "target_angle": 90.0 }
      }
    ],
    "dihedral_restraints_config": [
      {
        "atom_selection1": "chain A and (resid 5 to 50)",
        "atom_selection2": "chain A and (resid 51 to 100)",
        "atom_selection3": "chain A and (resid 101 to 150)",
        "atom_selection4": "chain A and (resid 151 to 224)",
        "start_sigma": 99999999,
        "stop_sigma": -1,
        "move": "1,4",
        "weight": 1.0,
        "harmonic": { "target_dihedral": 180.0 }
      }
    ],
    "conformer_restraints_config": {
      "start_sigma": 99999999,
      "stop_sigma": -1,
      "bond": { "weight": 1.0, "slack": 0.0 },
      "angle": { "weight": 1.0, "slack": 0.0 },
      "chiral": { "weight": 1.0, "slack": 0.05 },
      "cistrans": { "weight": 1.0, "slack": 0.0 },
      "vdw": { "weight": 1.0 }
    },
    "rmsd_restraints_config": [
      {
        "ref_pdb": "rmsd_ref.pdb",
        "harmonic": {"target_rmsd": 0.0},
        "weight": 1.0,
        "start_sigma": 99999999,
        "stop_sigma": 1.0,
        "pairing": "align",
        "best_effort": true,
        "atom_selection_ref_fit": "chain A and (resid 5 to 220)",
        "atom_selection_target_fit": "chain A and (resid 5 to 220)",
        "atom_selection_ref_calc": "chain A and (resid 90 to 180)",
        "atom_selection_target_calc": "chain A and (resid 90 to 180)"
      }
    ]
  }
}
```

## How to run

### Run

Save as `run_restr_example.sh` and run it on a GPU machine (`bash run_restr_example.sh`). Set
`MODEL_DIR` to your AF3 parameters directory and `DB_DIR` to the sequence databases:

```bash
#!/bin/bash
# alphafold3 RGI example runner (JAX backend). Run on a machine with a CUDA GPU.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to your AF3 model-parameters directory}"
DB_DIR="${DB_DIR:?set DB_DIR to your AF3 sequence-database directory}"
# AF3 enables no persistent XLA cache by default (~2 min recompile/run); this reuses it.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/${USER}_jax_cache}"

rm -rf out_restr_example   # else AF3 skip-existing early-returns on the prior output
python run_alphafold.py \
    --run_data_pipeline=True \
    --model_dir="$MODEL_DIR" \
    --db_dir="$DB_DIR" \
    --json_path=restr_example.json \
    --output_dir=out_restr_example
```

## Verify

With `verbose: true`, the log prints `built spec: n_active=.. bonds=.. ... distances=.. rmsd=..
group_angle=.. group_dihedral=..` — confirm the counts are non-zero for what you requested. AF3's
venv lacks gemmi, so run the workspace `check_dist.py` / `check_conf.py` with a gemmi-enabled venv
(e.g. `../chai-lab_restr/.venv/bin/python ../check_dist.py <pred.cif>`).
