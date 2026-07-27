# AlphaFold 3 — Restraint-Guided Inference (RGI)

[Documentation index](README.md) · [Configuration reference](config.md)

AlphaFold 3 + [`rgi_utils`](https://github.com/cddlab/rgi_utils) restraint-guided inference. Full
`restraints_config` schema & atom-selection DSL: [`config.md`](config.md).

> **Or generate it automatically:** the `generate-rgi-config` skill in Claude Code
> (`/generate-rgi-config`) or Codex (`$generate-rgi-config`) interviews you about the goal and
> writes a validated `restraints_config` where this tool expects it. Use it when hand-writing the
> full config below is unnecessary.

AF3 is the **JAX** tool: the restraint spec is built outside the `hk.scan` sampler and the pure
JIT-able minimizer closure (`get_minimizer()`) runs inside the compiled loop on each x0 prediction.

## Installation

The RGI code lives in the `cddlab/alphafold3_restr` fork. The base AF3 env is involved (it compiles
C++ components via scikit-build-core, and the model parameters must be obtained from Google) —
follow the upstream `docs/installation.md` for the full setup; the RGI delta is just the editable
fork install (the rgi_utils engine is declared in pyproject and comes with it):

```bash
git clone https://github.com/cddlab/alphafold3_restr.git
cd alphafold3_restr
uv venv && source .venv/bin/activate           # Python 3.12+
uv pip install -e .                            # compiles the C++ chem components (cmake/ninja) + pulls the rgi_utils engine
```

> For co-development of the engine, override the pinned dependency with a local editable
> checkout in a SEPARATE step: `uv pip install -e ../rgi_utils` (sibling clone).

- **Model parameters** are not redistributable: request them via Google's form (see the repo's
  `WEIGHTS_TERMS_OF_USE.md`) and point `--model_dir` at them.
- Run on a CUDA GPU.

## Configuration

AF3 reads RGI from a **`restraints_config` key inside the fold-input JSON** (beside
`sequences`/`modelSeeds`). Turn restraints on with:

1. **Per sequence** — `"conformer_restraints": true` on each protein, DNA, RNA,
   or ligand object enables only that chain.
2. **The `restraints_config` object** — the distance / angle / dihedral / conformer /
   RMSD restraints, plus config-only `custom` restraints (define your own — see config.md). The example below writes **every usable variable** with a concrete value; see
   [`config.md`](config.md) for the alternatives (restraint types and the RMSD
   `atom_selection_ref` / `atom_selection_target` shorthand).

### AlphaFold 3 notes

- **MSA**: AF3 has no ColabFold-style MSA *server*; it builds MSAs with a local genetic-search data
  pipeline. The run command passes `--run_data_pipeline=True` with `--db_dir` pointing at the
  sequence databases, so the JSON below carries **no `unpairedMsa`/`pairedMsa`/`templates`
  fields** — the pipeline builds them. (To skip the search, inline an `unpairedMsa` and pass
  `--run_data_pipeline=False`.)
- AF3 runs the **jax** backend — selected automatically because the AF3 glue grabs the pure
  minimizer via `get_minimizer()` (not a config key). The `gpu` flag is **inert** (the minimizer
  always runs on the model's device — to compute on CPU, run the whole process on the JAX CPU
  platform).
- AF3's minimizer converges near-target (~24-25 Å for the distance example); `max_iter: 2000`.
- **Nucleic residue names**: AF3 encodes `aatype` with the vocabulary that carries a GAP token
  after `UNK` (`… 20:UNK, 21:'-', 22:A, 23:G …`), while the plain `POLYMER_TYPES` list has no gap
  entry. Read with the wrong one, proteins stay correct (they sit below the gap) but every nucleic
  name shifts by one — an adenine token reads as `G`, a uridine as `DA`. The adapter resolves this
  from the batch's own `is_rna`/`is_dna` flags and logs a warning when it has to shift, so
  base-pair auto-detection and monomer-library lookups see the real base either way.
- **Nucleic-acid `conformer_restraints`**: prefer
  [`monomer_library`](config.md#monomer_library--refinement-targets-for-polymers-not-a-term) over
  the default reference-conformer targets. AF3 builds `ref_pos` by ETKDG-embedding the free CCD
  component, which is not refinement geometry (unconjugated exocyclic C-N, P-OH phosphate, and a
  seed-dependent embed), so plain `bond`/`angle` measurably *worsens* nucleotide geometry.

`resid` is the **per-chain 1-based ordinal**; there is **no top-level `start_sigma`**.

## Complete example (input JSON)

Save this as `restr_example.json`. The genetic-search data pipeline builds the MSA, so the JSON has
no MSA/template fields. It folds QBP + GLN **plus a short DNA duplex and an RNA duplex** and sets a
centroid distance, group angle, group dihedral, GLN conformer, whole-structure RMSD, a custom
(formula) restraint, and **Watson-Crick base pairs on the nucleic acids**. The custom entry keeps
both lobe-halves equidistant from the central domain — a difference of two distances, which no
single built-in can express (JSON has no comments, so the rationale lives here in prose). Chain ids
are explicit (`"id"`): protein **A**, ligand **B**, DNA strands **C**/**D**, RNA strands **E**/**F**;
both duplex strands are self-complementary palindromes (`GCATGC` / `GCAUGC`). It all runs under the
JAX minimizer (`lax.scan`) like every other restraint.

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
        "modifications": [],
        "conformer_restraints": true
      }
    },
    {
      "ligand": {
        "id": "B",
        "ccdCodes": ["GLN"],
        "conformer_restraints": true
      }
    },
    { "dna": { "id": "C", "sequence": "GCATGC", "modifications": [] } },
    { "dna": { "id": "D", "sequence": "GCATGC", "modifications": [] } },
    { "rna": { "id": "E", "sequence": "GCAUGC", "modifications": [] } },
    { "rna": { "id": "F", "sequence": "GCAUGC", "modifications": [] } }
  ],
  "restraints_config": {
    "verbose": true,
    "gpu": true,
    "method": "CG",
    "max_iter": 2000,
    "distance_restraints_config": [
      {
        "atom_selection1": "chain A and ((resid 5 to 84) or (resid 186 to 224))",
        "atom_selection2": "chain A and (resid 90 to 180)",
        "start_sigma": 99999999,
        "stop_sigma": -1,
        "move": "both",
        "weight": 1.0,
        "harmonic": { "target_distance": 25.0 }
      }
    ],
    "base_pair_restraints_config": [
      { "residue1": "chain C and resid 1", "residue2": "chain D and resid 6" },
      { "residue1": "chain C and resid 3", "residue2": "chain D and resid 4" },
      { "residue1": "chain E and resid 1", "residue2": "chain F and resid 6" },
      { "residue1": "chain E and resid 3", "residue2": "chain F and resid 4" }
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
      "plane": { "weight": 1.0 },
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
    ],
    "custom_restraints_config": [
      {
        "name": "equidistant",
        "energy": "(distance(L1, H) - distance(L2, H))**2",
        "selections": {
          "L1": "chain A and (resid 5 to 84)",
          "L2": "chain A and (resid 186 to 224)",
          "H": "chain A and (resid 90 to 180)"
        },
        "start_sigma": 99999999,
        "stop_sigma": -1,
        "weight": 1.0
      }
    ]
  }
}
```

## Run

Save as `run_restr_example.sh` and run it on a GPU machine (`bash run_restr_example.sh`). Set
`MODEL_DIR` to your AF3 parameters directory and `DB_DIR` to the sequence databases:

```bash
#!/bin/bash
# alphafold3 RGI example runner (JAX backend). Run on a machine with a CUDA GPU.
set -e
source .venv/bin/activate

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to your AF3 model-parameters directory}"
DB_DIR="${DB_DIR:?set DB_DIR to your AF3 sequence-database directory}"
# AF3 enables no persistent XLA cache by default (~2 min recompile/run); this reuses it.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/${USER}_jax_cache}"

# NB: AF3 early-returns if --output_dir already holds results — use a fresh dir to re-run.
python run_alphafold.py \
    --run_data_pipeline=True \
    --model_dir="$MODEL_DIR" \
    --db_dir="$DB_DIR" \
    --json_path=restr_example.json \
    --output_dir=out_restr_example
```

## Verify results

With `verbose: true`, the log prints `built spec: n_active=.. bonds=.. ... distances=.. rmsd=..
group_angle=.. group_dihedral=..` — confirm the counts are non-zero for what you requested. AF3's
venv lacks gemmi, so run the workspace `check_dist.py` / `check_conf.py` with a gemmi-enabled venv
(e.g. `../chai-lab_restr/.venv/bin/python ../check_dist.py <pred.cif>`).
