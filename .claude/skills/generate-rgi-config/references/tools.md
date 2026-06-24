# Per-tool placement, format, opt-in, and run command

The `restraints_config` **contents** are identical across all six tools — the distance /
angle / dihedral / conformer / rmsd / custom blocks are the same. What differs per tool is
only **(a) the file format, (b) where the config sits, (c) how a ligand opts into conformer
restraints, and (d) the run command**. Get (b) and (c) right or the restraint silently does
nothing.

Each tool also has a full, every-variable example in the repo: `doc/<tool>.md` ("Full
config"), and known-good fixtures at the repo root (`bench_in_<tool>_*`, `bench_m_<tool>_*`,
`bench_side_chai_*`). Prefer copying the **newer `bench_m_*`** fixtures — some older
`bench_in_*` ones predate the per-ligand conformer opt-in and carry a latent no-op.

## Placement + opt-in summary

| tool | format | where the config goes | ligand conformer opt-in |
|---|---|---|---|
| **boltz** (1/2) | YAML | `restraints_config:` nested in the input YAML (beside `sequences:`) | `conformer_restraints: true` next to the ligand's `ccd`/`smiles` |
| **protenix** | JSON | `restraints_config` nested in each job of the input **JSON list** (beside `name`/`sequences`) | `conformer_restraints: true` on the ligand object |
| **chai-lab** | YAML sidecar | the **whole sidecar file IS** the `restraints_config` (NOT nested); sequences in a separate FASTA | a `conformer_restraints: {<chain_id>: true}` map **in the sidecar** |
| **alphafold3** | JSON | `restraints_config` key in the fold-input JSON (beside `sequences`/`modelSeeds`) | `conformer_restraints: true` on the ligand object |
| **openfold-3** | JSON | `restraints_config` per query: `queries.<name>.restraints_config` | `conformer_restraints: true` on the ligand chain |
| **esmfold2** | Python dict | passed to `ESMFold2InputBuilder().fold(..., restraints_config=...)` | `conformer_restraints=True` on the `LigandInput` |

## Minimal placement examples

**boltz** (`input.yaml`):
```yaml
sequences:
  - protein: {id: A, sequence: "ADKK...", msa: ./out_msa/qbp.a3m}
  - ligand:  {id: B, ccd: ATP, conformer_restraints: true}   # <- opt-in lives here
restraints_config:
  verbose: true
  distance_restraints_config: [ ... ]
  conformer_restraints_config: { bond: {weight: 1}, angle: {weight: 1} }
```

**protenix** (`input.json`, a list):
```json
[{"name": "job1", "sequences": [
    {"proteinChain": {"sequence": "ADKK...", "count": 1}},
    {"ligand": {"ligand": "CCD_ATP", "count": 1, "conformer_restraints": true}}],
  "restraints_config": {"verbose": true, "distance_restraints_config": [ ... ]}}]
```

**chai** (`sidecar.yaml` — top level IS the config; pairs with a FASTA):
```yaml
verbose: true
conformer_restraints: {B: true}          # <- opt-in map keyed by chain id
distance_restraints_config: [ ... ]
conformer_restraints_config: { bond: {weight: 1} }
```

**openfold-3** (`query.json`):
```json
{"queries": {"job1": {"chains": [
    {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "ADKK..."},
    {"molecule_type": "ligand", "chain_ids": ["B"], "ccd_codes": "ATP", "conformer_restraints": true}],
  "restraints_config": {"verbose": true, "distance_restraints_config": [ ... ]}}}}
```

**alphafold3** (`fold_input.json`): a `"restraints_config": { ... }` key beside
`"sequences"` / `"modelSeeds"`; the ligand object carries `"conformer_restraints": true`.
AF3 forces the JAX backend, so `gpu` in the config is inert (run the process on the JAX CPU
platform to use CPU).

**esmfold2** (Python): build `RESTRAINTS_CONFIG = { ... }` and call
`ESMFold2InputBuilder().fold(model, spi, ..., restraints_config=RESTRAINTS_CONFIG)`, with
`LigandInput(id="B", ccd=["GLN"], conformer_restraints=True)` in the input.

## Run commands (GPU; usually via the tool's `sbatch_*.sh`)

| tool | run |
|---|---|
| boltz | `boltz predict input.yaml --seed 0 --out_dir out --model boltz2 --use_msa_server` |
| protenix | the tool's `sbatch_*.sh` / `protenix predict`. **Run on sm_89 (RTX 4090: `q1`/`q3`/`af3`), NOT Blackwell (`q4`/`maxq`)** — its fused kernels silently emit all-NaN coords there |
| chai | `python -m chai_lab.main fold input.fasta out --restraints-config-path sidecar.yaml --num-diffn-timesteps 200 --num-diffn-samples 2 --seed 0 --use-msa-server --use-templates-server --no-use-esm-embeddings` (FASTA + out_dir are **positional**) |
| alphafold3 | AF3's `run_alphafold.py` with `--json_path input.json` (+ `--model_dir`); MSA via local genetic search, no server |
| openfold-3 | `pixi run -e openfold3-cuda12 run_openfold predict --query-json query.json --output-dir out --num-diffusion-samples 2 --use-msa-server false --use-templates false` |
| esmfold2 | a Python script (see `esm_restr/restr_example.py` + `sbatch_esm_example.sh`); single-sequence, no MSA. Note the esmfold2 copy-over gotcha in `doc/esmfold2_restr.md` — without it the per-step hook is silently absent |

GPU work goes through `sbatch` on a compute node, never the login node. See the workspace
`CLAUDE.md` ("GPU work goes through sbatch") and each `doc/<tool>.md` for the full,
copy-pasteable run script.

## Which Python runs the validator

`.claude/skills/generate-rgi-config/scripts/validate_config.py` needs `rgi_utils`
(numpy-only) and, for YAML files, `pyyaml`:

- **JSON** (protenix/AF3/openfold): `rgi_utils/.venv/bin/python` is enough.
- **YAML** (boltz/chai): use a tool venv that also has pyyaml, e.g.
  `boltz_restr/.venv/bin/python`.
