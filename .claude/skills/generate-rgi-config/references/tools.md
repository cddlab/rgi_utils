# Per-tool placement, format, opt-in, and run command

The `restraints_config` **contents** are identical across all seven tools — the distance /
angle / dihedral / conformer / rmsd / base_pair / custom blocks are the same. What differs per tool is
only **(a) the file format, (b) where the config sits, (c) how an entity opts into conformer
restraints, and (d) the run command**. Get (b) and (c) right or the restraint silently does
nothing.

Each tool also has a full, every-variable example in the repo: `doc/<tool>.md` ("Full
config"), and known-good fixtures at the repo root (`bench_in_<tool>_*`, `bench_m_<tool>_*`,
`bench_side_chai_*`). Prefer copying the **newer `bench_m_*`** fixtures — some older
`bench_in_*` ones predate the per-ligand conformer opt-in and carry a latent no-op.

## Placement + opt-in summary

| tool | format | where the config goes | conformer opt-in |
|---|---|---|---|
| **boltz** (1/2) | YAML | `restraints_config:` nested in the input YAML (beside `sequences:`) | `conformer_restraints: true` next to the ligand's `ccd`/`smiles` |
| **protenix** | JSON | `restraints_config` nested in each job of the input **JSON list** (beside `name`/`sequences`) | `conformer_restraints: true` on the ligand object |
| **OpenDDE** | JSON | `restraints_config` nested in each job of the input **JSON list** (beside `name`/`modelSeeds`/`sequences`) | `conformer_restraints: true` on the sequence entity |
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

**OpenDDE** (`input.json`, a list):
```json
[{"name": "job1", "modelSeeds": [0], "sequences": [
    {"proteinChain": {"id": ["A"], "sequence": "ADKK...", "count": 1}},
    {"ligand": {"id": ["B"], "ligand": "CCD_ATP", "count": 1,
                 "conformer_restraints": true}}],
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
AF3 runs the JAX backend (selected automatically — its glue calls `get_minimizer()`), so `gpu`
in the config is inert (run the process on the JAX CPU platform to use CPU).

**esmfold2** (Python): build `RESTRAINTS_CONFIG = { ... }` and call
`ESMFold2InputBuilder().fold(model, spi, ..., restraints_config=RESTRAINTS_CONFIG)`, with
`LigandInput(id="B", ccd=["GLN"], conformer_restraints=True)` in the input.

## Run commands (GPU; usually via the tool's `sbatch_*.sh`)

| tool | run |
|---|---|
| boltz | `boltz predict input.yaml --seed 0 --out_dir out --model boltz2 --use_msa_server` |
| protenix | the tool's `sbatch_*.sh` / `protenix predict`. **Run on sm_89 (RTX 4090), NOT Blackwell (sm_120)** — its fused kernels silently emit all-NaN coords there |
| OpenDDE | `opendde pred -i input.json -o out -n opendde_v1 --use_msa false --use_template false --use_rna_msa false --sample 1 --step 200 --cycle 4` |
| chai | `python -m chai_lab.main fold input.fasta out --restraints-config-path sidecar.yaml --num-diffn-timesteps 200 --num-diffn-samples 2 --seed 0 --use-msa-server --use-templates-server --no-use-esm-embeddings` (FASTA + out_dir are **positional**) |
| alphafold3 | AF3's `run_alphafold.py` with `--json_path input.json` (+ `--model_dir`); MSA via local genetic search, no server |
| openfold-3 | `pixi run -e openfold3-cuda12 run_openfold predict --query-json query.json --output-dir out --num-diffusion-samples 2 --use-msa-server false --use-templates false` |
| esmfold2 | a Python script (see `esm_restr/restr_example.py` + `sbatch_esm_example.sh`); single-sequence, no MSA. Note the esmfold2 copy-over gotcha in `doc/esmfold2_restr.md` — without it the per-step hook is silently absent |

GPU work goes through `sbatch` on a compute node, never the login node. See the workspace
`AGENTS.md` (also exposed as `CLAUDE.md`; "GPU work goes through sbatch") and each
`doc/<tool>.md` for the full, copy-pasteable run script.

## Which Python runs the validator

The bundled `scripts/validate_config.py` needs `rgi_utils` (numpy-only) and, for YAML
files, `pyyaml`. Resolve `SKILL_DIR` to the directory containing the skill's `SKILL.md`,
then run:

```bash
uv run --project <rgi-utils-dir> --frozen --with pyyaml \
  python "$SKILL_DIR/scripts/validate_config.py" <file>
```
