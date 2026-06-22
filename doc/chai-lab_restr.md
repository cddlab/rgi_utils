# chai-lab_restr — Restraint-Guided Inference (RGI)

Chai-1 + [`rgi_utils`](https://github.com/cddlab/rgi_utils) restraint-guided inference. Full
`restraints_config` schema & atom-selection DSL: [`config.md`](config.md).

## Install

The RGI code lives in the `cddlab/chai-lab_restr` fork (branch `rgi-integration`) — install **that
fork**, not the upstream PyPI `chai_lab`, which has no RGI hooks. Run on a CUDA GPU with bfloat16
support (RTX 4090 / sm_89 works).

```bash
git clone -b rgi-integration https://github.com/cddlab/chai-lab_restr.git
cd chai-lab_restr
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e .
uv pip install "rgi_utils[torch] @ git+https://github.com/cddlab/rgi_utils.git@rgi-integration"
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124  # PyPI default is +cpu!
uv pip install pyyaml                                                                 # chai1.py imports yaml, not a chai dep
```

## Configuring restraints

chai is the **odd one out**: the file you pass to `--restraints-config-path` **IS** the
`restraints_config` dict at top level — do **not** nest it under a `restraints_config:` key (that is
the boltz/protenix style). chai loads it verbatim with `yaml.safe_load`. The sequences live in a
separate FASTA (the ligand is a SMILES string).

chai's FASTA can't carry a per-ligand flag (the header parser accepts only `name=`), so chai's
conformer opt-in lives **in the sidecar** as a `conformer_restraints` map keyed by ligand chain id
(the same chain id you use in `atom_selection`, e.g. `B`): set it `true` to enable that ligand's
bond/angle/chiral/cistrans/VdW conformer restraints. A ligand absent from the map (or set `false`)
is left unrestrained even when a `conformer_restraints_config` block is present. chai drops
intra-ligand bond orders at every layer, so the adapter rebuilds the molecule from the source SMILES
(Kekulized → correct valence + aromaticity + stereo) — bond/angle/chiral all apply.

The sidecar below writes **every usable variable** with a concrete value; see
[`config.md`](config.md) for the alternatives (restraint types, config-only `custom`
restraints, RMSD `atom_selection` shorthand). `resid` is the **per-chain 1-based ordinal** (qualify protein groups with `chain A and
(...)`). There is **no top-level `start_sigma`**.

## Full config (sidecar + FASTA)

Save the FASTA as `restr_example.fasta`:

```text
>protein|name=qbp
ADKKLVVATDTAFVPFEFKQGDKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPALQTKNVDLALAGITITDERKKAIDFSDGYYKSGLLVMVKANNNDVKSVKDLDGKVVAVKSGTGSVDYAKANIKTKDLRQFPNIDNAYMELGTNRADAVLHDTPNILYFIKTAGNGQFKAVGDSLEAQQYGIAFPKGSDELRDKVNGALKTLRENGTYNEIYKKWFGTEPK
>ligand|name=gln
N[C@@H](CCC(N)=O)C(=O)O
```

Save the sidecar as `restr_example.yaml` (this whole file is the `restraints_config` dict, with a
centroid distance, group angle, group dihedral, GLN conformer, and whole-structure RMSD restraint).
The run command passes `--use-msa-server --use-templates-server`, so chai fetches MSAs/templates
from the ColabFold server.

```yaml
verbose: true
gpu: true
backend: torch
method: "CG"
max_iter: 1000
distance_restraints_config:
  - atom_selection1: "chain A and ((resid 5 to 84) or (resid 186 to 224))"
    atom_selection2: "chain A and (resid 90 to 180)"
    start_sigma: 99999999
    stop_sigma: -1
    move: both
    harmonic:
      target_distance: 25.0
angle_restraints_config:
  - atom_selection1: "chain A and (resid 5 to 84)"
    atom_selection2: "chain A and (resid 90 to 180)"
    atom_selection3: "chain A and (resid 186 to 224)"
    start_sigma: 99999999
    stop_sigma: -1
    move: "1,3"
    weight: 1.0
    harmonic:
      target_angle: 90.0
dihedral_restraints_config:
  - atom_selection1: "chain A and (resid 5 to 50)"
    atom_selection2: "chain A and (resid 51 to 100)"
    atom_selection3: "chain A and (resid 101 to 150)"
    atom_selection4: "chain A and (resid 151 to 224)"
    start_sigma: 99999999
    stop_sigma: -1
    move: "1,4"
    weight: 1.0
    harmonic:
      target_dihedral: 180.0
conformer_restraints:
  B: true                      # opt the ligand (chain B) into conformer restraints
conformer_restraints_config:
  start_sigma: 99999999
  stop_sigma: -1
  bond: {weight: 1.0, slack: 0.0}
  angle: {weight: 1.0, slack: 0.0}
  chiral: {weight: 1.0, slack: 0.05}
  cistrans: {weight: 1.0, slack: 0.0}
  vdw: {weight: 1.0}
rmsd_restraints_config:
  - ref_pdb: rmsd_ref.pdb
    target_rmsd: 0.0
    weight: 1.0
    start_sigma: 99999999
    stop_sigma: 1.0
    pairing: align
    best_effort: true
    atom_selection_ref_fit: "chain A and (resid 5 to 220)"
    atom_selection_target_fit: "chain A and (resid 5 to 220)"
    atom_selection_ref_calc: "chain A and (resid 90 to 180)"
    atom_selection_target_calc: "chain A and (resid 90 to 180)"
```

## How to run

### Run

Save as `run_restr_example.sh` and run it on a GPU machine (`bash run_restr_example.sh`). Note that
the FASTA and out_dir are **positional** arguments:

```bash
#!/bin/bash
# chai-lab RGI example runner. Run on a machine with a CUDA GPU.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

export PATH="$HOME/.local/bin:$PATH"
python -c "import yaml" 2>/dev/null || uv pip install pyyaml   # chai1.py imports yaml (not a chai dep)
export CHAI_DOWNLOADS_DIR="${CHAI_DOWNLOADS_DIR:-$HOME/.cache/chai}"

rm -rf out_restr_example
# fasta & out_dir are POSITIONAL args
python -m chai_lab.main fold \
    restr_example.fasta \
    out_restr_example \
    --restraints-config-path restr_example.yaml \
    --num-diffn-timesteps 200 --num-diffn-samples 2 --seed 0 \
    --use-msa-server --use-templates-server --no-use-esm-embeddings
```

## Verify

With `verbose: true`, `setup` logs `built spec: n_active=.. bonds=.. angles=.. chirals=..
distances=.. rmsd=.. group_angle=.. group_dihedral=..` — confirm the counts are non-zero for what
you requested. Cross-check with the workspace helpers (chai's venv has gemmi/rdkit):
`../check_dist.py <pred.cif>` (centroid distance vs 25 Å) and `../check_conf.py <pred.cif> GLN`
(ligand geometry).
