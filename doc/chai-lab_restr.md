# chai-lab_restr — Restraint-Guided Inference (RGI)

Chai-1 + [`rgi_utils`](https://github.com/cddlab/rgi_utils) restraint-guided inference. Full
`restraints_config` schema & atom-selection DSL: [`config.md`](config.md).

> **Or generate it automatically:** the `generate-rgi-config` skill in Claude Code (`/generate-rgi-config`) or Codex (`$generate-rgi-config`) interviews you about the goal and writes a validated `restraints_config` placed where this tool expects it — handy when hand-writing the config below is more than you need.

## Install

The RGI code lives in the `cddlab/chai-lab_restr` fork — install **that fork**, not the upstream
PyPI `chai_lab`, which has no RGI hooks. Run on a CUDA GPU with bfloat16 support (RTX 4090 / sm_89
works).

```bash
git clone https://github.com/cddlab/chai-lab_restr.git
cd chai-lab_restr
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e .                                                                   # also pulls the rgi_utils engine (declared in requirements.in)
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124  # PyPI default is +cpu!
uv pip install pyyaml                                                                 # chai1.py imports yaml, not a chai dep
```

> For co-development of the engine, override the pinned dependency with a local editable
> checkout in a SEPARATE step: `uv pip install -e ../rgi_utils` (sibling clone).

## Configuring restraints

chai is the **odd one out**: the file you pass to `--restraints-config-path` **IS** the
`restraints_config` dict at top level — do **not** nest it under a `restraints_config:` key (that is
the boltz/protenix style). chai loads it verbatim with `yaml.safe_load`. The sequences live in a
separate FASTA (the ligand is a SMILES string).

Chai's FASTA cannot carry the flag, so conformer opt-in lives **in the sidecar** as a
`conformer_restraints` map keyed by chain id. Set each protein, DNA, RNA, or ligand
chain independently; absent/false chains remain unrestrained. Chai drops
intra-ligand bond orders at every layer, so the adapter rebuilds the molecule from the source SMILES
(Kekulized → correct valence + aromaticity + stereo) — bond/angle/chiral/plane/cistrans all apply.

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
method: "CG"
max_iter: 1000
distance_restraints_config:
  - atom_selection1: "chain A and ((resid 5 to 84) or (resid 186 to 224))"
    atom_selection2: "chain A and (resid 90 to 180)"
    start_sigma: 99999999
    stop_sigma: -1
    move: both
    weight: 1.0            # no-op for a lone restraint; balances over-constrained coupling only
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
  A: true                      # protein chain
  B: true                      # ligand chain
conformer_restraints_config:
  start_sigma: 99999999
  stop_sigma: -1
  bond: {weight: 1.0, slack: 0.0}
  angle: {weight: 1.0, slack: 0.0}
  chiral: {weight: 1.0, slack: 0.05}
  plane: {weight: 1.0}   # best-fit-plane over rings + sp2 groups (opt-in); GLN's amide + carboxyl groups -> plane=2
  cistrans: {weight: 1.0, slack: 0.0}
  vdw: {weight: 1.0}
rmsd_restraints_config:
  - ref_pdb: rmsd_ref.pdb
    harmonic: {target_rmsd: 0.0}
    weight: 1.0
    start_sigma: 99999999
    stop_sigma: 1.0
    pairing: align
    best_effort: true
    atom_selection_ref_fit: "chain A and (resid 5 to 220)"
    atom_selection_target_fit: "chain A and (resid 5 to 220)"
    atom_selection_ref_calc: "chain A and (resid 90 to 180)"
    atom_selection_target_calc: "chain A and (resid 90 to 180)"
custom_restraints_config:
  # Define your OWN restraint as a formula (no Python). This one keeps both
  # lobe-halves (L1, L2) equidistant from the central domain (H) — a difference of
  # two distances, which no single built-in restraint can express. See config.md
  # for the full vocabulary (distance/angle/dihedral/rg/...) and the code path.
  - name: equidistant
    energy: "(distance(L1, H) - distance(L2, H))**2"
    selections:
      L1: "chain A and (resid 5 to 84)"
      L2: "chain A and (resid 186 to 224)"
      H: "chain A and (resid 90 to 180)"
    start_sigma: 99999999
    stop_sigma: -1
    weight: 1.0
```

## How to run

### Run

Save as `run_restr_example.sh` and run it on a GPU machine (`bash run_restr_example.sh`). Note that
the FASTA and out_dir are **positional** arguments:

```bash
#!/bin/bash
# chai-lab RGI example runner. Run on a machine with a CUDA GPU.
set -e
source .venv/bin/activate

export PATH="$HOME/.local/bin:$PATH"
python -c "import yaml" 2>/dev/null || uv pip install pyyaml   # chai1.py imports yaml (not a chai dep)
export CHAI_DOWNLOADS_DIR="${CHAI_DOWNLOADS_DIR:-$HOME/.cache/chai}"

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
plane=.. cistrans=.. distances=.. rmsd=.. group_angle=.. group_dihedral=..` — confirm the counts are non-zero for what
you requested. Cross-check with the workspace helpers (chai's venv has gemmi/rdkit):
`../check_dist.py <pred.cif>` (centroid distance vs 25 Å) and `../check_conf.py <pred.cif> GLN`
(ligand geometry).
