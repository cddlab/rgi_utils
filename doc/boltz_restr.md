# boltz_restr — Restraint-Guided Inference (RGI)

Boltz-1/2 + [`rgi_utils`](https://github.com/cddlab/rgi_utils) restraint-guided inference. Full
`restraints_config` schema & atom-selection DSL: [`config.md`](config.md).

## Install

The RGI code lives in the `cddlab/boltz_restr` fork (branch `rgi-integration`) — install **that
fork**, not the upstream PyPI `boltz`, which has no RGI hooks. Run on a machine with a CUDA GPU
(RTX 4090 / sm_89 works).

```bash
git clone -b rgi-integration https://github.com/cddlab/boltz_restr.git
cd boltz_restr
uv venv && source .venv/bin/activate           # Python 3.11+
uv pip install -e ".[cuda]"
# the shared engine (torch extra); PyPI default torch is +cpu, the cuda extra above pins CUDA
uv pip install "rgi_utils[torch] @ git+https://github.com/cddlab/rgi_utils.git@rgi-integration"
```

> For co-development of the engine, install it editable from a sibling checkout instead of the
> git URL: `uv pip install -e ../rgi_utils[torch]`.

## Configuring restraints

boltz reads RGI from a **top-level `restraints_config:` key nested inside the input YAML** (the
same YAML that lists the sequences). Two things turn restraints on:

1. **Per ligand** — add `conformer_restraints: true` next to the ligand's `ccd`/`smiles` to enable
   the bond/angle/chiral/cistrans/VdW conformer restraints for that ligand.
2. **The `restraints_config:` block** — the distance / angle / dihedral / conformer /
   RMSD restraints, plus config-only `custom` restraints (define your own — see config.md). The example below writes **every usable variable** with a concrete value; see
   [`config.md`](config.md) for what each does, the alternative restraint types (`flat-bottomed`
   etc.), and the RMSD `atom_selection_ref`/`atom_selection_target` shorthand.

Key conventions (identical across all tools): `resid` is the **per-chain 1-based ordinal** (resets
at each chain; a ligand atom gets its own ordinal), so qualify protein groups with `chain A and
(...)` to avoid sweeping in the ligand. There is **no top-level `start_sigma`** — it is set per
distance/RMSD/group entry and once for all conformer terms.

## Full config (input file)

Save this as `restr_example.yaml`. It folds QBP (glutamine-binding protein) with its natural ligand
GLN and the full RGI restraint set — centroid distance, group angle, group dihedral, GLN conformer,
and whole-structure RMSD — with every variable spelled out. The run command passes
`--use_msa_server`, so boltz fetches the MSA from the ColabFold server.

```yaml
sequences:
  - protein:
      id: [A]
      sequence: ADKKLVVATDTAFVPFEFKQGDKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPALQTKNVDLALAGITITDERKKAIDFSDGYYKSGLLVMVKANNNDVKSVKDLDGKVVAVKSGTGSVDYAKANIKTKDLRQFPNIDNAYMELGTNRADAVLHDTPNILYFIKTAGNGQFKAVGDSLEAQQYGIAFPKGSDELRDKVNGALKTLRENGTYNEIYKKWFGTEPK
  - ligand:
      id: [B]
      ccd: GLN
      conformer_restraints: true

restraints_config:
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
```

## How to run

### Run

Save this as `run_restr_example.sh` and run it on a GPU machine (`bash run_restr_example.sh`):

```bash
#!/bin/bash
# boltz RGI example runner. Run on a machine with a CUDA GPU.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

rm -rf out_restr_example
boltz predict restr_example.yaml \
    --seed 0 --out_dir out_restr_example --model boltz2 --use_msa_server
```

## Verify

With `verbose: true`, the `setup` log prints `built spec: n_active=.. bonds=.. angles=.. chirals=..
distances=.. rmsd=.. group_angle=.. group_dihedral=.. ...` — confirm the counts are non-zero for
what you requested (a `finalize` term reading `0.00000` because the spec has 0 of that restraint is
a silent no-op, not "satisfied"). The workspace root carries helper scripts (run with any
gemmi/rdkit-enabled venv):

```bash
.venv/bin/python ../check_dist.py out_restr_example/**/*.cif    # centroid dist of the two groups vs 25 Å
.venv/bin/python ../check_conf.py out_restr_example/**/*.cif GLN # ligand bond/angle RMS vs RDKit ideal
```
