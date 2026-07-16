# openfold-3_restr — Restraint-Guided Inference (RGI)

OpenFold3-preview + [`rgi_utils`](https://github.com/cddlab/rgi_utils) restraint-guided inference.
Full `restraints_config` schema & atom-selection DSL: [`config.md`](config.md).

> **Or generate it automatically:** the `generate-rgi-config` skill in Claude Code (`/generate-rgi-config`) interviews you about the goal and writes a validated `restraints_config` placed where this tool expects it — handy when hand-writing the config below is more than you need.

## Install

The RGI code lives in the `cddlab/openfold-3_restr` fork — install **that fork**, not the upstream
PyPI `openfold3`. The `rgi_utils` engine is declared in `pixi.toml` (no `[torch]` extra — it uses the
conda torch, not a pip wheel), so `pixi install` pulls it automatically. Run on a CUDA GPU (RTX 4090
/ sm_89 works).

```bash
git clone https://github.com/cddlab/openfold-3_restr.git
cd openfold-3_restr
pixi install -e openfold3-cuda12        # builds the env + pulls rgi_utils (declared in pixi.toml)
printf '\n\n\nno\n' | pixi run -e openfold3-cuda12 setup_openfold   # fetch model params to ~/.openfold3
export OPENFOLD_CACHE="$HOME/.openfold3"
```

> For co-development of the engine, override with a local editable checkout AFTER pixi install:
> `pixi run -e openfold3-cuda12 python -m pip install -e ../rgi_utils` (sibling clone).

## Configuring restraints

OpenFold reads RGI from a **`restraints_config` field per query** in the input JSON
(`queries.<name>.restraints_config`). Two things turn conformer restraints on:

1. **Per ligand** — add `"conformer_restraints": true` to the ligand chain to enable its
   bond/angle/chiral/plane/cistrans/VdW conformer restraints. Without it that ligand is left
   unrestrained even when a `conformer_restraints_config` block is present.
2. **The `restraints_config` block** — the distance / angle / dihedral / conformer / RMSD
   restraints (below). Ligands are identified by `molecule_type_id == LIGAND` and accept `ccd_codes`.

The example below writes **every usable variable** with a concrete value (distance / angle /
dihedral / conformer / RMSD, plus config-only `custom`); see [`config.md`](config.md) for the alternatives
(restraint types, RMSD `atom_selection` shorthand). `resid` is the **per-chain 1-based ordinal**
(qualify protein groups with `chain A and (...)`). There is **no top-level `start_sigma`**.

## Full config (input file)

Save this as `restr_example.json`. Folds QBP + GLN with a centroid distance, group angle, group
dihedral, GLN conformer, whole-structure RMSD, and a custom (formula) restraint, every variable
spelled out. The custom entry keeps both lobe-halves equidistant from the central domain — a
difference of two distances, which no single built-in can express (JSON has no comments, so the
rationale lives here in prose). The run command passes `--use-msa-server true`, so OpenFold fetches
the MSA from the ColabFold server.

```json
{
  "queries": {
    "qbp_rgi_example": {
      "chains": [
        {
          "molecule_type": "protein",
          "chain_ids": ["A"],
          "sequence": "ADKKLVVATDTAFVPFEFKQGDKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPALQTKNVDLALAGITITDERKKAIDFSDGYYKSGLLVMVKANNNDVKSVKDLDGKVVAVKSGTGSVDYAKANIKTKDLRQFPNIDNAYMELGTNRADAVLHDTPNILYFIKTAGNGQFKAVGDSLEAQQYGIAFPKGSDELRDKVNGALKTLRENGTYNEIYKKWFGTEPK"
        },
        {
          "molecule_type": "ligand",
          "chain_ids": ["B"],
          "ccd_codes": "GLN",
          "conformer_restraints": true
        }
      ],
      "restraints_config": {
        "verbose": true,
        "gpu": true,
        "method": "CG",
        "max_iter": 1000,
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
  }
}
```

## How to run

### Run

Save as `run_restr_example.sh` and run it on a GPU machine (`bash run_restr_example.sh`):

```bash
#!/bin/bash
# openfold-3 RGI example runner (pixi env). Run on a machine with a CUDA GPU.
set -e

export OPENFOLD_CACHE="${OPENFOLD_CACHE:-$HOME/.openfold3}"

pixi run -e openfold3-cuda12 run_openfold predict \
    --query-json restr_example.json \
    --output-dir out_restr_example \
    --num-diffusion-samples 2 --use-msa-server true --use-templates false
```

## Verify

With `verbose: true`, the log prints `built spec: n_active=.. bonds=.. ... distances=.. rmsd=..
group_angle=.. group_dihedral=..` — confirm the counts are non-zero for what you requested.
Cross-check with the workspace helpers using a gemmi/rdkit-enabled venv: `../check_dist.py
<pred.cif>` (centroid distance vs 25 Å) and `../check_conf.py <pred.cif> GLN` (ligand geometry).
