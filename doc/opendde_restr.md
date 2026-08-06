# OpenDDE — Restraint-Guided Inference (RGI)

[Documentation index](README.md) · [Configuration reference](config.md)

OpenDDE + [`rgi_utils`](https://github.com/cddlab/rgi_utils) restraint-guided inference. Full
`restraints_config` schema & atom-selection DSL: [`config.md`](config.md).

> **Or generate it automatically:** the `generate-rgi-config` skill in Claude Code
> (`/generate-rgi-config`) or Codex (`$generate-rgi-config`) interviews you about the goal and
> writes a validated `restraints_config` where this tool expects it. Use it when hand-writing the
> full config below is unnecessary.

## Installation

The RGI code lives in the `cddlab/OpenDDE_restr` fork — install **that fork**, not upstream
OpenDDE, which has no RGI hooks. The integration supports CPython 3.12 and 3.13 and has been
GPU-validated on an RTX 4090 (sm_89).

```bash
git clone https://github.com/cddlab/OpenDDE_restr.git
cd OpenDDE_restr
git switch rgi-integration
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install --torch-backend cu126 -e ".[gpu]"  # also pulls rgi_utils (declared in pyproject)
```

> For co-development of the engine, override the declared dependency with a local editable
> checkout in a SEPARATE step: `uv pip install -e ../rgi_utils` (sibling clone).

OpenDDE also needs its checkpoint and common runtime data under `OPENDDE_ROOT_DIR`; follow the
fork's `docs/inference_instructions.md` or run `scripts/download_opendde_data.sh`.

## Configuration

OpenDDE reads RGI from a **`restraints_config` key nested inside each fold-input object** of its
JSON list (beside `name`, `modelSeeds`, and `sequences`). Two things turn restraints on:

1. **Per sequence** — add `"conformer_restraints": true` to each protein, DNA, RNA, or ligand
   entity whose local geometry should be restrained. The default is false.
2. **The `restraints_config` object** — the distance / angle / dihedral / conformer / RMSD
   restraints, plus config-only `custom` restraints (define your own — see `config.md`). The
   example below writes every usable variable with a concrete value; see [`config.md`](config.md)
   for the alternative restraint types and RMSD selection shorthands.

Explicit `id` lists are recommended because OpenDDE otherwise assigns chain IDs. In the selection
DSL, `resid` is the **per-chain 1-based token ordinal** and resets for each chain. Qualify protein
groups with `chain A and (...)`. There is **no top-level `start_sigma`** — it is set per
distance/RMSD/group entry and once for all conformer terms.

## Complete example (input JSON)

Save this as `restr_example.json`. It folds QBP with its GLN ligand plus short DNA and RNA duplexes
and exercises a centroid distance, group angle, group dihedral, selected plane, GLN conformer,
whole-structure RMSD, a custom formula, and Watson-Crick base pairs. The custom term keeps both
lobe halves equidistant from the central domain. Both duplex sequences are self-complementary
palindromes, so the identical strands pair antiparallel. JSON has no comments, so the rationale is
given here rather than embedded in the file.

```json
[
  {
    "name": "qbp_rgi_example",
    "modelSeeds": [0],
    "sequences": [
      {
        "proteinChain": {
          "id": ["A"],
          "sequence": "ADKKLVVATDTAFVPFEFKQGDKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPALQTKNVDLALAGITITDERKKAIDFSDGYYKSGLLVMVKANNNDVKSVKDLDGKVVAVKSGTGSVDYAKANIKTKDLRQFPNIDNAYMELGTNRADAVLHDTPNILYFIKTAGNGQFKAVGDSLEAQQYGIAFPKGSDELRDKVNGALKTLRENGTYNEIYKKWFGTEPK",
          "count": 1
        }
      },
      {
        "ligand": {
          "id": ["B"],
          "ligand": "CCD_GLN",
          "count": 1,
          "conformer_restraints": true
        }
      },
      {
        "dnaSequence": {
          "id": ["C"],
          "sequence": "GCATGC",
          "count": 1
        }
      },
      {
        "dnaSequence": {
          "id": ["D"],
          "sequence": "GCATGC",
          "count": 1
        }
      },
      {
        "rnaSequence": {
          "id": ["E"],
          "sequence": "GCAUGC",
          "count": 1
        }
      },
      {
        "rnaSequence": {
          "id": ["F"],
          "sequence": "GCAUGC",
          "count": 1
        }
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
          "harmonic": {"target_distance": 25.0}
        }
      ],
      "base_pair_restraints_config": [
        {"residue1": "chain C and resid 1", "residue2": "chain D and resid 6"},
        {"residue1": "chain C and resid 3", "residue2": "chain D and resid 4"},
        {"residue1": "chain E and resid 1", "residue2": "chain F and resid 6"},
        {"residue1": "chain E and resid 3", "residue2": "chain F and resid 4"}
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
          "harmonic": {"target_angle": 90.0}
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
          "harmonic": {"target_dihedral": 180.0}
        }
      ],
      "plane_restraints_config": [
        {
          "atom_selection1": "chain A and (resid 5 to 20)",
          "start_sigma": 99999999,
          "stop_sigma": -1,
          "move": "all",
          "weight": 1.0,
          "flat-bottomed2": {"target_plane2": 0.1}
        }
      ],
      "conformer_restraints_config": {
        "start_sigma": 99999999,
        "stop_sigma": -1,
        "bond": {"weight": 1.0, "slack": 0.0},
        "angle": {"weight": 1.0, "slack": 0.0},
        "chiral": {"weight": 1.0, "slack": 0.05},
        "plane": {"weight": 1.0},
        "cistrans": {"weight": 1.0, "slack": 0.0},
        "vdw": {"weight": 1.0}
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
]
```

## Run

Save this as `run_restr_example.sh` and run it on a CUDA GPU. The example lets OpenDDE build an MSA
but disables templates and RNA MSA; change those flags to match the runtime data available locally.

```bash
#!/bin/bash
# OpenDDE RGI example runner. The integration is validated on an RTX 4090 (sm_89).
set -e
source .venv/bin/activate

opendde pred \
  -i restr_example.json \
  -o out_restr_example \
  -n opendde_v1 \
  --use_msa true \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 4
```

No RGI CLI flag is required. An input job without `restraints_config` follows the upstream path.
Native training-free guidance is independently enabled with `--use_tfg_guidance true`; when both
are active, each diffusion step applies native TFG first and RGI second.

## Verify results

With `verbose: true`, the setup log prints `built spec: n_active=.. bonds=.. angles=.. chirals=..
plane=.. cistrans=.. distances=.. rmsd=.. group_angle=.. group_dihedral=.. custom=..` — confirm
that every requested count is non-zero. A zero residual for a term whose count is zero is a silent
no-op, not evidence that the restraint was satisfied. Cross-check the result with the workspace
helpers in a gemmi/RDKit-enabled environment: `../check_dist.py <pred.cif>` measures the centroid
distance and `../check_conf.py <pred.cif> GLN` checks ligand geometry.

## Integration details

- `rgi_utils.opendde.OpenDDEAdapter` reads atom metadata and real bond orders from OpenDDE's
  stashed Biotite `AtomArray`, reference geometry from `ref_pos`, and the pre-expansion
  `residue_level_atom_to_token_idx` mapping when structural tokens are enabled.
- OpenDDE constructs a fresh `CombinedRestraints` for each structure and carries it through
  chunked sampling. The hook optimizes the denoised x0 estimate before the native Euler update.
- After RGI changes x0, the noisy anchor is rigidly aligned to the corrected estimate, preventing
  the native step scale from extrapolating the restraint displacement into geometry distortion.
- SMILES bond orders are Kekulized and preserved, enabling cis/trans and non-ring planar conformer
  terms. Native TFG and RGI remain composable in the same diffusion step.
