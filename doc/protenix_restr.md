# Protenix — Restraint-Guided Inference (RGI)

[Documentation index](README.md) · [Configuration reference](config.md)

Protenix + [`rgi_utils`](https://github.com/cddlab/rgi_utils) restraint-guided inference. Full
`restraints_config` schema & atom-selection DSL: [`config.md`](config.md).

> **Or generate it automatically:** the `generate-rgi-config` skill in Claude Code
> (`/generate-rgi-config`) or Codex (`$generate-rgi-config`) interviews you about the goal and
> writes a validated `restraints_config` where this tool expects it. Use it when hand-writing the
> full config below is unnecessary.

## Installation

The RGI code lives in the `cddlab/protenix_restr` fork — install **that fork**, not the upstream
PyPI `protenix`, which has no RGI hooks.

```bash
git clone https://github.com/cddlab/protenix_restr.git
cd protenix_restr
uv venv && source .venv/bin/activate           # Python 3.11+
uv pip install -e .                             # also pulls the rgi_utils engine (declared in requirements.txt)
```

> For co-development of the engine, override the pinned dependency with a local editable
> checkout in a SEPARATE step: `uv pip install -e ../rgi_utils` (sibling clone).

> **Run protenix on sm_89 (e.g. RTX 4090), NOT on Blackwell (sm_120).** On Blackwell its
> cuequivariance fused kernels **silently** emit all-NaN coordinates (no crash, exit 0) even for a
> bare fold with no restraints. A NaN output is almost always this, not the restraint; confirm by
> re-running on an sm_89 GPU. Run on a machine with a CUDA GPU.

## Configuration

protenix reads RGI from a **`restraints_config` key nested inside each fold-input object** of the
input JSON (the input is a JSON *list* of fold jobs; the key sits beside `name`/`sequences`). Turn
restraints on with:

1. **Per sequence** — `"conformer_restraints": true` on each protein, DNA, RNA, or
   ligand object enables conformer restraints for only that chain.
2. **The `restraints_config` object** — the distance / angle / dihedral / conformer /
   RMSD restraints, plus config-only `custom` restraints (define your own — see config.md). The example below writes **every usable variable** with a concrete value; see
   [`config.md`](config.md) for what each does, the alternative restraint types
   (`flat-bottomed` etc.), and the RMSD `atom_selection_ref`/`atom_selection_target` shorthand.

`resid` is the **per-chain 1-based ordinal** (qualify protein groups with `chain A and (...)`).
There is **no top-level `start_sigma`**.

## Complete example (input JSON)

Save this as `restr_example.json`. Folds QBP + GLN **plus a short DNA duplex and an RNA duplex** with
a centroid distance, group angle, group dihedral, GLN conformer, whole-structure RMSD, a custom
(formula) restraint, and **Watson-Crick base pairs on the nucleic acids**, every variable spelled
out. The custom entry keeps both lobe-halves equidistant from the central domain — a difference of
two distances, which no single built-in can express (JSON has no comments, so the rationale lives
here in prose). protenix takes **no `id` field** on sequences — it assigns chain letters **by list
order**, so here the chains are protein **A**, ligand **B**, DNA strands **C**/**D**, RNA strands
**E**/**F** (the `base_pair` selectors below reference exactly those). Both duplex strands are
self-complementary palindromes (`GCATGC` / `GCAUGC`), so identical antiparallel strands pair. The run
command passes `--use_msa true`, so protenix runs its (ColabFold-compatible) MSA search.

```json
[
  {
    "name": "qbp_rgi_example",
    "sequences": [
      { "proteinChain": { "sequence": "ADKKLVVATDTAFVPFEFKQGDKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPALQTKNVDLALAGITITDERKKAIDFSDGYYKSGLLVMVKANNNDVKSVKDLDGKVVAVKSGTGSVDYAKANIKTKDLRQFPNIDNAYMELGTNRADAVLHDTPNILYFIKTAGNGQFKAVGDSLEAQQYGIAFPKGSDELRDKVNGALKTLRENGTYNEIYKKWFGTEPK", "count": 1, "conformer_restraints": true } },
      { "ligand": { "ligand": "CCD_GLN", "count": 1, "conformer_restraints": true } },
      { "dnaSequence": { "sequence": "GCATGC", "count": 1 } },
      { "dnaSequence": { "sequence": "GCATGC", "count": 1 } },
      { "rnaSequence": { "sequence": "GCAUGC", "count": 1 } },
      { "rnaSequence": { "sequence": "GCAUGC", "count": 1 } }
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
]
```

## Run

Save as `run_restr_example.sh` and run it on a GPU machine (`bash run_restr_example.sh`):

```bash
#!/bin/bash
# protenix RGI example runner. Run on an sm_89 CUDA GPU (Blackwell sm_120 emits silent all-NaN coords).
set -e
source .venv/bin/activate

protenix pred -i restr_example.json -o out_restr_example \
    --use_default_params true --use_msa true --seeds 0 --step 200 --sample 1 --cycle 4
```

## Verify results

With `verbose: true`, `setup` logs `built spec: n_active=.. bonds=.. angles=.. chirals=.. plane=..
cistrans=.. distances=.. rmsd=.. group_angle=.. group_dihedral=..` — confirm the counts are non-zero for what you requested.
Cross-check the result with the workspace helpers (any gemmi/rdkit venv): `../check_dist.py
<pred.cif>` (centroid distance vs 25 Å) and `../check_conf.py <pred.cif> GLN` (ligand geometry). If
the output is all-NaN, you almost certainly ran on Blackwell — re-run on an sm_89 GPU.
