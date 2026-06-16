# esmfold2_restr — Restraint-Guided Inference (RGI)

ESMFold2 + [`rgi_utils`](https://github.com/cddlab/rgi_utils) restraint-guided inference. Full
`restraints_config` schema & atom-selection DSL: [`config.md`](config.md).

ESMFold2 (ESM3-based) folds in **single-sequence mode** (a language-model folder — no MSA, hence no
MSA-server option).

ESMFold2 spans **two** repos — install both on `rgi-integration`:
- **`transformers_restr`** — the model + diffusion loop
  (`src/transformers/models/esmfold2/modeling_esmfold2_common.py`, where the per-step
  `restraints.minimize` hook lives).
- **`esm_restr`** — the user API `ESMFold2InputBuilder.fold` (`esm/models/esmfold2/`), which builds
  the adapter + `CombinedRestraints` and threads restraints through `forward()` into `sample()`.

## Install

ESMFold2 uses a **pixi** environment. Install the **local** `transformers_restr` fork + `rgi_utils`
editable so they override the pyproject git deps. Clone all three repos as siblings. Run on a CUDA
GPU (RTX 4090 / sm_89; this pixi env's torch is cu124, no Blackwell sm_120 kernels).

```bash
git clone -b rgi-integration https://github.com/cddlab/esm_restr.git
git clone -b rgi-integration https://github.com/cddlab/transformers_restr.git   # sibling
git clone -b rgi-integration https://github.com/cddlab/rgi_utils.git            # sibling
cd esm_restr
PIXI="${PIXI:-$([ -x ../.pixi-bin/pixi ] && echo ../.pixi-bin/pixi || echo pixi)}"

"$PIXI" install
"$PIXI" run python -m pip install -e ../transformers_restr -e "../rgi_utils[torch]"

# CRITICAL — `pip install -e ../transformers_restr` does NOT win over the real `transformers`
# dir pixi already put in site-packages (same package name → no editable finder), so the STALE
# copy WITHOUT the per-step `restraints.minimize` hook imports and the restraint silently never
# runs (rgi_utils IS editable, so setup/finalize still log — it looks like an unrestrained
# "stall", not an error). Copy the esmfold2 model dir over the installed one, then assert the hook:
SP=$("$PIXI" run python -c "import transformers,os;print(os.path.dirname(transformers.__file__))")
cp -rf ../transformers_restr/src/transformers/models/esmfold2/. "$SP/models/esmfold2/"
"$PIXI" run python -c "import inspect; from transformers.models.esmfold2 import modeling_esmfold2_common as m; assert 'restraints.minimize' in inspect.getsource(m), 'esmfold2 hook missing — copy-over failed'"
```

To use Blackwell (sm_120): add a cu128 `[tool.pixi.pypi-options]` extra-index + `pixi update torch`
and remove the cu124-pinned `cuequivariance` (esmfold2 falls back to pure torch).

## Configuring restraints

ESMFold2's API is **Pythonic**: `restraints_config` is a plain **Python dict** passed to
`ESMFold2InputBuilder().fold(model, spi, restraints_config=...)` — not a YAML/JSON sidecar. The dict
schema is identical to the other tools.

A ligand = one token/atom, so `token_bonds` carries intra-ligand connectivity and bond ORDERS ride
on `ChainInfo.ligand_bond_orders` (CCD via `get_ligand_ccd_bonds`, SMILES via Kekulized 3-tuples) —
so the conformer cistrans term works for both CCD and SMILES ligands.

The `RESTRAINTS_CONFIG` dict below writes **every usable variable** with a concrete value (distance
/ angle / dihedral / conformer / RMSD); see [`config.md`](config.md) for the
alternatives (restraint types, RMSD `atom_selection` shorthand). `resid` is the **per-chain 1-based
ordinal** (qualify protein groups with `chain A and (...)`). There is **no top-level `start_sigma`**.

## Full config (Python script)

Save this as `restr_example.py`. Folds QBP + its GLN ligand with a centroid distance, group angle,
group dihedral, GLN conformer, and whole-structure RMSD restraint, every variable spelled out.

```python
"""ESMFold2 RGI (restraint-guided inference) example via rgi_utils."""

from __future__ import annotations

from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    LigandInput,
    ProteinInput,
    StructurePredictionInput,
)

QBP = (
    "ADKKLVVATDTAFVPFEFKQGDKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPALQTKNVDLALAGITITDERKK"
    "AIDFSDGYYKSGLLVMVKANNNDVKSVKDLDGKVVAVKSGTGSVDYAKANIKTKDLRQFPNIDNAYMELGTNRADAV"
    "LHDTPNILYFIKTAGNGQFKAVGDSLEAQQYGIAFPKGSDELRDKVNGALKTLRENGTYNEIYKKWFGTEPK"
)

RESTRAINTS_CONFIG = {
    "verbose": True,
    "gpu": True,
    "backend": "torch",
    "method": "CG",
    "max_iter": 1000,
    "distance_restraints_config": [
        {
            "atom_selection1": "chain A and ((resid 5 to 84) or (resid 186 to 224))",
            "atom_selection2": "chain A and (resid 90 to 180)",
            "start_sigma": 99999999,
            "stop_sigma": -1,
            "move": "both",
            "harmonic": {"target_distance": 25.0},
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
            "harmonic": {"target_angle": 90.0},
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
            "harmonic": {"target_dihedral": 180.0},
        }
    ],
    "conformer_restraints_config": {
        "start_sigma": 99999999,
        "stop_sigma": -1,
        "bond": {"weight": 1.0, "slack": 0.0},
        "angle": {"weight": 1.0, "slack": 0.0},
        "chiral": {"weight": 1.0, "slack": 0.05},
        "cistrans": {"weight": 1.0, "slack": 0.0},
        "vdw": {"weight": 1.0},
    },
    "rmsd_restraints_config": [
        {
            "ref_pdb": "rmsd_ref.pdb",
            "target_rmsd": 0.0,
            "weight": 1.0,
            "start_sigma": 99999999,
            "stop_sigma": 1.0,
            "pairing": "align",
            "best_effort": True,
            "atom_selection_ref_fit": "chain A and (resid 5 to 220)",
            "atom_selection_target_fit": "chain A and (resid 5 to 220)",
            "atom_selection_ref_calc": "chain A and (resid 90 to 180)",
            "atom_selection_target_calc": "chain A and (resid 90 to 180)",
        }
    ],
}


def main() -> None:
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda()
    model.train(False)  # inference / eval mode

    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence=QBP),
            LigandInput(id="B", ccd=["GLN"]),  # glutamine — QBP's natural ligand
        ]
    )

    result = ESMFold2InputBuilder().fold(
        model,
        spi,
        num_loops=3,
        num_sampling_steps=200,
        seed=0,
        restraints_config=RESTRAINTS_CONFIG,
    )
    with open("out_esm.cif", "w") as fh:
        fh.write(result.complex.to_mmcif())
    print("wrote out_esm.cif")


if __name__ == "__main__":
    main()
```

## How to run

### Run

Save as `run_restr_example.sh` and run it on a GPU machine (`bash run_restr_example.sh`). Note the
**copy-over + assert** after the editable install — without it the per-step hook is silently absent
and restraints never apply (the run looks like an unrestrained stall, not an error):

```bash
#!/bin/bash
# ESMFold2 RGI example runner (pixi env). Run on an sm_89 CUDA GPU (RTX 4090): the pixi
# env's torch is cu124 (no Blackwell sm_120 kernels).
set -e
cd "$(dirname "$0")"

PIXI="${PIXI:-$([ -x ../.pixi-bin/pixi ] && echo ../.pixi-bin/pixi || echo pixi)}"

# Build the env + install the LOCAL transformers fork (esmfold2 with the RGI hook) and rgi_utils.
"$PIXI" install
"$PIXI" run python -m pip install -e ../transformers_restr -e "../rgi_utils[torch]"

# CRITICAL: copy the esmfold2 model dir over the installed transformers, else the stale copy
# WITHOUT the per-step `restraints.minimize` hook imports and the restraint silently never runs.
SP=$("$PIXI" run python -c "import transformers,os;print(os.path.dirname(transformers.__file__))")
cp -rf ../transformers_restr/src/transformers/models/esmfold2/. "$SP/models/esmfold2/"
"$PIXI" run python -c "import inspect; from transformers.models.esmfold2 import modeling_esmfold2_common as m; assert 'restraints.minimize' in inspect.getsource(m), 'esmfold2 hook missing'"

"$PIXI" run python restr_example.py
```

## Verify

With `verbose: True`, the `setup` log prints `built spec: n_active=.. bonds=.. ... distances=..
rmsd=.. group_angle=.. group_dihedral=..` — confirm the counts are non-zero. The esm pixi env has no
gemmi, so run the centroid check with another tool's venv: `../chai-lab_restr/.venv/bin/python
../check_dist.py out_esm.cif`. If setup/finalize log but the structure is unchanged (an unrestrained
"stall"), the copy-over step above was skipped — re-run it and assert the hook is present.
