# esmfold2_restr — Restraint-Guided Inference (RGI)

ESMFold2 + [`rgi_utils`](https://github.com/cddlab/rgi_utils) restraint-guided inference. Full
`restraints_config` schema & atom-selection DSL: [`config.md`](config.md).

> **Or generate it automatically:** the `generate-rgi-config` skill in Claude Code (`/generate-rgi-config`) interviews you about the goal and writes a validated `restraints_config` placed where this tool expects it — handy when hand-writing the config below is more than you need.

ESMFold2 (ESM3-based) folds in **single-sequence mode** (a language-model folder — no MSA, hence no
MSA-server option).

ESMFold2 spans **two** repos — install both on `rgi-integration`:
- **`transformers_restr`** — the model + diffusion loop
  (`src/transformers/models/esmfold2/modeling_esmfold2_common.py`, where the per-step
  `restraints.minimize` hook lives).
- **`esm_restr`** — the user API `ESMFold2InputBuilder.fold` (`esm/models/esmfold2/`), which builds
  the adapter + `CombinedRestraints` and threads restraints through `forward()` into `sample()`.

## Install

ESMFold2 uses a **pixi** environment. Both engines are declared as dependencies in `esm_restr`'s
`pyproject.toml`: the **`transformers_restr`** fork (installed as the `transformers` package — it
carries the per-step `restraints.minimize` hook) and the `rgi_utils` engine. So `pixi install` pulls
both, with no extra steps. Run on a CUDA GPU (RTX 4090 / sm_89; this pixi env's torch is cu124, no
Blackwell sm_120 kernels).

```bash
git clone https://github.com/cddlab/esm_restr.git
cd esm_restr
pixi install                                     # pulls transformers_restr (hooked) + rgi_utils
```

To use Blackwell (sm_120): add a cu128 `[tool.pixi.pypi-options]` extra-index + `pixi update torch`
and remove the cu124-pinned `cuequivariance` (esmfold2 falls back to pure torch).

### Co-development

Clone `transformers_restr` / `rgi_utils` as siblings and override the git deps AFTER `pixi install`
(e.g. `pixi run python -m pip install -e ../rgi_utils`). Editing the **esmfold2 hook** is the one
catch: `pip install -e ../transformers_restr` does **not** win over the installed `transformers`
(same package name → no editable finder), so copy the model dir over and assert the hook:

```bash
SP=$(pixi run python -c "import transformers,os;print(os.path.dirname(transformers.__file__))")
cp -rf ../transformers_restr/src/transformers/models/esmfold2/. "$SP/models/esmfold2/"
pixi run python -c "import inspect; from transformers.models.esmfold2 import modeling_esmfold2_common as m; assert 'restraints.minimize' in inspect.getsource(m), 'esmfold2 hook missing — copy-over failed'"
```

## Configuring restraints

ESMFold2's API is **Pythonic**: `restraints_config` is a plain **Python dict** passed to
`ESMFold2InputBuilder().fold(model, spi, restraints_config=...)` — not a YAML/JSON sidecar. The dict
schema is identical to the other tools.

Conformer restraints are **per-ligand opt-in**: set `conformer_restraints=True` on a `LigandInput`
to enable its bond/angle/chiral/plane/cistrans/VdW restraints. A ligand left at the default (`False`) is
unrestrained even when a `conformer_restraints_config` block is present.

A ligand = one token/atom, so `token_bonds` carries intra-ligand connectivity and bond ORDERS ride
on `ChainInfo.ligand_bond_orders` (CCD via `get_ligand_ccd_bonds`, SMILES via Kekulized 3-tuples) —
so the conformer cistrans term (and the non-ring sp2 half of `plane`) work for both CCD and SMILES
ligands. `plane`'s ring half needs only ring topology + reference coplanarity, so it fires even without
bond orders.

The `RESTRAINTS_CONFIG` dict below writes **every usable variable** with a concrete value (distance
/ angle / dihedral / conformer / RMSD, plus config-only `custom`); see [`config.md`](config.md) for the
alternatives (restraint types, RMSD `atom_selection` shorthand). `resid` is the **per-chain 1-based
ordinal** (qualify protein groups with `chain A and (...)`). There is **no top-level `start_sigma`**.

## Full config (Python script)

Save this as `restr_example.py`. Folds QBP + its GLN ligand with a centroid distance, group angle,
group dihedral, GLN conformer, whole-structure RMSD, and a custom (formula) restraint, every
variable spelled out. Because ESMFold2's API is already Python, the custom **code path**
(`CombinedRestraints.add_custom(fn=...)`) is equally available — see config.md.

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
        "plane": {"weight": 1.0},  # best-fit-plane, rings + sp2 groups (opt-in); GLN -> plane=2
        "cistrans": {"weight": 1.0, "slack": 0.0},
        "vdw": {"weight": 1.0},
    },
    "rmsd_restraints_config": [
        {
            "ref_pdb": "rmsd_ref.pdb",
            "harmonic": {"target_rmsd": 0.0},
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
    # Define your OWN restraint as a formula (no Python beyond this dict). This one keeps
    # both lobe-halves (L1, L2) equidistant from the central domain (H) — a difference of
    # two distances, which no single built-in restraint can express. See config.md for the
    # full vocabulary and the code (ctx-function) path.
    "custom_restraints_config": [
        {
            "name": "equidistant",
            "energy": "(distance(L1, H) - distance(L2, H))**2",
            "selections": {
                "L1": "chain A and (resid 5 to 84)",
                "L2": "chain A and (resid 186 to 224)",
                "H": "chain A and (resid 90 to 180)",
            },
            "start_sigma": 99999999,
            "stop_sigma": -1,
            "weight": 1.0,
        }
    ],
}


def main() -> None:
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda()
    model.train(False)  # inference / eval mode

    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence=QBP),
            LigandInput(id="B", ccd=["GLN"], conformer_restraints=True),  # glutamine — QBP's natural ligand
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

Save as `run_restr_example.sh` and run it on a GPU machine (`bash run_restr_example.sh`). A fresh
`pixi install` ships the hooked transformers (the `transformers_restr` dep), so the runner is just
build-env + fold:

```bash
#!/bin/bash
# ESMFold2 RGI example runner (pixi env). Run on an sm_89 CUDA GPU (RTX 4090): the pixi
# env's torch is cu124 (no Blackwell sm_120 kernels).
set -e

pixi install                     # pulls transformers_restr (hooked) + rgi_utils
pixi run python restr_example.py
```

## Verify

With `verbose: True`, the `setup` log prints `built spec: n_active=.. bonds=.. ... distances=..
rmsd=.. group_angle=.. group_dihedral=..` — confirm the counts are non-zero. The esm pixi env has no
gemmi, so run the centroid check with another tool's venv: `../chai-lab_restr/.venv/bin/python
../check_dist.py out_esm.cif`. If setup/finalize log but the structure is unchanged (an unrestrained
"stall"), the imported `transformers` lacks the esmfold2 hook — confirm the `transformers_restr` dep
installed (a co-dev editable needs the copy-over from the Install section).
