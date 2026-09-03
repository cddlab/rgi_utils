"""ESMFold2 RGI example -- dual-ref RMSD morph -> midpoint of 1GGG(open)/1WDN(closed), target 3.0 A.

Single-sequence fold (no MSA). Build/activate the esm_restr pixi env first; run via run.sh.
"""

from __future__ import annotations

import os

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    ProteinInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

SEQUENCE = "ADKKLVVATDTAFVPFEFKQGDKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPALQTKNVDLALAGITITDERKKAIDFSDGYYKSGLLVMVKANNNDVKSVKDLDGKVVAVKSGTGSVDYAKANIKTKDLRQFPNIDNAYMELGTNRADAVLHDTPNILYFIKTAGNGQFKAVGDSLEAQQYGIAFPKGSDELRDKVNGALKTLRENGTYNEIYKKWFGTEPK"  # noqa: E501

RESTRAINTS_CONFIG = {
    "verbose": True,
    "gpu": True,
    "max_iter": 1000,
    "method": "CG",
    "rmsd_restraints_config": [
        {
            "ref_cif": "1GGG.cif",
            "atom_selection_ref_fit": "chain A and name CA",
            "atom_selection_target_fit": "chain A and name CA",
            "atom_selection_ref_calc": "chain A and name CA",
            "atom_selection_target_calc": "chain A and name CA",
            "pairing": "align",
            "start_sigma": 99999999,
            "stop_sigma": 1.0,
            "harmonic": {"target_rmsd": 3.0},
        },
        {
            "ref_cif": "1WDN.cif",
            "atom_selection_ref_fit": "chain A and name CA",
            "atom_selection_target_fit": "chain A and name CA",
            "atom_selection_ref_calc": "chain A and name CA",
            "atom_selection_target_calc": "chain A and name CA",
            "pairing": "align",
            "start_sigma": 99999999,
            "stop_sigma": 1.0,
            "harmonic": {"target_rmsd": 3.0},
        },
    ],
}


def main() -> None:
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda()
    model.train(False)

    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=SEQUENCE)])

    result = ESMFold2InputBuilder().fold(
        model,
        spi,
        num_loops=20,
        num_sampling_steps=200,
        seed=0,
        restraints_config=RESTRAINTS_CONFIG,
    )
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_rmsd.cif")
    with open(out, "w") as fh:
        fh.write(result.complex.to_mmcif())
    print("wrote", out)


if __name__ == "__main__":
    main()
