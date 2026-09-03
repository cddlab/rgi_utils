"""ESMFold2 RGI example -- centroid distance -> 25.0 A (QBP).

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
    "max_iter": 100,
    "method": "CG",
    "distance_restraints_config": [
        {
            "atom_selection1": "(resid 4 to 83) or (resid 185 to 223)",
            "atom_selection2": "(resid 89 to 179)",
            "start_sigma": 99999999,
            "harmonic": {"target_distance": 25.0},
        }
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
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_distance.cif")
    with open(out, "w") as fh:
        fh.write(result.complex.to_mmcif())
    print("wrote", out)


if __name__ == "__main__":
    main()
