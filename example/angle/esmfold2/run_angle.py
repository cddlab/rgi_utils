"""ESMFold2 RGI example -- group-centroid angle -> 72.85 deg (ADK NMP-CORE-LID).

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

SEQUENCE = "MRIILLGAPGAGKGTQAQFIMEKYGIPQISTGDMLRAAVKSGSELGKQAKDIMDAGKLVTDELVIALVKERIAQEDCRNGFLLDGFPRTIPQADAMKEAGINVDYVLEFDVPDELIVDRIVGRRVHAPSGRVYHVKFNPPKVEGKDDVTGEELTTRKDDQEETVRKRLVEYHQMTAPLIGYYSKEAEAGNTKYAKVDGTKPVAEVRADLEKILG"  # noqa: E501

RESTRAINTS_CONFIG = {
    "verbose": True,
    "gpu": True,
    "max_iter": 100,
    "method": "CG",
    "angle_restraints_config": [
        {
            "atom_selection1": "(resid 30 to 59)",
            "atom_selection2": "(resid 1 to 29) or (resid 60 to 121) or "
            "(resid 160 to 214)",
            "atom_selection3": "(resid 122 to 159)",
            "start_sigma": 99999999,
            "harmonic": {"target_angle": 72.85},
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
        num_loops=3,
        num_sampling_steps=200,
        seed=0,
        restraints_config=RESTRAINTS_CONFIG,
    )
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_angle.cif")
    with open(out, "w") as fh:
        fh.write(result.complex.to_mmcif())
    print("wrote", out)


if __name__ == "__main__":
    main()
