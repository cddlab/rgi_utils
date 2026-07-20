"""Minimal CUDA smoke test for polymer conformer restraints."""

from __future__ import annotations

import numpy as np
import torch

from rgi_utils.atom_context import AtomRecord
from rgi_utils.combined import CombinedRestraints

NAMES = ["N", "CA", "C", "O", "CB"]
REFERENCE = np.array(
    [
        [-1.20, 0.60, 0.00],
        [0.00, 0.00, 0.00],
        [1.50, 0.10, 0.00],
        [2.10, 1.10, 0.00],
        [-0.10, -0.80, 1.20],
    ],
    dtype=np.float32,
)


class Adapter:
    def iter_atoms(self):
        for resid in (1, 2):
            for local, name in enumerate(NAMES):
                yield AtomRecord(
                    chain="A",
                    resid=resid,
                    index=(resid - 1) * len(NAMES) + local,
                    name=name,
                    mol_type="protein",
                    resname="ALA",
                )

    def get_elements(self):
        return np.array([7, 6, 6, 8, 6] * 2)

    def get_reference_positions(self):
        return np.concatenate([REFERENCE, REFERENCE])

    def get_reference_space_uid(self):
        return np.repeat(np.arange(2), len(NAMES))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    restraints = CombinedRestraints()
    restraints.setup(
        Adapter(),
        config={
            "gpu": True,
            "max_iter": 100,
            "conformer_restraints_config": {
                "polymer_types": ["protein"],
                "bond": {},
                "angle": {},
                "chiral": {},
                "vdw": {"max_neighbors": 8},
            },
        },
    )
    coords = np.concatenate([REFERENCE, REFERENCE + np.array([5.0, 0.0, 0.0])])
    coords = torch.as_tensor(coords, device="cuda")
    before = abs(float(torch.linalg.vector_norm(coords[2] - coords[5])) - 1.329)
    restraints.minimize(coords, istep=0, sigma=100.0)
    after = abs(float(torch.linalg.vector_norm(coords[2] - coords[5])) - 1.329)
    assert bool(torch.isfinite(coords).all())
    assert after < before
    print(
        f"PASS device={torch.cuda.get_device_name()} peptide_error={before:.4f}->{after:.4f}"
    )


if __name__ == "__main__":
    main()
