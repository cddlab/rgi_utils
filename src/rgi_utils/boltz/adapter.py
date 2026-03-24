from __future__ import annotations

from typing import Iterator

import torch

from rgi_utils.atom_context import AtomRecord


class BoltzFeatsAdapter:
    """Adapter that converts boltz feats dict to the FrameworkAdapter protocol."""

    def __init__(self, feats: dict) -> None:
        self.feats = feats

    def iter_atoms(self) -> Iterator[AtomRecord]:
        """Iterate over all non-padded atoms, yielding AtomRecord for each."""
        feats = self.feats
        asym_id_token = feats["asym_id"]
        _atom_pad_mask = feats["atom_pad_mask"]
        atom_to_token = feats["atom_to_token"]
        record = feats["record"]

        # compute per-atom asym_id by matrix-multiplying atom_to_token with token
        # asym_ids
        asym_id_atom = torch.bmm(
            atom_to_token.float(), asym_id_token.unsqueeze(-1).float()
        ).squeeze(-1).long()
        # asym_id_atom shape: (batch, natoms) — filter by pad mask (batch=0 only)
        asym_id_atom_b0 = asym_id_atom[0]
        atom_to_token_b0 = atom_to_token[0]

        for chain in record[0].chains:
            chain_id = chain.chain_id
            chain_sites = torch.where(asym_id_atom_b0 == chain_id)[0].tolist()
            for global_padded_idx in chain_sites:
                token_idx = torch.argmax(atom_to_token_b0[global_padded_idx, :]).item()
                yield AtomRecord(
                    chain=chain.chain_name,
                    resid=int(token_idx) + 1,
                    index=int(global_padded_idx),
                )
