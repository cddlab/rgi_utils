# Adapter protocol

The adapter is the bridge between a tool's data structures and rgi_utils.
rgi_utils never imports your framework — it only calls the methods below and
consumes the records they yield.

## Records (from `rgi_utils.atom_context`)

- `AtomRecord(chain: str, resid: int, index: int)` — one real (non-padding) atom.
  - `chain`: chain id/name; matches the selection DSL token `chain A`.
  - `resid`: **1-based residue/token ordinal WITHIN the chain** (resets to 1 at
    each chain). This cross-tool convention makes `chain B and resid 5` select
    the same atom in every tool — it is NOT a cumulative/global token index.
  - `index`: **global flat index** of this atom in the coordinate tensor handed
    to `minimize` (i.e. its row after reshaping coords to `(n_atom, 3)`).
- `LigandConf(mol, conf_coords, global_indices, invert_chirality=False)`
  - `mol`: RDKit mol of the ligand heavy atoms (`Chem.RemoveHs` if it has Hs).
    Its bonds drive bond/angle restraints; its chiral tags drive chiral volumes.
  - `conf_coords`: `(n_atoms, 3)` reference coordinates — the ideal geometry to
    pull toward — in `mol` atom order.
  - `global_indices`: `(n_atoms,)` global flat indices of those atoms.

## Methods

| method | needed for | returns |
|---|---|---|
| `iter_atoms()` | distance restraints | `Iterator[AtomRecord]` over every real atom |
| `num_atoms()` | conformer / VdW | padded atom count (`int`) |
| `get_elements()` | dynamic ligand-protein VdW | `(num_atoms,)` atomic numbers, padding → 0 |
| `iter_ligand_confs()` | conformer / VdW | `Iterator[LigandConf]`, one per ligand |

`setup()` calls `iter_atoms` (through `DistanceData.resolve_sites`),
`iter_ligand_confs`, and `get_elements` as needed. Omitting an optional method
just disables that feature (e.g. no `iter_ligand_confs` → distance-only tool).

## Where the adapter lives

- **PyTorch tool**: the adapter can live in `rgi_utils/<tool>/adapter.py`,
  because it receives a plain dict/array and imports no framework code (keeps
  the dependency direction rgi_utils → nothing). boltz and protenix do this.
- **JAX tool**: put the adapter *in the tool* (e.g.
  `src/<tool>/.../restraints/adapter.py`). JAX tools usually need
  framework-specific machinery (CCD lookup, atom-name decode) that would invert
  the dependency if placed in rgi_utils. AF3 does this.

## Worked example 1 — boltz (reads a feats dict, batch 0)

```python
class BoltzFeatsAdapter:
    def __init__(self, feats): self.feats = feats
    def iter_atoms(self):
        # asym_id per atom + atom->token map give chain + per-chain resid;
        # global atom index is the row in the padded coordinate tensor.
        for chain in self.feats["record"][0].chains:
            sites = torch.where(asym_id_atom == chain.chain_id)[0].tolist()
            tok2resid = {t: i+1 for i, t in enumerate(sorted(set(tokens_of(sites))))}
            for gidx in sites:
                yield AtomRecord(chain.chain_name, tok2resid[token_of(gidx)], int(gidx))
    def iter_ligand_confs(self):
        # feats["ligand_mols"] = {asym_id: CCD RDKit mol} (tool exposes this)
        for chain in self.feats["record"][0].chains:
            mol = ligand_mols.get(chain.chain_id)
            if mol is None: continue
            mol = Chem.RemoveHs(mol)
            gidx = atoms_of_chain(chain.chain_id)         # global indices
            yield LigandConf(mol, mol.GetConformer().GetPositions(), gidx)
```
Key point: boltz had to *expose* `feats["ligand_mols"]` (a tool-specific data
publication) so the adapter could find each ligand's mol. That is a legitimate
tool-side change; the restraint logic still lives in rgi_utils.

## Worked example 2 — protenix (biotite AtomArray)

```python
class ProtenixAdapter:
    def __init__(self, feats): self.atom_array = feats["atom_array"]
    def iter_atoms(self):
        for i in range(len(self.atom_array)):
            yield AtomRecord(str(self.atom_array.label_asym_id[i]),
                             int(self.atom_array.res_id[i]), i)
    def iter_ligand_confs(self):
        # rebuild the ligand mol from the atom_array subset: elements + bonds +
        # 3D coords, then AssignStereochemistryFrom3D for chiral tags. mol atom i
        # corresponds to global_indices[i] by construction (no atom-order guesswork).
        for chain_id in hetero_chain_ids:
            idxs = atoms_of(chain_id)
            mol = build_rdkit_mol(elements[idxs], coords[idxs], bonds_local(idxs))
            yield LigandConf(mol, coords[idxs], idxs)
```

## Worked example 3 — AF3 (CCD-based batch, JAX; adapter in the tool)

AF3 coordinates are `(num_tokens, max_atoms_per_token, 3)`, so
`flat_idx = token_idx * max_atoms_per_token + within_token_idx`
(ligand atoms use `within = 0`).

```python
class AF3RestraintAdapter:
    def iter_atoms(self):
        # per-chain 1-based resid from asym_id; flat_idx = token*max + within
        for token_idx, aint in enumerate(self.token_asym_ids):
            for within in range(self.max_atoms_per_token):
                if not self.ref_mask[token_idx, within]: continue
                yield AtomRecord(self.asym_int_to_chain[int(aint)],
                                 per_chain_resid[token_idx],
                                 token_idx * self.max_atoms_per_token + within)
    def iter_ligand_confs(self):
        for chain in ligand_chains:
            mol = ccd_or_smiles_mol(chain)          # CCD-by-name or SMILES
            flat, kept = self._flat_indices(chain)  # decode ref_atom_name_chars
            if 0 < len(kept) < mol.GetNumAtoms():
                mol = subset_mol(mol, kept)          # drop CCD-only atoms (leaving)
            yield LigandConf(mol, ref_pos.reshape(-1,3)[flat], flat)
```
AF3-specific machinery that *must* stay in this adapter: CCD-by-name mol lookup,
the `ref_atom_name_chars` decode, the leaving-atom subset (see pitfalls), the
flat-index formula, and the per-chain resid counter. Everything downstream
(spec, energy, optim) is rgi_utils.

## Full source

The five complete adapters are the ground truth:
- `rgi_utils/src/rgi_utils/boltz/adapter.py` — feats dict + exposed `ligand_mols`
- `rgi_utils/src/rgi_utils/protenix/adapter.py` — biotite AtomArray (real bonds + coords)
- `rgi_utils/src/rgi_utils/chai/adapter.py` — reference conformer, NO bonds → `build_ligand_mol(perceive_bonds=True)`
- `rgi_utils/src/rgi_utils/openfold3/adapter.py` — AtomArray with zeroed coords → geometry from `ref_pos`; ligand by `molecule_type_id`
- `<af3>/src/alphafold3/model/restraints/adapter.py` — CCD-by-name, JAX, in-tool

The chai and openfold3 adapters are worth reading specifically for the "tool exposes
an incomplete ligand picture" cases (pitfalls 10–12).
