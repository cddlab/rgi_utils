# RGI documentation

Use the tool-specific guide for installation, config placement, and the run command. Use the
shared [`restraints_config` reference](config.md) for restraint semantics and defaults. See the
[FAQ](FAQ.md) for common failure modes and troubleshooting guidance.

## Tool guides

| Predictor         | Backend | Config placement                                              | Guide                              |
| ----------------- | ------- | ------------------------------------------------------------- | ---------------------------------- |
| Boltz-1 / Boltz-2 | PyTorch | `restraints_config` in the input YAML                         | [Boltz](boltz_restr.md)            |
| AlphaFold 3       | JAX     | `restraints_config` in the fold-input JSON                    | [AlphaFold 3](alphafold3_restr.md) |
| Protenix v1 / v2  | PyTorch | `restraints_config` in each fold-input JSON object            | [Protenix](protenix_restr.md)      |
| ESMFold2          | PyTorch | Python dict passed to `fold()`                                | [ESMFold2](esmfold2_restr.md)      |
| OpenFold 3        | PyTorch | `queries.<name>.restraints_config` in the input JSON          | [OpenFold 3](openfold-3_restr.md)  |
| Chai-1            | PyTorch | Top-level sidecar YAML passed with `--restraints-config-path` | [Chai-1](chai-lab_restr.md)        |
| OpenDDE v1        | PyTorch | `restraints_config` in each fold-input JSON object            | [OpenDDE](opendde_restr.md)        |

## Recommended workflow

1. Open the guide for your predictor and follow its installation and config-placement rules.
2. Define the restraints using the [configuration reference](config.md).
3. Validate the input with the repository's `generate-rgi-config` skill before submitting a GPU
   job.
4. Run with `verbose: true` and verify that every requested restraint has a non-zero count in the
   `built spec:` log.

## Conventions shared by every tool

- `resid` is the per-chain, 1-based residue or token ordinal. It is not an author residue number
  or a global index. Qualify residue selections with `chain`.
- Sigma and step activation windows are alternatives. Do not set both on the same restraint.
- Conformer terms require both `conformer_restraints_config` and a per-molecule opt-in. The opt-in
  location differs by tool.
- The backend is inferred from the predictor. `gpu` selects the PyTorch device and is inert for
  AlphaFold 3's JAX path.
- Config validation checks schema and selection syntax, but it cannot prove that a selection
  matches the intended atoms. The runtime `built spec:` counts provide that check.
