# rgi_utils RGI examples

Minimal, ready-to-run Restraint-Guided Inference (RGI) samples: **4 restraint types x 7
predictors**. Each `<type>/<tool>/` (and `custom/dist-diff/<tool>/`) is one representative
restraint with a `run.sh`. Restraint blocks are the bench-rgi minimal configs; the runners
fetch the MSA from a server where supported; OpenDDE disables external feature searches and
AlphaFold3 uses its local data pipeline, as detailed below.

| type | system | restraint |
|---|---|---|
| `distance/` | QBP (226 aa) | centroid distance of two lobe groups -> 25.0 A |
| `angle/` | adenylate kinase (214 aa) | NMP-CORE-LID centroid angle -> 72.85 deg |
| `rmsd/` | QBP | dual-ref morph to the midpoint of 1GGG (open) / 1WDN (closed), target 3.0 A, released below sigma 1.0 |
| `custom/dist-diff/` | DgoT transporter (419 aa) | custom energy `(d(A,B) - d(C,D)) -> 0` (a difference of two centroid distances) |

Tools: `boltz-2`, `protenix-v2`, `opendde`, `alphafold3`, `openfold-3`, `chai`,
`esmfold2`.

## Run

```bash
bash distance/boltz-2/run.sh          # one example
```

Each `run.sh` locates the workspace root, activates/uses the matching fork's env, and folds.

## Prerequisites

- **The matching fork must exist as a sibling of `rgi_utils/`** (e.g. `../boltz_restr`,
  `../esm_restr`) with its venv/pixi env built. `run.sh` finds it automatically. See each
  fork's `rgi_utils/doc/<tool>.md` for install steps.
- **Run on a GPU compute node** — not a shared login node. GPU generations: RTX 4090 (sm_89)
  and Blackwell (sm_120).
- **protenix runs on sm_89 only** (Blackwell emits silent all-NaN). The default esm/chai torch
  is cu124 (sm_89) too.
- **OpenDDE examples disable MSA, template, and RNA-MSA searches**. Install its checkpoint and
  common runtime files first; the large search databases are unnecessary for these examples.
- **AlphaFold3 is the one non-self-contained example**: it has no ColabFold MSA server, so its
  `run.sh` runs the genetic-search data pipeline, requiring external `MODEL_DIR` (weights) and
  `DB_DIR` (sequence databases, hundreds of GB). A local no-DB fallback is documented inline in
  `alphafold3/*/run.sh`.

## Notes

- The `rmsd/` reference structures (`1GGG.cif`, `1WDN.cif`) are **not stored in the repo** --
  each `rmsd/*/run.sh` `wget`s them from RCSB into its own directory at run time (needs network
  on the compute node). They are byte-identical to the RCSB deposits.
- The `rmsd/` example deliberately restrains **nothing but the two RMSD terms** -- no polymer
  conformer restraint is layered on top, so the moved CA atoms leave the local geometry to the
  predictor. Adding one measurably WORSENS stereochemistry: a 2x2 ablation (boltz2, QBP, 3 seeds,
  MolProbity medians) gave clashscore 4.81 unrestrained / 5.94 RMSD-only / 27.58 conformer-only /
  22.91 conformer+RMSD. `max_iter: 1000` (vs 100 elsewhere) is what lets the inner CG satisfy the
  two COMPETING targets each step; `stop_sigma: 1.0` releases them for the final low-noise steps
  so the model can heal any strain it was held in.
- Most selections use bare `resid N to M` because QBP / ADK / DgoT are single-chain and
  ligand-free; OpenDDE examples use explicit `chain A` as a safer template. **On a system with a
  ligand or multiple chains, qualify each group with `chain A and (...)`** or the bare `resid`
  range will also sweep in the ligand's atoms.
- `resid` is the per-chain 1-based ordinal (not the author residue number). Full schema:
  `../doc/config.md`.
- Set `verbose: true` (already on) and check the `setup` log line `built spec: ... distances=..
  group_angle=.. rmsd=..` — the count must be non-zero, or the restraint silently did nothing.
  The `rmsd/` example inverts this for the conformer half: its correct signal is an **absence**
  (`conformer=False` in the setup line, `n_rmsd=2`). The opt-in and the config block must be
  dropped TOGETHER — either one alone is a no-op whose intent nothing in the file records.
