# rgi_utils RGI examples

Minimal, ready-to-run Restraint-Guided Inference (RGI) samples: **4 restraint types x 6
predictors**. Each `<type>/<tool>/` (and `custom/dist-diff/<tool>/`) is one representative
restraint with a `run.sh`. Restraint blocks are the bench-rgi minimal configs; the runners
fetch the MSA from a server so the examples are self-contained (AlphaFold3 excepted, below).

| type | system | restraint |
|---|---|---|
| `distance/` | QBP (226 aa) | centroid distance of two lobe groups -> 25.0 A |
| `angle/` | adenylate kinase (214 aa) | NMP-CORE-LID centroid angle -> 72.85 deg |
| `rmsd/` | QBP | dual-ref morph to the midpoint of 1GGG (open) / 1WDN (closed), target 3.0 A, with a protein conformer restraint holding backbone geometry |
| `custom/dist-diff/` | DgoT transporter (419 aa) | custom energy `(d(A,B) - d(C,D)) -> 0` (a difference of two centroid distances) |

Tools: `boltz-2`, `protenix-v2`, `alphafold3`, `openfold-3`, `chai`, `esmfold2`.

## Run

```bash
bash distance/boltz-2/run.sh          # one example
```

Each `run.sh` locates the workspace root, activates/uses the matching fork's env, and folds.

## Prerequisites

- **The matching fork must exist as a sibling of `rgi_utils/`** (e.g. `../boltz_restr`,
  `../esm_restr`) with its venv/pixi env built. `run.sh` finds it automatically. See each
  fork's `rgi_utils/doc/<tool>.md` for install steps.
- **GPU via sbatch** — never the login node. Partitions: `q1`/`q3`/`af3` = RTX 4090 (sm_89),
  `q4`/`maxq` = sm_120.
- **protenix runs on sm_89 only** (Blackwell emits silent all-NaN). The default esm/chai torch
  is cu124 (sm_89) too.
- **AlphaFold3 is the one non-self-contained example**: it has no ColabFold MSA server, so its
  `run.sh` runs the genetic-search data pipeline, requiring external `MODEL_DIR` (weights) and
  `DB_DIR` (sequence databases, hundreds of GB). A local no-DB fallback is documented inline in
  `alphafold3/*/run.sh`.

## Notes

- Selections use bare `resid N to M` (no `chain A and`) because QBP / ADK / DgoT are all
  single-chain, ligand-free. **On a system with a ligand or multiple chains, qualify each group
  with `chain A and (...)`** or the bare `resid` range will also sweep in the ligand's atoms.
- `resid` is the per-chain 1-based ordinal (not the author residue number). Full schema:
  `../doc/config.md`.
- Set `verbose: true` (already on) and check the `setup` log line `built spec: ... distances=..
  group_angle=.. rmsd=..` — the count must be non-zero, or the restraint silently did nothing.
