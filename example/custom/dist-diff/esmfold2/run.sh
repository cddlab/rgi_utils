#!/bin/bash
# esmfold2 RGI example -- custom dist-diff: (d(A,B)-d(C,D)) -> 0.0 (DgoT)
# Restraint config = bench-rgi minimal; MSA is fetched from a server so the example is
# self-contained. (AlphaFold3 is the exception -- it needs external model params + DBs.)
# GPU only: run on a compute node via sbatch, NOT the login node.
# Requires the esm_restr (+ transformers_restr) checkout to exist as a sibling of rgi_utils/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/rgi_utils" ]; do WS="$(dirname "$WS")"; done
PIXI="$WS/esm_restr/.pixi-bin/pixi"; [ -x "$PIXI" ] || PIXI=pixi
cd "$HERE"
# ESMFold2 folds single-sequence (no MSA). Runs on sm_89 (the pixi env torch is cu124).
"$PIXI" run --manifest-path "$WS/esm_restr/pyproject.toml" \
    env PYTHONPATH="$WS/transformers_restr/src" python "$HERE/run_custom.py"
