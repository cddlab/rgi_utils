#!/bin/bash
# openfold-3 RGI example -- custom dist-diff: (d(A,B)-d(C,D)) -> 0.0 (DgoT)
# Restraint config = bench-rgi minimal; MSA is fetched from a server so the example is
# self-contained. (AlphaFold3 is the exception -- it needs external model params + DBs.)
# GPU only: run on a GPU compute node (not a shared login node).
# Requires the openfold-3_restr checkout to exist as a sibling of rgi_utils/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/rgi_utils" ]; do WS="$(dirname "$WS")"; done
export OPENFOLD_CACHE="${OPENFOLD_CACHE:-$HOME/.openfold3}"
PIXI="$WS/openfold-3_restr/.pixi-bin/pixi"; [ -x "$PIXI" ] || PIXI=pixi
cd "$HERE"
"$PIXI" run --manifest-path "$WS/openfold-3_restr/pixi.toml" -e openfold3-cuda12 \
    run_openfold predict --query-json "$HERE/dgot_0.00.json" --output-dir "$HERE/out" \
    --num-diffusion-samples 1 --use-msa-server true --use-templates false
