#!/bin/bash
# openfold-3 RGI example -- group-centroid angle -> 72.85 deg (ADK NMP-CORE-LID)
# Restraint config = bench-rgi minimal; MSA is fetched from a server so the example is
# self-contained. (AlphaFold3 is the exception -- it needs external model params + DBs.)
# GPU only: run on a compute node via sbatch, NOT the login node.
# Requires the openfold-3_restr checkout to exist as a sibling of rgi_utils/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/rgi_utils" ]; do WS="$(dirname "$WS")"; done
export OPENFOLD_CACHE="${OPENFOLD_CACHE:-$HOME/.openfold3}"
PIXI="$WS/openfold-3_restr/.pixi-bin/pixi"; [ -x "$PIXI" ] || PIXI=pixi
cd "$HERE"
"$PIXI" run --manifest-path "$WS/openfold-3_restr/pixi.toml" -e openfold3-cuda12 \
    run_openfold predict --query-json "$HERE/adk_72.85.json" --output-dir "$HERE/out" \
    --num-diffusion-samples 1 --use-msa-server true --use-templates false
