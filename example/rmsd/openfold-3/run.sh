#!/bin/bash
# openfold-3 RGI example -- dual-ref RMSD morph -> midpoint of 1GGG(open)/1WDN(closed), target 3.0 A
# Restraint config = bench-rgi minimal; MSA is fetched from a server so the example is
# self-contained. (AlphaFold3 is the exception -- it needs external model params + DBs.)
# GPU only: run on a GPU compute node (not a shared login node).
# Requires the openfold-3_restr checkout to exist as a sibling of rgi_utils/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/rgi_utils" ]; do WS="$(dirname "$WS")"; done
# Reference structures (1GGG open / 1WDN closed) are downloaded from RCSB at run time
# instead of being stored in the repo. The config's ref_cif uses the bare filename.
( cd "$HERE" && for pdb in 1GGG 1WDN; do
    [ -f "$pdb.cif" ] || wget -q "https://files.rcsb.org/download/$pdb.cif"
done )
export OPENFOLD_CACHE="${OPENFOLD_CACHE:-$HOME/.openfold3}"
PIXI="$WS/openfold-3_restr/.pixi-bin/pixi"; [ -x "$PIXI" ] || PIXI=pixi
cd "$HERE"
"$PIXI" run --manifest-path "$WS/openfold-3_restr/pixi.toml" -e openfold3-cuda12 \
    run_openfold predict --query-json "$HERE/qbp_3.00.json" --output-dir "$HERE/out" \
    --num-diffusion-samples 1 --use-msa-server true --use-templates false
