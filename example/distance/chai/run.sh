#!/bin/bash
# chai RGI example -- centroid distance -> 25.0 A (QBP)
# Restraint config = bench-rgi minimal; MSA is fetched from a server so the example is
# self-contained. (AlphaFold3 is the exception -- it needs external model params + DBs.)
# GPU only: run on a GPU compute node (not a shared login node).
# Requires the chai-lab_restr checkout to exist as a sibling of rgi_utils/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/rgi_utils" ]; do WS="$(dirname "$WS")"; done
source "$WS/chai-lab_restr/.venv/bin/activate"
cd "$HERE"
export CHAI_DOWNLOADS_DIR="${CHAI_DOWNLOADS_DIR:-$HOME/.cache/chai}"
python -m chai_lab.main fold qbp_25.00.fasta out \
    --restraints-config-path qbp_25.00.yaml \
    --num-diffn-samples 1 --seed 0 \
    --use-msa-server --use-templates-server --no-use-esm-embeddings
