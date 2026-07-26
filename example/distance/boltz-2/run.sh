#!/bin/bash
# boltz-2 RGI example -- centroid distance -> 25.0 A (QBP)
# Restraint config = bench-rgi minimal; MSA is fetched from a server so the example is
# self-contained. (AlphaFold3 is the exception -- it needs external model params + DBs.)
# GPU only: run on a GPU compute node (not a shared login node).
# Requires the boltz_restr checkout to exist as a sibling of rgi_utils/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/rgi_utils" ]; do WS="$(dirname "$WS")"; done
source "$WS/boltz_restr/.venv/bin/activate"
cd "$HERE"
boltz predict qbp_25.00.yaml --out_dir out --model boltz2 --use_msa_server --seed 0
