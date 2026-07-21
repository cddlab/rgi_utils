#!/bin/bash
# chai RGI example -- dual-ref RMSD morph -> midpoint of 1GGG(open)/1WDN(closed), target 3.0 A
# Restraint config = bench-rgi minimal; MSA is fetched from a server so the example is
# self-contained. (AlphaFold3 is the exception -- it needs external model params + DBs.)
# GPU only: run on a GPU compute node (not a shared login node).
# Requires the chai-lab_restr checkout to exist as a sibling of rgi_utils/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/rgi_utils" ]; do WS="$(dirname "$WS")"; done
# Reference structures (1GGG open / 1WDN closed) are downloaded from RCSB at run time
# instead of being stored in the repo. The config's ref_cif uses the bare filename.
( cd "$HERE" && for pdb in 1GGG 1WDN; do
    [ -f "$pdb.cif" ] || wget -q "https://files.rcsb.org/download/$pdb.cif"
done )
source "$WS/chai-lab_restr/.venv/bin/activate"
cd "$HERE"
export CHAI_DOWNLOADS_DIR="${CHAI_DOWNLOADS_DIR:-$HOME/.cache/chai}"
python -m chai_lab.main fold qbp_3.00.fasta out \
    --restraints-config-path qbp_3.00.yaml \
    --num-diffn-samples 1 --seed 0 \
    --use-msa-server --use-templates-server --no-use-esm-embeddings
