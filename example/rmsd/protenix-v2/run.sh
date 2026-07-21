#!/bin/bash
# protenix-v2 RGI example -- dual-ref RMSD morph -> midpoint of 1GGG(open)/1WDN(closed), target 3.0 A
# Restraint config = bench-rgi minimal; MSA is fetched from a server so the example is
# self-contained. (AlphaFold3 is the exception -- it needs external model params + DBs.)
# GPU only: run on a compute node via sbatch, NOT the login node.
# Requires the protenix_restr checkout to exist as a sibling of rgi_utils/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/rgi_utils" ]; do WS="$(dirname "$WS")"; done
# Reference structures (1GGG open / 1WDN closed) are downloaded from RCSB at run time
# instead of being stored in the repo. The config's ref_cif uses the bare filename.
( cd "$HERE" && for pdb in 1GGG 1WDN; do
    [ -f "$pdb.cif" ] || wget -q "https://files.rcsb.org/download/$pdb.cif"
done )
source "$WS/protenix_restr/.venv/bin/activate"
cd "$HERE"
# protenix must run on sm_89 (e.g. RTX 4090); Blackwell emits silent all-NaN coords.
protenix pred -i qbp_3.00.json -o out \
    --use_default_params true --use_msa true --seeds 0 --step 200 --sample 1 --cycle 4
