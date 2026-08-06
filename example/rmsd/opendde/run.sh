#!/bin/bash
# OpenDDE RGI example -- dual-ref RMSD morph -> midpoint of 1GGG(open)/1WDN(closed), target 3.0 A
# External feature searches are disabled; the OpenDDE checkpoint and common runtime
# files must already be installed. Run on a GPU compute node, not a login node.
# Requires the OpenDDE_restr checkout to exist as a sibling of rgi_utils/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/rgi_utils" ]; do WS="$(dirname "$WS")"; done
# Reference structures are downloaded from RCSB at run time and are not stored here.
( cd "$HERE" && for pdb in 1GGG 1WDN; do
    [ -f "$pdb.cif" ] || wget -q "https://files.rcsb.org/download/$pdb.cif"
done )
source "$WS/OpenDDE_restr/.venv/bin/activate"
cd "$HERE"
opendde pred -i qbp_3.00.json -o out -n opendde_v1 \
    --use_msa false --use_template false --use_rna_msa false \
    --sample 1 --step 200 --cycle 4
