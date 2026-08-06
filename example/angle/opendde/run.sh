#!/bin/bash
# OpenDDE RGI example -- group-centroid angle -> 72.85 deg (ADK NMP-CORE-LID)
# External feature searches are disabled; the OpenDDE checkpoint and common runtime
# files must already be installed. Run on a GPU compute node, not a login node.
# Requires the OpenDDE_restr checkout to exist as a sibling of rgi_utils/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/rgi_utils" ]; do WS="$(dirname "$WS")"; done
source "$WS/OpenDDE_restr/.venv/bin/activate"
cd "$HERE"
opendde pred -i adk_72.85.json -o out -n opendde_v1 \
    --use_msa false --use_template false --use_rna_msa false \
    --sample 1 --step 200 --cycle 4
