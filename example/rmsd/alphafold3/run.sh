#!/bin/bash
# alphafold3 RGI example -- dual-ref RMSD morph -> midpoint of 1GGG(open)/1WDN(closed), target 3.0 A
# Restraint config = bench-rgi minimal; MSA is fetched from a server so the example is
# self-contained. (AlphaFold3 is the exception -- it needs external model params + DBs.)
# GPU only: run on a compute node via sbatch, NOT the login node.
# Requires the alphafold3_restr checkout to exist as a sibling of rgi_utils/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE"; while [ "$WS" != / ] && [ ! -d "$WS/rgi_utils" ]; do WS="$(dirname "$WS")"; done
source "$WS/alphafold3_restr/.venv/bin/activate"
cd "$HERE"
# AF3 has NO ColabFold MSA server -> it runs the genetic-search data pipeline, which
# needs the AF3 model weights + sequence databases (hundreds of GB). AF3 is therefore
# the ONE example that is not fully self-contained.
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to your AF3 model-parameters directory}"
DB_DIR="${DB_DIR:?set DB_DIR to your AF3 sequence-database directory}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/${USER}_jax_cache}"
python "$WS/alphafold3_restr/run_alphafold.py" \
    --run_data_pipeline=True --model_dir="$MODEL_DIR" --db_dir="$DB_DIR" \
    --json_path=qbp_3.00.json --output_dir=out
# Local fallback (skip the DB search): add
#   "unpairedMsaPath": "/home/hori/works/misc/impl_rgi/bench-rgi/distance/fixtures/qbp.a3m"
# to the protein object in qbp_3.00.json, then pass --run_data_pipeline=False.
