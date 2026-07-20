#!/bin/bash
#SBATCH -p q1,q3,af3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%2
#SBATCH -J rgi_poly_qbp
#SBATCH -o /mnt/lustre01/hori/rgi_utils_confrestr_alt/logs/qbp_%A_%a.out

set -euo pipefail

REPO=/home/hori/works/misc/impl_rgi/rgi_utils_confrestr_alt
BOLTZ=/home/hori/works/misc/impl_rgi/boltz_restr
BASE=/mnt/lustre01/hori/rgi_utils_confrestr_alt/e2e_${SLURM_ARRAY_JOB_ID}
mkdir -p "$BASE"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    TAG=rmsd_only_target0_seed0
    INPUT=$REPO/e2e/rmsd_only_qbp_target0.yaml
    SEED=0
    ;;
  1)
    TAG=polymer_target0_seed0
    INPUT=$REPO/e2e/polymer_qbp_target0.yaml
    SEED=0
    ;;
  2)
    TAG=polymer_target0_seed1
    INPUT=$REPO/e2e/polymer_qbp_target0.yaml
    SEED=1
    ;;
  3)
    TAG=polymer_target05_seed0
    INPUT=$REPO/e2e/polymer_qbp_target05.yaml
    SEED=0
    ;;
esac

echo "tag=$TAG seed=$SEED host=$(hostname)"
cd "$REPO"
export PYTHONPATH="$REPO/src"
export UV_PROJECT_ENVIRONMENT="$BOLTZ/.venv"
uv run --no-sync boltz predict \
  --seed "$SEED" \
  --out_dir "$BASE/$TAG" \
  --model boltz2 \
  "$INPUT"
