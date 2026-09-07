#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/wangx/home/duxinxu/miniconda3/envs/landau-pic-surrogate/bin/python"
RESULT_ROOT="${PROJECT_ROOT}/results/continuum_v1_macrostep_round10/formal"
DEVICE="${1:-cuda:1}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# These seeds were fixed from validation before the diagnostic test was opened.
declare -A SELECTED_SEED=(
  [fno_single]=1
  [fno_history4]=2
  [unet_history4]=1
)

for ARM in fno_single fno_history4 unet_history4; do
  SEED="${SELECTED_SEED[${ARM}]}"
  RUN_ROOT="${RESULT_ROOT}/${ARM}/seed${SEED}"
  OUTPUT_DIR="${RUN_ROOT}/test_t80"
  if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
    echo "${ARM} seed ${SEED}: test rollout already complete; skipping"
    continue
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_continuum_v1_macrostep_rollouts.py" \
    --checkpoint "${RUN_ROOT}/best.pt" \
    --output-dir "${OUTPUT_DIR}" \
    --split test \
    --allow-test \
    --maximum-time 80 \
    --device "${DEVICE}"
done
