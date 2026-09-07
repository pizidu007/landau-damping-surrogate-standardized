#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-3}"
CPU_SET="${CPU_SET:-0,1}"
PROJECT_ROOT="/wangx/home/duxinxu/projects/landau-damping-surrogate-standardized"
PYTHON_BIN="/wangx/home/duxinxu/miniconda3/envs/landau-pic-surrogate/bin/python"
CONFIG="${PROJECT_ROOT}/configs/training/continuum_v1_closure_baseline.json"
RESULT_ROOT="${PROJECT_ROOT}/results/continuum_v1_closure_baseline"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PROJECT_ROOT}/src"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${RESULT_ROOT}"
for SEED in 0 1 2; do
  OUTPUT_DIR="${RESULT_ROOT}/seed${SEED}"
  if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
    echo "seed ${SEED} already complete; skipping"
    continue
  fi
  mkdir -p "${OUTPUT_DIR}"
  echo "starting seed ${SEED} on physical GPU ${GPU_ID}, CPU set ${CPU_SET}"
  COMPLETED=0
  for BATCH_SIZE in 8192 4096 2048; do
    ATTEMPT_LOG="${OUTPUT_DIR}/attempt_batch${BATCH_SIZE}.log"
    echo "seed ${SEED}: trying batch size ${BATCH_SIZE}"
    if taskset -c "${CPU_SET}" "${PYTHON_BIN}" \
      "${PROJECT_ROOT}/scripts/train_continuum_v1_closure.py" \
      --config "${CONFIG}" \
      --output-dir "${OUTPUT_DIR}" \
      --seed "${SEED}" \
      --device cuda:0 \
      --batch-size "${BATCH_SIZE}" \
      2>&1 | tee "${ATTEMPT_LOG}"; then
      COMPLETED=1
      break
    fi
    if grep -q "OutOfMemoryError" "${ATTEMPT_LOG}"; then
      echo "seed ${SEED}: CUDA memory changed; reducing batch size"
    else
      echo "seed ${SEED}: non-OOM failure; not retrying"
      exit 1
    fi
  done
  if [[ "${COMPLETED}" -ne 1 ]]; then
    echo "seed ${SEED}: all safe batch sizes exhausted"
    exit 1
  fi
done
