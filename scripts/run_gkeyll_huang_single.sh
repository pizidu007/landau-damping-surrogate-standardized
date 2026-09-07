#!/usr/bin/env bash
set -euo pipefail

mode="${1:-preflight}"
if [[ "$mode" != "preflight" && "$mode" != "formal" ]]; then
  echo "Usage: $0 [preflight|formal]" >&2
  exit 2
fi

project_root="/wangx/home/duxinxu/projects/landau-damping-surrogate-standardized"
dataset_root="/wangx/home/duxinxu/datasets/landau-damping-surrogate-standardized/gkeyll/paper_match_single_v1"
gkeyll_env="/wangx/home/duxinxu/software/micromamba-root-v1/envs/gkeyll-build"
gkeyll_bin="/wangx/home/duxinxu/software/gkylsoft-paper-match/gkeyll/bin/gkeyll"
source_input="$project_root/configs/gkeyll/huang2025_k0p35_a0p10_1x1v_p2.lua"

if [[ "$mode" == "preflight" ]]; then
  run_dir="$dataset_root/preflight"
else
  run_dir="$dataset_root/raw"
fi
mkdir -p "$run_dir" "$dataset_root/logs" "$dataset_root/provenance"

if [[ ! -x "$gkeyll_bin" ]]; then
  echo "Gkeyll executable is missing: $gkeyll_bin" >&2
  exit 1
fi

input_name="huang2025_k0p35_a0p10_1x1v_p2.lua"
cp "$source_input" "$run_dir/$input_name"

if [[ -z "${GPU_ID:-}" ]]; then
  GPU_ID="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export LD_LIBRARY_PATH="$gkeyll_env/lib:/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}"

timestamp="$(date +%Y%m%dT%H%M%S%z)"
log="$dataset_root/logs/${mode}_${timestamp}.log"
{
  echo "mode=$mode"
  echo "timestamp=$timestamp"
  echo "host=$(hostname)"
  echo "physical_gpu=$GPU_ID"
  echo "input_sha256=$(sha256sum "$run_dir/$input_name" | awk '{print $1}')"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader
} | tee "$log"

cd "$run_dir"
if [[ "$mode" == "preflight" ]]; then
  export GKYL_T_END=0.1
  export GKYL_NUM_FRAMES=20
else
  unset GKYL_T_END GKYL_NUM_FRAMES || true
fi

"$gkeyll_bin" -g "$input_name" 2>&1 | tee -a "$log"
