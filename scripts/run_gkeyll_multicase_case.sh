#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 K ALPHA PHYSICAL_GPU" >&2
  exit 2
fi

k_value="$1"
alpha_value="$2"
gpu_id="$3"
project_root="/wangx/home/duxinxu/projects/landau-damping-surrogate-standardized"
dataset_root="${GKYL_DATASET_ROOT:-/wangx/home/duxinxu/datasets/landau-damping-surrogate-standardized/gkeyll/paper_match_multicase_v1}"
gkeyll_env="/wangx/home/duxinxu/software/micromamba-root-v1/envs/gkeyll-build"
gkeyll_bin="/wangx/home/duxinxu/software/gkylsoft-paper-match/gkeyll/bin/gkeyll"
source_input="$project_root/configs/gkeyll/huang2025_multicase_1x1v_p2.lua"

case_id="$(printf 'k%.3f_a%.3f' "$k_value" "$alpha_value" | tr '.' 'p')"
run_dir="$dataset_root/cases/$case_id/raw"
log_dir="$dataset_root/logs"
provenance_dir="$dataset_root/cases/$case_id/provenance"
mkdir -p "$run_dir" "$log_dir" "$provenance_dir"

if [[ -f "$provenance_dir/stat.json" ]] && compgen -G "$run_dir/*-elc_M0_8000.gkyl" >/dev/null; then
  echo "Skipping already completed Gkeyll case in $run_dir"
  exit 0
fi
if find "$run_dir" -maxdepth 1 -name '*.gkyl' -print -quit | grep -q .; then
  echo "Refusing to overwrite incomplete Gkeyll frames in $run_dir" >&2
  exit 1
fi
if [[ ! -x "$gkeyll_bin" ]]; then
  echo "Missing Gkeyll executable: $gkeyll_bin" >&2
  exit 1
fi

input_name="huang2025_${case_id}_1x1v_p2.lua"
cp "$source_input" "$run_dir/$input_name"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export LD_LIBRARY_PATH="$gkeyll_env/lib:/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}"
export GKYL_K0="$k_value"
export GKYL_ALPHA="$alpha_value"
export GKYL_T_END="${GKYL_T_END:-40.0}"
export GKYL_NUM_FRAMES="${GKYL_NUM_FRAMES:-8000}"
export GKYL_DISTRIBUTION_FRAME_STRIDE="${GKYL_DISTRIBUTION_FRAME_STRIDE:-2000}"

timestamp="$(date +%Y%m%dT%H%M%S%z)"
log="$log_dir/${case_id}_${timestamp}.log"
{
  echo "case_id=$case_id"
  echo "k=$GKYL_K0"
  echo "alpha=$GKYL_ALPHA"
  echo "t_end=$GKYL_T_END"
  echo "num_frames=$GKYL_NUM_FRAMES"
  echo "distribution_frame_stride=$GKYL_DISTRIBUTION_FRAME_STRIDE"
  echo "physical_gpu=$gpu_id"
  echo "input_sha256=$(sha256sum "$run_dir/$input_name" | awk '{print $1}')"
  echo "binary_sha256=$(sha256sum "$gkeyll_bin" | awk '{print $1}')"
  echo "vlasov_library_sha256=$(sha256sum /wangx/home/duxinxu/software/gkylsoft-paper-match/gkeyll/lib/libg0vlasov.so | awk '{print $1}')"
} | tee "$log"

cd "$run_dir"
"$gkeyll_bin" -g "$input_name" 2>&1 | tee -a "$log"
cp "${input_name%.lua}-stat.json" "$provenance_dir/stat.json"
cp "$run_dir/$input_name" "$provenance_dir/input.lua"
