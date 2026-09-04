#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="${1:-$repo_dir/outputs/8gpu_sweep_$(date +%Y%m%d_%H%M%S)}"
config_path="${2:-$repo_dir/config/experiment.yaml}"
mkdir -p "$run_root"

for gpu in $(seq 0 7); do
  seed=$((100 + gpu))
  CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HOME="$repo_dir/model_cache" \
    TMPDIR="$repo_dir/.tmp" \
    PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$repo_dir/.venv/bin/python" -m self_evolve.runner \
      --config "$config_path" \
      --device cuda:0 \
      --seed "$seed" \
      --output "$run_root/seed_$seed" \
      >"$run_root/seed_$seed.log" 2>&1 &
done
wait
echo "Eight-seed sweep completed: $run_root"
