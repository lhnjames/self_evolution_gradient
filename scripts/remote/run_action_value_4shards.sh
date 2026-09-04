#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
data_root="${ALFWORLD_DATA_ROOT:-$repo_root/alfworld_data}"
config_path="${ALFWORLD_CONFIG:-$repo_root/references/SEED/agent_system/environments/env_package/alfworld/configs/config_tw.yaml}"
output_root="${OUTPUT_ROOT:-$repo_root/outputs/action_value_alignment}"
num_shards="${NUM_SHARDS:-4}"
expert_type="${EXPERT_TYPE:-handcoded}"
value_seed="${VALUE_SEED:-20260902}"

mkdir -p "$output_root/logs"

run_shard() {
  local shard="$1"
  for split in valid_seen valid_unseen; do
    PYTHONPATH="$repo_root/src" "$python_bin" "$repo_root/scripts/evaluate_alfworld_action_values.py" \
      --config "$config_path" \
      --data-root "$data_root" \
      --decision-file "$repo_root/data/alfworld_expert_large/${split}.jsonl" \
      --split "$split" \
      --output-dir "$output_root/${split}_shard_${shard}" \
      --expert-type "$expert_type" \
      --gamma 0.95 \
      --max-steps 50 \
      --seed "$value_seed" \
      --shard-index "$shard" \
      --num-shards "$num_shards" \
      --resume
  done
}

pids=()
for shard in $(seq 0 $((num_shards - 1))); do
  run_shard "$shard" >"$output_root/logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
