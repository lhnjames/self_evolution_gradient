#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
model_path="${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to the local Qwen checkpoint directory}"
output_root="${OUTPUT_ROOT:-$repo_root/outputs/gradient_probe}"
controls_root="${CONTROLS_ROOT:-$repo_root/outputs/skill_controls_3b}"
baseline_root="${BASELINE_ROOT:-$repo_root/outputs/alfworld_large_3b}"

mkdir -p "$output_root/logs"

run_probe() {
  local gpu="$1"
  export CUDA_VISIBLE_DEVICES="$gpu"

  PYTHONPATH="$repo_root/src" "$python_bin" "$repo_root/scripts/probe_skill_parameter_gradients.py" \
    --decision-file "$repo_root/data/alfworld_expert_large/valid_seen.jsonl" \
    --baseline-trace "$baseline_root/valid_seen_merged/trace.jsonl" \
    --control-traces \
      "$controls_root/valid_seen_shard_0/trace.jsonl" \
      "$controls_root/valid_seen_shard_1/trace.jsonl" \
      "$controls_root/valid_seen_shard_2/trace.jsonl" \
      "$controls_root/valid_seen_shard_3/trace.jsonl" \
    --model-path "$model_path" \
    --output-dir "$output_root/valid_seen_shard_${gpu}" \
    --device cuda:0 \
    --sample-size 64 \
    --shard-index "$gpu" \
    --num-shards 4 \
    --batch-size 4

  PYTHONPATH="$repo_root/src" "$python_bin" "$repo_root/scripts/probe_skill_parameter_gradients.py" \
    --decision-file "$repo_root/data/alfworld_expert_large/valid_unseen.jsonl" \
    --baseline-trace "$baseline_root/valid_unseen_merged/trace.jsonl" \
    --control-traces \
      "$controls_root/valid_unseen_shard_0/trace.jsonl" \
      "$controls_root/valid_unseen_shard_1/trace.jsonl" \
      "$controls_root/valid_unseen_shard_2/trace.jsonl" \
      "$controls_root/valid_unseen_shard_3/trace.jsonl" \
    --model-path "$model_path" \
    --output-dir "$output_root/valid_unseen_shard_${gpu}" \
    --device cuda:0 \
    --sample-size 64 \
    --shard-index "$gpu" \
    --num-shards 4 \
    --batch-size 4
}

run_probe 0 >"$output_root/logs/gpu0.log" 2>&1 &
run_probe 1 >"$output_root/logs/gpu1.log" 2>&1 &
run_probe 2 >"$output_root/logs/gpu2.log" 2>&1 &
run_probe 3 >"$output_root/logs/gpu3.log" 2>&1 &

wait
