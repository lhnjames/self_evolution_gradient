#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
seed_model_path="${SEED_MODEL_PATH:?Set SEED_MODEL_PATH to the local SEED checkpoint directory}"
tokenizer_path="${BASE_TOKENIZER_PATH:?Set BASE_TOKENIZER_PATH to the base Qwen checkpoint directory}"
output_root="${OUTPUT_ROOT:-$repo_root/outputs/seed_checkpoint_plain}"

mkdir -p "$output_root/logs"

run_shard() {
  local gpu="$1"
  local seen_offset="$2"
  local seen_count="$3"
  local unseen_offset="$4"
  local unseen_count="$5"
  export CUDA_VISIBLE_DEVICES="$gpu"

  PYTHONPATH="$repo_root/src" "$python_bin" "$repo_root/scripts/score_alfworld_checkpoint.py" \
    --decision-file "$repo_root/data/alfworld_expert_large/valid_seen.jsonl" \
    --model-path "$seed_model_path" \
    --tokenizer-path "$tokenizer_path" \
    --condition-name seed_checkpoint_same_plain_prompt \
    --output-dir "$output_root/valid_seen_shard_${gpu}" \
    --device cuda:0 \
    --decision-offset "$seen_offset" \
    --max-decisions "$seen_count" \
    --batch-size 4

  PYTHONPATH="$repo_root/src" "$python_bin" "$repo_root/scripts/score_alfworld_checkpoint.py" \
    --decision-file "$repo_root/data/alfworld_expert_large/valid_unseen.jsonl" \
    --model-path "$seed_model_path" \
    --tokenizer-path "$tokenizer_path" \
    --condition-name seed_checkpoint_same_plain_prompt \
    --output-dir "$output_root/valid_unseen_shard_${gpu}" \
    --device cuda:0 \
    --decision-offset "$unseen_offset" \
    --max-decisions "$unseen_count" \
    --batch-size 4
}

run_shard 0 0 111 0 132 >"$output_root/logs/gpu0.log" 2>&1 &
run_shard 1 111 111 132 132 >"$output_root/logs/gpu1.log" 2>&1 &
run_shard 2 222 111 264 132 >"$output_root/logs/gpu2.log" 2>&1 &
run_shard 3 333 111 396 131 >"$output_root/logs/gpu3.log" 2>&1 &

wait
