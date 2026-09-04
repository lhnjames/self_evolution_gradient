#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/data/user/agent_self_evolution_gradient}
delta_path=${DELTA_PATH:-}
model_path=${MODEL_PATH:-$root/model}
condition=${CONDITION_NAME:?Set CONDITION_NAME}
output_root=${OUTPUT_ROOT:?Set OUTPUT_ROOT}
mkdir -p "$output_root/logs"
delta_args=()
if [[ -n "$delta_path" ]]; then delta_args=(--parameter-delta "$delta_path"); fi

run_shard() {
  local gpu="$1"
  local seen_offset="$2"
  local seen_count="$3"
  local unseen_offset="$4"
  local unseen_count="$5"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/src" \
    "$root/.venv/bin/python" "$root/scripts/score_alfworld_checkpoint.py" \
    --decision-file "$root/data/alfworld_expert_large/valid_seen.jsonl" \
    --model-path "$model_path" \
    --tokenizer-path "$root/model" \
    "${delta_args[@]}" --force-float32 \
    --condition-name "$condition" \
    --output-dir "$output_root/valid_seen_shard_${gpu}" \
    --device cuda:0 --decision-offset "$seen_offset" --max-decisions "$seen_count" --batch-size 4

  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/src" \
    "$root/.venv/bin/python" "$root/scripts/score_alfworld_checkpoint.py" \
    --decision-file "$root/data/alfworld_expert_large/valid_unseen.jsonl" \
    --model-path "$model_path" \
    --tokenizer-path "$root/model" \
    "${delta_args[@]}" --force-float32 \
    --condition-name "$condition" \
    --output-dir "$output_root/valid_unseen_shard_${gpu}" \
    --device cuda:0 --decision-offset "$unseen_offset" --max-decisions "$unseen_count" --batch-size 4
}

run_shard 0 0 111 0 132 >"$output_root/logs/gpu0.log" 2>&1 &
run_shard 1 111 111 132 132 >"$output_root/logs/gpu1.log" 2>&1 &
run_shard 2 222 111 264 132 >"$output_root/logs/gpu2.log" 2>&1 &
run_shard 3 333 111 396 131 >"$output_root/logs/gpu3.log" 2>&1 &
wait

for split in valid_seen valid_unseen; do
  PYTHONPATH="$root/src" "$root/.venv/bin/python" "$root/scripts/merge_alfworld_shards.py" \
    --output-dir "$output_root/${split}_merged" \
    "$output_root"/${split}_shard_0 "$output_root"/${split}_shard_1 \
    "$output_root"/${split}_shard_2 "$output_root"/${split}_shard_3
done

touch "$output_root/SCORING_COMPLETE"
echo "Candidate scoring complete: $output_root"
