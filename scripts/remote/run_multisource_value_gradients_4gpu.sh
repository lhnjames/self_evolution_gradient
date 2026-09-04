#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/data/user/agent_self_evolution_gradient}
output_root=${OUTPUT_ROOT:-$root/outputs/multisource_value_gradient_v1}
source_count=${SOURCE_COUNT:-12}
holdout_count=${HOLDOUT_COUNT:-6}
mlp_step=${MLP_STEP:-0.0125}
last_four_step=${LAST_FOUR_STEP:-0.006}
mkdir -p "$output_root/logs"
touch "$output_root/TMUX_RUN_STARTED"

splits=(valid_seen valid_unseen valid_seen valid_unseen)
groups=(last_mlp last_mlp last_four_blocks last_four_blocks)
steps=("$mlp_step" "$mlp_step" "$last_four_step" "$last_four_step")
pids=()
for gpu in 0 1 2 3; do
  split=${splits[$gpu]}
  group=${groups[$gpu]}
  step=${steps[$gpu]}
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/src" \
    "$root/.venv/bin/python" "$root/scripts/probe_multisource_value_gradients.py" \
    --decision-file "$root/data/alfworld_expert_large/${split}.jsonl" \
    --value-trace "$root/outputs/action_value_alignment/analysis/trace.jsonl" \
    --base-model "$root/model" \
    --tokenizer-path "$root/model" \
    --output-file "$output_root/${split}_${group}.json" \
    --device cuda:0 \
    --split "$split" \
    --parameter-group "$group" \
    --step-norm "$step" \
    --source-count "$source_count" \
    --holdout-count "$holdout_count" \
    --sample-seed 20260903 \
    >"$output_root/logs/${split}_${group}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed )); then
  echo "At least one multisource probe failed; inspect logs." >&2
  exit 1
fi

for split in valid_seen valid_unseen; do
  for group in last_mlp last_four_blocks; do
    "$root/.venv/bin/python" -c \
      'import json,sys; r=json.load(open(sys.argv[1])); n=int(sys.argv[2]); assert len(r["verb_results"]) == 5; assert all(len(v["single_source_updates"]) == n for v in r["verb_results"])' \
      "$output_root/${split}_${group}.json" "$source_count"
  done
done
touch "$output_root/TMUX_RUN_COMPLETE"
echo "Multisource value-gradient probe completed successfully."
