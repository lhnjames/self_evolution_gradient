#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/data/user/agent_self_evolution_gradient}
output_root=${OUTPUT_ROOT:-$root/outputs/conflict_aware_accumulation_300x_v1}
source_count=${SOURCE_COUNT:-12}
holdout_count=${HOLDOUT_COUNT:-6}
mlp_norm=${MLP_NORM:-0.375}
last_four_norm=${LAST_FOUR_NORM:-0.18}
mkdir -p "$output_root/logs"
touch "$output_root/TMUX_RUN_STARTED"

splits=(valid_seen valid_unseen valid_seen valid_unseen)
groups=(last_mlp last_mlp last_four_blocks last_four_blocks)
norms=("$mlp_norm" "$mlp_norm" "$last_four_norm" "$last_four_norm")
pids=()
for gpu in 0 1 2 3; do
  split=${splits[$gpu]}
  group=${groups[$gpu]}
  norm=${norms[$gpu]}
  delta_args=()
  if [[ "$split" == "valid_seen" ]]; then
    delta_args=(--delta-output-dir "$output_root/deltas")
  fi
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/src:$root/scripts" \
    "$root/.venv/bin/python" "$root/scripts/probe_conflict_aware_accumulation.py" \
    --decision-file "$root/data/alfworld_expert_large/${split}.jsonl" \
    --value-trace "$root/outputs/action_value_alignment/analysis/trace.jsonl" \
    --base-model "$root/model" \
    --tokenizer-path "$root/model" \
    --output-file "$output_root/${split}_${group}.json" \
    --device cuda:0 \
    --split "$split" \
    --parameter-group "$group" \
    --single-action-total-norm "$norm" \
    --source-count "$source_count" \
    --holdout-count "$holdout_count" \
    --sample-seed 20260903 \
    "${delta_args[@]}" \
    >"$output_root/logs/${split}_${group}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if (( failed )); then
  echo "At least one conflict-aware accumulation probe failed; inspect logs." >&2
  exit 1
fi
for split in valid_seen valid_unseen; do
  for group in last_mlp last_four_blocks; do
    "$root/.venv/bin/python" -c \
      'import json,sys; r=json.load(open(sys.argv[1])); assert len(r["strategy_results"]) == 3; assert all(len(s["steps"]) == 60 for s in r["strategy_results"])' \
      "$output_root/${split}_${group}.json"
  done
done
touch "$output_root/TMUX_RUN_COMPLETE"
echo "Conflict-aware accumulation completed successfully."
