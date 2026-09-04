#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/data/user/agent_self_evolution_gradient}
output_root=${OUTPUT_ROOT:-$root/outputs/multisource_dose_response_v1}
mkdir -p "$output_root/logs"
touch "$output_root/TMUX_RUN_STARTED"

splits=(valid_seen valid_unseen valid_seen valid_unseen)
groups=(last_mlp last_mlp last_four_blocks last_four_blocks)
step_lists=(
  "0.00125 0.00375 0.0125 0.0375"
  "0.00125 0.00375 0.0125 0.0375"
  "0.0006 0.0018 0.006 0.018"
  "0.0006 0.0018 0.006 0.018"
)
pids=()
for gpu in 0 1 2 3; do
  split=${splits[$gpu]}
  group=${groups[$gpu]}
  read -r -a steps <<<"${step_lists[$gpu]}"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/src:$root/scripts" \
    "$root/.venv/bin/python" "$root/scripts/probe_multisource_dose_response.py" \
    --decision-file "$root/data/alfworld_expert_large/${split}.jsonl" \
    --value-trace "$root/outputs/action_value_alignment/analysis/trace.jsonl" \
    --base-model "$root/model" \
    --tokenizer-path "$root/model" \
    --output-file "$output_root/${split}_${group}.json" \
    --device cuda:0 \
    --split "$split" \
    --parameter-group "$group" \
    --step-norms "${steps[@]}" \
    --source-count 12 \
    --holdout-count 6 \
    --sample-seed 20260903 \
    >"$output_root/logs/${split}_${group}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if (( failed )); then
  echo "At least one dose-response process failed." >&2
  exit 1
fi
for split in valid_seen valid_unseen; do
  for group in last_mlp last_four_blocks; do
    "$root/.venv/bin/python" -c \
      'import json,sys; r=json.load(open(sys.argv[1])); assert len(r["verb_results"])==5; assert all(len(v["dose_updates"])==4 for v in r["verb_results"])' \
      "$output_root/${split}_${group}.json"
  done
done
touch "$output_root/TMUX_RUN_COMPLETE"
echo "Multisource dose response completed successfully."
