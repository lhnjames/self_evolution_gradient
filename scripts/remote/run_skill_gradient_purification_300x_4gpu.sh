#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/data/user/agent_self_evolution_gradient}
output_root=${OUTPUT_ROOT:-$root/outputs/skill_gradient_purification_300x_v1}
source_count=${SOURCE_COUNT:-12}
holdout_count=${HOLDOUT_COUNT:-6}
step_norm=${STEP_NORM:-0.18}
seeds=(20260904 20260921 20260938 20260955)
mkdir -p "$output_root/logs"
touch "$output_root/TMUX_RUN_STARTED"

pids=()
for gpu in 0 1 2 3; do
  seed=${seeds[$gpu]}
  seed_root="$output_root/seed_${seed}"
  mkdir -p "$seed_root/deltas"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/src:$root/scripts" \
    "$root/.venv/bin/python" "$root/scripts/probe_skill_gradient_purification.py" \
    --decision-file "$root/data/alfworld_expert_large/valid_seen.jsonl" \
    --value-trace "$root/outputs/action_value_alignment/analysis/trace.jsonl" \
    --base-score-trace "$root/outputs/horizontal_300x_vs_seed_fp32/valid_seen_base_fp32/trace.jsonl" \
    --base-model "$root/model" --tokenizer-path "$root/model" \
    --output-file "$seed_root/results.json" --delta-output-dir "$seed_root/deltas" \
    --device cuda:0 --split valid_seen --parameter-group last_four_blocks \
    --step-norm "$step_norm" --verbs go open close \
    --source-count "$source_count" --holdout-count "$holdout_count" \
    --sample-seed "$seed" \
    >"$output_root/logs/seed_${seed}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if (( failed )); then
  echo "At least one skill-gradient purification probe failed; inspect logs." >&2
  exit 1
fi

inputs=()
for seed in "${seeds[@]}"; do
  inputs+=(--input "$output_root/seed_${seed}/results.json")
done
PYTHONPATH="$root/src:$root/scripts" "$root/.venv/bin/python" \
  "$root/scripts/analyze_skill_gradient_purification.py" \
  "${inputs[@]}" --value-trace "$root/outputs/action_value_alignment/analysis/trace.jsonl" \
  --output-dir "$output_root/analysis" \
  >"$output_root/analysis.stdout.log" 2>&1
touch "$output_root/TMUX_RUN_COMPLETE"
sed -n '1,160p' "$output_root/analysis/REPORT.md"
