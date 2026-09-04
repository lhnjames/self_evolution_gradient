#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/data/user/agent_self_evolution_gradient}
score_root=${SCORE_ROOT:?Set SCORE_ROOT}
label=${COMPARISON_LABEL:?Set COMPARISON_LABEL}
analysis_root=${ANALYSIS_ROOT:-$score_root/analysis}
base_score_root=${BASE_SCORE_ROOT:-}
mkdir -p "$analysis_root"

candidate_args=()
value_args=()
base_args=()
split_args=()
for split in valid_seen valid_unseen; do
  split_args+=(--split "$split")
  if [[ -n "$base_score_root" ]]; then
    base_args+=(--base-trace "$base_score_root/${split}_merged/trace.jsonl")
  else
    base_args+=(--base-trace "$root/baseline_traces/${split}.jsonl")
  fi
  for path in "$root"/outputs/action_value_alignment/${split}_shard_*/trace.jsonl; do
    value_args+=(--value-trace "$path")
  done
  for path in "$score_root"/${split}_shard_*/trace.jsonl; do
    candidate_args+=(--seed-trace "$path")
  done
done

PYTHONPATH="$root/src" "$root/.venv/bin/python" \
  "$root/scripts/analyze_action_value_alignment.py" \
  "${split_args[@]}" "${value_args[@]}" "${base_args[@]}" "${candidate_args[@]}" \
  --output-dir "$analysis_root" --bootstrap-samples 10000 --seed 20260903 \
  --practical-threshold 0.80 --comparison-label "$label" \
  >"$analysis_root/analysis.stdout.log" 2>&1
touch "$analysis_root/ANALYSIS_COMPLETE"
sed -n '1,160p' "$analysis_root/REPORT.md"
