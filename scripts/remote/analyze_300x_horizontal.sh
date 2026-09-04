#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/data/user/agent_self_evolution_gradient}
score_root=${SCORE_ROOT:-$root/outputs/horizontal_300x_vs_seed_fp32}
analysis_root=${ANALYSIS_ROOT:-$score_root/analysis}
mkdir -p "$analysis_root"
while [[ ! -f "$score_root/SCORING_COMPLETE" ]]; do
  date '+%F %T waiting for SCORING_COMPLETE'
  sleep 30
done

value_args=()
for split in valid_seen valid_unseen; do
  for path in "$root"/outputs/action_value_alignment/${split}_shard_*/trace.jsonl; do
    value_args+=(--value-trace "$path")
  done
done

for condition in seed_fp32 direct300_last4 project300_last4; do
  label="$condition"
  PYTHONPATH="$root/src" "$root/.venv/bin/python" \
    "$root/scripts/analyze_action_value_alignment.py" \
    --split valid_seen --split valid_unseen \
    "${value_args[@]}" \
    --base-trace "$score_root/valid_seen_base_fp32/trace.jsonl" \
    --base-trace "$score_root/valid_unseen_base_fp32/trace.jsonl" \
    --seed-trace "$score_root/valid_seen_${condition}/trace.jsonl" \
    --seed-trace "$score_root/valid_unseen_${condition}/trace.jsonl" \
    --output-dir "$analysis_root/$condition" --bootstrap-samples 10000 \
    --seed 20260903 --practical-threshold 0.80 --required-relative-improvement 0.30 \
    --comparison-label "$label" \
    >"$analysis_root/${condition}.stdout.log" 2>&1
done
PYTHONPATH="$root/src:$root/scripts" "$root/.venv/bin/python" \
  "$root/scripts/summarize_300x_horizontal.py" \
  --analysis-root "$analysis_root" --output-dir "$analysis_root/summary" \
  --required-relative-improvement 0.30 \
  >"$analysis_root/summary.stdout.log" 2>&1
touch "$analysis_root/ANALYSIS_COMPLETE"
sed -n '1,180p' "$analysis_root/summary/REPORT.md"
for condition in seed_fp32 direct300_last4 project300_last4; do
  sed -n '1,100p' "$analysis_root/$condition/REPORT.md"
done
