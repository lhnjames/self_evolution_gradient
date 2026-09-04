#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output_root="$repo_root/outputs/action_value_alignment"
analysis_root="$output_root/analysis"
mkdir -p "$analysis_root"

while ! test -e "$output_root/TMUX_RUN_COMPLETE"; do
  if test -e "$output_root/TMUX_RUN_FAILED"; then
    echo 'The tmux evaluation run failed; inspect the value-run and value-status windows.'
    exit 1
  fi
  echo "$(date '+%F %T') waiting for TMUX_RUN_COMPLETE"
  sleep 30
done

value_args=()
seed_args=()
base_args=()
split_args=()
for split in valid_seen valid_unseen; do
  split_args+=(--split "$split")
  base_args+=(--base-trace "$repo_root/baseline_traces/${split}.jsonl")
  for path in "$output_root"/${split}_shard_*/trace.jsonl; do
    value_args+=(--value-trace "$path")
  done
  for path in "$repo_root"/outputs/seed_checkpoint_plain/${split}_shard_*/trace.jsonl; do
    seed_args+=(--seed-trace "$path")
  done
done

PYTHONPATH="$repo_root/src" "$repo_root/.venv/bin/python" \
  "$repo_root/scripts/analyze_action_value_alignment.py" \
  "${split_args[@]}" \
  "${value_args[@]}" \
  "${base_args[@]}" \
  "${seed_args[@]}" \
  --output-dir "$analysis_root" \
  --bootstrap-samples 10000 \
  --seed 20260902 \
  --practical-threshold 0.80 \
  | tee "$analysis_root/analysis.stdout.log"

touch "$analysis_root/ANALYSIS_COMPLETE"
echo
echo "Analysis complete: $analysis_root/REPORT.md"

