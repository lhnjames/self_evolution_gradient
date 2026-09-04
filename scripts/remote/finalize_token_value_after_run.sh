#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/data/user/agent_self_evolution_gradient}
output_root=${OUTPUT_ROOT:-$root/outputs/token_value_alignment}
while [[ ! -f "$output_root/TMUX_RUN_COMPLETE" ]]; do
  date '+%F %T waiting for TMUX_RUN_COMPLETE'
  sleep 30
done

traces=()
for split in valid_seen valid_unseen; do
  for shard in 0 1 2 3; do
    path="$output_root/${split}_shard_${shard}/trace.jsonl"
    [[ -f "$path" ]] || { echo "Missing $path" >&2; exit 1; }
    traces+=(--trace "$path")
  done
done
PYTHONPATH="$root/src" "$root/.venv/bin/python" \
  "$root/scripts/analyze_token_value_alignment.py" \
  "${traces[@]}" \
  --output-dir "$output_root/analysis" \
  --bootstrap-samples 10000 \
  >"$output_root/analysis.stdout.log" 2>&1
sed -n '1,240p' "$output_root/analysis/REPORT.md"
echo "Analysis complete: $output_root/analysis/REPORT.md"

