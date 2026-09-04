#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/data/user/agent_self_evolution_gradient}
output_root=${OUTPUT_ROOT:-$root/outputs/multisource_value_gradient_v1}
while [[ ! -f "$output_root/TMUX_RUN_COMPLETE" ]]; do
  date '+%F %T waiting for TMUX_RUN_COMPLETE'
  sleep 30
done

PYTHONPATH="$root/src" "$root/.venv/bin/python" \
  "$root/scripts/analyze_multisource_value_gradients.py" \
  --inputs \
    "$output_root/valid_seen_last_mlp.json" \
    "$output_root/valid_unseen_last_mlp.json" \
    "$output_root/valid_seen_last_four_blocks.json" \
    "$output_root/valid_unseen_last_four_blocks.json" \
  --output-dir "$output_root/analysis" \
  --bootstrap-samples 10000 \
  >"$output_root/analysis.stdout.log" 2>&1
sed -n '1,360p' "$output_root/analysis/REPORT.md"
echo "Analysis complete: $output_root/analysis/REPORT.md"
