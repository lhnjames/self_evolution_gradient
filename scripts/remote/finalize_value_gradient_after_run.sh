#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/data/user/agent_self_evolution_gradient}
output_root=${OUTPUT_ROOT:-$root/outputs/value_gradient_probe_v1}
while [[ ! -f "$output_root/TMUX_RUN_COMPLETE" ]]; do
  date '+%F %T waiting for TMUX_RUN_COMPLETE'
  sleep 30
done

seen=()
unseen=()
for shard in 0 1 2 3; do
  seen+=("$output_root/valid_seen_shard_${shard}")
  unseen+=("$output_root/valid_unseen_shard_${shard}")
done
PYTHONPATH="$root/src" "$root/.venv/bin/python" \
  "$root/scripts/analyze_value_parameter_gradients.py" \
  --seen-shards "${seen[@]}" \
  --unseen-shards "${unseen[@]}" \
  --output-dir "$output_root/analysis" \
  --bootstrap-samples 10000 \
  >"$output_root/analysis.stdout.log" 2>&1
sed -n '1,280p' "$output_root/analysis/REPORT.md"
echo "Analysis complete: $output_root/analysis/REPORT.md"

