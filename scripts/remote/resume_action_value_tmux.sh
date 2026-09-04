#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output_root="$repo_root/outputs/action_value_alignment"
mkdir -p "$output_root"

for marker in TMUX_RUN_STARTED TMUX_RUN_COMPLETE TMUX_RUN_FAILED; do
  if test -e "$output_root/$marker"; then
    unlink "$output_root/$marker"
  fi
done
touch "$output_root/TMUX_RUN_STARTED"

if env \
  NUM_SHARDS=32 \
  EXPERT_TYPE=handcoded \
  VALUE_SEED=20260902 \
  OUTPUT_ROOT="$output_root" \
  bash "$repo_root/scripts/remote/run_action_value_4shards.sh"; then
  touch "$output_root/TMUX_RUN_COMPLETE"
  echo "Action-value evaluation completed successfully."
else
  status=$?
  touch "$output_root/TMUX_RUN_FAILED"
  echo "Action-value evaluation failed with status $status."
  exit "$status"
fi

