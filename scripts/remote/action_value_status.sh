#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

date '+%F %T %Z'
printf 'host: '
hostname
printf 'evaluator_processes: '
pgrep -fc '[e]valuate_alfworld_action_values.py.*outputs/action_value_alignment/valid' || true

for root in action_value_alignment action_value_alignment_planner; do
  printf '\n%s\n' "$root"
  for split in valid_seen valid_unseen; do
    states=$(find "outputs/$root" -path "*${split}_shard_*/trace.jsonl" -type f \
      -exec wc -l {} + 2>/dev/null | tail -n 1 | awk '{print $1}')
    printf '  %-12s states=%s\n' "$split" "${states:-0}"
  done
  errors=$(rg -l 'Traceback|RuntimeError|FileNotFoundError' "outputs/$root/logs" 2>/dev/null | wc -l)
  printf '  error_logs=%s\n' "$errors"
done

printf '\nmarkers\n'
for marker in \
  outputs/action_value_alignment/TMUX_RUN_STARTED \
  outputs/action_value_alignment/TMUX_RUN_COMPLETE \
  outputs/action_value_alignment/TMUX_RUN_FAILED \
  outputs/action_value_alignment/analysis/ANALYSIS_COMPLETE; do
  if test -e "$marker"; then
    stat -c '  %y %n' "$marker"
  fi
done

printf '\ngpus (index, memory MiB)\n'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits

