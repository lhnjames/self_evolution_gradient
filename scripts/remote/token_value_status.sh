#!/usr/bin/env bash
set -u

root=${ROOT:-/data/user/agent_self_evolution_gradient}
output_root=${OUTPUT_ROOT:-$root/outputs/token_value_alignment}
date '+%F %T %Z'
echo "host: $(hostname)"
printf 'probe_processes: '
pgrep -fc '[p]robe_action_token_logits.py' || true
for split in valid_seen valid_unseen; do
  count=$(find "$output_root" -path "*/${split}_shard_*/trace.jsonl" -type f -exec wc -l {} + 2>/dev/null | awk 'END {print $1+0}')
  target=527
  [[ "$split" == valid_seen ]] && target=444
  printf '%-12s states=%s/%s\n' "$split" "$count" "$target"
done
printf 'trace_size: '
du -sh "$output_root" 2>/dev/null | awk '{print $1}' || echo 0
printf 'error_logs: '
grep -lE 'Traceback|CUDA out of memory|AssertionError|ValueError' "$output_root/logs"/*.log 2>/dev/null | wc -l
echo "markers:"
find "$output_root" -maxdepth 2 -type f \( -name 'TMUX_*' -o -name 'ANALYSIS_COMPLETE' \) -printf '  %TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort
echo "gpus 0-3 (index, memory MiB, util %):"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | sed -n '1,4p'
