#!/usr/bin/env bash
set -u
root=${ROOT:-/data/user/agent_self_evolution_gradient}
output_root=${OUTPUT_ROOT:-$root/outputs/sequential_value_accumulation_v1}
source_count=${SOURCE_COUNT:-12}
date '+%F %T %Z'
printf 'probe_processes: '
pgrep -fc '[p]robe_sequential_value_accumulation.py' || true
for log in "$output_root"/logs/*.log; do
  [[ -e "$log" ]] || continue
  completed=$(grep -cE ' complete$' "$log" 2>/dev/null || true)
  steps=$(grep -cE ' step=[0-9]+/[0-9]+ ' "$log" 2>/dev/null || true)
  printf '%-52s verbs=%s/5 steps=%s/%s\n' "$(basename "$log")" "$completed" "$steps" "$((5 * source_count))"
done
printf 'error_logs: '
grep -lE 'Traceback|CUDA out of memory|AssertionError|RuntimeError|ValueError' "$output_root"/logs/*.log 2>/dev/null | wc -l
echo "markers:"
find "$output_root" -maxdepth 2 -type f \( -name 'TMUX_*' -o -name 'ANALYSIS_COMPLETE' \) -printf '  %TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort
echo "gpus 0-3:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | sed -n '1,4p'

