#!/usr/bin/env bash
set -u

root=${ROOT:-/data/user/agent_self_evolution_gradient}
sample_size=${SAMPLE_SIZE:-128}
output_root=${OUTPUT_ROOT:-$root/outputs/value_gradient_probe_v1}
date '+%F %T %Z'
echo "host: $(hostname)"
printf 'probe_processes: '
pgrep -fc '[p]robe_value_parameter_gradients.py' || true
for split in valid_seen valid_unseen; do
  progress=$(grep -hE '^\[[0-9]+/[0-9]+\]' "$output_root/logs/${split}_gpu"*.log 2>/dev/null | wc -l)
  completed=$(
    "$root/.venv/bin/python" -c \
      'import json,glob,sys; print(sum(len(json.load(open(p))["states"]) for p in glob.glob(sys.argv[1])))' \
      "$output_root/${split}_shard_*/results.json" 2>/dev/null || echo 0
  )
  printf '%-12s progress=%s/%s completed_artifacts=%s\n' "$split" "$progress" "$sample_size" "$completed"
done
printf 'error_logs: '
grep -lE 'Traceback|CUDA out of memory|AssertionError|RuntimeError|ValueError' "$output_root/logs"/*.log 2>/dev/null | wc -l
echo "markers:"
find "$output_root" -maxdepth 2 -type f \( -name 'TMUX_*' -o -name 'ANALYSIS_COMPLETE' \) -printf '  %TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort
echo "gpus 0-3 (index, memory MiB, util %):"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | sed -n '1,4p'
