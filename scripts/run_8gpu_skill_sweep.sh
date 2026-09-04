#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/hanning/agent_self_evolution_gradient
CONFIG=${1:-$ROOT/config/skill_experiment.yaml}
OUTPUT=${2:-$ROOT/outputs/8gpu_skill_experiment}
export TMPDIR=$ROOT/.tmp
export HF_HOME=$ROOT/model_cache
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT"

pids=()
for gpu in $(seq 0 7); do
  seed=$((200 + gpu))
  CUDA_VISIBLE_DEVICES=$gpu \
    "$ROOT/.venv/bin/python" -m self_evolve.skill_runner \
      --config "$CONFIG" --output "$OUTPUT/seed_$seed" --device cuda:0 --seed "$seed" \
      >"$OUTPUT/seed_$seed.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
