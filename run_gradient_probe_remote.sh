#!/usr/bin/env bash
set -euo pipefail

cd /data/user/agent_self_evolution_gradient
mkdir -p outputs/gradient_probe/logs

run_probe() {
  local gpu="$1"
  export CUDA_VISIBLE_DEVICES="$gpu"

  PYTHONPATH=src .venv/bin/python scripts/probe_skill_parameter_gradients.py \
    --decision-file alfworld_expert_large/valid_seen.jsonl \
    --baseline-trace baseline_traces/valid_seen.jsonl \
    --control-traces \
      outputs/skill_controls_3b/valid_seen_shard_0/trace.jsonl \
      outputs/skill_controls_3b/valid_seen_shard_1/trace.jsonl \
      outputs/skill_controls_3b/valid_seen_shard_2/trace.jsonl \
      outputs/skill_controls_3b/valid_seen_shard_3/trace.jsonl \
    --model-path model \
    --output-dir "outputs/gradient_probe/valid_seen_shard_${gpu}" \
    --device cuda:0 \
    --sample-size 64 \
    --shard-index "$gpu" \
    --num-shards 4 \
    --batch-size 4

  PYTHONPATH=src .venv/bin/python scripts/probe_skill_parameter_gradients.py \
    --decision-file alfworld_expert_large/valid_unseen.jsonl \
    --baseline-trace baseline_traces/valid_unseen.jsonl \
    --control-traces \
      outputs/skill_controls_3b/valid_unseen_shard_0/trace.jsonl \
      outputs/skill_controls_3b/valid_unseen_shard_1/trace.jsonl \
      outputs/skill_controls_3b/valid_unseen_shard_2/trace.jsonl \
      outputs/skill_controls_3b/valid_unseen_shard_3/trace.jsonl \
    --model-path model \
    --output-dir "outputs/gradient_probe/valid_unseen_shard_${gpu}" \
    --device cuda:0 \
    --sample-size 64 \
    --shard-index "$gpu" \
    --num-shards 4 \
    --batch-size 4
}

run_probe 0 >outputs/gradient_probe/logs/gpu0.log 2>&1 &
run_probe 1 >outputs/gradient_probe/logs/gpu1.log 2>&1 &
run_probe 2 >outputs/gradient_probe/logs/gpu2.log 2>&1 &
run_probe 3 >outputs/gradient_probe/logs/gpu3.log 2>&1 &

wait
