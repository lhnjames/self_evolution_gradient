#!/usr/bin/env bash
set -euo pipefail

cd /data/user/agent_self_evolution_gradient
mkdir -p outputs/seed_checkpoint_plain/logs

run_shard() {
  local gpu="$1"
  local seen_offset="$2"
  local seen_count="$3"
  local unseen_offset="$4"
  local unseen_count="$5"
  export CUDA_VISIBLE_DEVICES="$gpu"

  PYTHONPATH=src .venv/bin/python scripts/score_alfworld_checkpoint.py \
    --decision-file alfworld_expert_large/valid_seen.jsonl \
    --model-path seed_model \
    --tokenizer-path model \
    --condition-name seed_checkpoint_same_plain_prompt \
    --output-dir "outputs/seed_checkpoint_plain/valid_seen_shard_${gpu}" \
    --device cuda:0 \
    --decision-offset "$seen_offset" \
    --max-decisions "$seen_count" \
    --batch-size 4

  PYTHONPATH=src .venv/bin/python scripts/score_alfworld_checkpoint.py \
    --decision-file alfworld_expert_large/valid_unseen.jsonl \
    --model-path seed_model \
    --tokenizer-path model \
    --condition-name seed_checkpoint_same_plain_prompt \
    --output-dir "outputs/seed_checkpoint_plain/valid_unseen_shard_${gpu}" \
    --device cuda:0 \
    --decision-offset "$unseen_offset" \
    --max-decisions "$unseen_count" \
    --batch-size 4
}

run_shard 0 0 111 0 132 >outputs/seed_checkpoint_plain/logs/gpu0.log 2>&1 &
run_shard 1 111 111 132 132 >outputs/seed_checkpoint_plain/logs/gpu1.log 2>&1 &
run_shard 2 222 111 264 132 >outputs/seed_checkpoint_plain/logs/gpu2.log 2>&1 &
run_shard 3 333 111 396 131 >outputs/seed_checkpoint_plain/logs/gpu3.log 2>&1 &

wait
