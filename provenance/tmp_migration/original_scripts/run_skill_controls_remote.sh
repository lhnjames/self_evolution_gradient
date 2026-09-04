#!/usr/bin/env bash
set -euo pipefail

cd /data/user/agent_self_evolution_gradient
mkdir -p outputs/skill_controls_3b/logs

run_shard() {
  local gpu="$1"
  local seen_offset="$2"
  local seen_count="$3"
  local unseen_offset="$4"
  local unseen_count="$5"

  export CUDA_VISIBLE_DEVICES="$gpu"
  PYTHONPATH=src .venv/bin/python scripts/score_alfworld_skill_controls.py \
    --decision-file alfworld_expert_large/valid_seen.jsonl \
    --model-path model \
    --skills-path references/SkillRL/memory_data/alfworld/claude_style_skills.json \
    --output-dir "outputs/skill_controls_3b/valid_seen_shard_${gpu}" \
    --device cuda:0 \
    --decision-offset "$seen_offset" \
    --max-decisions "$seen_count" \
    --batch-size 4

  PYTHONPATH=src .venv/bin/python scripts/score_alfworld_skill_controls.py \
    --decision-file alfworld_expert_large/valid_unseen.jsonl \
    --model-path model \
    --skills-path references/SkillRL/memory_data/alfworld/claude_style_skills.json \
    --output-dir "outputs/skill_controls_3b/valid_unseen_shard_${gpu}" \
    --device cuda:0 \
    --decision-offset "$unseen_offset" \
    --max-decisions "$unseen_count" \
    --batch-size 4
}

run_shard 0 0 111 0 132 >outputs/skill_controls_3b/logs/gpu0.log 2>&1 &
run_shard 1 111 111 132 132 >outputs/skill_controls_3b/logs/gpu1.log 2>&1 &
run_shard 2 222 111 264 132 >outputs/skill_controls_3b/logs/gpu2.log 2>&1 &
run_shard 3 333 111 396 131 >outputs/skill_controls_3b/logs/gpu3.log 2>&1 &

wait
