#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/data/user/agent_self_evolution_gradient}
gradient_root=${GRADIENT_ROOT:-$root/outputs/conflict_aware_accumulation_300x_v1}
output_root=${OUTPUT_ROOT:-$root/outputs/horizontal_300x_vs_seed_fp32}
mkdir -p "$output_root/logs"

run_condition() {
  local gpu="$1"
  local name="$2"
  local model_path="$3"
  local delta_path="$4"
  local delta_args=()
  if [[ -n "$delta_path" ]]; then delta_args=(--parameter-delta "$delta_path"); fi
  for split in valid_seen valid_unseen; do
    env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/src" \
      "$root/.venv/bin/python" "$root/scripts/score_alfworld_checkpoint.py" \
      --decision-file "$root/data/alfworld_expert_large/${split}.jsonl" \
      --model-path "$model_path" --tokenizer-path "$root/model" \
      "${delta_args[@]}" --force-float32 --condition-name "$name" \
      --output-dir "$output_root/${split}_${name}" --device cuda:0 --batch-size 4
  done
}

run_condition 0 base_fp32 "$root/model" "" >"$output_root/logs/base_gpu0.log" 2>&1 &
pid0=$!
run_condition 1 seed_fp32 "$root/seed_model" "" >"$output_root/logs/seed_gpu1.log" 2>&1 &
pid1=$!
run_condition 2 direct300_last4 "$root/model" \
  "$gradient_root/deltas/valid_seen_last_four_blocks_direct.pt" \
  >"$output_root/logs/direct_gpu2.log" 2>&1 &
pid2=$!
run_condition 3 project300_last4 "$root/model" \
  "$gradient_root/deltas/valid_seen_last_four_blocks_project.pt" \
  >"$output_root/logs/project_gpu3.log" 2>&1 &
pid3=$!

failed=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
  if ! wait "$pid"; then failed=1; fi
done
if (( failed )); then
  echo "At least one 300x horizontal scoring condition failed" >&2
  exit 1
fi
touch "$output_root/SCORING_COMPLETE"
echo "FP32 Base/SEED/direct300/project300 scoring complete."
