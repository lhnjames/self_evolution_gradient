#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/data/user/agent_self_evolution_gradient}
candidate_delta=${CANDIDATE_DELTA:?Set CANDIDATE_DELTA}
candidate_label=${CANDIDATE_LABEL:?Set CANDIDATE_LABEL}
output_root=${OUTPUT_ROOT:-$root/outputs/full_episode_300x_compare}
config=${CONFIG:-$root/config/alfworld_online_3b_remote_50.yaml}
mkdir -p "$output_root/logs"

run_condition() {
  local gpu="$1"
  local condition="$2"
  local split="$3"
  local model_path="$4"
  local delta_path="$5"
  local extra=()
  if [[ -n "$delta_path" ]]; then extra=(--parameter-delta "$delta_path"); fi
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/src" \
    "$root/.venv/bin/python" -m self_evolve.alfworld_online \
    --config "$config" --policy plain --split "$split" --device cuda:0 \
    --model-path "$model_path" --force-float32 "${extra[@]}" \
    --output-dir "$output_root/${condition}_${split}"
}

(
  run_condition 0 base valid_seen "$root/model" ""
  run_condition 0 base valid_unseen "$root/model" ""
) >"$output_root/logs/base_gpu0.log" 2>&1 &
pid0=$!
(
  run_condition 1 seed valid_seen "$root/seed_model" ""
  run_condition 1 seed valid_unseen "$root/seed_model" ""
) >"$output_root/logs/seed_gpu1.log" 2>&1 &
pid1=$!
run_condition 2 "$candidate_label" valid_seen "$root/model" "$candidate_delta" \
  >"$output_root/logs/candidate_seen_gpu2.log" 2>&1 &
pid2=$!
run_condition 3 "$candidate_label" valid_unseen "$root/model" "$candidate_delta" \
  >"$output_root/logs/candidate_unseen_gpu3.log" 2>&1 &
pid3=$!

failed=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
  if ! wait "$pid"; then failed=1; fi
done
if (( failed )); then
  echo "At least one full-episode comparison failed" >&2
  exit 1
fi
touch "$output_root/ROLLOUT_COMPLETE"
echo "Full-episode Base/SEED/$candidate_label comparison complete."
