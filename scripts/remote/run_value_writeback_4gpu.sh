#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/data/user/agent_self_evolution_gradient}
num_shards=${NUM_SHARDS:-4}
sample_size=${SAMPLE_SIZE:-32}
controls_per_bucket=${CONTROLS_PER_BUCKET:-1}
output_root=${OUTPUT_ROOT:-$root/outputs/value_gradient_writeback_v1}
mkdir -p "$output_root/logs"
touch "$output_root/TMUX_RUN_STARTED"

pids=()
for shard in $(seq 0 $((num_shards - 1))); do
  (
    for split in valid_seen valid_unseen; do
      env CUDA_VISIBLE_DEVICES="$shard" PYTHONPATH="$root/src" \
        "$root/.venv/bin/python" "$root/scripts/probe_value_gradient_writeback.py" \
        --decision-file "$root/data/alfworld_expert_large/${split}.jsonl" \
        --value-trace "$root/outputs/action_value_alignment/analysis/trace.jsonl" \
        --base-model "$root/model" \
        --tokenizer-path "$root/model" \
        --output-file "$output_root/${split}_shard_${shard}.json" \
        --device cuda:0 \
        --split "$split" \
        --sample-size "$sample_size" \
        --sample-seed 20260903 \
        --shard-index "$shard" \
        --num-shards "$num_shards" \
        --controls-per-bucket "$controls_per_bucket" \
        --step-norm 0.01 \
        --target-source-kl 0.0001 \
        --calibration-steps 3 \
        --calibration-tolerance 0.25 \
        >"$output_root/logs/${split}_gpu${shard}.log" 2>&1
    done
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed )); then
  echo "At least one writeback shard failed; inspect $output_root/logs" >&2
  exit 1
fi

for split in valid_seen valid_unseen; do
  total=$(
    "$root/.venv/bin/python" -c \
      'import json,glob,sys; print(sum(len(json.load(open(p))["sources"]) for p in glob.glob(sys.argv[1])))' \
      "$output_root/${split}_shard_*.json"
  )
  if [[ "$total" != "$sample_size" ]]; then
    echo "Completeness failure: $split=$total/$sample_size" >&2
    exit 1
  fi
done
touch "$output_root/TMUX_RUN_COMPLETE"
echo "Value-gradient writeback probing completed successfully."
