#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/data/user/agent_self_evolution_gradient}
num_shards=${NUM_SHARDS:-4}
top_k=${TOP_K_MOVERS:-12}
output_root=${OUTPUT_ROOT:-$root/outputs/token_value_alignment}
mkdir -p "$output_root/logs"
touch "$output_root/TMUX_RUN_STARTED"

pids=()
for shard in $(seq 0 $((num_shards - 1))); do
  (
    for split in valid_seen valid_unseen; do
      env CUDA_VISIBLE_DEVICES="$shard" PYTHONPATH="$root/src" \
        "$root/.venv/bin/python" "$root/scripts/probe_action_token_logits.py" \
        --decision-file "$root/data/alfworld_expert_large/${split}.jsonl" \
        --value-trace "$root/outputs/action_value_alignment/analysis/trace.jsonl" \
        --base-model "$root/model" \
        --seed-model "$root/seed_model" \
        --tokenizer-path "$root/model" \
        --output-dir "$output_root/${split}_shard_${shard}" \
        --device cuda:0 \
        --split "$split" \
        --shard-index "$shard" \
        --num-shards "$num_shards" \
        --batch-size 4 \
        --top-k-movers "$top_k" \
        --resume
    done
  ) >"$output_root/logs/gpu${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed )); then
  echo "At least one token-value shard failed; inspect $output_root/logs" >&2
  exit 1
fi

seen=$(find "$output_root" -path '*/valid_seen_shard_*/trace.jsonl' -type f -exec wc -l {} + | awk 'END {print $1+0}')
unseen=$(find "$output_root" -path '*/valid_unseen_shard_*/trace.jsonl' -type f -exec wc -l {} + | awk 'END {print $1+0}')
if [[ "$seen" != 444 || "$unseen" != 527 ]]; then
  echo "Completeness failure: valid_seen=$seen/444 valid_unseen=$unseen/527" >&2
  exit 1
fi
touch "$output_root/TMUX_RUN_COMPLETE"
echo "Token-value full-vocabulary probing completed successfully."

