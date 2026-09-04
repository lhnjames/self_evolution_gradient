#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/data/user/agent_self_evolution_gradient}
gradient_root=${GRADIENT_ROOT:-$root/outputs/skill_gradient_purification_300x_v1}
output_root=${OUTPUT_ROOT:-$root/outputs/skill_gradient_routed_300x_v1}
aggregate=${AGGREGATE:-purified12}
seeds=(20260904 20260921 20260938 20260955)
mkdir -p "$output_root/logs"
touch "$output_root/TMUX_RUN_STARTED"

pids=()
for gpu in 0 1 2 3; do
  seed=${seeds[$gpu]}
  (
    for route_mode in oracle_skill base_predicted_skill; do
      for split in valid_seen valid_unseen; do
        name="${aggregate}_${route_mode}_seed_${seed}"
        env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/src:$root/scripts" \
          "$root/.venv/bin/python" "$root/scripts/score_skill_conditioned_delta_bank.py" \
          --decision-file "$root/data/alfworld_expert_large/${split}.jsonl" \
          --base-score-trace "$root/outputs/horizontal_300x_vs_seed_fp32/${split}_base_fp32/trace.jsonl" \
          --base-model "$root/model" --tokenizer-path "$root/model" \
          --delta-manifest "$gradient_root/seed_${seed}/deltas/manifest.json" \
          --aggregate "$aggregate" --route-mode "$route_mode" --condition-name "$name" \
          --output-dir "$output_root/${split}_${name}" --device cuda:0
      done
    done
  ) >"$output_root/logs/seed_${seed}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if (( failed )); then
  echo "At least one routed skill-gradient score failed; inspect logs." >&2
  exit 1
fi
touch "$output_root/SCORING_COMPLETE"

value_args=()
for split in valid_seen valid_unseen; do
  for path in "$root"/outputs/action_value_alignment/${split}_shard_*/trace.jsonl; do
    value_args+=(--value-trace "$path")
  done
done
analysis_args=()
for route_mode in oracle_skill base_predicted_skill; do
  for seed in "${seeds[@]}"; do
    name="${aggregate}_${route_mode}_seed_${seed}"
    analysis_dir="$output_root/analysis/$name"
    PYTHONPATH="$root/src" "$root/.venv/bin/python" \
      "$root/scripts/analyze_action_value_alignment.py" \
      --split valid_seen --split valid_unseen "${value_args[@]}" \
      --base-trace "$root/outputs/horizontal_300x_vs_seed_fp32/valid_seen_base_fp32/trace.jsonl" \
      --base-trace "$root/outputs/horizontal_300x_vs_seed_fp32/valid_unseen_base_fp32/trace.jsonl" \
      --seed-trace "$output_root/valid_seen_${name}/trace.jsonl" \
      --seed-trace "$output_root/valid_unseen_${name}/trace.jsonl" \
      --output-dir "$analysis_dir" --bootstrap-samples 10000 \
      --seed "$seed" --required-relative-improvement 0.30 --comparison-label "$name" \
      >"$output_root/analysis_${name}.log" 2>&1
    analysis_args+=(--seed-analysis "$analysis_dir/results.json")
    analysis_args+=(--seed-analysis-label "${route_mode}_seed_${seed}")
  done
done
PYTHONPATH="$root/src:$root/scripts" "$root/.venv/bin/python" \
  "$root/scripts/summarize_skill_gradient_routed.py" \
  --analysis-root "$output_root/analysis" "${analysis_args[@]}" \
  --seed-baseline-analysis "$root/outputs/horizontal_300x_vs_seed_fp32/analysis/seed_fp32/results.json" \
  --output-dir "$output_root/analysis/summary" --required-relative-improvement 0.30 \
  >"$output_root/summary.log" 2>&1
touch "$output_root/ANALYSIS_COMPLETE"
sed -n '1,160p' "$output_root/analysis/summary/REPORT.md"
