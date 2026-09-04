#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CATEGORIES = (
    "source",
    "learned_source",
    "future_source",
    "same_action_holdout",
    "same_task_different_action",
    "different_task_different_action",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def bootstrap(values, samples, seed):
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(array), size=(samples, len(array)))
    draws = np.mean(array[chosen], axis=1)
    return {
        "mean": float(np.mean(array)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "verb_positive_rate": float(np.mean(array > 0)),
    }


def summarize_states(states, category):
    if category == "source":
        selected = [x for x in states if x["relationship"] == "source"]
    else:
        selected = [x for x in states if x["accumulation_role"] == category]
    if not selected:
        return None
    return {
        "count": len(selected),
        "mean_value_delta": float(np.mean([x["expected_value_delta"] for x in selected])),
        "mean_kl": float(np.mean([x["kl_baseline_to_updated"] for x in selected])),
        "mean_total_variation": float(np.mean([x["total_variation"] for x in selected])),
        "top_flip_rate": float(np.mean([x["baseline_top_index"] != x["updated_top_index"] for x in selected])),
        "top_value_harm_rate": float(np.mean([x["top_value_delta"] < 0 for x in selected])),
    }


def analyze_condition(result, samples, seed):
    per_verb = {}
    validation = {"repeat_max": 0.0, "restore_max": 0.0, "reproduction_max": 0.0}
    for verb_row in result["verb_results"]:
        verb = verb_row["verb"]
        validation["repeat_max"] = max(validation["repeat_max"], verb_row["baseline_repeat_max_absolute_error"])
        validation["restore_max"] = max(validation["restore_max"], verb_row["restore_max_absolute_error"])
        per_verb[verb] = {}
        for step in verb_row["steps"]:
            validation["reproduction_max"] = max(
                validation["reproduction_max"], step["gradient_score_reproduction_max_error"]
            )
            per_verb[verb][str(step["step_index"])] = {
                "parameter_l2_distance_from_base": step["parameter_l2_distance_from_base"],
                **{
                    category: summary
                    for category in CATEGORIES
                    if (summary := summarize_states(step["states"], category)) is not None
                },
            }

    aggregate = {}
    counter = 0
    for step_index in range(1, result["source_count_per_verb"] + 1):
        key = str(step_index)
        aggregate[key] = {}
        drift = [per_verb[verb][key]["parameter_l2_distance_from_base"] for verb in result["verbs"]]
        aggregate[key]["parameter_l2_distance_from_base"] = bootstrap(
            drift, samples, seed + counter
        )
        counter += 1
        for category in CATEGORIES:
            available = [per_verb[verb][key][category] for verb in result["verbs"] if category in per_verb[verb][key]]
            if not available:
                continue
            aggregate[key][category] = {
                metric: float(np.mean([row[metric] for row in available]))
                for metric in ("mean_value_delta", "mean_kl", "mean_total_variation", "top_flip_rate", "top_value_harm_rate")
            }
            aggregate[key][category]["value_delta_inference"] = bootstrap(
                [row["mean_value_delta"] for row in available], samples, seed + counter
            )
            counter += 1
    return {
        "metadata": {key: result[key] for key in (
            "split", "parameter_group", "parameter_count", "dtype", "objective", "protocol",
            "total_nominal_parameter_l2_budget", "increment_parameter_l2_norm",
            "source_count_per_verb", "holdout_count_per_category_per_verb", "verbs", "sample_seed",
        )},
        "validation": validation,
        "aggregate_over_verbs": aggregate,
        "per_verb": per_verb,
    }


def build_report(analysis):
    lines = [
        "# 在线重算价值梯度的 12 步顺序累积",
        "",
        "每一步都在当前参数上重新计算当前源状态的长期价值梯度；单步参数 L2 为正式 10× 总预算的 1/12。置信区间按 action verb 聚类。",
        "",
    ]
    checkpoints = (1, 2, 4, 8, 12)
    for split in ("valid_seen", "valid_unseen"):
        lines.extend([f"## {split}", ""])
        for group in ("last_mlp", "last_four_blocks"):
            data = analysis[split][group]
            valid = data["validation"]
            lines.extend([
                f"### {group}",
                "",
                f"恢复/基线重算/梯度复现最大误差：{valid['restore_max']:.3g} / {valid['repeat_max']:.3g} / {valid['reproduction_max']:.3g}。",
                "",
                "| 累计经验 | 实际参数漂移 L2 | 已学习源 ΔV | 未来源 ΔV | 独立同动作 ΔV [95% CI] | 同任务异动作 ΔV | 异任务异动作 ΔV | 同动作 top flip / harm |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for step in checkpoints:
                row = data["aggregate_over_verbs"][str(step)]
                same = row["same_action_holdout"]
                inference = same["value_delta_inference"]
                learned = row.get("learned_source")
                future = row.get("future_source")
                learned_text = f"{learned['mean_value_delta']:+.6f}" if learned else "—"
                future_text = f"{future['mean_value_delta']:+.6f}" if future else "—"
                lines.append(
                    f"| {step} | {row['parameter_l2_distance_from_base']['mean']:.6f} | {learned_text} | {future_text} | {same['mean_value_delta']:+.6f} [{inference['ci95'][0]:+.6f},{inference['ci95'][1]:+.6f}] | {row['same_task_different_action']['mean_value_delta']:+.6f} | {row['different_task_different_action']['mean_value_delta']:+.6f} | {same['top_flip_rate']:.1%} / {same['top_value_harm_rate']:.1%} |"
                )
            lines.append("")
    lines.extend([
        "## 判据",
        "",
        "顺序累积成立需要独立同动作留出随经验数总体提高、最终跨 verb 区间为正，同时已学习源保持、异动作控制伤害不随步数失控。否则应停在共同方向机制结论，不保存候选 checkpoint。",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    loaded = {}
    for path in args.inputs:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        loaded[(result["split"], result["parameter_group"])] = result
    expected = {(s, g) for s in ("valid_seen", "valid_unseen") for g in ("last_mlp", "last_four_blocks")}
    if set(loaded) != expected:
        raise ValueError(f"Incomplete inputs: {sorted(loaded)}")
    analysis = {"bootstrap_samples": args.bootstrap_samples, "bootstrap_cluster": "action_verb"}
    counter = 0
    for split in ("valid_seen", "valid_unseen"):
        analysis[split] = {}
        for group in ("last_mlp", "last_four_blocks"):
            analysis[split][group] = analyze_condition(
                loaded[(split, group)], args.bootstrap_samples, args.seed + counter * 100_003
            )
            counter += 1
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = build_report(analysis)
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "ANALYSIS_COMPLETE").touch()
    print(report)


if __name__ == "__main__":
    main()
