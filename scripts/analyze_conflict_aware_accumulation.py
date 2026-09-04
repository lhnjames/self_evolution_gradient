#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CATEGORIES = ("source", "same_action_holdout", "same_task_different_action", "different_task_different_action")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def summarize(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "mean_value_delta": float(np.mean([x["expected_value_delta"] for x in rows])),
        "mean_kl": float(np.mean([x["kl_baseline_to_updated"] for x in rows])),
        "mean_total_variation": float(np.mean([x["total_variation"] for x in rows])),
        "top_flip_rate": float(np.mean([x["baseline_top_index"] != x["updated_top_index"] for x in rows])),
        "top_value_harm_rate": float(np.mean([x["top_value_delta"] < 0 for x in rows])),
    }


def clustered_bootstrap(values: dict[str, float], samples: int, seed: int) -> dict:
    array = np.asarray([values[k] for k in sorted(values)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.mean(array[rng.integers(0, len(array), size=(samples, len(array)))], axis=1)
    return {
        "mean": float(np.mean(array)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "verb_positive_rate": float(np.mean(array > 0)),
    }


def analyze_one(result: dict, samples: int, seed: int) -> dict:
    output = {
        "metadata": {key: result[key] for key in (
            "split", "parameter_group", "parameter_count", "protocol", "reference_definition",
            "single_action_total_norm", "joint_rms_nominal_norm", "increment_parameter_l2_norm",
            "source_count_per_verb", "holdout_count_per_category_per_verb", "verbs", "strategies",
            "checkpoints", "sample_seed", "unique_evaluation_state_count",
        )},
        "validation": {
            "repeat_max": result["baseline_repeat_max_absolute_error"],
            "restore_max": result["restore_max_absolute_error"],
            "reproduction_max": max(
                step["gradient_score_reproduction_max_error"]
                for strategy in result["strategy_results"] for step in strategy["steps"]
            ),
        },
        "strategies": {},
    }
    counter = 0
    for strategy in result["strategy_results"]:
        checkpoints = {}
        for checkpoint in strategy["checkpoints"]:
            categories = {}
            for category in CATEGORIES:
                selected = [x for x in checkpoint["states"] if x["relationship"] == category]
                overall = summarize(selected)
                by_verb = {
                    verb: summarize([x for x in selected if x["anchor_verb"] == verb])
                    for verb in result["verbs"]
                }
                overall["value_delta_inference"] = clustered_bootstrap(
                    {verb: row["mean_value_delta"] for verb, row in by_verb.items()},
                    samples, seed + counter,
                )
                counter += 1
                categories[category] = {"overall": overall, "by_anchor_verb": by_verb}
            checkpoints[str(checkpoint["round_index"])] = {
                "experience_count": checkpoint["experience_count"],
                "accepted_count": checkpoint["accepted_count"],
                "projected_count": checkpoint["projected_count"],
                "parameter_l2_distance_from_base": checkpoint["parameter_l2_distance_from_base"],
                "categories": categories,
            }
        output["strategies"][strategy["strategy"]] = {
            "accepted_count": strategy["accepted_count"],
            "projected_count": strategy["projected_count"],
            "negative_cosine_count": sum(
                step["raw_cosine_to_running_reference"] is not None
                and step["raw_cosine_to_running_reference"] < 0
                for step in strategy["steps"]
            ),
            "checkpoints": checkpoints,
        }
    return output


def build_report(analysis: dict) -> str:
    lines = [
        "# 多动作冲突感知梯度累积",
        "",
        "五种动作各 12 条经验按轮交错。`direct` 全写入；`positive_filter` 对当前梯度与历史累计方向余弦小于 0 的经验跳过；`project` 消除该负向分量并保持相同单步参数 L2。剂量按动作数平方根归一，使五动作近似正交时的联合 RMS 参数剂量等于此前单动作 10× 正式剂量。",
        "",
    ]
    for split in ("valid_seen", "valid_unseen"):
        lines.extend([f"## {split}", ""])
        for group in ("last_mlp", "last_four_blocks"):
            data = analysis[split][group]
            valid = data["validation"]
            lines.extend([
                f"### {group}",
                "",
                f"重算/恢复/梯度复现最大误差：{valid['repeat_max']:.3g} / {valid['restore_max']:.3g} / {valid['reproduction_max']:.3g}。",
                "",
                "| 策略 | 轮次/经验 | 接受/投影 | 实际漂移 L2 | 同动作留出 ΔV [verb-bootstrap 95% CI] | 同任务异动作 ΔV | 异任务异动作 ΔV | 同动作 flip / harm |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for strategy_name in ("direct", "positive_filter", "project"):
                strategy = data["strategies"][strategy_name]
                for round_index in data["metadata"]["checkpoints"]:
                    row = strategy["checkpoints"][str(round_index)]
                    same = row["categories"]["same_action_holdout"]["overall"]
                    same_ci = same["value_delta_inference"]["ci95"]
                    control = row["categories"]["same_task_different_action"]["overall"]
                    unrelated = row["categories"]["different_task_different_action"]["overall"]
                    lines.append(
                        f"| {strategy_name} | {round_index}/{row['experience_count']} | "
                        f"{row['accepted_count']}/{row['projected_count']} | {row['parameter_l2_distance_from_base']:.6f} | "
                        f"{same['mean_value_delta']:+.6f} [{same_ci[0]:+.6f},{same_ci[1]:+.6f}] | "
                        f"{control['mean_value_delta']:+.6f} | {unrelated['mean_value_delta']:+.6f} | "
                        f"{same['top_flip_rate']:.1%} / {same['top_value_harm_rate']:.1%} |"
                    )
            lines.append("")
    lines.extend([
        "## 进入完整回合评测的门槛",
        "",
        "冲突感知策略必须在两种 split 上同时满足：最终同动作留出价值为正；相对 direct 减少同任务异动作伤害；同动作最高价值动作伤害不高于 2%；且效果不是仅由显著更小的实际参数漂移解释。满足后才保存该策略的参数增量并运行配对的完整 ALFWorld 回合。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    loaded = {}
    for path in args.inputs:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        loaded[(result["split"], result["parameter_group"])] = result
    expected = {(s, g) for s in ("valid_seen", "valid_unseen")
                for g in ("last_mlp", "last_four_blocks")}
    if set(loaded) != expected:
        raise ValueError(f"Incomplete inputs: {sorted(loaded)}")
    analysis = {"bootstrap_samples": args.bootstrap_samples, "bootstrap_cluster": "anchor_action_verb"}
    counter = 0
    for split in ("valid_seen", "valid_unseen"):
        analysis[split] = {}
        for group in ("last_mlp", "last_four_blocks"):
            analysis[split][group] = analyze_one(
                loaded[(split, group)], args.bootstrap_samples, args.seed + counter * 100_003
            )
            counter += 1
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = build_report(analysis)
    (output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "ANALYSIS_COMPLETE").touch()
    print(report)


if __name__ == "__main__":
    main()
