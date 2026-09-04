#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BOOTSTRAP_SAMPLES = 10_000


def verb_bootstrap(values, seed):
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    draws = np.mean(array[selected], axis=1)
    return {
        "mean": float(np.mean(array)),
        "cluster_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "verb_positive_rate": float(np.mean(array > 0)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def summarize(result):
    rows = defaultdict(list)
    validation = {"restore_max": 0.0, "gradient_reproduction_max": 0.0}
    per_verb = {}
    for verb_row in result["verb_results"]:
        verb = verb_row["verb"]
        validation["gradient_reproduction_max"] = max(
            validation["gradient_reproduction_max"],
            verb_row["gradient_score_reproduction_max_error"],
        )
        per_verb[verb] = {}
        for step, update in zip(result["step_norms"], verb_row["dose_updates"], strict=True):
            validation["restore_max"] = max(validation["restore_max"], update["restore_max_absolute_error"])
            by_category = defaultdict(list)
            for state in update["states"]:
                by_category[state["relationship"]].append(state)
                if state["relationship"] != "source":
                    by_category["all_holdout"].append(state)
            per_verb[verb][str(step)] = {}
            for category, states in by_category.items():
                item = {
                    "count": len(states),
                    "mean_value_delta": float(np.mean([x["expected_value_delta"] for x in states])),
                    "mean_kl": float(np.mean([x["kl_baseline_to_updated"] for x in states])),
                    "mean_total_variation": float(np.mean([x["total_variation"] for x in states])),
                    "top_flip_rate": float(np.mean([x["baseline_top_index"] != x["updated_top_index"] for x in states])),
                    "top_value_improve_rate": float(np.mean([x["top_value_delta"] > 0 for x in states])),
                    "top_value_harm_rate": float(np.mean([x["top_value_delta"] < 0 for x in states])),
                }
                per_verb[verb][str(step)][category] = item
                rows[(step, category)].append(item)
    aggregate = {}
    for position, ((step, category), items) in enumerate(rows.items()):
        aggregate.setdefault(str(step), {})[category] = {
            key: float(np.mean([item[key] for item in items]))
            for key in (
                "mean_value_delta", "mean_kl", "mean_total_variation", "top_flip_rate",
                "top_value_improve_rate", "top_value_harm_rate",
            )
        }
        aggregate[str(step)][category]["mean_value_delta_inference"] = verb_bootstrap(
            [item["mean_value_delta"] for item in items], 20260903 + position * 1009
        )
    contrasts = {}
    for step in result["step_norms"]:
        key = str(step)
        differences = [
            per_verb[verb][key]["same_action_holdout"]["mean_value_delta"]
            - per_verb[verb][key]["different_task_different_action"]["mean_value_delta"]
            for verb in result["verbs"]
        ]
        contrasts[key] = verb_bootstrap(differences, 20260903 + int(step * 1e9) % 1_000_003)
    return {
        "validation": validation,
        "aggregate_over_verbs": aggregate,
        "same_action_minus_unrelated": contrasts,
        "per_verb": per_verb,
    }


def build_report(analysis):
    lines = [
        "# 多源价值梯度剂量响应",
        "",
        "每个动作的方向由 12 个独立 episode 源梯度的原始均值构成；每类留出含 6 个独立 episode。扫描参数 L2 剂量，目标是找到不再只是微扰、同时无关状态损害仍可控的正式比较尺度。",
        "",
    ]
    for split in ("valid_seen", "valid_unseen"):
        lines.extend([f"## {split}", ""])
        for group in ("last_mlp", "last_four_blocks"):
            data = analysis[split][group]
            lines.extend(
                [
                    f"### {group}",
                    "",
                    "| L2 step | source ΔV | all holdout ΔV | same-action ΔV | unrelated ΔV | holdout TV | holdout top flip | top improve | top harm |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for step, categories in data["aggregate_over_verbs"].items():
                source = categories["source"]
                holdout = categories["all_holdout"]
                same = categories["same_action_holdout"]
                unrelated = categories["different_task_different_action"]
                lines.append(
                    f"| {float(step):.6g} | {source['mean_value_delta']:+.6f} | {holdout['mean_value_delta']:+.6f} | {same['mean_value_delta']:+.6f} | {unrelated['mean_value_delta']:+.6f} | {holdout['mean_total_variation']:.3%} | {holdout['top_flip_rate']:.1%} | {holdout['top_value_improve_rate']:.1%} | {holdout['top_value_harm_rate']:.1%} |"
                )
            lines.extend(
                [
                    "",
                    "| L2 step | same-action ΔV 的 verb-cluster 95% CI | 正向 verb | same-action − unrelated ΔV [95% CI] |",
                    "|---:|---:|---:|---:|",
                ]
            )
            for step, categories in data["aggregate_over_verbs"].items():
                inference = categories["same_action_holdout"]["mean_value_delta_inference"]
                contrast = data["same_action_minus_unrelated"][step]
                lines.append(
                    f"| {float(step):.6g} | [{inference['cluster_ci95'][0]:+.6f}, {inference['cluster_ci95'][1]:+.6f}] | {inference['verb_positive_rate']:.0%} | {contrast['mean']:+.6f} [{contrast['cluster_ci95'][0]:+.6f}, {contrast['cluster_ci95'][1]:+.6f}] |"
                )
            lines.append("")
    lines.extend(
        [
            "## 选择规则",
            "",
            "正式多源/单源比较应选择能产生清晰同动作价值增益和非零 top-1 翻转、但无关状态 top-value harm 未明显扩大的最小剂量。若不存在这样的剂量，应把结论记为局部梯度不可安全放大，而不是继续增加步长。",
            "",
        ]
    )
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
    analysis = {
        split: {group: summarize(loaded[(split, group)]) for group in ("last_mlp", "last_four_blocks")}
        for split in ("valid_seen", "valid_unseen")
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = build_report(analysis)
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "ANALYSIS_COMPLETE").touch()
    print(report)


if __name__ == "__main__":
    main()
