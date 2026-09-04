#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CATEGORIES = (
    "source",
    "same_action_holdout",
    "same_task_different_action",
    "different_task_different_action",
    "all_holdout",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def spearman(x, y) -> float:
    left = ranks(np.asarray(x, dtype=np.float64))
    right = ranks(np.asarray(y, dtype=np.float64))
    if np.std(left) <= 1e-20 or np.std(right) <= 1e-20:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def clustered_mean(rows, samples: int, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for verb, value in rows:
        grouped[verb].append(float(value))
    keys = sorted(grouped)
    means = np.asarray([np.mean(grouped[key]) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        chosen = rng.integers(0, len(keys), size=(count, len(keys)))
        draws[start : start + count] = np.mean(means[chosen], axis=1)
    return {
        "observations": len(rows),
        "verb_clusters": len(keys),
        "mean": float(np.mean(means)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "verb_positive_rate": float(np.mean(means > 0)),
    }


def bootstrap_spearman(records, samples: int, seed: int) -> dict[str, Any]:
    by_verb: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in records:
        by_verb[row["verb"]].append(row)
    verbs = sorted(by_verb)
    observed = spearman(
        [row["gradient_cosine"] for row in records],
        [row["target_value_delta"] for row in records],
    )
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        chosen = rng.integers(0, len(verbs), size=len(verbs))
        sample = [row for index in chosen for row in by_verb[verbs[index]]]
        value = spearman(
            [row["gradient_cosine"] for row in sample],
            [row["target_value_delta"] for row in sample],
        )
        if math.isfinite(value):
            draws.append(value)
    return {
        "directed_pairs": len(records),
        "verb_clusters": len(verbs),
        "spearman": observed,
        "cluster_bootstrap_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_positive": float(np.mean(np.asarray(draws) > 0)),
    }


def load_inputs(paths: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for path in paths:
        row = json.loads(Path(path).read_text(encoding="utf-8"))
        key = (row["split"], row["parameter_group"])
        if key in result:
            raise ValueError(f"Duplicate condition {key}")
        if [item["verb"] for item in row["verb_results"]] != row["verbs"]:
            raise ValueError(f"Incomplete verb ordering in {path}")
        result[key] = row
    expected = {
        (split, group)
        for split in ("valid_seen", "valid_unseen")
        for group in ("last_mlp", "last_four_blocks")
    }
    if set(result) != expected:
        raise ValueError(f"Input completeness failure: {sorted(result)}")
    return result


def analyze_condition(result: dict[str, Any], samples: int, seed: int) -> dict[str, Any]:
    comparisons: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conflicts = []
    coherence = []
    max_restore = 0.0
    max_repeat = 0.0
    max_reproduction = 0.0
    for verb_row in result["verb_results"]:
        verb = verb_row["verb"]
        source_indices = verb_row["selection"]["sources"]
        multi_states = verb_row["mean_gradient_update"]["states"]
        singles = verb_row["single_source_updates"]
        max_restore = max(
            max_restore,
            verb_row["mean_gradient_update"]["restore_max_absolute_error"],
            *(row["restore_max_absolute_error"] for row in singles),
        )
        max_repeat = max(max_repeat, verb_row["baseline_repeat_max_absolute_error"])
        max_reproduction = max(max_reproduction, verb_row["gradient_score_reproduction_max_error"])
        coherence.append(
            {
                "verb": verb,
                "mean_norm_ratio": verb_row["mean_norm_over_mean_individual_norm"],
                "mean_cosine_to_sources": float(np.mean(verb_row["mean_direction_cosines_to_sources"])),
                "minimum_cosine_to_sources": float(np.min(verb_row["mean_direction_cosines_to_sources"])),
            }
        )
        for position, multi_state in enumerate(multi_states):
            category = multi_state["relationship"]
            if category == "source":
                source_position = source_indices.index(multi_state["global_decision_index"])
                single_delta = singles[source_position]["states"][position]["expected_value_delta"]
            else:
                single_delta = float(np.mean([
                    update["states"][position]["expected_value_delta"] for update in singles
                ]))
            row = {
                "verb": verb,
                "global_decision_index": multi_state["global_decision_index"],
                "multi_value_delta": multi_state["expected_value_delta"],
                "single_comparator_value_delta": single_delta,
                "multi_minus_single": multi_state["expected_value_delta"] - single_delta,
            }
            comparisons[category].append(row)
            if category != "source":
                comparisons["all_holdout"].append(row)

        for pair in verb_row["pairwise_source_gradients"]:
            conflicts.extend(
                [
                    {
                        "verb": verb,
                        "gradient_source_index": pair["left_source_index"],
                        "target_source_index": pair["right_source_index"],
                        "gradient_cosine": pair["gradient_cosine"],
                        "target_value_delta": pair["left_update_on_right_value_delta"],
                    },
                    {
                        "verb": verb,
                        "gradient_source_index": pair["right_source_index"],
                        "target_source_index": pair["left_source_index"],
                        "gradient_cosine": pair["gradient_cosine"],
                        "target_value_delta": pair["right_update_on_left_value_delta"],
                    },
                ]
            )

    summaries = {}
    counter = 0
    for category in CATEGORIES:
        rows = comparisons[category]
        summaries[category] = {
            field: clustered_mean(
                [(row["verb"], row[field]) for row in rows], samples, seed + counter
            )
            for counter, field in enumerate(
                ("multi_value_delta", "single_comparator_value_delta", "multi_minus_single"),
                start=counter,
            )
        }
        counter += 3

    cosines = np.asarray([row["gradient_cosine"] for row in conflicts], dtype=np.float64)
    deltas = np.asarray([row["target_value_delta"] for row in conflicts], dtype=np.float64)
    edges = np.quantile(cosines, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = []
    for index in range(4):
        selected = (cosines >= edges[index]) & (
            cosines <= edges[index + 1] if index == 3 else cosines < edges[index + 1]
        )
        bins.append(
            {
                "quartile": index + 1,
                "cosine_range": [float(edges[index]), float(edges[index + 1])],
                "count": int(np.sum(selected)),
                "mean_value_delta": float(np.mean(deltas[selected])),
                "positive_transfer_rate": float(np.mean(deltas[selected] > 0)),
            }
        )
    per_verb_conflict = {}
    for verb in result["verbs"]:
        selected = [row for row in conflicts if row["verb"] == verb]
        per_verb_conflict[verb] = {
            "spearman": spearman(
                [row["gradient_cosine"] for row in selected],
                [row["target_value_delta"] for row in selected],
            ),
            "mean_cosine": float(np.mean([row["gradient_cosine"] for row in selected])),
            "mean_value_delta": float(np.mean([row["target_value_delta"] for row in selected])),
            "positive_transfer_rate": float(np.mean([row["target_value_delta"] > 0 for row in selected])),
        }
    return {
        "metadata": {
            key: result[key]
            for key in (
                "split", "parameter_group", "parameter_count", "dtype", "objective",
                "step_protocol", "parameter_delta_l2_norm", "source_count_per_verb",
                "holdout_count_per_category_per_verb", "verbs", "sample_seed",
            )
        },
        "validation": {
            "maximum_restore_parameter_error": max_restore,
            "maximum_baseline_repeat_score_error": max_repeat,
            "maximum_gradient_score_reproduction_error": max_reproduction,
        },
        "gradient_coherence_by_verb": coherence,
        "multi_vs_single": summaries,
        "multi_vs_single_rows": dict(comparisons),
        "conflict_geometry": {
            "overall": bootstrap_spearman(conflicts, samples, seed + 100_000),
            "cosine_quartiles": bins,
            "per_verb": per_verb_conflict,
            "directed_pair_rows": conflicts,
        },
    }


def ci(item):
    return f"{item['mean']:+.6f} [{item['ci95'][0]:+.6f},{item['ci95'][1]:+.6f}]"


def build_report(analysis):
    example = analysis["valid_seen"]["last_mlp"]["metadata"]
    source_count = example["source_count_per_verb"]
    holdout_count = example["holdout_count_per_category_per_verb"]
    lines = [
        "# 多源长期价值梯度共同方向与冲突因果实验",
        "",
        f"每个动作使用 {source_count} 个不同 episode 的源状态，每类留出使用 {holdout_count} 个不同 episode；多源原始梯度均值归一化后，与每个单源梯度使用完全相同的参数 L2 写回预算。置信区间以 action verb 为聚类单位；只有 5 个 verb cluster，因此同时保留逐 verb 结果。",
        "",
    ]
    for split in ("valid_seen", "valid_unseen"):
        lines.extend([f"## {split}", ""])
        for group in ("last_mlp", "last_four_blocks"):
            data = analysis[split][group]
            valid = data["validation"]
            lines.extend(
                [
                    f"### {group}",
                    "",
                    f"L2 步长：{data['metadata']['parameter_delta_l2_norm']}; 恢复/基线重算/梯度分数复现最大误差：{valid['maximum_restore_parameter_error']:.3g} / {valid['maximum_baseline_repeat_score_error']:.3g} / {valid['maximum_gradient_score_reproduction_error']:.3g}。",
                    "",
                    "| 评估集合 | 多源均值方向 ΔV | 单源比较量 ΔV | 多源减单源 |",
                    "|---|---:|---:|---:|",
                ]
            )
            for category in CATEGORIES:
                row = data["multi_vs_single"][category]
                lines.append(
                    f"| {category} | {ci(row['multi_value_delta'])} | {ci(row['single_comparator_value_delta'])} | {ci(row['multi_minus_single'])} |"
                )
            conflict = data["conflict_geometry"]["overall"]
            lines.extend(
                [
                    "",
                    f"源间精确梯度 cosine 与真实定向写回 ΔV 的 Spearman：**{conflict['spearman']:+.4f}**，verb-cluster 95% CI [{conflict['cluster_bootstrap_ci95'][0]:+.4f},{conflict['cluster_bootstrap_ci95'][1]:+.4f}]，定向 pairs={conflict['directed_pairs']}。",
                    "",
                    "| cosine 四分位 | cosine 范围 | 平均目标 ΔV | 正迁移率 |",
                    "|---|---:|---:|---:|",
                ]
            )
            for row in data["conflict_geometry"]["cosine_quartiles"]:
                lines.append(
                    f"| Q{row['quartile']} | [{row['cosine_range'][0]:+.4f},{row['cosine_range'][1]:+.4f}] | {row['mean_value_delta']:+.6f} | {row['positive_transfer_rate']:.1%} |"
                )
            lines.extend(
                [
                    "",
                    "| verb | mean-gradient norm ratio | mean direction→source cosine | pairwise cosine↔ΔV Spearman |",
                    "|---|---:|---:|---:|",
                ]
            )
            coherence = {row["verb"]: row for row in data["gradient_coherence_by_verb"]}
            for verb in data["metadata"]["verbs"]:
                row = coherence[verb]
                correlation = data["conflict_geometry"]["per_verb"][verb]["spearman"]
                lines.append(
                    f"| {verb} | {row['mean_norm_ratio']:.4f} | {row['mean_cosine_to_sources']:.4f} | {correlation:+.4f} |"
                )
            lines.append("")
    lines.extend(
        [
            "## 解释边界",
            "",
            "均值方向是共同成分的科学探针，不是最终算法。五个 verb cluster 的区间只能作为跨动作类型复现证据；不能把同一 verb 内的多对状态当成完全独立样本。只有在均值方向优于单源、且梯度几何能预测真实交叉写回时，才应继续顺序累积实验。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    inputs = load_inputs(args.inputs)
    analysis = {"bootstrap_samples": args.bootstrap_samples, "bootstrap_cluster": "action_verb"}
    counter = 0
    for split in ("valid_seen", "valid_unseen"):
        analysis[split] = {}
        for group in ("last_mlp", "last_four_blocks"):
            analysis[split][group] = analyze_condition(
                inputs[(split, group)], args.bootstrap_samples, args.seed + counter * 100_003
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
