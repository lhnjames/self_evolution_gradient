#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen-files", nargs="+", required=True)
    parser.add_argument("--unseen-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def load_split(paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources: dict[int, dict[str, Any]] = {}
    metadata = None
    for path in paths:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        current = {
            key: result[key]
            for key in (
                "split",
                "dtype",
                "writeback_scope",
                "sample_size_requested_across_all_shards",
                "selected_source_indices_all_shards",
                "controls_per_relationship_bucket",
                "relationship_buckets",
                "objectives",
                "parameter_groups",
                "parameter_step_l2_norm",
                "target_source_kl",
                "calibration_steps",
                "calibration_tolerance",
            )
        }
        if metadata is None:
            metadata = current
        elif current != metadata:
            raise ValueError("Shard metadata differ")
        for source in result["sources"]:
            index = int(source["source_global_decision_index"])
            if index in sources:
                raise ValueError(f"Duplicate source {index}")
            sources[index] = source
    if metadata is None:
        raise ValueError("No input files")
    expected = set(metadata["selected_source_indices_all_shards"])
    if set(sources) != expected:
        raise ValueError(f"Completeness failure: have {len(sources)}, expected {len(expected)}")
    return [sources[index] for index in sorted(sources)], metadata


def cluster_summary(
    rows: list[tuple[int, float]], samples: int, seed: int
) -> dict[str, Any]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for source, value in rows:
        if math.isfinite(float(value)):
            grouped[source].append(float(value))
    keys = sorted(grouped)
    means = np.asarray([np.mean(grouped[key]) for key in keys], dtype=np.float64)
    if not len(means):
        return {"observations": 0, "sources": 0, "mean": math.nan, "ci95": [math.nan, math.nan]}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        choices = rng.integers(0, len(means), size=(count, len(means)))
        draws[start : start + count] = np.mean(means[choices], axis=1)
    return {
        "observations": len(rows),
        "sources": len(means),
        "mean": float(np.mean(means)),
        "median_source_mean": float(np.median(means)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_probability_positive": float(np.mean(draws > 0.0)),
        "source_positive_rate": float(np.mean(means > 0.0)),
    }


def analyze_split(
    paths: list[str], samples: int, seed: int
) -> dict[str, Any]:
    sources, metadata = load_split(paths)
    flattened: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    indexed: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    maximum_restore_error = 0.0
    maximum_repeat_error = 0.0
    calibration_hits = []
    calibrated_steps: dict[tuple[str, str], list[float]] = defaultdict(list)
    for source in sources:
        source_index = int(source["source_global_decision_index"])
        maximum_repeat_error = max(maximum_repeat_error, source["baseline_repeat_max_absolute_error"])
        for objective, objective_row in source["objectives"].items():
            for group, group_row in objective_row["groups"].items():
                maximum_restore_error = max(maximum_restore_error, group_row["restore_max_absolute_error"])
                calibrated_steps[(objective, group)].append(group_row["actual_parameter_delta_norm"])
                source_state = group_row["states"][0]
                calibration_hits.append(
                    abs(source_state["kl_baseline_to_updated"] - metadata["target_source_kl"])
                    <= metadata["calibration_tolerance"] * metadata["target_source_kl"]
                )
                for state in group_row["states"]:
                    relationship = state["relationship"]
                    flattened[(objective, group, relationship)].append((source_index, state))
                    indexed[(source_index, int(state["global_decision_index"]), objective, group)] = state

    summaries = {}
    counter = 0
    fields = (
        "expected_value_delta",
        "optimal_mass_delta",
        "expert_probability_delta",
        "kl_baseline_to_updated",
        "total_variation",
        "top_value_delta",
    )
    relationships = ["source", *metadata["relationship_buckets"], "all_heldout"]
    for objective in metadata["objectives"]:
        summaries[objective] = {}
        for group in metadata["parameter_groups"]:
            summaries[objective][group] = {}
            for relationship in relationships:
                if relationship == "all_heldout":
                    items = [
                        item
                        for bucket in metadata["relationship_buckets"]
                        for item in flattened[(objective, group, bucket)]
                    ]
                else:
                    items = flattened[(objective, group, relationship)]
                field_rows = {}
                for field in fields:
                    field_rows[field] = cluster_summary(
                        [(source, state[field]) for source, state in items],
                        samples,
                        seed + counter,
                    )
                    counter += 1
                summaries[objective][group][relationship] = field_rows

    contrasts = {}
    for objective in ("value_expectation", "value_optimal_set"):
        contrasts[objective] = {}
        for group in metadata["parameter_groups"]:
            contrasts[objective][group] = {}
            for relationship in relationships:
                rows = []
                if relationship == "all_heldout":
                    value_items = [
                        item
                        for bucket in metadata["relationship_buckets"]
                        for item in flattened[(objective, group, bucket)]
                    ]
                else:
                    value_items = flattened[(objective, group, relationship)]
                for source_index, value_state in value_items:
                    key = (
                        source_index,
                        int(value_state["global_decision_index"]),
                        "expert_nll_control",
                        group,
                    )
                    control = indexed[key]
                    rows.append(
                        (
                            source_index,
                            value_state["expected_value_delta"] - control["expected_value_delta"],
                        )
                    )
                contrasts[objective][group][relationship] = cluster_summary(
                    rows, samples, seed + counter
                )
                counter += 1

    return {
        "metadata": metadata,
        "source_count": len(sources),
        "validation": {
            "maximum_baseline_repeat_score_error": maximum_repeat_error,
            "maximum_restore_parameter_error": maximum_restore_error,
            "source_kl_calibration_hit_rate": float(np.mean(calibration_hits)),
        },
        "actual_parameter_delta_norms": {
            objective: {
                group: {
                    "mean": float(np.mean(calibrated_steps[(objective, group)])),
                    "min": float(np.min(calibrated_steps[(objective, group)])),
                    "max": float(np.max(calibrated_steps[(objective, group)])),
                }
                for group in metadata["parameter_groups"]
            }
            for objective in metadata["objectives"]
        },
        "summaries": summaries,
        "expected_value_delta_vs_expert_nll_control": contrasts,
    }


def ci_text(item: dict[str, Any]) -> str:
    return f"{item['mean']:+.6f} [{item['ci95'][0]:+.6f}, {item['ci95'][1]:+.6f}]"


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# ALFWorld 价值梯度微小写回实验",
        "",
        "每个条件只在内存中沿源状态的负梯度更新；每次测量后逐参数精确恢复。不同参数区通过源状态候选分布 KL=1e-4 校准，因此比较的是等功能扰动，而非等原始梯度尺度。置信区间按源状态聚类 bootstrap。",
        "",
    ]
    for split in ("valid_seen", "valid_unseen"):
        data = result[split]
        valid = data["validation"]
        lines.extend(
            [
                f"## {split}",
                "",
                f"源状态数：{data['source_count']}；恢复最大误差：{valid['maximum_restore_parameter_error']:.3g}；基线重算最大误差：{valid['maximum_baseline_repeat_score_error']:.3g}；KL 校准命中率：{valid['source_kl_calibration_hit_rate']:.1%}。",
                "",
                "### 源状态：折扣价值期望变化（均值与 95% CI）",
                "",
                "| 目标 | 参数区 | Δ价值 | 正值源比例 | 源 KL |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for objective in data["metadata"]["objectives"]:
            for group in data["metadata"]["parameter_groups"]:
                row = data["summaries"][objective][group]["source"]
                value = row["expected_value_delta"]
                kl = row["kl_baseline_to_updated"]
                lines.append(
                    f"| {objective} | {group} | {ci_text(value)} | {value['source_positive_rate']:.1%} | {kl['mean']:.6g} |"
                )
        lines.extend(
            [
                "",
                "### 四类留出合并：折扣价值期望变化",
                "",
                "| 目标 | 参数区 | 留出 Δ价值 | 正值源比例 |",
                "|---|---|---:|---:|",
            ]
        )
        for objective in data["metadata"]["objectives"]:
            for group in data["metadata"]["parameter_groups"]:
                value = data["summaries"][objective][group]["all_heldout"]["expected_value_delta"]
                lines.append(
                    f"| {objective} | {group} | {ci_text(value)} | {value['source_positive_rate']:.1%} |"
                )
        for objective in ("value_expectation", "value_optimal_set"):
            lines.extend(
                [
                    "",
                    f"### 留出迁移：{objective} 的 Δ价值",
                    "",
                    "| 参数区 | 同任务同动作 | 同任务异动作 | 异任务同动作 | 异任务异动作 |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for group in data["metadata"]["parameter_groups"]:
                summaries = data["summaries"][objective][group]
                cells = [ci_text(summaries[relationship]["expected_value_delta"]) for relationship in data["metadata"]["relationship_buckets"]]
                lines.append(f"| {group} | " + " | ".join(cells) + " |")
        lines.extend(
            [
                "",
                "### 相对专家 NLL 对照：留出 Δ价值差",
                "",
                "正值表示价值目标在等源 KL 下优于专家 NLL；下表汇总四类留出状态。",
                "",
                "| 价值目标 | 参数区 | 四类留出配对差（均值与 95% CI） |",
                "|---|---|---:|",
            ]
        )
        for objective in ("value_expectation", "value_optimal_set"):
            for group in data["metadata"]["parameter_groups"]:
                item = data["expected_value_delta_vs_expert_nll_control"][objective][group]["all_heldout"]
                lines.append(f"| {objective} | {group} | {ci_text(item)} |")
        lines.append("")
    lines.extend(
        [
            "## 解释边界",
            "",
            "该实验检验局部、一阶、单状态更新能否迁移；它不是训练算法，也不做多步累计。若留出 CI 跨零，不能把源状态的必然改善解释为可泛化学习规则。专家 NLL 只用于等 KL 的行为克隆对照。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = {
        "valid_seen": analyze_split(args.seen_files, args.bootstrap_samples, args.seed),
        "valid_unseen": analyze_split(args.unseen_files, args.bootstrap_samples, args.seed + 1_000_003),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "source_global_decision_index",
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(build_report(result), encoding="utf-8")
    (output / "ANALYSIS_COMPLETE").touch()
    print(build_report(result))


if __name__ == "__main__":
    main()
