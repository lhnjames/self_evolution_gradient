#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


RELATIONSHIPS = (
    "source",
    "same_action_holdout",
    "same_task_different_action",
    "different_task_different_action",
)
METHODS = ("single12_average", "mean12", "purified12")


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "positive_rate": float(np.mean(array > 0.0)),
    }


def state_summary(
    rows: list[dict[str, Any]], optimal_by_index: dict[int, float]
) -> dict[str, Any]:
    baseline = np.asarray([float(row["baseline_expected_value"]) for row in rows])
    updated = np.asarray([float(row["updated_expected_value"]) for row in rows])
    deltas = updated - baseline
    base_top = np.asarray(
        [
            int(
                row.get(
                    "baseline_top_is_value_optimal",
                    np.isclose(
                        row["baseline_top_value"],
                        optimal_by_index[int(row["global_decision_index"])],
                        rtol=0.0,
                        atol=1e-12,
                    ),
                )
            )
            for row in rows
        ]
    )
    updated_top = np.asarray(
        [
            int(
                row.get(
                    "updated_top_is_value_optimal",
                    np.isclose(
                        row["updated_top_value"],
                        optimal_by_index[int(row["global_decision_index"])],
                        rtol=0.0,
                        atol=1e-12,
                    ),
                )
            )
            for row in rows
        ]
    )
    base_top_rate = float(base_top.mean())
    updated_top_rate = float(updated_top.mean())
    return {
        "expected_value_delta": describe(deltas.tolist()),
        "baseline_expected_value": float(baseline.mean()),
        "updated_expected_value": float(updated.mean()),
        "relative_expected_value_gain": float(
            (updated.mean() - baseline.mean()) / max(abs(baseline.mean()), 1e-12)
        ),
        "base_top_value_optimal_rate": base_top_rate,
        "updated_top_value_optimal_rate": updated_top_rate,
        "top_value_optimal_rate_gain_points": updated_top_rate - base_top_rate,
        "relative_top_value_optimal_rate_gain": (
            None
            if base_top_rate <= 1e-12
            else (updated_top_rate - base_top_rate) / base_top_rate
        ),
        "top_value_harm_rate": float(
            np.mean(
                [row["updated_top_value"] < row["baseline_top_value"] for row in rows]
            )
        ),
    }


def update_rows(verb_row: dict[str, Any], method: str) -> list[dict[str, Any]]:
    if method == "single12_average":
        return [
            {**state, "source_replica": position}
            for position, update in enumerate(verb_row["single_source_updates"])
            for state in update["states"]
        ]
    return verb_row["aggregate_updates"][method]["states"]


def summarize(paths: list[str], value_trace: str) -> dict[str, Any]:
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    value_rows = [json.loads(line) for line in Path(value_trace).open(encoding="utf-8") if line.strip()]
    optimal_by_index = {
        int(row["global_decision_index"]): float(row["discounted_optimal_value"])
        for row in value_rows
        if row["split"] == "valid_seen"
    }
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    top_harm: dict[tuple[str, str], list[float]] = defaultdict(list)
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skill_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    per_seed_skill = []
    for record in records:
        seed = int(record["sample_seed"])
        for verb_row in record["verb_results"]:
            verb = verb_row["verb"]
            for method in METHODS:
                rows = update_rows(verb_row, method)
                for relationship in RELATIONSHIPS:
                    selected = [row for row in rows if row["relationship"] == relationship]
                    deltas = [float(row["expected_value_delta"]) for row in selected]
                    harms = [float(row["updated_top_value"] < row["baseline_top_value"]) for row in selected]
                    grouped[(method, relationship)].extend(deltas)
                    top_harm[(method, relationship)].extend(harms)
                    grouped_rows[(method, relationship)].extend(selected)
                    if relationship == "same_action_holdout":
                        skill_rows[(method, verb)].extend(selected)
                    per_seed_skill.append(
                        {
                            "seed": seed,
                            "verb": verb,
                            "method": method,
                            "relationship": relationship,
                            "mean_expected_value_delta": float(np.mean(deltas)),
                            "top_value_harm_rate": float(np.mean(harms)),
                        }
                    )
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for relationship in RELATIONSHIPS:
            summary[method][relationship] = state_summary(
                grouped_rows[(method, relationship)], optimal_by_index
            )
    paired = {}
    keys = {(r["seed"], r["verb"], r["relationship"]) for r in per_seed_skill}
    lookup = {
        (r["seed"], r["verb"], r["relationship"], r["method"]): r
        for r in per_seed_skill
    }
    for comparison, left, right in (
        ("mean_minus_single", "mean12", "single12_average"),
        ("purified_minus_single", "purified12", "single12_average"),
        ("purified_minus_mean", "purified12", "mean12"),
    ):
        paired[comparison] = {}
        for relationship in RELATIONSHIPS:
            differences = [
                lookup[(*key, left)]["mean_expected_value_delta"]
                - lookup[(*key, right)]["mean_expected_value_delta"]
                for key in sorted(keys)
                if key[2] == relationship
            ]
            paired[comparison][relationship] = describe(differences)
    by_skill = {
        method: {
            verb: state_summary(skill_rows[(method, verb)], optimal_by_index)
            for verb in sorted({row["verb"] for record in records for row in record["verb_results"]})
        }
        for method in METHODS
    }
    return {
        "input_files": paths,
        "seeds": sorted({int(record["sample_seed"]) for record in records}),
        "methods": list(METHODS),
        "summary": summary,
        "same_action_holdout_by_skill": by_skill,
        "paired_seed_skill_differences": paired,
        "per_seed_skill": per_seed_skill,
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# 300× 相似技能梯度净化实验",
        "",
        f"独立采样种子：{', '.join(map(str, result['seeds']))}。所有方法使用相同参数 L2 写回预算。",
        "`single12_average` 表示 12 个单失败梯度分别写回后的平均，而不是先平均梯度。",
        "",
        "| 方法 | 同技能留出 ΔV | 正迁移率 | 同技能 top harm | 异技能同任务 ΔV |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        same = result["summary"][method]["same_action_holdout"]
        control = result["summary"][method]["same_task_different_action"]
        delta = same["expected_value_delta"]
        lines.append(
            f"| {method} | {delta['mean']:+.6f} | {100 * delta['positive_rate']:.1f}% | "
            f"{100 * same['top_value_harm_rate']:.1f}% | "
            f"{control['expected_value_delta']['mean']:+.6f} |"
        )
    lines.extend(["", "## 30% 门槛", ""])
    for method in METHODS:
        same = result["summary"][method]["same_action_holdout"]
        lines.append(
            f"- {method}：同技能留出概率价值相对提升 "
            f"`{100 * same['relative_expected_value_gain']:+.2f}%`；失败状态 top-value 救回 "
            f"`{100 * same['top_value_optimal_rate_gain_points']:+.2f}` 个百分点。"
        )
    lines.extend(["", "## 配对差值", ""])
    for name, comparisons in result["paired_seed_skill_differences"].items():
        value = comparisons["same_action_holdout"]
        lines.append(
            f"- {name}：同技能留出 `{value['mean']:+.6f}`，"
            f"{value['positive_rate'] * 100:.1f}% 的 seed×skill 条件为正。"
        )
    lines.extend(["", "## 逐技能 300× 结果", ""])
    lines.extend(
        [
            "| 方法 | skill | 相对概率价值提升 | top-value 救回 | top harm |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        for verb, value in result["same_action_holdout_by_skill"][method].items():
            lines.append(
                f"| {method} | {verb} | {100 * value['relative_expected_value_gain']:+.2f}% | "
                f"{100 * value['top_value_optimal_rate_gain_points']:+.2f} 点 | "
                f"{100 * value['top_value_harm_rate']:.2f}% |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--value-trace", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = summarize(args.input, args.value_trace)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (destination / "REPORT.md").write_text(render(result), encoding="utf-8")
    print(render(result))


if __name__ == "__main__":
    main()
