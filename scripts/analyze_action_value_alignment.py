#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from self_evolve.action_value import probability_value_metrics, softmax, spearman


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite(values: Iterable[float]) -> np.ndarray:
    return np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)


def describe(values: Iterable[float]) -> dict[str, float | int]:
    array = finite(values)
    if not len(array):
        return {"count": 0, "mean": math.nan, "median": math.nan}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    field: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[field])
        if math.isfinite(value):
            grouped[row["episode_key"]].append(value)
    keys = sorted(grouped)
    if not keys:
        return {"observed": math.nan, "ci95": [math.nan, math.nan]}
    sums = np.asarray([sum(grouped[key]) for key in keys], dtype=np.float64)
    counts = np.asarray([len(grouped[key]) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        selected = rng.integers(0, len(keys), size=(count, len(keys)))
        draws[start : start + count] = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return {
        "observed": float(sums.sum() / counts.sum()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_positive": float(np.mean(draws > 0.0)),
        "clusters": len(keys),
    }


def prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def group_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    result = {}
    for name, members in sorted(grouped.items()):
        result[name] = {
            "states": len(members),
            "episodes": len({row["episode_key"] for row in members}),
            "expected_discounted_value_delta": describe(
                row["discounted_expected_value_delta"] for row in members
            ),
            "expected_success_delta": describe(
                row["success_expected_value_delta"] for row in members
            ),
            "top_discounted_value_delta": describe(
                row["discounted_top_value_delta"] for row in members
            ),
            "delta_logit_value_spearman": describe(
                row["delta_logit_value_spearman"] for row in members
            ),
            "base_top_value_optimal_rate": float(
                np.mean([row["discounted_base_top_is_value_optimal"] for row in members])
            ),
            "seed_top_value_optimal_rate": float(
                np.mean([row["discounted_seed_top_is_value_optimal"] for row in members])
            ),
        }
    return result


def align(
    value_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    values_by_index = {int(row["global_decision_index"]): row for row in value_rows}
    seed_by_index = {int(row["global_decision_index"]): row for row in seed_rows}
    if len(values_by_index) != len(value_rows):
        raise ValueError("duplicate action-value state indices")
    if len(seed_by_index) != len(seed_rows):
        raise ValueError("duplicate SEED state indices")

    aligned = []
    for index in sorted(values_by_index):
        if index >= len(base_rows) or index not in seed_by_index:
            raise ValueError(f"missing base or SEED row for state {index}")
        value_row = values_by_index[index]
        base = base_rows[index]
        seed = seed_by_index[index]
        actions = list(value_row["admissible_actions"])
        candidates = value_row["candidate_values"]
        if len(actions) != len(candidates):
            raise ValueError(f"incomplete candidate values at state {index}")
        if actions != list(base["admissible_actions"]) or actions != list(seed["admissible_actions"]):
            raise ValueError(f"action mismatch at state {index}")
        for field in ("task_type", "step_index", "expert_action"):
            if value_row[field] != base[field] or value_row[field] != seed[field]:
                raise ValueError(f"{field} mismatch at state {index}")

        base_score_field = base["normalized_scores"]
        if isinstance(base_score_field, dict):
            base_score_field = base_score_field["plain"]
        base_scores = np.asarray(base_score_field, dtype=np.float64)
        seed_scores = np.asarray(seed["normalized_scores"], dtype=np.float64)
        successes = np.asarray([item["won"] for item in candidates], dtype=np.float64)
        discounted = np.asarray(
            [item["discounted_success"] for item in candidates], dtype=np.float64
        )
        success_metrics = prefixed(
            "success", probability_value_metrics(base_scores, seed_scores, successes)
        )
        discounted_metrics = prefixed(
            "discounted", probability_value_metrics(base_scores, seed_scores, discounted)
        )
        base_p, seed_p = softmax(base_scores), softmax(seed_scores)
        delta_p = seed_p - base_p
        gold = actions.index(value_row["expert_action"])
        base_top, seed_top = int(np.argmax(base_scores)), int(np.argmax(seed_scores))
        if base_top != gold and seed_top == gold:
            transition = "rescued_to_expert"
        elif base_top == gold and seed_top != gold:
            transition = "harmed_from_expert"
        elif base_top == gold and seed_top == gold:
            transition = "stable_expert"
        else:
            transition = "stable_nonexpert" if base_top == seed_top else "changed_nonexpert"

        action_rows = [
            {
                "candidate_index": i,
                "action": action,
                "is_recorded_expert_action": int(i == gold),
                "won": int(successes[i]),
                "recovery_steps": int(candidates[i]["recovery_steps"]),
                "discounted_success": float(discounted[i]),
                "base_probability": float(base_p[i]),
                "seed_probability": float(seed_p[i]),
                "probability_delta": float(delta_p[i]),
                "base_score": float(base_scores[i]),
                "seed_score": float(seed_scores[i]),
                "score_delta": float(seed_scores[i] - base_scores[i]),
            }
            for i, action in enumerate(actions)
        ]
        receivers = sorted(action_rows, key=lambda row: row["probability_delta"], reverse=True)
        donors = sorted(action_rows, key=lambda row: row["probability_delta"])
        aligned.append(
            {
                "split": value_row["split"],
                "global_decision_index": index,
                "episode_key": value_row["episode_key"],
                "episode_id": value_row["episode_id"],
                "task_type": value_row["task_type"],
                "step_index": value_row["step_index"],
                "action_verb": value_row["action_verb"],
                "expert_action": value_row["expert_action"],
                "base_top_action": actions[base_top],
                "seed_top_action": actions[seed_top],
                "expert_transition": transition,
                "candidate_count": len(actions),
                "successful_candidate_rate": float(np.mean(successes)),
                "binary_value_is_heterogeneous": int(np.ptp(successes) > 0),
                "discounted_value_is_heterogeneous": int(np.ptp(discounted) > 1e-12),
                "expert_success": int(successes[gold]),
                "expert_discounted_value": float(discounted[gold]),
                "expert_is_discounted_value_optimal": int(
                    np.isclose(discounted[gold], np.max(discounted), rtol=0.0, atol=1e-12)
                ),
                "delta_logit_value_spearman": spearman(seed_scores - base_scores, discounted),
                "delta_probability_value_spearman": spearman(delta_p, discounted),
                "largest_probability_receiver": receivers[0],
                "largest_probability_donor": donors[0],
                **success_metrics,
                **discounted_metrics,
                "actions": action_rows,
            }
        )
    return aligned


def analyze_rows(
    rows: list[dict[str, Any]],
    samples: int,
    seed: int,
    practical_threshold: float,
    required_relative_improvement: float = 0.30,
) -> dict[str, Any]:
    metrics = [
        "success_expected_value_delta",
        "discounted_expected_value_delta",
        "discounted_top_value_delta",
        "discounted_added_minus_removed_value",
        "discounted_probability_on_value_optimal_delta",
        "delta_logit_value_spearman",
        "delta_probability_value_spearman",
        "discounted_base_expected_value",
        "discounted_seed_expected_value",
        "discounted_base_top_value",
        "discounted_seed_top_value",
    ]
    seed_top_rate = float(
        np.mean([row["discounted_seed_top_is_value_optimal"] for row in rows])
    )
    base_expected = float(np.mean([row["discounted_base_expected_value"] for row in rows]))
    comparison_expected = float(np.mean([row["discounted_seed_expected_value"] for row in rows]))
    relative_expected_gain = (comparison_expected - base_expected) / max(abs(base_expected), 1e-12)
    base_top_rate = float(np.mean([row["discounted_base_top_is_value_optimal"] for row in rows]))
    relative_top_rate_gain = (seed_top_rate - base_top_rate) / max(base_top_rate, 1e-12)
    return {
        "states": len(rows),
        "episodes": len({row["episode_key"] for row in rows}),
        "candidate_actions": sum(row["candidate_count"] for row in rows),
        "candidate_success_rate": float(
            sum(row["successful_candidate_rate"] * row["candidate_count"] for row in rows)
            / sum(row["candidate_count"] for row in rows)
        ),
        "states_with_binary_value_variation": sum(
            row["binary_value_is_heterogeneous"] for row in rows
        ),
        "states_with_discounted_value_variation": sum(
            row["discounted_value_is_heterogeneous"] for row in rows
        ),
        "expert_action_value_optimal_rate": float(
            np.mean([row["expert_is_discounted_value_optimal"] for row in rows])
        ),
        "base_top_value_optimal_rate": base_top_rate,
        "seed_top_value_optimal_rate": seed_top_rate,
        "base_expected_discounted_value": base_expected,
        "comparison_expected_discounted_value": comparison_expected,
        "relative_expected_discounted_value_gain": relative_expected_gain,
        "relative_top_value_optimal_rate_gain": relative_top_rate_gain,
        "required_relative_improvement": required_relative_improvement,
        "meets_required_relative_expected_value_improvement": bool(
            relative_expected_gain >= required_relative_improvement
        ),
        "practical_threshold": practical_threshold,
        "meets_practical_threshold": bool(seed_top_rate >= practical_threshold),
        "gap_to_practical_threshold": seed_top_rate - practical_threshold,
        "descriptive": {field: describe(row[field] for row in rows) for field in metrics},
        "episode_cluster_bootstrap": {
            field: cluster_bootstrap(rows, field, samples, seed + offset)
            for offset, field in enumerate(metrics)
        },
        "by_expert_transition": group_summary(rows, "expert_transition"),
        "by_task_type": group_summary(rows, "task_type"),
        "by_action_verb": group_summary(rows, "action_verb"),
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def number(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:+.6f}"


def render_report(results: dict[str, Any], comparison_label: str = "SEED") -> str:
    lines = [
        f"# ALFWorld 候选动作长期价值 × {comparison_label} 输出变化",
        "",
        "价值定义：在记录状态强制执行一个候选动作，随后由 ALFWorld 官方 expert 恢复；",
        "在总计 50 步预算内成功记为 1，同时报告 `0.95^(恢复步数-1)` 的折扣成功值。",
        "该实验测量的是 expert-recovery 条件价值，不等同于 Base/SEED 自身 rollout 价值。",
        "",
    ]
    for split, summary in results["splits"].items():
        desc = summary["descriptive"]
        boot = summary["episode_cluster_bootstrap"]["discounted_expected_value_delta"]
        lines.extend(
            [
                f"## {split}",
                "",
                f"- {summary['states']} 个状态、{summary['episodes']} 个独立 game trial、{summary['candidate_actions']} 个候选动作。",
                f"- 候选动作在预算内的总体恢复成功率：{pct(summary['candidate_success_rate'])}。",
                f"- 有二元成功差异的状态：{summary['states_with_binary_value_variation']}；有折扣价值差异的状态：{summary['states_with_discounted_value_variation']}。",
                f"- expert 动作 / Base top-1 / {comparison_label} top-1 为最高价值动作的比例：{pct(summary['expert_action_value_optimal_rate'])} / {pct(summary['base_top_value_optimal_rate'])} / {pct(summary['seed_top_value_optimal_rate'])}。",
                f"- {100.0 * summary['practical_threshold']:.0f}% 实用门槛：{'通过' if summary['meets_practical_threshold'] else '未通过'}（差值 {100.0 * summary['gap_to_practical_threshold']:+.2f} 点）。",
                f"- {comparison_label} 概率加权折扣价值变化均值：{number(desc['discounted_expected_value_delta']['mean'])}。",
                f"- Base / {comparison_label} 概率加权折扣价值：{summary['base_expected_discounted_value']:.6f} / {summary['comparison_expected_discounted_value']:.6f}；相对提升 {pct(summary['relative_expected_discounted_value_gain'])}（30% 硬门槛：{'通过' if summary['meets_required_relative_expected_value_improvement'] else '未通过'}）。",
                f"- 最高价值动作命中率相对提升：{pct(summary['relative_top_value_optimal_rate_gain'])}。",
                f"- episode-cluster 95% CI：[{boot['ci95'][0]:+.6f}, {boot['ci95'][1]:+.6f}]。",
                f"- {comparison_label} Δlogit 与动作价值的状态内 Spearman 均值：{number(desc['delta_logit_value_spearman']['mean'])}。",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", required=True)
    parser.add_argument("--value-trace", action="append", required=True)
    parser.add_argument("--base-trace", action="append", required=True)
    parser.add_argument("--seed-trace", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--practical-threshold", type=float, default=0.80)
    parser.add_argument("--comparison-label", default="SEED")
    parser.add_argument("--required-relative-improvement", type=float, default=0.30)
    args = parser.parse_args()
    counts = {len(args.split), len(args.base_trace)}
    if len(counts) != 1:
        raise ValueError("--split and --base-trace counts must match")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {"comparison_label": args.comparison_label, "splits": {}}
    all_aligned = []
    for split_index, split in enumerate(args.split):
        value_paths = [path for path in args.value_trace if f"/{split}_" in path]
        seed_paths = [path for path in args.seed_trace if f"/{split}_" in path]
        if not value_paths or not seed_paths:
            raise ValueError(f"could not identify value/SEED traces for {split}")
        value_rows = [row for path in value_paths for row in read_jsonl(path)]
        seed_rows = [row for path in seed_paths for row in read_jsonl(path)]
        base_rows = read_jsonl(args.base_trace[split_index])
        aligned_rows = align(value_rows, base_rows, seed_rows)
        all_aligned.extend(aligned_rows)
        all_results["splits"][split] = analyze_rows(
            aligned_rows,
            args.bootstrap_samples,
            args.seed + split_index * 100,
            args.practical_threshold,
            args.required_relative_improvement,
        )

    trace_path = output_dir / "trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in all_aligned:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        render_report(all_results, args.comparison_label), encoding="utf-8"
    )
    print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
