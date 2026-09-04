#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


OBJECTIVES = ("value_expectation", "value_optimal_set", "expert_nll_control")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen-shards", nargs="+", required=True)
    parser.add_argument("--unseen-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def finite(values: Iterable[float]) -> np.ndarray:
    return np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=np.float64)


def describe(values: Iterable[float]) -> dict[str, Any]:
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
    states: list[dict[str, Any]], values: Sequence[float], samples: int, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for state, value in zip(states, values, strict=True):
        if math.isfinite(float(value)):
            grouped[state["episode_key"]].append(float(value))
    episodes = sorted(grouped)
    if not episodes:
        return {"observed": math.nan, "ci95": [math.nan, math.nan], "clusters": 0}
    sums = np.asarray([math.fsum(grouped[key]) for key in episodes], dtype=np.float64)
    counts = np.asarray([len(grouped[key]) for key in episodes], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        chosen = rng.integers(0, len(episodes), size=(count, len(episodes)))
        draws[start : start + count] = sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)
    return {
        "observed": float(sums.sum() / counts.sum()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_positive": float(np.mean(draws > 0.0)),
        "clusters": len(episodes),
    }


def load_split(paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    states = []
    rows_by_index = {}
    array_rows: dict[int, dict[str, np.ndarray]] = {}
    common_metadata = None
    for raw_path in paths:
        root = Path(raw_path)
        result = json.loads((root / "results.json").read_text(encoding="utf-8"))
        metadata = {
            "split": result["split"],
            "sample_size_requested": result["sample_size_requested"],
            "sample_seed": result["sample_seed"],
            "selected_global_indices_all_shards": result["selected_global_indices_all_shards"],
            "objectives": result["objectives"],
            "parameter_groups": result["parameter_groups"],
        }
        if common_metadata is None:
            common_metadata = metadata
        elif metadata != common_metadata:
            raise ValueError("Shard metadata differ")
        arrays = np.load(root / "gradient_sketches.npz")
        indices = arrays["global_decision_indices"]
        for local_index, state in enumerate(result["states"]):
            global_index = int(state["global_decision_index"])
            if global_index != int(indices[local_index]) or global_index in rows_by_index:
                raise ValueError(f"Duplicate or misaligned gradient state {global_index}")
            rows_by_index[global_index] = state
            array_rows[global_index] = {
                name: arrays[name][local_index].astype(np.float32)
                for name in arrays.files
                if name != "global_decision_indices"
            }
    assert common_metadata is not None
    expected = set(common_metadata["selected_global_indices_all_shards"])
    if set(rows_by_index) != expected:
        raise ValueError(
            f"Sample completeness failure: have {len(rows_by_index)}, expected {len(expected)}"
        )
    states = [rows_by_index[index] for index in sorted(rows_by_index)]
    arrays = {
        name: np.stack([array_rows[index][name] for index in sorted(rows_by_index)])
        for name in next(iter(array_rows.values()))
    }
    return states, arrays, common_metadata


def pairwise_commonality(states: list[dict[str, Any]], sketches: np.ndarray) -> dict[str, Any]:
    normalized = sketches / np.maximum(np.linalg.norm(sketches, axis=1, keepdims=True), 1e-20)
    similarities = normalized @ normalized.T
    buckets: dict[str, list[float]] = defaultdict(list)
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            if states[left]["episode_key"] == states[right]["episode_key"]:
                continue
            same_task = states[left]["task_type"] == states[right]["task_type"]
            same_verb = states[left]["action_verb"] == states[right]["action_verb"]
            label = (
                f"{'same' if same_task else 'different'}_task__"
                f"{'same' if same_verb else 'different'}_verb"
            )
            buckets[label].append(float(similarities[left, right]))
    return {
        name: {
            "pairs": len(values),
            "cosine": describe(values),
            "positive_rate": float(np.mean(np.asarray(values) > 0.0)),
        }
        for name, values in sorted(buckets.items())
    }


def analyze_split(
    paths: list[str], samples: int, seed: int
) -> dict[str, Any]:
    states, sketches, metadata = load_split(paths)
    groups = sorted(metadata["parameter_groups"])
    summary = {}
    counter = 0
    for objective in OBJECTIVES:
        summary[objective] = {}
        for group in groups:
            metrics = [state["objectives"][objective][group] for state in states]
            fields = (
                "base_gradient_norm",
                "seed_gradient_norm",
                "seed_to_base_gradient_norm_ratio",
                "base_seed_gradient_cosine",
                "base_gradient_dot_parameter_delta",
                "seed_gradient_dot_parameter_delta",
                "base_descent_parameter_delta_cosine",
                "seed_descent_parameter_delta_cosine",
            )
            group_result = {}
            for field in fields:
                values = [metric[field] for metric in metrics]
                group_result[field] = describe(values)
                group_result[f"{field}_bootstrap"] = cluster_bootstrap(
                    states, values, samples, seed + counter
                )
                counter += 1
            for model_name in ("base", "seed"):
                key = f"{objective}__{model_name}__{group}"
                group_result[f"{model_name}_coordinate_sketch_concentration"] = float(
                    np.linalg.norm(np.mean(sketches[key], axis=0))
                )
                group_result[f"{model_name}_pairwise_commonality"] = pairwise_commonality(
                    states, sketches[key]
                )
            summary[objective][group] = group_result

    cross_objective = {}
    for group in groups:
        cross_objective[group] = {}
        for model_name in ("base", "seed"):
            value_key = f"value_expectation__{model_name}__{group}"
            for control in ("value_optimal_set", "expert_nll_control"):
                control_key = f"{control}__{model_name}__{group}"
                values = np.sum(sketches[value_key] * sketches[control_key], axis=1)
                name = f"{model_name}_value_expectation_vs_{control}_coordinate_cosine"
                cross_objective[group][name] = describe(values)
                cross_objective[group][f"{name}_bootstrap"] = cluster_bootstrap(
                    states, values, samples, seed + counter
                )
                counter += 1

    slices = {}
    for field in ("task_type", "action_verb", "expert_transition"):
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, state in enumerate(states):
            grouped[str(state[field])].append(index)
        slices[field] = {}
        for name, indices in sorted(grouped.items()):
            metrics = [
                states[index]["objectives"]["value_expectation"]["selected_union"]
                for index in indices
            ]
            slices[field][name] = {
                "states": len(indices),
                "base_descent_parameter_delta_cosine": describe(
                    metric["base_descent_parameter_delta_cosine"] for metric in metrics
                ),
                "base_seed_gradient_cosine": describe(
                    metric["base_seed_gradient_cosine"] for metric in metrics
                ),
            }
    return {
        "states": len(states),
        "episodes": len({state["episode_key"] for state in states}),
        "sample_seed": metadata["sample_seed"],
        "max_base_score_reproduction_error": max(
            state["max_base_score_reproduction_error"] for state in states
        ),
        "max_seed_score_reproduction_error": max(
            state["max_seed_score_reproduction_error"] for state in states
        ),
        "parameter_groups": metadata["parameter_groups"],
        "objectives": summary,
        "cross_objective": cross_objective,
        "slices": slices,
    }


def fmt(value: float, digits: int = 4) -> str:
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def interval(item: dict[str, Any]) -> str:
    return f"{fmt(item['observed'])} [{fmt(item['ci95'][0])}, {fmt(item['ci95'][1])}]"


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# ALFWorld 长期价值目标的 Base/SEED 参数梯度结构",
        "",
        "主目标 `value_expectation` 是 `-E_softmax(action score)[V_expert-recovery]`；",
        "`value_optimal_set` 是最高价值动作集合的负对数概率；expert NLL 仅作为 control。",
        "参数组有意重叠；跨经验方向使用 4096 维固定坐标 sketch，单状态 Base/SEED 与真实参数差分对齐为全参数精确内积。",
        "",
    ]
    for split, analysis in result["splits"].items():
        lines.extend(
            [
                f"## {split}",
                "",
                f"- 分层样本 {analysis['states']} 个状态、{analysis['episodes']} 个独立 trial。",
                f"- 动作分数最大复现误差 Base/SEED："
                f"{analysis['max_base_score_reproduction_error']:.3g}/"
                f"{analysis['max_seed_score_reproduction_error']:.3g}。",
                "",
                "### 主价值目标",
                "",
                "`descent × delta-theta cosine` 为正表示真实 SEED 参数变化与该状态的价值改进下降方向同向。",
                "",
                "| parameter group | params | Base/SEED gradient cosine [CI] | Base descent×delta-theta [CI] | Seed/Base norm ratio [CI] |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for group, item in analysis["objectives"]["value_expectation"].items():
            meta = analysis["parameter_groups"][group]
            lines.append(
                f"| {group} | {meta['parameter_count']} | "
                f"{interval(item['base_seed_gradient_cosine_bootstrap'])} | "
                f"{interval(item['base_descent_parameter_delta_cosine_bootstrap'])} | "
                f"{interval(item['seed_to_base_gradient_norm_ratio_bootstrap'])} |"
            )
        union = analysis["cross_objective"]["selected_union"]
        lines.extend(
            [
                "",
                "### 价值目标与控制目标的坐标-sketch方向",
                "",
                "| comparison | mean cosine |",
                "|---|---:|",
            ]
        )
        for name, item in union.items():
            if name.endswith("_bootstrap"):
                continue
            lines.append(f"| {name} | {fmt(item['mean'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    result = {
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "splits": {
            "valid_seen": analyze_split(
                args.seen_shards, args.bootstrap_samples, args.seed
            ),
            "valid_unseen": analyze_split(
                args.unseen_shards, args.bootstrap_samples, args.seed + 10_000
            ),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = render_report(result)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    (output_dir / "ANALYSIS_COMPLETE").touch()
    print(report, end="")


if __name__ == "__main__":
    main()
