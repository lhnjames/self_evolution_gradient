#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from self_evolve.action_value import spearman


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def read_rows(paths: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    keys = [(row["split"], int(row["global_decision_index"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate split/state rows in token traces")
    return sorted(rows, key=lambda row: (row["split"], row["global_decision_index"]))


def finite(values: Iterable[float]) -> np.ndarray:
    return np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=np.float64)


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
    rows: list[dict[str, Any]], field: str, samples: int, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row.get(field, math.nan))
        if math.isfinite(value):
            grouped[str(row["episode_key"])].append(value)
    keys = sorted(grouped)
    if not keys:
        return {"observed": math.nan, "ci95": [math.nan, math.nan], "clusters": 0}
    sums = np.asarray([math.fsum(grouped[key]) for key in keys], dtype=np.float64)
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


def summarize_state(row: dict[str, Any]) -> dict[str, Any]:
    actions = row["actions"]
    roles = sorted(
        {role for action in actions for role in action["role_score_delta_contributions"]}
    )
    values = np.asarray([action["discounted_value"] for action in actions], dtype=np.float64)
    probability_delta = np.asarray(
        [action["action_probability_delta"] for action in actions], dtype=np.float64
    )
    added = np.clip(probability_delta, 0.0, None)
    removed = np.clip(-probability_delta, 0.0, None)
    moved = float(np.sum(added))
    result: dict[str, Any] = {
        "split": row["split"],
        "global_decision_index": row["global_decision_index"],
        "episode_key": row["episode_key"],
        "episode_id": row["episode_id"],
        "task_type": row["task_type"],
        "step_index": row["step_index"],
        "expert_transition": row["expert_transition"],
        "discounted_expected_value_delta": row["discounted_expected_value_delta"],
        "base_top_is_value_optimal": row["base_top_is_value_optimal"],
        "seed_top_is_value_optimal": row["seed_top_is_value_optimal"],
        "candidate_actions": row["candidate_actions"],
        "candidate_tokens": row["candidate_tokens"],
        "unique_prefix_nodes": row["unique_prefix_nodes"],
        "prompt_was_truncated": row["prompt_was_truncated"],
        "max_base_score_reproduction_error": row["max_base_score_reproduction_error"],
        "max_seed_score_reproduction_error": row["max_seed_score_reproduction_error"],
        "moved_action_probability_mass": moved,
    }
    for role in roles:
        contributions = np.asarray(
            [action["role_score_delta_contributions"].get(role, 0.0) for action in actions],
            dtype=np.float64,
        )
        result[f"role::{role}::value_spearman"] = spearman(contributions, values)
        if moved > 1e-12:
            receiver = float(np.dot(added, contributions) / moved)
            donor = float(np.dot(removed, contributions) / moved)
            result[f"role::{role}::receiver_mean"] = receiver
            result[f"role::{role}::donor_mean"] = donor
            result[f"role::{role}::receiver_minus_donor"] = receiver - donor
        else:
            result[f"role::{role}::receiver_mean"] = math.nan
            result[f"role::{role}::donor_mean"] = math.nan
            result[f"role::{role}::receiver_minus_donor"] = math.nan

    grouped_nodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in row["nodes"]:
        child_roles = sorted(
            {role for child in node["candidate_children"] for role in child["roles"]}
        )
        node_role = child_roles[0] if len(child_roles) == 1 else "mixed"
        grouped_nodes[node_role].append(node)
    for role, nodes in grouped_nodes.items():
        result[f"node::{role}::vocabulary_tv"] = float(
            np.mean([node["vocabulary_total_variation"] for node in nodes])
        )
        result[f"node::{role}::candidate_mass_delta"] = float(
            np.mean([node["candidate_child_mass_delta"] for node in nodes])
        )
        multichild = [node for node in nodes if len(node["candidate_children"]) > 1]
        branch_deltas = []
        correlations = []
        top_receiver_admissible = []
        for node in multichild:
            children = node["candidate_children"]
            branch_values = np.asarray(
                [child["branch_value"]["mean"] for child in children], dtype=np.float64
            )
            child_deltas = np.asarray(
                [child["candidate_conditional_probability_delta"] for child in children],
                dtype=np.float64,
            )
            branch_deltas.append(float(np.dot(child_deltas, branch_values)))
            correlations.append(spearman(child_deltas, branch_values))
            candidate_ids = {child["token_id"] for child in children}
            top_receiver_admissible.append(
                int(node["top_probability_receivers"][0]["token_id"] in candidate_ids)
            )
        result[f"node::{role}::multichild_nodes"] = len(multichild)
        result[f"node::{role}::branch_value_delta"] = (
            float(np.mean(branch_deltas)) if branch_deltas else math.nan
        )
        result[f"node::{role}::child_delta_value_spearman"] = (
            float(np.mean(finite(correlations))) if len(finite(correlations)) else math.nan
        )
        result[f"node::{role}::top_receiver_admissible_rate"] = (
            float(np.mean(top_receiver_admissible)) if top_receiver_admissible else math.nan
        )
    return result


def analyze_split(
    raw_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, Any]:
    actions = [action for row in raw_rows for action in row["actions"]]
    nodes = [node for row in raw_rows for node in row["nodes"]]
    roles = sorted(
        {role for action in actions for role in action["role_score_delta_contributions"]}
    )
    node_roles = sorted(
        {
            (child_roles[0] if len(child_roles) == 1 else "mixed")
            for node in nodes
            for child_roles in [
                sorted({role for child in node["candidate_children"] for role in child["roles"]})
            ]
        }
    )

    absolute_totals = {
        role: math.fsum(
            abs(float(action["role_score_delta_contributions"].get(role, 0.0)))
            for action in actions
        )
        for role in roles
    }
    absolute_total = math.fsum(absolute_totals.values())
    role_results = {}
    for offset, role in enumerate(roles):
        contributions = [
            float(action["role_score_delta_contributions"].get(role, 0.0))
            for action in actions
        ]
        optimal = [
            value for value, action in zip(contributions, actions, strict=True) if action["is_value_optimal"]
        ]
        nonoptimal = [
            value
            for value, action in zip(contributions, actions, strict=True)
            if not action["is_value_optimal"]
        ]
        receivers = [
            value
            for value, action in zip(contributions, actions, strict=True)
            if action["action_probability_delta"] > 1e-12
        ]
        donors = [
            value
            for value, action in zip(contributions, actions, strict=True)
            if action["action_probability_delta"] < -1e-12
        ]
        spearman_field = f"role::{role}::value_spearman"
        contrast_field = f"role::{role}::receiver_minus_donor"
        role_results[role] = {
            "absolute_score_delta_share": (
                absolute_totals[role] / absolute_total if absolute_total > 0 else math.nan
            ),
            "all_action_contribution": describe(contributions),
            "value_optimal_action_contribution": describe(optimal),
            "nonoptimal_action_contribution": describe(nonoptimal),
            "probability_receiver_contribution": describe(receivers),
            "probability_donor_contribution": describe(donors),
            "state_value_spearman": describe(row.get(spearman_field, math.nan) for row in state_rows),
            "state_value_spearman_bootstrap": cluster_bootstrap(
                state_rows, spearman_field, samples, seed + 100 + offset
            ),
            "state_receiver_minus_donor": describe(
                row.get(contrast_field, math.nan) for row in state_rows
            ),
            "state_receiver_minus_donor_bootstrap": cluster_bootstrap(
                state_rows, contrast_field, samples, seed + 200 + offset
            ),
        }

    node_results = {}
    for offset, role in enumerate(node_roles):
        role_nodes = []
        for node in nodes:
            child_roles = sorted(
                {item for child in node["candidate_children"] for item in child["roles"]}
            )
            actual = child_roles[0] if len(child_roles) == 1 else "mixed"
            if actual == role:
                role_nodes.append(node)
        fields = {
            "vocabulary_total_variation": "vocabulary_tv",
            "candidate_child_mass_delta": "candidate_mass_delta",
            "branch_value_delta": "branch_value_delta",
            "child_delta_value_spearman": "child_delta_value_spearman",
            "top_receiver_admissible_rate": "top_receiver_admissible_rate",
        }
        result: dict[str, Any] = {
            "nodes": len(role_nodes),
            "multichild_nodes": sum(len(node["candidate_children"]) > 1 for node in role_nodes),
            "vocabulary_total_variation": describe(
                node["vocabulary_total_variation"] for node in role_nodes
            ),
            "candidate_child_mass_delta": describe(
                node["candidate_child_mass_delta"] for node in role_nodes
            ),
        }
        for raw_name, state_suffix in list(fields.items())[2:]:
            state_field = f"node::{role}::{state_suffix}"
            result[raw_name] = describe(row.get(state_field, math.nan) for row in state_rows)
            result[f"{raw_name}_bootstrap"] = cluster_bootstrap(
                state_rows, state_field, samples, seed + 300 + offset
            )
        node_results[role] = result

    transitions: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        grouped[row["expert_transition"]].append(row)
    for name, members in sorted(grouped.items()):
        transitions[name] = {
            "states": len(members),
            "discounted_expected_value_delta": describe(
                row["discounted_expected_value_delta"] for row in members
            ),
            "role_receiver_minus_donor": {
                role: describe(
                    row.get(f"role::{role}::receiver_minus_donor", math.nan)
                    for row in members
                )
                for role in roles
            },
        }

    return {
        "states": len(raw_rows),
        "episodes": len({row["episode_key"] for row in raw_rows}),
        "candidate_actions": len(actions),
        "candidate_tokens": sum(row["candidate_tokens"] for row in raw_rows),
        "unique_prefix_nodes": len(nodes),
        "prompt_truncations": sum(row["prompt_was_truncated"] for row in raw_rows),
        "max_base_score_reproduction_error": max(
            row["max_base_score_reproduction_error"] for row in raw_rows
        ),
        "max_seed_score_reproduction_error": max(
            row["max_seed_score_reproduction_error"] for row in raw_rows
        ),
        "roles": role_results,
        "node_roles": node_results,
        "by_expert_transition": transitions,
    }


def fmt(value: float, digits: int = 4) -> str:
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def render_report(results: dict[str, Any]) -> str:
    lines = [
        "# ALFWorld 有价值动作变化的逐 token / 全词表数值结构",
        "",
        "同一 plain prompt、同一 Base tokenizer、BF16，并严格复用阶段一的 batch=4 与右填充语义。",
        "动作级 score delta 被逐 token 精确分解；每个唯一候选前缀另保存全词表概率总变差与最大 donor/receiver。",
        "分支价值使用该 token 后可达候选动作的平均 expert-recovery 折扣价值，仅用于机制描述，不等同于模型 rollout Q 值。",
        "",
    ]
    for split, result in results["splits"].items():
        lines.extend(
            [
                f"## {split}",
                "",
                f"- {result['states']} 个状态、{result['episodes']} 个 trial、"
                f"{result['candidate_actions']} 个动作、{result['candidate_tokens']} 个动作 token。",
                f"- 唯一候选前缀节点：{result['unique_prefix_nodes']}；prompt 截断：{result['prompt_truncations']}。",
                f"- 阶段一动作分数最大复现误差（Base / SEED）："
                f"{result['max_base_score_reproduction_error']:.3g} / "
                f"{result['max_seed_score_reproduction_error']:.3g}。",
                "",
                "### 动作 score delta 的语义角色分解",
                "",
                "| role | 绝对变化占比 | value Spearman [cluster 95% CI] | receiver−donor role delta [CI] |",
                "|---|---:|---:|---:|",
            ]
        )
        for role, item in result["roles"].items():
            corr = item["state_value_spearman_bootstrap"]
            contrast = item["state_receiver_minus_donor_bootstrap"]
            lines.append(
                f"| {role} | {100 * item['absolute_score_delta_share']:.2f}% | "
                f"{fmt(corr['observed'])} [{fmt(corr['ci95'][0])}, {fmt(corr['ci95'][1])}] | "
                f"{fmt(contrast['observed'])} [{fmt(contrast['ci95'][0])}, {fmt(contrast['ci95'][1])}] |"
            )
        lines.extend(
            [
                "",
                "### 唯一前缀处的全词表与候选分支变化",
                "",
                "| next-token role | nodes | multi-child | vocab TV | candidate mass Δ | branch value Δ [CI] |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for role, item in result["node_roles"].items():
            branch = item["branch_value_delta_bootstrap"]
            lines.append(
                f"| {role} | {item['nodes']} | {item['multichild_nodes']} | "
                f"{fmt(item['vocabulary_total_variation']['mean'])} | "
                f"{fmt(item['candidate_child_mass_delta']['mean'])} | "
                f"{fmt(branch['observed'])} [{fmt(branch['ci95'][0])}, {fmt(branch['ci95'][1])}] |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    raw_rows = read_rows(args.trace)
    if not raw_rows:
        raise ValueError("No token trace rows")
    state_rows = [summarize_state(row) for row in raw_rows]
    for row, summary in zip(raw_rows, state_rows, strict=True):
        for action in row["actions"]:
            reconstructed = math.fsum(action["role_score_delta_contributions"].values())
            if abs(reconstructed - action["normalized_score_delta"]) > 1e-6:
                raise AssertionError("Role decomposition does not reconstruct action score delta")
        if max(
            summary["max_base_score_reproduction_error"],
            summary["max_seed_score_reproduction_error"],
        ) > 1e-4:
            raise AssertionError("Token logits do not reproduce phase-one action scores")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in state_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    splits = {}
    for offset, split in enumerate(sorted({row["split"] for row in raw_rows})):
        split_raw = [row for row in raw_rows if row["split"] == split]
        split_states = [row for row in state_rows if row["split"] == split]
        splits[split] = analyze_split(
            split_raw, split_states, args.bootstrap_samples, args.seed + offset * 1000
        )
    results = {
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "splits": splits,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(render_report(results), encoding="utf-8")
    (output_dir / "ANALYSIS_COMPLETE").touch()
    print(render_report(results), end="")


if __name__ == "__main__":
    main()
