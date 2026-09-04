#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TEACHERS = (
    "evolved_skill",
    "mismatched_skill",
    "reformatted_skill",
    "anti_skill",
    "length_matched_placebo",
)


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray([value for value in values if math.isfinite(value)])
    return float(array.mean()) if len(array) else float("nan")


def bootstrap(states: list[dict[str, Any]], teacher: str, samples: int, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for state in states:
        grouped[state["episode_key"]].append(
            state["conditions"][teacher]["cosine_with_verified"]
        )
    episodes = sorted(grouped)
    sums = np.asarray([np.sum(grouped[key]) for key in episodes], dtype=np.float64)
    counts = np.asarray([len(grouped[key]) for key in episodes], dtype=np.float64)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(episodes), size=(samples, len(episodes)))
    draws = sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)
    return {
        "observed": float(sums.sum() / counts.sum()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_positive": float(np.mean(draws > 0)),
    }


def contrast_bootstrap(
    states: list[dict[str, Any]], left: str, right: str, samples: int, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for state in states:
        grouped[state["episode_key"]].append(
            state["conditions"][left]["cosine_with_verified"]
            - state["conditions"][right]["cosine_with_verified"]
        )
    episodes = sorted(grouped)
    sums = np.asarray([np.sum(grouped[key]) for key in episodes], dtype=np.float64)
    counts = np.asarray([len(grouped[key]) for key in episodes], dtype=np.float64)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(episodes), size=(samples, len(episodes)))
    draws = sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)
    return {
        "observed": float(sums.sum() / counts.sum()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_positive": float(np.mean(draws > 0)),
    }


def pairwise_commonality(states: list[dict[str, Any]], gradients: np.ndarray) -> dict[str, Any]:
    similarity = gradients.astype(np.float32) @ gradients.astype(np.float32).T
    buckets: dict[str, list[float]] = defaultdict(list)
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            if states[left]["episode_key"] == states[right]["episode_key"]:
                continue
            same_task = states[left]["task_type"] == states[right]["task_type"]
            same_stage = states[left]["action_verb"] == states[right]["action_verb"]
            label = f"{'same' if same_task else 'different'}_task__{'same' if same_stage else 'different'}_stage"
            buckets[label].append(float(similarity[left, right]))
    return {
        name: {
            "pairs": len(values),
            "mean_cosine": finite_mean(values),
            "median_cosine": float(np.median(values)),
            "positive_rate": finite_mean(value > 0 for value in values),
        }
        for name, values in sorted(buckets.items())
    }


def alignment_slices(
    states: list[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        grouped[state[field]].append(state)
    result = {}
    for label, group in sorted(grouped.items()):
        result[label] = {
            "states": len(group),
            "episodes": len({state["episode_key"] for state in group}),
            "conditions": {
                teacher: {
                    "mean_cosine_with_verified": finite_mean(
                        state["conditions"][teacher]["cosine_with_verified"]
                        for state in group
                    ),
                    "positive_alignment_rate": finite_mean(
                        state["conditions"][teacher]["cosine_with_verified"] > 0
                        for state in group
                    ),
                }
                for teacher in TEACHERS
            },
        }
    return result


def load_shards(paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], int]:
    records = []
    gradient_rows: dict[int, dict[str, np.ndarray]] = {}
    parameter_count = None
    for path in paths:
        root = Path(path)
        result = json.loads((root / "results.json").read_text(encoding="utf-8"))
        parameter_count = parameter_count or result["parameter_count"]
        if result["parameter_count"] != parameter_count:
            raise ValueError("parameter counts differ across shards")
        arrays = np.load(root / "normalized_gradients.npz")
        for local_index, state in enumerate(result["states"]):
            global_index = int(state["global_index"])
            records.append(state)
            gradient_rows[global_index] = {
                name: arrays[name][local_index] for name in TEACHERS
            }
    records.sort(key=lambda state: state["global_index"])
    gradients = {
        name: np.stack([gradient_rows[state["global_index"]][name] for state in records])
        for name in TEACHERS
    }
    return records, gradients, int(parameter_count)


def analyze_split(paths: list[str], samples: int, seed: int) -> dict[str, Any]:
    states, gradients, parameter_count = load_shards(paths)
    condition_summary = {}
    for index, teacher in enumerate(TEACHERS):
        values = [state["conditions"][teacher]["cosine_with_verified"] for state in states]
        norms = [state["conditions"][teacher]["gradient_norm"] for state in states]
        unit_gradients = gradients[teacher].astype(np.float32)
        condition_summary[teacher] = {
            "mean_cosine_with_verified": finite_mean(values),
            "positive_alignment_rate": finite_mean(value > 0 for value in values),
            "mean_gradient_norm": finite_mean(norms),
            "unit_gradient_concentration": float(np.linalg.norm(unit_gradients.mean(axis=0))),
            "episode_bootstrap": bootstrap(states, teacher, samples, seed + index),
            "pairwise_commonality": pairwise_commonality(states, unit_gradients),
        }
    cross_condition = {}
    evolved = gradients["evolved_skill"].astype(np.float32)
    for teacher in TEACHERS[1:]:
        other = gradients[teacher].astype(np.float32)
        values = np.sum(evolved * other, axis=1)
        cross_condition[f"evolved_vs_{teacher}"] = {
            "mean_cosine": float(values.mean()),
            "median_cosine": float(np.median(values)),
            "negative_rate": float(np.mean(values < 0)),
        }
    paired_alignment_contrasts = {
        f"evolved_skill_minus_{teacher}": contrast_bootstrap(
            states, "evolved_skill", teacher, samples, seed + 100 + index
        )
        for index, teacher in enumerate(TEACHERS[1:])
    }
    return {
        "states": len(states),
        "episodes": len({state["episode_key"] for state in states}),
        "parameter_count": parameter_count,
        "mean_plain_score_max_abs_error": finite_mean(
            state["plain_score_max_abs_error"] for state in states
        ),
        "max_plain_score_abs_error": max(
            state["plain_score_max_abs_error"] for state in states
        ),
        "conditions": condition_summary,
        "cross_condition": cross_condition,
        "paired_alignment_contrasts": paired_alignment_contrasts,
        "by_task_type": alignment_slices(states, "task_type"),
        "by_action_verb": alignment_slices(states, "action_verb"),
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# ALFWorld RMSNorm parameter-gradient probe",
        "",
        "| split | teacher | states | mean cosine with verified | 95% CI | positive rate | unit-gradient concentration |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split, analysis in result["splits"].items():
        for name, condition in analysis["conditions"].items():
            ci = condition["episode_bootstrap"]["ci95"]
            lines.append(
                f"| {split} | {name} | {analysis['states']} | "
                f"{condition['mean_cosine_with_verified']:+.4f} | "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                f"{condition['positive_alignment_rate']:.4f} | "
                f"{condition['unit_gradient_concentration']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen-shards", nargs="+", required=True)
    parser.add_argument("--unseen-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    result = {
        "parameter_subspace": "all_qwen_rmsnorm_scale_parameters",
        "splits": {
            "valid_seen": analyze_split(
                args.seen_shards, args.bootstrap_samples, args.seed
            ),
            "valid_unseen": analyze_split(
                args.unseen_shards, args.bootstrap_samples, args.seed + 100
            ),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    report = render(result)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
