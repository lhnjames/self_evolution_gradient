#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


BASELINE_NAMES = ("plain", "evolved_skill", "mismatched_skill")
CONTROL_NAMES = (
    "reformatted_skill",
    "anti_skill",
    "task_only_skill",
    "general_only_skill",
    "length_matched_placebo",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def softmax(scores: np.ndarray) -> np.ndarray:
    values = np.exp(scores - scores.max())
    return values / values.sum()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else float("nan")


def margin(scores: np.ndarray, gold: int) -> float:
    return float(scores[gold] - np.delete(scores, gold).max())


def categorical_kl(q: np.ndarray, p: np.ndarray) -> float:
    q, p = np.clip(q, 1e-12, 1.0), np.clip(p, 1e-12, 1.0)
    return float(np.sum(q * (np.log(q) - np.log(p))))


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return float(array.mean()) if len(array) else float("nan")


def correlation(left: Iterable[float], right: Iterable[float]) -> float:
    pairs = [(x, y) for x, y in zip(left, right) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    x, y = (np.asarray([pair[index] for pair in pairs]) for index in (0, 1))
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def combine_rows(baseline_path: str, control_paths: list[str]) -> list[dict[str, Any]]:
    baseline = read_jsonl(baseline_path)
    controls = [row for path in control_paths for row in read_jsonl(path)]
    controls.sort(key=lambda row: row["global_decision_index"])
    if len(baseline) != len(controls):
        raise ValueError(f"baseline={len(baseline)} controls={len(controls)}")
    rows = []
    for index, (base, control) in enumerate(zip(baseline, controls)):
        for field in ("episode_id", "task_type", "step_index", "expert_action", "admissible_actions"):
            if base[field] != control[field]:
                raise ValueError(f"row {index} mismatch in {field}")
        scores = {name: base["normalized_scores"][name] for name in BASELINE_NAMES}
        scores.update(control["normalized_scores"])
        row = {
            "episode_key": control["episode_key"],
            "task_type": base["task_type"],
            "action_verb": base["action_verb"],
            "gold_index": int(base["gold_index"]),
            "scores": scores,
            "reference_context_token_length": control["reference_context_token_length"],
            "control_context_token_lengths": control["control_context_token_lengths"],
            "reference_prompt_overflow": control["reference_prompt_overflow"],
            "control_prompt_overflows": control["control_prompt_overflows"],
        }
        add_metrics(row)
        rows.append(row)
    return rows


def add_metrics(row: dict[str, Any]) -> None:
    scores = {name: np.asarray(value, dtype=np.float64) for name, value in row["scores"].items()}
    probabilities = {name: softmax(value) for name, value in scores.items()}
    gold = row["gold_index"]
    z0, p0 = scores["plain"], probabilities["plain"]
    verified = -p0.copy()
    verified[gold] += 1.0
    metrics = {}
    directions = {}
    for name, zq in scores.items():
        pq = probabilities[name]
        direction = zq - z0
        direction -= direction.mean()
        directions[name] = direction
        metrics[name] = {
            "correct": int(np.argmax(zq) == gold),
            "gold_probability": float(pq[gold]),
            "nll": float(-math.log(max(pq[gold], 1e-12))),
            "margin": margin(zq, gold),
            "kl_from_plain": categorical_kl(pq, p0),
            "gold_logp_gain": float(math.log(max(pq[gold], 1e-12)) - math.log(max(p0[gold], 1e-12))),
            "margin_gain": margin(zq, gold) - margin(z0, gold),
            "gradient_cosine": 0.0 if name == "plain" else cosine(verified, direction),
            "first_order_gain": float(np.dot(verified, direction)),
        }
    correct_direction = directions["evolved_skill"]
    for name in CONTROL_NAMES + ("mismatched_skill",):
        metrics[name]["direction_cosine_with_evolved"] = cosine(
            directions[name], correct_direction
        )
    row["metrics"] = metrics
    row["top1_reformatted_agrees_with_evolved"] = int(
        np.argmax(scores["reformatted_skill"]) == np.argmax(scores["evolved_skill"])
    )


def summarize(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    metrics = [row["metrics"][name] for row in rows]
    result = {"decisions": len(rows), "episodes": len({row["episode_key"] for row in rows})}
    for metric in (
        "correct",
        "gold_probability",
        "nll",
        "margin",
        "kl_from_plain",
        "gold_logp_gain",
        "margin_gain",
        "gradient_cosine",
        "first_order_gain",
    ):
        result[metric] = finite_mean(item[metric] for item in metrics)
    if name != "plain":
        result["positive_gradient_rate"] = finite_mean(
            item["gradient_cosine"] > 0
            for item in metrics
            if math.isfinite(item["gradient_cosine"])
        )
    if "direction_cosine_with_evolved" in metrics[0]:
        result["direction_cosine_with_evolved"] = finite_mean(
            item["direction_cosine_with_evolved"] for item in metrics
        )
    return result


def cluster_bootstrap(
    rows: list[dict[str, Any]], left: str, right: str, metric: str, samples: int, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["episode_key"]].append(
            row["metrics"][left][metric] - row["metrics"][right][metric]
        )
    episodes = sorted(grouped)
    sums = np.asarray([np.nansum(grouped[key]) for key in episodes], dtype=np.float64)
    counts = np.asarray(
        [np.sum(np.isfinite(grouped[key])) for key in episodes], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(episodes), size=(samples, len(episodes)))
    draws = sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)
    observed = sums.sum() / counts.sum()
    return {
        "observed": float(observed),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_positive": float(np.mean(draws > 0)),
    }


def analyze_split(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    names = BASELINE_NAMES + CONTROL_NAMES
    contrasts = {}
    for index, name in enumerate(names[1:]):
        for metric in ("correct", "gold_logp_gain", "margin_gain", "gradient_cosine"):
            contrasts[f"{name}_minus_plain__{metric}"] = cluster_bootstrap(
                rows, name, "plain", metric, samples, seed + index * 10
            )
    for index, name in enumerate(("mismatched_skill",) + CONTROL_NAMES):
        for metric in ("correct", "gold_logp_gain", "margin_gain"):
            contrasts[f"evolved_skill_minus_{name}__{metric}"] = cluster_bootstrap(
                rows, "evolved_skill", name, metric, samples, seed + 100 + index * 10
            )
    return {
        "conditions": {name: summarize(rows, name) for name in names},
        "reformatted_top1_agreement_with_evolved": finite_mean(
            row["top1_reformatted_agrees_with_evolved"] for row in rows
        ),
        "reformatted_effect_correlation": correlation(
            (row["metrics"]["evolved_skill"]["margin_gain"] for row in rows),
            (row["metrics"]["reformatted_skill"]["margin_gain"] for row in rows),
        ),
        "anti_effect_correlation": correlation(
            (row["metrics"]["evolved_skill"]["margin_gain"] for row in rows),
            (row["metrics"]["anti_skill"]["margin_gain"] for row in rows),
        ),
        "placebo_length_exact_rate": finite_mean(
            row["control_context_token_lengths"]["length_matched_placebo"]
            == row["reference_context_token_length"]
            for row in rows
        ),
        "prompt_truncation": {
            "evolved_skill_rate": finite_mean(
                row["reference_prompt_overflow"] > 0 for row in rows
            ),
            "evolved_skill_mean_overflow_tokens": finite_mean(
                row["reference_prompt_overflow"] for row in rows
            ),
            **{
                f"{name}_rate": finite_mean(
                    row["control_prompt_overflows"][name] > 0 for row in rows
                )
                for name in CONTROL_NAMES
            },
        },
        "bootstrap_contrasts": contrasts,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# ALFWorld semantic skill-control results",
        "",
        "| split | condition | top-1 | NLL | margin gain | KL | gradient cosine | cosine with evolved |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, analysis in result["splits"].items():
        for name, row in analysis["conditions"].items():
            direction = row.get("direction_cosine_with_evolved", float("nan"))
            lines.append(
                f"| {split} | {name} | {row['correct']:.4f} | {row['nll']:.4f} | "
                f"{row['margin_gain']:+.4f} | {row['kl_from_plain']:.4f} | "
                f"{row['gradient_cosine']:+.4f} | {direction:+.4f} |"
            )
    lines.extend(["", "All confidence intervals are episode-cluster bootstraps.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen-baseline", required=True)
    parser.add_argument("--unseen-baseline", required=True)
    parser.add_argument("--seen-controls", nargs="+", required=True)
    parser.add_argument("--unseen-controls", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    all_rows = {
        "valid_seen": combine_rows(args.seen_baseline, args.seen_controls),
        "valid_unseen": combine_rows(args.unseen_baseline, args.unseen_controls),
    }
    result = {
        "bootstrap_samples": args.bootstrap_samples,
        "splits": {
            split: analyze_split(rows, args.bootstrap_samples, args.seed + index)
            for index, (split, rows) in enumerate(all_rows.items())
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    report = render_report(result)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
