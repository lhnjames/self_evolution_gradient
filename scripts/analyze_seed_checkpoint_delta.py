#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from self_evolve.seed_checkpoint_delta import centered, cosine, minimal_top1_repair, softmax


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return float(array.mean()) if len(array) else float("nan")


def categorical_kl(q: np.ndarray, p: np.ndarray) -> float:
    q = np.clip(q, 1e-12, 1.0)
    p = np.clip(p, 1e-12, 1.0)
    return float(np.sum(q * (np.log(q) - np.log(p))))


def align_rows(base_path: str, checkpoint_paths: list[str]) -> list[dict[str, Any]]:
    base_rows = read_jsonl(base_path)
    checkpoint_rows = [row for path in checkpoint_paths for row in read_jsonl(path)]
    checkpoint_rows.sort(key=lambda row: row["global_decision_index"])
    if len(base_rows) != len(checkpoint_rows):
        raise ValueError(f"base={len(base_rows)} checkpoint={len(checkpoint_rows)}")
    aligned = []
    for index, (base, checkpoint) in enumerate(zip(base_rows, checkpoint_rows)):
        for field in ("task_type", "step_index", "expert_action", "admissible_actions", "gold_index"):
            if base[field] != checkpoint[field]:
                raise ValueError(f"row {index} mismatch in {field}")
        z0 = np.asarray(base["normalized_scores"]["plain"], dtype=np.float64)
        z1 = np.asarray(checkpoint["normalized_scores"], dtype=np.float64)
        p0, p1 = softmax(z0), softmax(z1)
        gold = int(base["gold_index"])
        pred0, pred1 = int(np.argmax(z0)), int(np.argmax(z1))
        delta = centered(z1 - z0)
        verified = np.zeros_like(p0)
        verified[gold] = 1.0
        verified -= p0
        skill_delta = centered(
            np.asarray(base["normalized_scores"]["evolved_skill"], dtype=np.float64) - z0
        )
        repair = minimal_top1_repair(z0, gold)
        repair_metrics = None
        if pred0 != gold and repair.gap > 1e-12:
            repair_metrics = {
                "competitor_index": repair.competitor_index,
                "gap": repair.gap,
                "minimal_l2": repair.l2_norm,
                "checkpoint_cosine_with_minimal_repair": cosine(delta, repair.delta),
                "checkpoint_norm_ratio_to_minimal": float(
                    np.linalg.norm(delta) / repair.l2_norm
                ),
                "checkpoint_projection_in_minimal_units": float(
                    np.dot(delta, repair.delta) / np.dot(repair.delta, repair.delta)
                ),
                "original_competitor_margin_change": float(
                    (z1[gold] - z1[repair.competitor_index])
                    - (z0[gold] - z0[repair.competitor_index])
                ),
            }
        aligned.append(
            {
                "global_index": index,
                "episode_key": checkpoint.get("episode_key", checkpoint.get("gamefile", base["episode_id"])),
                "task_type": base["task_type"],
                "action_verb": base["action_verb"],
                "gold_index": gold,
                "base_correct": int(pred0 == gold),
                "checkpoint_correct": int(pred1 == gold),
                "rescue": int(pred0 != gold and pred1 == gold),
                "harm": int(pred0 == gold and pred1 != gold),
                "base_gold_probability": float(p0[gold]),
                "checkpoint_gold_probability": float(p1[gold]),
                "base_nll": -math.log(max(float(p0[gold]), 1e-12)),
                "checkpoint_nll": -math.log(max(float(p1[gold]), 1e-12)),
                "checkpoint_kl_from_base": categorical_kl(p1, p0),
                "checkpoint_delta_l2": float(np.linalg.norm(delta)),
                "checkpoint_verified_cosine": cosine(delta, verified),
                "checkpoint_first_order_gain": float(np.dot(delta, verified)),
                "checkpoint_gold_logp_gain": float(
                    math.log(max(float(p1[gold]), 1e-12))
                    - math.log(max(float(p0[gold]), 1e-12))
                ),
                "checkpoint_cosine_with_prompt_skill": cosine(delta, skill_delta),
                "repair": repair_metrics,
            }
        )
    return aligned


def cluster_bootstrap(
    rows: list[dict[str, Any]], metric, samples: int, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = metric(row)
        if math.isfinite(value):
            grouped[row["episode_key"]].append(value)
    episodes = sorted(grouped)
    sums = np.asarray([np.sum(grouped[key]) for key in episodes], dtype=np.float64)
    counts = np.asarray([len(grouped[key]) for key in episodes], dtype=np.float64)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(episodes), size=(samples, len(episodes)))
    draws = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return {
        "observed": float(sums.sum() / counts.sum()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_positive": float(np.mean(draws > 0)),
    }


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def slice_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return {
        label: {
            "states": len(group),
            "episodes": len({row["episode_key"] for row in group}),
            "base_top1": finite_mean(row["base_correct"] for row in group),
            "checkpoint_top1": finite_mean(row["checkpoint_correct"] for row in group),
            "top1_delta": finite_mean(
                row["checkpoint_correct"] - row["base_correct"] for row in group
            ),
            "rescue_rate": finite_mean(row["rescue"] for row in group),
            "harm_rate": finite_mean(row["harm"] for row in group),
            "mean_verified_cosine": finite_mean(
                row["checkpoint_verified_cosine"] for row in group
            ),
            "mean_kl": finite_mean(row["checkpoint_kl_from_base"] for row in group),
        }
        for label, group in sorted(groups.items())
    }


def repair_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"states": 0}
    return {
        "states": len(rows),
        "minimal_l2": quantiles(row["repair"]["minimal_l2"] for row in rows),
        "checkpoint_cosine_with_minimal_repair": quantiles(
            row["repair"]["checkpoint_cosine_with_minimal_repair"] for row in rows
        ),
        "checkpoint_norm_ratio_to_minimal": quantiles(
            row["repair"]["checkpoint_norm_ratio_to_minimal"] for row in rows
        ),
        "checkpoint_projection_in_minimal_units": quantiles(
            row["repair"]["checkpoint_projection_in_minimal_units"] for row in rows
        ),
    }


def repair_distance_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    distances = np.asarray([row["repair"]["minimal_l2"] for row in rows], dtype=np.float64)
    boundaries = np.quantile(distances, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = []
    for index, (low, high) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if index == 3:
            selected = [
                row for row in rows if low <= row["repair"]["minimal_l2"] <= high
            ]
        else:
            selected = [
                row for row in rows if low <= row["repair"]["minimal_l2"] < high
            ]
        bins.append(
            {
                "quartile": index + 1,
                "lower": float(low),
                "upper": float(high),
                "states": len(selected),
                "rescue_rate": finite_mean(row["checkpoint_correct"] for row in selected),
                "mean_checkpoint_cosine_with_minimal_repair": finite_mean(
                    row["repair"]["checkpoint_cosine_with_minimal_repair"]
                    for row in selected
                ),
                "mean_checkpoint_projection_in_minimal_units": finite_mean(
                    row["repair"]["checkpoint_projection_in_minimal_units"]
                    for row in selected
                ),
            }
        )
    return bins


def analyze_split(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    wrong = [row for row in rows if not row["base_correct"] and row["repair"] is not None]
    rescued_wrong = [row for row in wrong if row["checkpoint_correct"]]
    unresolved_wrong = [row for row in wrong if not row["checkpoint_correct"]]
    return {
        "states": len(rows),
        "episodes": len({row["episode_key"] for row in rows}),
        "base": {
            "top1": finite_mean(row["base_correct"] for row in rows),
            "mean_gold_probability": finite_mean(row["base_gold_probability"] for row in rows),
            "nll": finite_mean(row["base_nll"] for row in rows),
        },
        "seed_checkpoint": {
            "top1": finite_mean(row["checkpoint_correct"] for row in rows),
            "mean_gold_probability": finite_mean(
                row["checkpoint_gold_probability"] for row in rows
            ),
            "nll": finite_mean(row["checkpoint_nll"] for row in rows),
            "mean_kl_from_base": finite_mean(row["checkpoint_kl_from_base"] for row in rows),
            "mean_delta_l2": finite_mean(row["checkpoint_delta_l2"] for row in rows),
            "mean_verified_cosine": finite_mean(
                row["checkpoint_verified_cosine"] for row in rows
            ),
            "positive_verified_cosine_rate": finite_mean(
                row["checkpoint_verified_cosine"] > 0
                for row in rows
                if math.isfinite(row["checkpoint_verified_cosine"])
            ),
            "mean_cosine_with_prompt_skill": finite_mean(
                row["checkpoint_cosine_with_prompt_skill"] for row in rows
            ),
        },
        "transitions": {
            "rescued": int(sum(row["rescue"] for row in rows)),
            "harmed": int(sum(row["harm"] for row in rows)),
            "rescue_rate_all": finite_mean(row["rescue"] for row in rows),
            "harm_rate_all": finite_mean(row["harm"] for row in rows),
            "repair_rate_among_base_errors": finite_mean(
                row["checkpoint_correct"] for row in rows if not row["base_correct"]
            ),
        },
        "bootstrap": {
            "top1_delta": cluster_bootstrap(
                rows,
                lambda row: row["checkpoint_correct"] - row["base_correct"],
                samples,
                seed,
            ),
            "gold_logp_gain": cluster_bootstrap(
                rows, lambda row: row["checkpoint_gold_logp_gain"], samples, seed + 1
            ),
            "verified_cosine": cluster_bootstrap(
                rows, lambda row: row["checkpoint_verified_cosine"], samples, seed + 2
            ),
        },
        "base_error_repair_distance": {
            "gap": quantiles(row["repair"]["gap"] for row in wrong),
            "minimal_l2": quantiles(row["repair"]["minimal_l2"] for row in wrong),
        },
        "seed_vs_minimal_repair_on_base_errors": {
            "cosine": quantiles(
                row["repair"]["checkpoint_cosine_with_minimal_repair"] for row in wrong
            ),
            "norm_ratio": quantiles(
                row["repair"]["checkpoint_norm_ratio_to_minimal"] for row in wrong
            ),
            "projection_in_minimal_units": quantiles(
                row["repair"]["checkpoint_projection_in_minimal_units"] for row in wrong
            ),
        },
        "base_error_outcome_groups": {
            "rescued": repair_group_summary(rescued_wrong),
            "unresolved": repair_group_summary(unresolved_wrong),
            "distance_quartiles": repair_distance_bins(wrong),
        },
        "by_task_type": slice_summary(rows, "task_type"),
        "by_action_verb": slice_summary(rows, "action_verb"),
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Base Qwen vs SEED checkpoint on identical ALFWorld expert states",
        "",
        "Both checkpoints use the existing plain prompt, identical admissible commands, and the same sequence scorer.",
        "",
        "| split | base top-1 | SEED top-1 | delta | base NLL | SEED NLL | KL(SEED||base) | verified cosine |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, analysis in result["splits"].items():
        top1_delta = analysis["seed_checkpoint"]["top1"] - analysis["base"]["top1"]
        lines.append(
            f"| {split} | {analysis['base']['top1']:.4f} | "
            f"{analysis['seed_checkpoint']['top1']:.4f} | {top1_delta:+.4f} | "
            f"{analysis['base']['nll']:.4f} | {analysis['seed_checkpoint']['nll']:.4f} | "
            f"{analysis['seed_checkpoint']['mean_kl_from_base']:.4f} | "
            f"{analysis['seed_checkpoint']['mean_verified_cosine']:+.4f} |"
        )
    lines.extend(["", "Confidence intervals in results.json use episode-cluster bootstrap.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen-base", required=True)
    parser.add_argument("--unseen-base", required=True)
    parser.add_argument("--seen-checkpoint", nargs="+", required=True)
    parser.add_argument("--unseen-checkpoint", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    rows = {
        "valid_seen": align_rows(args.seen_base, args.seen_checkpoint),
        "valid_unseen": align_rows(args.unseen_base, args.unseen_checkpoint),
    }
    result = {
        "comparison": "Qwen2.5-3B-Instruct_vs_Seed-AlfWorld-3B",
        "prompt_mode": "existing_plain_prompt_for_both_models",
        "bootstrap_samples": args.bootstrap_samples,
        "splits": {
            split: analyze_split(split_rows, args.bootstrap_samples, args.seed + index * 100)
            for index, (split, split_rows) in enumerate(rows.items())
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in rows.items():
        with (output_dir / f"{split}_trace.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    report = render_report(result)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
