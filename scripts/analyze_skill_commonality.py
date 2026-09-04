#!/usr/bin/env python3
"""Diagnose whether ALFWorld skill shifts form reusable output-edit directions.

This script is deliberately observational: it does not fit a gate or propose a
new policy.  It reuses cached candidate sequence scores, restores unique
gamefile-level episode keys, and measures raw skill effects and their alignment
with the negative expert cross-entropy gradient in output-logit space.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DOSES = (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0)
BOOTSTRAP_METRICS = (
    "accuracy_delta",
    "gold_logp_gain",
    "margin_gain",
    "first_order_gain",
    "alignment_cosine",
    "correct_minus_mismatch_margin",
)


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    values = np.exp(shifted)
    return values / values.sum()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def margin(scores: np.ndarray, gold: int) -> float:
    if len(scores) == 1:
        return 0.0
    alternatives = np.delete(scores, gold)
    return float(scores[gold] - alternatives.max())


def kl_divergence(q: np.ndarray, p: np.ndarray) -> float:
    q_safe = np.clip(q, 1e-12, 1.0)
    p_safe = np.clip(p, 1e-12, 1.0)
    return float(np.sum(q_safe * (np.log(q_safe) - np.log(p_safe))))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation(left: Iterable[float], right: Iterable[float], *, ranked: bool = False) -> float:
    pairs = [(x, y) for x, y in zip(left, right) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if ranked:
        x, y = rankdata(x), rankdata(y)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rows(trace_path: str | Path, decision_path: str | Path, split: str) -> list[dict[str, Any]]:
    traces = _read_jsonl(trace_path)
    decisions = [row for row in _read_jsonl(decision_path) if not row.get("is_trivial", False)]
    if len(traces) != len(decisions):
        raise ValueError(f"{split}: {len(traces)} trace rows != {len(decisions)} decisions")
    enriched = []
    for index, (trace, decision) in enumerate(zip(traces, decisions)):
        checks = ("episode_id", "task_type", "step_index", "expert_action", "admissible_actions")
        for field in checks:
            if trace[field] != decision[field]:
                raise ValueError(f"{split} row {index}: alignment mismatch in {field}")
        row = dict(trace)
        row["split"] = split
        row["gamefile"] = decision["gamefile"]
        row["episode_key"] = decision["gamefile"]
        enriched.append(row)
    return enriched


def verb_projection(probabilities: np.ndarray, candidates: list[str], index: dict[str, int]) -> np.ndarray:
    projected = np.zeros(len(index), dtype=np.float64)
    for probability, candidate in zip(probabilities, candidates):
        verb = candidate.split(maxsplit=1)[0] if candidate.strip() else "<empty>"
        projected[index[verb]] += probability
    return projected


def classify_flip(plain_correct: bool, skill_correct: bool, margin_gain: float) -> str:
    if not plain_correct and skill_correct:
        return "rescue"
    if plain_correct and not skill_correct:
        return "harm"
    if margin_gain > 1e-9:
        return "reinforce"
    if margin_gain < -1e-9:
        return "weaken"
    return "inert"


def add_effects(rows: list[dict[str, Any]], verb_index: dict[str, int]) -> None:
    for row in rows:
        scores = row["normalized_scores"]
        z0 = np.asarray(scores["plain"], dtype=np.float64)
        zs = np.asarray(scores["evolved_skill"], dtype=np.float64)
        zw = np.asarray(scores["mismatched_skill"], dtype=np.float64)
        p0, ps, pw = softmax(z0), softmax(zs), softmax(zw)
        gold = int(row["gold_index"])
        expert_update = -p0
        expert_update[gold] += 1.0  # negative CE gradient at the plain distribution
        skill_direction = zs - z0
        skill_direction -= skill_direction.mean()
        wrong_direction = zw - z0
        wrong_direction -= wrong_direction.mean()
        margin_gain = margin(zs, gold) - margin(z0, gold)
        wrong_margin_gain = margin(zw, gold) - margin(z0, gold)
        plain_correct = int(np.argmax(z0) == gold)
        skill_correct = int(np.argmax(zs) == gold)
        wrong_correct = int(np.argmax(zw) == gold)

        p0_verb = verb_projection(p0, row["admissible_actions"], verb_index)
        ps_verb = verb_projection(ps, row["admissible_actions"], verb_index)
        pw_verb = verb_projection(pw, row["admissible_actions"], verb_index)
        gold_verb = row["action_verb"]
        expert_verb_update = -p0_verb
        expert_verb_update[verb_index[gold_verb]] += 1.0

        row["effect"] = {
            "plain_correct": plain_correct,
            "skill_correct": skill_correct,
            "mismatch_correct": wrong_correct,
            "accuracy_delta": skill_correct - plain_correct,
            "gold_probability_gain": float(ps[gold] - p0[gold]),
            "gold_logp_gain": float(math.log(max(ps[gold], 1e-12)) - math.log(max(p0[gold], 1e-12))),
            "margin_gain": margin_gain,
            "mismatch_margin_gain": wrong_margin_gain,
            "correct_minus_mismatch_margin": margin_gain - wrong_margin_gain,
            "first_order_gain": float(np.dot(expert_update, skill_direction)),
            "mismatch_first_order_gain": float(np.dot(expert_update, wrong_direction)),
            "alignment_cosine": cosine(expert_update, skill_direction),
            "mismatch_alignment_cosine": cosine(expert_update, wrong_direction),
            "direction_norm": float(np.linalg.norm(skill_direction)),
            "kl_from_plain": kl_divergence(ps, p0),
            "verb_gold_probability_gain": float(
                ps_verb[verb_index[gold_verb]] - p0_verb[verb_index[gold_verb]]
            ),
            "verb_alignment_cosine": cosine(expert_verb_update, ps_verb - p0_verb),
            "mismatch_verb_alignment_cosine": cosine(expert_verb_update, pw_verb - p0_verb),
            "flip": classify_flip(bool(plain_correct), bool(skill_correct), margin_gain),
        }
        row["schema_shift"] = ps_verb - p0_verb
        row["mismatch_schema_shift"] = pw_verb - p0_verb
        row["dose"] = {}
        for dose in DOSES:
            q = softmax(z0 + dose * skill_direction)
            row["dose"][dose] = {
                "correct": int(np.argmax(q) == gold),
                "gold_probability": float(q[gold]),
                "nll": float(-math.log(max(q[gold], 1e-12))),
                "margin": margin(z0 + dose * skill_direction, gold),
                "kl": kl_divergence(q, p0),
            }


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return float(array.mean()) if len(array) else float("nan")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flips = defaultdict(int)
    for row in rows:
        flips[row["effect"]["flip"]] += 1
    result = {
        "decisions": len(rows),
        "episodes": len({row["episode_key"] for row in rows}),
        "legacy_episode_ids": len({row["episode_id"] for row in rows}),
        "plain_accuracy": finite_mean(row["effect"]["plain_correct"] for row in rows),
        "skill_accuracy": finite_mean(row["effect"]["skill_correct"] for row in rows),
        "mismatch_accuracy": finite_mean(row["effect"]["mismatch_correct"] for row in rows),
        "flip_rates": {name: count / len(rows) for name, count in sorted(flips.items())},
    }
    for metric in (
        "accuracy_delta",
        "gold_probability_gain",
        "gold_logp_gain",
        "margin_gain",
        "first_order_gain",
        "alignment_cosine",
        "mismatch_margin_gain",
        "mismatch_first_order_gain",
        "mismatch_alignment_cosine",
        "correct_minus_mismatch_margin",
        "direction_norm",
        "kl_from_plain",
        "verb_gold_probability_gain",
        "verb_alignment_cosine",
        "mismatch_verb_alignment_cosine",
    ):
        result[metric] = finite_mean(row["effect"][metric] for row in rows)
    result["positive_margin_rate"] = finite_mean(row["effect"]["margin_gain"] > 0 for row in rows)
    result["positive_alignment_rate"] = finite_mean(
        row["effect"]["alignment_cosine"] > 0
        for row in rows
        if math.isfinite(row["effect"]["alignment_cosine"])
    )
    result["linear_actual_pearson"] = correlation(
        (row["effect"]["first_order_gain"] for row in rows),
        (row["effect"]["gold_logp_gain"] for row in rows),
    )
    result["linear_actual_spearman"] = correlation(
        (row["effect"]["first_order_gain"] for row in rows),
        (row["effect"]["gold_logp_gain"] for row in rows),
        ranked=True,
    )
    return result


def cluster_bootstrap(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["episode_key"]].append(row)
    episodes = sorted(grouped)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(episodes), size=(samples, len(episodes)))
    output = {}
    for metric in BOOTSTRAP_METRICS:
        sums = np.asarray(
            [np.nansum([row["effect"][metric] for row in grouped[key]]) for key in episodes],
            dtype=np.float64,
        )
        counts = np.asarray(
            [np.sum([math.isfinite(row["effect"][metric]) for row in grouped[key]]) for key in episodes]
        )
        episode_means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        decision_draws = sums[chosen].sum(axis=1) / np.maximum(counts[chosen].sum(axis=1), 1)
        episode_draws = np.nanmean(episode_means[chosen], axis=1)
        observed_decision = finite_mean(row["effect"][metric] for row in rows)
        observed_episode = float(np.nanmean(episode_means))
        output[metric] = {
            "decision_weighted": _bootstrap_row(observed_decision, decision_draws),
            "episode_weighted": _bootstrap_row(observed_episode, episode_draws),
        }
    return output


def _bootstrap_row(observed: float, draws: np.ndarray) -> dict[str, Any]:
    return {
        "observed": observed,
        "ci95": [float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))],
        "probability_positive": float(np.nanmean(draws > 0)),
    }


def dose_response(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for dose in DOSES:
        output[str(dose)] = {
            metric: finite_mean(row["dose"][dose][metric] for row in rows)
            for metric in ("correct", "gold_probability", "nll", "margin", "kl")
        }
    return output


def slice_results(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in key_fields)].append(row)
    output = {}
    for key, group in sorted(groups.items()):
        if len(group) < 5:
            continue
        name = " | ".join(key)
        output[name] = summarize(group)
    return output


def pairwise_commonality(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if left["episode_key"] == right["episode_key"]:
                continue
            same_task = left["task_type"] == right["task_type"]
            same_stage = left["action_verb"] == right["action_verb"]
            label = f"{'same' if same_task else 'different'}_task__{'same' if same_stage else 'different'}_stage"
            similarity = cosine(left[field], right[field])
            if math.isfinite(similarity):
                buckets[label].append(similarity)
    return {
        label: {
            "pairs": len(values),
            "mean_cosine": finite_mean(values),
            "median_cosine": float(np.median(values)),
            "positive_rate": finite_mean(value > 0 for value in values),
        }
        for label, values in sorted(buckets.items())
    }


def cross_split_prototypes(
    all_rows: dict[str, list[dict[str, Any]]], field: str = "schema_shift"
) -> dict[str, Any]:
    prototypes: dict[str, dict[tuple[str, str], np.ndarray]] = defaultdict(dict)
    counts: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    for split, rows in all_rows.items():
        groups: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        for row in rows:
            groups[(row["task_type"], row["action_verb"])].append(row[field])
        for key, vectors in groups.items():
            prototypes[split][key] = np.mean(vectors, axis=0)
            counts[split][key] = len(vectors)
    output = {}
    for left, right in (("train", "valid_seen"), ("train", "valid_unseen"), ("valid_seen", "valid_unseen")):
        rows = []
        for key in sorted(set(prototypes[left]) & set(prototypes[right])):
            if min(counts[left][key], counts[right][key]) < 3:
                continue
            similarity = cosine(prototypes[left][key], prototypes[right][key])
            if math.isfinite(similarity):
                rows.append(
                    {
                        "task_type": key[0],
                        "action_verb": key[1],
                        "left_count": counts[left][key],
                        "right_count": counts[right][key],
                        "cosine": similarity,
                    }
                )
        output[f"{left}__{right}"] = {
            "slices": rows,
            "mean_cosine": finite_mean(row["cosine"] for row in rows),
            "positive_rate": finite_mean(row["cosine"] > 0 for row in rows),
        }
    return output


def write_enriched_trace(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {key: value for key, value in row.items() if key not in {"effect", "schema_shift", "mismatch_schema_shift", "dose"}}
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")


def render_report(results: dict[str, Any]) -> str:
    lines = [
        "# ALFWorld skill commonality diagnostic",
        "",
        "This is an observational analysis over cached sequence scores; no new policy or gate is fitted.",
        "",
        "## Aggregate raw skill effects",
        "",
        "| split | decisions / true episodes | plain acc | skill acc | mismatch acc | margin gain | log-p gain | alignment cosine | positive alignment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, summary in results["aggregate"].items():
        lines.append(
            f"| {split} | {summary['decisions']} / {summary['episodes']} | "
            f"{summary['plain_accuracy']:.4f} | {summary['skill_accuracy']:.4f} | "
            f"{summary['mismatch_accuracy']:.4f} | {summary['margin_gain']:+.4f} | "
            f"{summary['gold_logp_gain']:+.4f} | {summary['alignment_cosine']:+.4f} | "
            f"{summary['positive_alignment_rate']:.4f} |"
        )
    lines.extend(["", "## Corrected episode-cluster bootstrap (decision-weighted)", ""])
    lines.extend(
        [
            "| split | metric | observed | 95% CI | P(>0) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for split, metrics in results["bootstrap"].items():
        for metric in ("accuracy_delta", "gold_logp_gain", "margin_gain", "alignment_cosine"):
            row = metrics[metric]["decision_weighted"]
            lines.append(
                f"| {split} | {metric} | {row['observed']:+.4f} | "
                f"[{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}] | {row['probability_positive']:.4f} |"
            )
    lines.extend(["", "## Dose response", ""])
    for split, doses in results["dose_response"].items():
        lines.extend(
            [
                f"### {split}",
                "",
                "| alpha | top-1 | NLL | margin | KL |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for dose, row in doses.items():
            lines.append(f"| {dose} | {row['correct']:.4f} | {row['nll']:.4f} | {row['margin']:+.4f} | {row['kl']:.4f} |")
        lines.append("")
    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- Alignment is measured in candidate-logit space against the negative expert CE gradient.",
            "- Verb-space cosine measures action-stage structure, not full object/receptacle logic.",
            "- The available cache contains correct and mismatched skills, but not paraphrase, negation, or length-matched placebo controls.",
            "- A stable output-space direction is necessary but not sufficient evidence for a reusable backbone-parameter edit.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "train": (args.train_trace, args.train_decisions),
        "valid_seen": (args.seen_trace, args.seen_decisions),
        "valid_unseen": (args.unseen_trace, args.unseen_decisions),
    }
    all_rows = {
        split: load_rows(trace, decisions, split)
        for split, (trace, decisions) in paths.items()
    }
    verbs = sorted(
        {
            candidate.split(maxsplit=1)[0] if candidate.strip() else "<empty>"
            for rows in all_rows.values()
            for row in rows
            for candidate in row["admissible_actions"]
        }
    )
    verb_index = {verb: index for index, verb in enumerate(verbs)}
    for rows in all_rows.values():
        add_effects(rows, verb_index)

    results = {
        "analysis_type": "observational_cached_output_distribution",
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "verb_vocabulary": verbs,
        "aggregate": {split: summarize(rows) for split, rows in all_rows.items()},
        "bootstrap": {
            split: cluster_bootstrap(rows, args.bootstrap_samples, args.seed + index)
            for index, (split, rows) in enumerate(all_rows.items())
        },
        "dose_response": {split: dose_response(rows) for split, rows in all_rows.items()},
        "by_task_type": {split: slice_results(rows, ("task_type",)) for split, rows in all_rows.items()},
        "by_action_verb": {split: slice_results(rows, ("action_verb",)) for split, rows in all_rows.items()},
        "by_task_and_verb": {
            split: slice_results(rows, ("task_type", "action_verb"))
            for split, rows in all_rows.items()
        },
        "pairwise_schema_commonality": {
            split: {
                "correct_skill": pairwise_commonality(rows, "schema_shift"),
                "mismatched_skill": pairwise_commonality(rows, "mismatch_schema_shift"),
            }
            for split, rows in all_rows.items()
        },
        "cross_split_schema_prototypes": {
            "correct_skill": cross_split_prototypes(all_rows, "schema_shift"),
            "mismatched_skill": cross_split_prototypes(all_rows, "mismatch_schema_shift"),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(render_report(results), encoding="utf-8")
    for split, rows in all_rows.items():
        write_enriched_trace(rows, output_dir / "enriched_traces" / f"{split}.jsonl")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trace", required=True)
    parser.add_argument("--seen-trace", required=True)
    parser.add_argument("--unseen-trace", required=True)
    parser.add_argument("--train-decisions", required=True)
    parser.add_argument("--seen-decisions", required=True)
    parser.add_argument("--unseen-decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    results = run(args)
    print(json.dumps({"output_dir": args.output_dir, "aggregate": results["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
