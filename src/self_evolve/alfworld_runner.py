from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from .alfworld_data import (
    AlfworldDecision,
    collect_expert_decisions,
    load_decisions,
    save_decisions,
)
from .alfworld_skills import AlfworldSkillBank, render_conditions
from .model import resolve_model_snapshot
from .sequence_scorer import SequenceActionScorer


def categorical_kl(q: torch.Tensor, p: torch.Tensor) -> float:
    eps = torch.finfo(q.dtype).tiny
    return float((q * (q.clamp_min(eps).log() - p.clamp_min(eps).log())).sum())


def trust_region_skill_distribution(
    plain: torch.Tensor,
    skill: torch.Tensor,
    kl_budget: float,
    iterations: int = 30,
) -> tuple[torch.Tensor, float]:
    """Geometric log-distribution interpolation under KL(q || plain) <= delta."""
    log_plain = plain.clamp_min(1e-12).log()
    log_skill = skill.clamp_min(1e-12).log()

    def project(rho: float) -> torch.Tensor:
        return torch.softmax(log_plain + rho * (log_skill - log_plain), dim=-1)

    if categorical_kl(skill, plain) <= kl_budget:
        return skill, 1.0
    low, high = 0.0, 1.0
    for _ in range(iterations):
        middle = (low + high) / 2
        if categorical_kl(project(middle), plain) <= kl_budget:
            low = middle
        else:
            high = middle
    return project(low), low


def _rank(probabilities: torch.Tensor, gold_index: int) -> int:
    order = torch.argsort(probabilities, descending=True)
    return int((order == gold_index).nonzero(as_tuple=False)[0, 0]) + 1


def _ece(rows: Sequence[dict[str, Any]], bins: int = 10) -> float:
    if not rows:
        return float("nan")
    total = len(rows)
    result = 0.0
    for lower_index in range(bins):
        lower = lower_index / bins
        upper = (lower_index + 1) / bins
        members = [
            row for row in rows
            if lower <= row["confidence"] <= upper and (lower_index == bins - 1 or row["confidence"] < upper)
        ]
        if members:
            accuracy = sum(row["correct"] for row in members) / len(members)
            confidence = sum(row["confidence"] for row in members) / len(members)
            result += len(members) / total * abs(accuracy - confidence)
    return result


def summarize_condition(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "top1_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "mean_expert_probability": sum(row["expert_probability"] for row in rows) / len(rows),
        "mean_nll": sum(row["nll"] for row in rows) / len(rows),
        "mean_reciprocal_rank": sum(1.0 / row["rank"] for row in rows) / len(rows),
        "mean_entropy": sum(row["entropy"] for row in rows) / len(rows),
        "mean_kl_from_plain": sum(row["kl_from_plain"] for row in rows) / len(rows),
        "ece": _ece(rows),
        "mean_confidence": sum(row["confidence"] for row in rows) / len(rows),
        "skill_selected_rate": sum(row.get("skill_selected", 0.0) for row in rows) / len(rows),
        "mean_rho": sum(row.get("rho", 0.0) for row in rows) / len(rows),
    }


def _group_summary(traces: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[str(trace[field])].append(trace)
    result = {}
    for key, members in sorted(grouped.items()):
        plain = [row["conditions"]["plain"] for row in members]
        skill = [row["conditions"]["evolved_skill"] for row in members]
        plain_acc = summarize_condition(plain)["top1_accuracy"]
        skill_acc = summarize_condition(skill)["top1_accuracy"]
        result[key] = {
            "count": len(members),
            "plain_accuracy": plain_acc,
            "evolved_skill_accuracy": skill_acc,
            "accuracy_delta": skill_acc - plain_acc,
            "expert_probability_delta": sum(
                skill_row["expert_probability"] - plain_row["expert_probability"]
                for plain_row, skill_row in zip(plain, skill)
            ) / len(members),
        }
    return result


def _condition_row(
    probabilities: torch.Tensor,
    plain: torch.Tensor,
    gold_index: int,
    selected: float = 0.0,
    rho: float = 0.0,
) -> dict[str, Any]:
    predicted = int(probabilities.argmax())
    expert_probability = float(probabilities[gold_index])
    entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return {
        "predicted_index": predicted,
        "correct": int(predicted == gold_index),
        "confidence": float(probabilities.max()),
        "expert_probability": expert_probability,
        "nll": -math.log(max(expert_probability, 1e-12)),
        "rank": _rank(probabilities, gold_index),
        "entropy": entropy,
        "kl_from_plain": categorical_kl(probabilities, plain),
        "skill_selected": selected,
        "rho": rho,
        "probabilities": [float(x) for x in probabilities],
    }


def evaluate(
    decisions: Sequence[AlfworldDecision],
    scorer: SequenceActionScorer,
    skill_bank: AlfworldSkillBank,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    skill_cfg = config["skills"]
    eval_cfg = config["evaluation"]
    traces: list[dict[str, Any]] = []
    trace_path = output_dir / "trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace_file:
        for decision_index, decision in enumerate(decisions):
            prompts = render_conditions(
                decision,
                skill_bank,
                general_top_k=int(skill_cfg["general_top_k"]),
                task_top_k=int(skill_cfg["task_top_k"]),
                mistakes_top_k=int(skill_cfg["mistakes_top_k"]),
                history_window=int(eval_cfg["history_window"]),
            )
            distributions = {}
            raw_scores = {}
            for name, prompt in prompts.items():
                scored = scorer.score(prompt, decision.admissible_actions)
                distributions[name] = scored.probabilities(float(eval_cfg["temperature"]))
                raw_scores[name] = [float(x) for x in scored.normalized_scores]

            plain = distributions["plain"]
            skill = distributions["evolved_skill"]
            wrong = distributions["mismatched_skill"]
            gold_index = decision.admissible_actions.index(decision.expert_action)

            trust, rho = trust_region_skill_distribution(
                plain, skill, float(eval_cfg["kl_budget"])
            )
            confidence_select = float(skill.max() >= plain.max())
            confidence_gate = skill if confidence_select else plain
            # Diagnostic upper bound only: this uses the expert label and must
            # never be interpreted as a deployable verifier.
            oracle_select = float(skill[gold_index] >= plain[gold_index])
            oracle_gate = skill if oracle_select else plain

            condition_rows = {
                "plain": _condition_row(plain, plain, gold_index),
                "evolved_skill": _condition_row(skill, plain, gold_index, selected=1.0, rho=1.0),
                "mismatched_skill": _condition_row(wrong, plain, gold_index, selected=1.0, rho=1.0),
                "trust_region_skill": _condition_row(trust, plain, gold_index, selected=1.0, rho=rho),
                "confidence_gated_skill": _condition_row(
                    confidence_gate, plain, gold_index, selected=confidence_select, rho=confidence_select
                ),
                "oracle_gated_skill_upper_bound": _condition_row(
                    oracle_gate, plain, gold_index, selected=oracle_select, rho=oracle_select
                ),
            }
            trace = {
                "decision_index": decision_index,
                "episode_id": decision.episode_id,
                "episode_key": decision.gamefile,
                "gamefile": decision.gamefile,
                "task_type": decision.task_type,
                "skill_category": skill_bank.category(decision),
                "step_index": decision.step_index,
                "action_verb": decision.action_verb,
                "goal": decision.goal,
                "expert_action": decision.expert_action,
                "gold_index": gold_index,
                "admissible_actions": list(decision.admissible_actions),
                "conditions": condition_rows,
                "normalized_scores": raw_scores,
            }
            traces.append(trace)
            trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
            trace_file.flush()
            print(
                f"[{decision_index + 1}/{len(decisions)}] {decision.action_verb}: "
                f"plain={condition_rows['plain']['correct']} "
                f"skill={condition_rows['evolved_skill']['correct']} "
                f"dp={condition_rows['evolved_skill']['expert_probability'] - condition_rows['plain']['expert_probability']:+.4f}",
                flush=True,
            )

    names = list(traces[0]["conditions"]) if traces else []
    conditions = {
        name: summarize_condition([trace["conditions"][name] for trace in traces])
        for name in names
    }
    probability_deltas = [
        trace["conditions"]["evolved_skill"]["expert_probability"]
        - trace["conditions"]["plain"]["expert_probability"]
        for trace in traces
    ]
    return {
        "conditions": conditions,
        "skill_effect": {
            "mean_expert_probability_delta": sum(probability_deltas) / len(probability_deltas),
            "positive_shift_rate": sum(delta > 1e-9 for delta in probability_deltas) / len(probability_deltas),
            "harmful_shift_rate": sum(delta < -1e-9 for delta in probability_deltas) / len(probability_deltas),
            "unchanged_shift_rate": sum(abs(delta) <= 1e-9 for delta in probability_deltas) / len(probability_deltas),
        },
        "by_action_verb": _group_summary(traces, "action_verb"),
        "by_task_type": _group_summary(traces, "task_type"),
    }


def run(config_path: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    overrides = overrides or {}
    if overrides.get("output_dir"):
        config["output_dir"] = overrides["output_dir"]
    if overrides.get("decision_file"):
        config["data"]["decision_file"] = overrides["decision_file"]
    if overrides.get("device"):
        config["model"]["device"] = overrides["device"]
    if overrides.get("decision_offset") is not None:
        config["evaluation"]["decision_offset"] = int(overrides["decision_offset"])
    if overrides.get("max_decisions") is not None:
        config["evaluation"]["max_decisions"] = int(overrides["max_decisions"])
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.resolved.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    data_cfg = config["data"]
    cache_path = output_dir / "expert_decisions.jsonl"
    if data_cfg.get("decision_file"):
        decisions = load_decisions(data_cfg["decision_file"])
        collection = {"loaded_from": data_cfg["decision_file"], "decisions": len(decisions)}
    else:
        decisions, collection = collect_expert_decisions(
            config_path=data_cfg["alfworld_config"],
            data_root=data_cfg["alfworld_data"],
            split=data_cfg["split"],
            episodes=int(data_cfg["episodes"]),
            seed=seed,
            max_steps=int(data_cfg["max_steps"]),
        )
        save_decisions(cache_path, decisions)

    if not bool(config["evaluation"].get("include_trivial", False)):
        decisions = [decision for decision in decisions if not decision.is_trivial]
    decision_offset = int(config["evaluation"].get("decision_offset", 0))
    decisions = decisions[decision_offset:]
    max_decisions = int(config["evaluation"].get("max_decisions", -1))
    if max_decisions > 0 and len(decisions) > max_decisions:
        decisions = decisions[:max_decisions]
    if not decisions:
        raise RuntimeError("No ALFWorld decisions remain after filtering")

    model_cfg = config["model"]
    model_path = model_cfg.get("path") or resolve_model_snapshot(model_cfg["cache_root"])
    scorer = SequenceActionScorer(
        model_path,
        device=model_cfg["device"],
        batch_size=int(model_cfg["batch_size"]),
        max_length=int(model_cfg["max_length"]),
        length_penalty=float(model_cfg["length_penalty"]),
    )
    bank = AlfworldSkillBank(config["skills"]["path"])
    results = {
        "seed": seed,
        "collection": collection,
        "evaluated_decisions": len(decisions),
        "model_path": model_path,
        **evaluate(decisions, scorer, bank, config, output_dir),
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--decision-file")
    parser.add_argument("--device")
    parser.add_argument("--decision-offset", type=int)
    parser.add_argument("--max-decisions", type=int)
    args = parser.parse_args()
    results = run(
        args.config,
        {
            "output_dir": args.output_dir,
            "decision_file": args.decision_file,
            "device": args.device,
            "decision_offset": args.decision_offset,
            "max_decisions": args.max_decisions,
        },
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
