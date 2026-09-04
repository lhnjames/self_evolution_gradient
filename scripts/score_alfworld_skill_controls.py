#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from self_evolve.alfworld_data import load_decisions
from self_evolve.alfworld_skill_controls import render_control_contexts
from self_evolve.alfworld_skills import AlfworldSkillBank, build_action_prompt
from self_evolve.sequence_scorer import SequenceActionScorer


def summarize(rows: list[dict], condition: str) -> dict[str, float]:
    metrics = [row["conditions"][condition] for row in rows]
    return {
        "count": len(metrics),
        "top1_accuracy": sum(item["correct"] for item in metrics) / len(metrics),
        "mean_expert_probability": sum(item["expert_probability"] for item in metrics) / len(metrics),
        "mean_nll": sum(item["nll"] for item in metrics) / len(metrics),
        "mean_entropy": sum(item["entropy"] for item in metrics) / len(metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--skills-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--decision-offset", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    parser.add_argument("--general-top-k", type=int, default=3)
    parser.add_argument("--task-top-k", type=int, default=3)
    parser.add_argument("--mistakes-top-k", type=int, default=2)
    args = parser.parse_args()

    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    decisions = decisions[args.decision_offset :]
    if args.max_decisions > 0:
        decisions = decisions[: args.max_decisions]
    scorer = SequenceActionScorer(
        args.model_path,
        args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        length_penalty=1.0,
    )
    bank = AlfworldSkillBank(args.skills_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"
    rows = []
    with trace_path.open("w", encoding="utf-8") as handle:
        for local_index, decision in enumerate(decisions):
            contexts = render_control_contexts(
                bank,
                decision,
                scorer.tokenizer,
                args.general_top_k,
                args.task_top_k,
                args.mistakes_top_k,
            )
            gold = decision.admissible_actions.index(decision.expert_action)
            normalized_scores = {}
            conditions = {}
            context_lengths = {}
            prompt_lengths = {}
            prompt_overflows = {}
            longest_candidate = max(
                len(scorer.tokenizer.encode(" " + candidate, add_special_tokens=False))
                for candidate in decision.admissible_actions
            )
            for name, context in contexts.items():
                context_lengths[name] = len(
                    scorer.tokenizer.encode(context, add_special_tokens=False)
                )
                prompt = build_action_prompt(decision, context, args.history_window)
                prompt_lengths[name] = len(
                    scorer.tokenizer.encode(prompt, add_special_tokens=True)
                )
                prompt_overflows[name] = max(
                    0, prompt_lengths[name] + longest_candidate - args.max_length
                )
                scored = scorer.score(prompt, decision.admissible_actions)
                probabilities = scored.probabilities()
                expert_probability = float(probabilities[gold])
                normalized_scores[name] = [float(value) for value in scored.normalized_scores]
                conditions[name] = {
                    "predicted_index": int(probabilities.argmax()),
                    "correct": int(int(probabilities.argmax()) == gold),
                    "expert_probability": expert_probability,
                    "nll": -math.log(max(expert_probability, 1e-12)),
                    "entropy": float(
                        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
                    ),
                }
            reference = bank.render(
                decision, args.general_top_k, args.task_top_k, args.mistakes_top_k
            )
            reference_length = len(
                scorer.tokenizer.encode(reference, add_special_tokens=False)
            )
            reference_prompt = build_action_prompt(
                decision, reference, args.history_window
            )
            reference_prompt_length = len(
                scorer.tokenizer.encode(reference_prompt, add_special_tokens=True)
            )
            if context_lengths["length_matched_placebo"] != reference_length:
                raise AssertionError("Placebo and reference skill token lengths differ")
            row = {
                "global_decision_index": args.decision_offset + local_index,
                "episode_id": decision.episode_id,
                "episode_key": decision.gamefile,
                "gamefile": decision.gamefile,
                "task_type": decision.task_type,
                "step_index": decision.step_index,
                "action_verb": decision.action_verb,
                "goal": decision.goal,
                "expert_action": decision.expert_action,
                "gold_index": gold,
                "admissible_actions": list(decision.admissible_actions),
                "reference_context_token_length": reference_length,
                "control_context_token_lengths": context_lengths,
                "reference_prompt_token_length": reference_prompt_length,
                "reference_prompt_overflow": max(
                    0, reference_prompt_length + longest_candidate - args.max_length
                ),
                "control_prompt_token_lengths": prompt_lengths,
                "control_prompt_overflows": prompt_overflows,
                "conditions": conditions,
                "normalized_scores": normalized_scores,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{local_index + 1}/{len(decisions)}] global={row['global_decision_index']} "
                + " ".join(f"{name}={value['correct']}" for name, value in conditions.items()),
                flush=True,
            )
    result = {
        "decision_file": args.decision_file,
        "decision_offset": args.decision_offset,
        "evaluated_decisions": len(decisions),
        "conditions": {
            name: summarize(rows, name) for name in rows[0]["conditions"]
        },
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
