#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from self_evolve.alfworld_data import load_decisions
from self_evolve.alfworld_skills import build_action_prompt
from self_evolve.sequence_scorer import SequenceActionScorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--parameter-delta")
    parser.add_argument("--force-float32", action="store_true")
    parser.add_argument(
        "--tokenizer-path",
        help="Optional tokenizer override; use the base tokenizer for a weight-only comparison.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--condition-name", default="checkpoint_plain")
    parser.add_argument("--decision-offset", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    args = parser.parse_args()

    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    decisions = decisions[args.decision_offset :]
    if args.max_decisions > 0:
        decisions = decisions[: args.max_decisions]
    if not decisions:
        raise ValueError("No decisions selected")

    scorer = SequenceActionScorer(
        args.model_path,
        args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        length_penalty=1.0,
        tokenizer_name_or_path=args.tokenizer_path,
        parameter_delta_path=args.parameter_delta,
        force_float32=args.force_float32,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with (output_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for local_index, decision in enumerate(decisions):
            prompt = build_action_prompt(decision, history_window=args.history_window)
            scored = scorer.score(prompt, decision.admissible_actions)
            probabilities = scored.probabilities()
            gold = decision.admissible_actions.index(decision.expert_action)
            predicted = int(probabilities.argmax())
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
                "prompt_mode": "existing_plain_prompt",
                "condition_name": args.condition_name,
                "normalized_scores": [float(value) for value in scored.normalized_scores],
                "predicted_index": predicted,
                "correct": int(predicted == gold),
                "expert_probability": float(probabilities[gold]),
                "nll": -math.log(max(float(probabilities[gold]), 1e-12)),
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{local_index + 1}/{len(decisions)}] global={row['global_decision_index']} "
                f"correct={row['correct']} p_gold={row['expert_probability']:.4f}",
                flush=True,
            )

    result = {
        "decision_file": args.decision_file,
        "model_path": args.model_path,
        "parameter_delta": args.parameter_delta,
        "tokenizer_path": args.tokenizer_path or args.model_path,
        "condition_name": args.condition_name,
        "prompt_mode": "existing_plain_prompt",
        "decision_offset": args.decision_offset,
        "evaluated_decisions": len(rows),
        "top1_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "mean_expert_probability": sum(row["expert_probability"] for row in rows) / len(rows),
        "mean_nll": sum(row["nll"] for row in rows) / len(rows),
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
