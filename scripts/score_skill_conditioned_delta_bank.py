#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_value_gradient_writeback import score_actions
from self_evolve.alfworld_data import load_decisions
from self_evolve.alfworld_skills import build_action_prompt
from self_evolve.skill_gradient_purification import route_from_action


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-file", required=True)
    parser.add_argument("--base-score-trace", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--delta-manifest", required=True)
    parser.add_argument("--aggregate", choices=("mean12", "purified12"), required=True)
    parser.add_argument("--route-mode", choices=("oracle_skill", "base_predicted_skill"), required=True)
    parser.add_argument("--condition-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("CUDA is required")
    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    base_rows = read_jsonl(args.base_score_trace)
    if len(base_rows) != len(decisions):
        raise ValueError("base score trace and decision file have different lengths")
    manifest_path = Path(args.delta_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "skill_conditioned_parameter_delta_bank_v1":
        raise ValueError("unsupported delta bank")
    route_paths = manifest["routes"][args.aggregate]
    verbs = sorted(route_paths)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, local_files_only=True
    ).to(device)
    model.eval()
    parameters = dict(model.named_parameters())

    routes = []
    for index, (decision, base) in enumerate(zip(decisions, base_rows, strict=True)):
        if int(base["global_decision_index"]) != index:
            raise ValueError(f"base trace index mismatch at {index}")
        if list(decision.admissible_actions) != list(base["admissible_actions"]):
            raise ValueError(f"candidate mismatch at {index}")
        if args.route_mode == "oracle_skill":
            route = decision.action_verb
        else:
            scores = base["normalized_scores"]
            if isinstance(scores, dict):
                scores = scores["plain"]
            route = route_from_action(decision.admissible_actions[int(np.argmax(scores))])
        routes.append(route if route in route_paths else "base")

    scores_by_index: dict[int, np.ndarray] = {}
    for index, (route, base) in enumerate(zip(routes, base_rows, strict=True)):
        if route == "base":
            scores = base["normalized_scores"]
            if isinstance(scores, dict):
                scores = scores["plain"]
            scores_by_index[index] = np.asarray(scores, dtype=np.float64)

    manifest_root = manifest_path.parent
    for verb in verbs:
        raw_path = Path(route_paths[verb])
        delta_path = raw_path if raw_path.is_absolute() else manifest_root / raw_path
        payload = torch.load(delta_path, map_location="cpu", weights_only=True)
        state_dict = payload["state_dict"]
        unknown = set(state_dict) - set(parameters)
        if unknown:
            raise ValueError(f"delta contains unknown parameters: {sorted(unknown)[:3]}")
        with torch.no_grad():
            for name, delta in state_dict.items():
                parameters[name].add_(delta.to(device=device, dtype=parameters[name].dtype))
        selected = [index for index, route in enumerate(routes) if route == verb]
        for position, index in enumerate(selected, start=1):
            decision = decisions[index]
            scores_by_index[index] = score_actions(
                model,
                tokenizer,
                build_action_prompt(decision, history_window=args.history_window),
                list(decision.admissible_actions),
                device,
                args.batch_size,
                args.max_length,
            )
            if position % 25 == 0 or position == len(selected):
                print(f"route={verb} scored={position}/{len(selected)}", flush=True)
        with torch.no_grad():
            for name, delta in state_dict.items():
                parameters[name].sub_(delta.to(device=device, dtype=parameters[name].dtype))
        del payload, state_dict

    if len(scores_by_index) != len(decisions):
        raise RuntimeError("not every decision received scores")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with (output_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for index, decision in enumerate(decisions):
            scores = scores_by_index[index]
            weights = np.exp(scores - scores.max())
            probabilities = weights / weights.sum()
            gold = decision.admissible_actions.index(decision.expert_action)
            predicted = int(np.argmax(scores))
            row = {
                "global_decision_index": index,
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
                "gradient_skill_route": routes[index],
                "route_mode": args.route_mode,
                "normalized_scores": [float(value) for value in scores],
                "predicted_index": predicted,
                "correct": int(predicted == gold),
                "expert_probability": float(probabilities[gold]),
                "nll": -math.log(max(float(probabilities[gold]), 1e-12)),
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "decision_file": args.decision_file,
        "base_model": args.base_model,
        "delta_manifest": args.delta_manifest,
        "aggregate": args.aggregate,
        "route_mode": args.route_mode,
        "condition_name": args.condition_name,
        "evaluated_decisions": len(rows),
        "edited_decisions": sum(route != "base" for route in routes),
        "route_counts": {route: routes.count(route) for route in [*verbs, "base"]},
        "top1_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "mean_expert_probability": sum(row["expert_probability"] for row in rows) / len(rows),
        "mean_nll": sum(row["nll"] for row in rows) / len(rows),
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
