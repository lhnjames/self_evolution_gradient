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

from probe_multisource_value_gradients import (
    ALLOWED_GROUPS,
    DEFAULT_VERBS,
    build_specs,
    capture_gradient,
    gradient_norm,
    select_verb_states,
)
from probe_value_gradient_writeback import (
    load_value_rows,
    score_actions,
    selected_parameter_groups,
)
from self_evolve.alfworld_data import load_decisions
from self_evolve.value_writeback import candidate_distribution_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-file", required=True)
    parser.add_argument("--value-trace", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--split", choices=("valid_seen", "valid_unseen"), required=True)
    parser.add_argument("--parameter-group", choices=ALLOWED_GROUPS, required=True)
    parser.add_argument("--total-step-norm", type=float, required=True)
    parser.add_argument("--verbs", nargs="+", default=list(DEFAULT_VERBS))
    parser.add_argument("--source-count", type=int, default=12)
    parser.add_argument("--holdout-count", type=int, default=6)
    parser.add_argument("--sample-seed", type=int, default=20260903)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    return parser.parse_args()


def parameter_distance(
    parameters: dict[str, torch.nn.Parameter],
    names: list[str],
    originals: list[torch.Tensor],
) -> float:
    total = torch.zeros((), dtype=torch.float64, device=originals[0].device)
    for name, original in zip(names, originals, strict=True):
        total.add_(torch.sum((parameters[name].detach().double() - original.double()).square()))
    return float(torch.sqrt(total).item())


def score_metrics(
    specs: list[dict[str, Any]],
    baselines: list[np.ndarray],
    updated: list[np.ndarray],
    learned_indices: set[int],
) -> list[dict[str, Any]]:
    result = []
    for spec, baseline, changed in zip(specs, baselines, updated, strict=True):
        relationship = spec["relationship"]
        if relationship == "source":
            accumulation_role = (
                "learned_source"
                if spec["global_decision_index"] in learned_indices
                else "future_source"
            )
        else:
            accumulation_role = relationship
        result.append(
            {
                "global_decision_index": spec["global_decision_index"],
                "relationship": relationship,
                "accumulation_role": accumulation_role,
                "episode_key": spec["episode_key"],
                "task_type": spec["task_type"],
                "action_verb": spec["action_verb"],
                **candidate_distribution_metrics(
                    baseline, changed, spec["values"], spec["expert_index"]
                ),
            }
        )
    return result


def mean_delta(states: list[dict[str, Any]], role: str) -> float:
    selected = [row["expected_value_delta"] for row in states if row["accumulation_role"] == role]
    return float(np.mean(selected)) if selected else math.nan


def main() -> None:
    args = parse_args()
    if args.source_count < 2 or args.holdout_count < 1 or args.total_step_norm <= 0:
        raise ValueError("Invalid source, holdout, or total step")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("CUDA is required")

    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    rows = load_value_rows(args.value_trace, args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, local_files_only=True
    ).to(device)
    parameters, groups = selected_parameter_groups(model)
    names = groups[args.parameter_group]
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)

    increment_norm = args.total_step_norm / args.source_count
    verb_results = []
    for verb_position, verb in enumerate(args.verbs, start=1):
        selection = select_verb_states(
            verb,
            rows,
            args.source_count,
            args.holdout_count,
            args.sample_seed + verb_position * 1_000_003,
        )
        specs = build_specs(selection, decisions, rows, args.history_window)
        baselines = [
            score_actions(
                model, tokenizer, spec["prompt"], spec["candidates"], device,
                args.batch_size, args.max_length,
            )
            for spec in specs
        ]
        repeated = [
            score_actions(
                model, tokenizer, spec["prompt"], spec["candidates"], device,
                args.batch_size, args.max_length,
            )
            for spec in specs
        ]
        repeat_error = max(
            float(np.max(np.abs(left - right)))
            for left, right in zip(baselines, repeated, strict=True)
        )
        originals = [parameters[name].detach().clone() for name in names]
        current_scores = baselines
        learned: set[int] = set()
        steps = []

        for step_index, source_index in enumerate(selection["sources"], start=1):
            source_position = selection["sources"].index(source_index)
            gradient, reproduction_error = capture_gradient(
                model=model,
                parameters=parameters,
                names=names,
                tokenizer=tokenizer,
                spec=specs[source_position],
                baseline_scores=current_scores[source_position],
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            norm = gradient_norm(gradient)
            if not math.isfinite(norm) or norm <= 1e-20:
                raise RuntimeError(f"Invalid gradient norm {norm}")
            scale = -increment_norm / norm
            with torch.no_grad():
                for name in names:
                    parameters[name].add_(gradient[name], alpha=scale)
            del gradient
            model.zero_grad(set_to_none=True)

            current_scores = [
                score_actions(
                    model, tokenizer, spec["prompt"], spec["candidates"], device,
                    args.batch_size, args.max_length,
                )
                for spec in specs
            ]
            learned.add(source_index)
            states = score_metrics(specs, baselines, current_scores, learned)
            drift = parameter_distance(parameters, names, originals)
            steps.append(
                {
                    "step_index": step_index,
                    "gradient_source_index": source_index,
                    "raw_gradient_norm": norm,
                    "increment_parameter_l2_norm": increment_norm,
                    "scale_on_raw_gradient": scale,
                    "gradient_score_reproduction_max_error": reproduction_error,
                    "parameter_l2_distance_from_base": drift,
                    "states": states,
                }
            )
            print(
                f"[{verb_position}/{len(args.verbs)}] {args.split}/{args.parameter_group}/{verb} "
                f"step={step_index}/{args.source_count} drift={drift:.6g} "
                f"learned_dV={mean_delta(states, 'learned_source'):+.6g} "
                f"future_dV={mean_delta(states, 'future_source'):+.6g} "
                f"heldout_dV={mean_delta(states, 'same_action_holdout'):+.6g}",
                flush=True,
            )

        with torch.no_grad():
            for name, original in zip(names, originals, strict=True):
                parameters[name].copy_(original)
        restore_error = max(
            float(torch.max(torch.abs(parameters[name].detach() - original)).item())
            for name, original in zip(names, originals, strict=True)
        )
        del originals
        model.zero_grad(set_to_none=True)
        verb_results.append(
            {
                "verb": verb,
                "selection": selection,
                "source_order": selection["sources"],
                "baseline_repeat_max_absolute_error": repeat_error,
                "restore_max_absolute_error": restore_error,
                "steps": steps,
            }
        )
        print(
            f"[{verb_position}/{len(args.verbs)}] {args.split}/{args.parameter_group}/{verb} complete",
            flush=True,
        )

    result = {
        "split": args.split,
        "parameter_group": args.parameter_group,
        "parameter_count": sum(parameters[name].numel() for name in names),
        "dtype": "float32",
        "objective": "online_recomputed_value_expectation",
        "protocol": "sequential_online_gradient_with_equal_increment_l2",
        "total_nominal_parameter_l2_budget": args.total_step_norm,
        "increment_parameter_l2_norm": increment_norm,
        "source_count_per_verb": args.source_count,
        "holdout_count_per_category_per_verb": args.holdout_count,
        "verbs": list(args.verbs),
        "sample_seed": args.sample_seed,
        "verb_results": verb_results,
    }
    destination = Path(args.output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(destination), "verbs": len(verb_results)}), flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()

