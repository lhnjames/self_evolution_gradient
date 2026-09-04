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
    gradient_dot,
    gradient_norm,
    select_verb_states,
)
from probe_sequential_value_accumulation import parameter_distance
from probe_value_gradient_writeback import load_value_rows, score_actions, selected_parameter_groups
from self_evolve.alfworld_data import load_decisions
from self_evolve.value_writeback import candidate_distribution_metrics


STRATEGIES = ("direct", "positive_filter", "project")


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
    parser.add_argument("--single-action-total-norm", type=float, required=True)
    parser.add_argument("--verbs", nargs="+", default=list(DEFAULT_VERBS))
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--source-count", type=int, default=12)
    parser.add_argument("--holdout-count", type=int, default=6)
    parser.add_argument("--sample-seed", type=int, default=20260903)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[1, 4, 8, 12])
    parser.add_argument(
        "--delta-output-dir",
        help="Optionally save each strategy's final selected-parameter delta for episode evaluation.",
    )
    return parser.parse_args()


def normalized_add(reference: dict[str, torch.Tensor] | None,
                   gradient: dict[str, torch.Tensor], norm: float) -> dict[str, torch.Tensor]:
    if reference is None:
        return {name: tensor.clone().mul_(1.0 / norm) for name, tensor in gradient.items()}
    for name in reference:
        reference[name].add_(gradient[name], alpha=1.0 / norm)
    return reference


def score_spec(model: Any, tokenizer: Any, spec: dict[str, Any], device: torch.device,
               batch_size: int, max_length: int) -> np.ndarray:
    return score_actions(model, tokenizer, spec["prompt"], spec["candidates"], device,
                         batch_size, max_length)


def score_panel(model: Any, tokenizer: Any, unique_specs: dict[int, dict[str, Any]],
                device: torch.device, batch_size: int, max_length: int) -> dict[int, np.ndarray]:
    return {
        index: score_spec(model, tokenizer, spec, device, batch_size, max_length)
        for index, spec in unique_specs.items()
    }


def panel_metrics(panels: dict[str, list[dict[str, Any]]], baselines: dict[int, np.ndarray],
                  updated: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for anchor_verb, specs in panels.items():
        for spec in specs:
            index = spec["global_decision_index"]
            rows.append({
                "anchor_verb": anchor_verb,
                "global_decision_index": index,
                "relationship": spec["relationship"],
                "episode_key": spec["episode_key"],
                "task_type": spec["task_type"],
                "action_verb": spec["action_verb"],
                **candidate_distribution_metrics(
                    baselines[index], updated[index], spec["values"], spec["expert_index"]
                ),
            })
    return rows


def main() -> None:
    args = parse_args()
    if args.source_count < 2 or args.holdout_count < 1 or args.single_action_total_norm <= 0:
        raise ValueError("Invalid source, holdout, or norm")
    checkpoints = sorted(set(args.checkpoints))
    if not checkpoints or checkpoints[-1] != args.source_count or checkpoints[0] < 1:
        raise ValueError("Checkpoints must include source-count and be within its range")
    if checkpoints[-1] > args.source_count:
        raise ValueError("Checkpoint exceeds source-count")
    if len(set(args.strategies)) != len(args.strategies):
        raise ValueError("Duplicate strategies")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("CUDA is required")

    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    value_rows = load_value_rows(args.value_trace, args.split)
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

    selections: dict[str, dict[str, Any]] = {}
    panels: dict[str, list[dict[str, Any]]] = {}
    source_specs: dict[tuple[str, int], dict[str, Any]] = {}
    unique_specs: dict[int, dict[str, Any]] = {}
    for verb_position, verb in enumerate(args.verbs, start=1):
        selection = select_verb_states(
            verb, value_rows, args.source_count, args.holdout_count,
            args.sample_seed + verb_position * 1_000_003,
        )
        specs = build_specs(selection, decisions, value_rows, args.history_window)
        selections[verb] = selection
        panels[verb] = specs
        for source_position in range(args.source_count):
            source_specs[(verb, source_position)] = specs[source_position]
        for spec in specs:
            unique_specs.setdefault(spec["global_decision_index"], spec)

    baselines = score_panel(
        model, tokenizer, unique_specs, device, args.batch_size, args.max_length
    )
    repeated = score_panel(
        model, tokenizer, unique_specs, device, args.batch_size, args.max_length
    )
    repeat_error = max(
        float(np.max(np.abs(baselines[index] - repeated[index]))) for index in baselines
    )
    originals = {name: parameters[name].detach().clone() for name in names}

    # If the five action directions were orthogonal, this makes their joint RMS
    # dose equal to the previously selected single-action 10x dose.
    increment_norm = args.single_action_total_norm / (args.source_count * math.sqrt(len(args.verbs)))
    results = []
    for strategy_position, strategy in enumerate(args.strategies, start=1):
        with torch.no_grad():
            for name in names:
                parameters[name].copy_(originals[name])
        model.zero_grad(set_to_none=True)
        reference: dict[str, torch.Tensor] | None = None
        reference_norm = 0.0
        steps = []
        checkpoint_rows = []
        accepted_count = 0
        projected_count = 0

        for round_index in range(args.source_count):
            offset = round_index % len(args.verbs)
            verb_order = list(args.verbs[offset:]) + list(args.verbs[:offset])
            for verb in verb_order:
                spec = source_specs[(verb, round_index)]
                current_scores = score_spec(
                    model, tokenizer, spec, device, args.batch_size, args.max_length
                )
                gradient, reproduction_error = capture_gradient(
                    model=model,
                    parameters=parameters,
                    names=names,
                    tokenizer=tokenizer,
                    spec=spec,
                    baseline_scores=current_scores,
                    device=device,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )
                raw_norm = gradient_norm(gradient)
                if not math.isfinite(raw_norm) or raw_norm <= 1e-20:
                    raise RuntimeError(f"Invalid gradient norm {raw_norm}")
                dot = gradient_dot(gradient, reference) if reference is not None else None
                raw_cosine = dot / max(raw_norm * reference_norm, 1e-30) if dot is not None else None
                accepted = strategy != "positive_filter" or raw_cosine is None or raw_cosine >= 0
                projection_coefficient = 0.0
                if strategy == "project" and dot is not None and dot < 0:
                    projection_coefficient = dot / max(reference_norm * reference_norm, 1e-30)
                    with torch.no_grad():
                        for name in names:
                            gradient[name].add_(reference[name], alpha=-projection_coefficient)
                    projected_count += 1
                applied_norm = gradient_norm(gradient) if accepted else 0.0
                if accepted:
                    if not math.isfinite(applied_norm) or applied_norm <= 1e-20:
                        raise RuntimeError(f"Invalid applied gradient norm {applied_norm}")
                    with torch.no_grad():
                        for name in names:
                            parameters[name].add_(gradient[name], alpha=-increment_norm / applied_norm)
                    reference = normalized_add(reference, gradient, applied_norm)
                    reference_norm = gradient_norm(reference)
                    accepted_count += 1
                drift = parameter_distance(parameters, names, list(originals.values()))
                steps.append({
                    "step_index": len(steps) + 1,
                    "round_index": round_index + 1,
                    "verb": verb,
                    "gradient_source_index": spec["global_decision_index"],
                    "raw_gradient_norm": raw_norm,
                    "raw_cosine_to_running_reference": raw_cosine,
                    "accepted": accepted,
                    "projection_coefficient": projection_coefficient,
                    "applied_gradient_norm": applied_norm,
                    "increment_parameter_l2_norm": increment_norm if accepted else 0.0,
                    "gradient_score_reproduction_max_error": reproduction_error,
                    "parameter_l2_distance_from_base": drift,
                })
                del gradient
                model.zero_grad(set_to_none=True)

            if round_index + 1 in checkpoints:
                updated = score_panel(
                    model, tokenizer, unique_specs, device, args.batch_size, args.max_length
                )
                states = panel_metrics(panels, baselines, updated)
                checkpoint_rows.append({
                    "round_index": round_index + 1,
                    "experience_count": len(args.verbs) * (round_index + 1),
                    "accepted_count": accepted_count,
                    "projected_count": projected_count,
                    "parameter_l2_distance_from_base": steps[-1]["parameter_l2_distance_from_base"],
                    "states": states,
                })
                same = [x["expected_value_delta"] for x in states
                        if x["relationship"] == "same_action_holdout"]
                control = [x["expected_value_delta"] for x in states
                           if x["relationship"] == "same_task_different_action"]
                print(
                    f"[{strategy_position}/{len(args.strategies)}] {args.split}/"
                    f"{args.parameter_group}/{strategy} round={round_index + 1} "
                    f"accepted={accepted_count}/{len(steps)} drift={steps[-1]['parameter_l2_distance_from_base']:.6g} "
                    f"same={np.mean(same):+.6g} control={np.mean(control):+.6g}",
                    flush=True,
                )

        results.append({
            "strategy": strategy,
            "accepted_count": accepted_count,
            "projected_count": projected_count,
            "steps": steps,
            "checkpoints": checkpoint_rows,
        })
        if args.delta_output_dir:
            delta_dir = Path(args.delta_output_dir)
            delta_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format": "selected_parameter_delta_v1",
                    "base_model": args.base_model,
                    "split": args.split,
                    "parameter_group": args.parameter_group,
                    "strategy": strategy,
                    "state_dict": {
                        name: (parameters[name].detach() - originals[name]).to(
                            device="cpu", dtype=torch.float16
                        )
                        for name in names
                    },
                },
                delta_dir / f"{args.split}_{args.parameter_group}_{strategy}.pt",
            )
        del reference
        model.zero_grad(set_to_none=True)

    with torch.no_grad():
        for name in names:
            parameters[name].copy_(originals[name])
    restore_error = max(
        float(torch.max(torch.abs(parameters[name].detach() - originals[name])).item())
        for name in names
    )

    result = {
        "split": args.split,
        "parameter_group": args.parameter_group,
        "parameter_count": sum(parameters[name].numel() for name in names),
        "dtype": "float32",
        "objective": "online_recomputed_value_expectation",
        "protocol": "five_action_interleaved_conflict_aware_accumulation",
        "reference_definition": "running_sum_of_unit_applied_loss_gradients",
        "single_action_total_norm": args.single_action_total_norm,
        "joint_rms_nominal_norm": args.single_action_total_norm,
        "increment_parameter_l2_norm": increment_norm,
        "source_count_per_verb": args.source_count,
        "holdout_count_per_category_per_verb": args.holdout_count,
        "verbs": list(args.verbs),
        "strategies": list(args.strategies),
        "checkpoints": checkpoints,
        "sample_seed": args.sample_seed,
        "unique_evaluation_state_count": len(unique_specs),
        "selection_by_verb": selections,
        "baseline_repeat_max_absolute_error": repeat_error,
        "restore_max_absolute_error": restore_error,
        "strategy_results": results,
    }
    destination = Path(args.output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(destination), "strategies": len(results)}), flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
