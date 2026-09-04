#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_value_gradient_writeback import (
    load_value_rows,
    objective_coefficients,
    score_actions,
    selected_parameter_groups,
    state_spec,
)
from self_evolve.alfworld_data import load_decisions
from self_evolve.value_writeback import candidate_distribution_metrics


DEFAULT_VERBS = ("go", "open", "close", "take", "move")
ALLOWED_GROUPS = ("last_mlp", "last_four_blocks")


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
    parser.add_argument("--step-norm", type=float, required=True)
    parser.add_argument("--verbs", nargs="+", default=list(DEFAULT_VERBS))
    parser.add_argument("--source-count", type=int, default=8)
    parser.add_argument("--holdout-count", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=20260903)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    return parser.parse_args()


def is_heterogeneous(row: dict[str, Any]) -> bool:
    values = [float(action["discounted_success"]) for action in row["actions"]]
    return max(values) - min(values) > 1e-12


def one_state_per_episode(
    indices: list[int], rows: dict[int, dict[str, Any]], rng: random.Random
) -> list[int]:
    shuffled = list(indices)
    rng.shuffle(shuffled)
    result = []
    episodes = set()
    for index in shuffled:
        episode = rows[index]["episode_key"]
        if episode not in episodes:
            result.append(index)
            episodes.add(episode)
    return result


def round_robin_tasks(
    indices: list[int], rows: dict[int, dict[str, Any]], count: int, rng: random.Random
) -> list[int]:
    by_task: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_task[rows[index]["task_type"]].append(index)
    tasks = sorted(by_task)
    rng.shuffle(tasks)
    for values in by_task.values():
        rng.shuffle(values)
    selected = []
    depth = 0
    while len(selected) < count:
        added = False
        for task in tasks:
            if depth < len(by_task[task]):
                selected.append(by_task[task][depth])
                added = True
                if len(selected) == count:
                    return selected
        if not added:
            break
        depth += 1
    raise ValueError(f"Only {len(selected)} distinct-episode states available; need {count}")


def choose_matched_controls(
    *,
    source_indices: list[int],
    rows: dict[int, dict[str, Any]],
    verb: str,
    count: int,
    used_episodes: set[str],
    seed: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    rng = random.Random(seed)
    all_indices = [index for index, row in rows.items() if is_heterogeneous(row)]

    def pick(require_different_task: bool) -> list[tuple[int, int]]:
        result = []
        source_order = list(source_indices)
        rng.shuffle(source_order)
        for source_index in source_order:
            source = rows[source_index]
            candidates = []
            for index in all_indices:
                target = rows[index]
                if target["episode_key"] in used_episodes or target["action_verb"] == verb:
                    continue
                same_task = target["task_type"] == source["task_type"]
                if same_task == require_different_task:
                    continue
                candidates.append(index)
            rng.shuffle(candidates)
            if not candidates:
                continue
            chosen = candidates[0]
            used_episodes.add(rows[chosen]["episode_key"])
            result.append((chosen, source_index))
            if len(result) == count:
                return result
        label = "different-task" if require_different_task else "same-task"
        if len(result) < count:
            raise ValueError(f"Only {len(result)} independent {label} controls for {verb}; need {count}")
        return result

    same_task = pick(require_different_task=False)
    different_task = pick(require_different_task=True)
    return same_task, different_task


def select_verb_states(
    verb: str,
    rows: dict[int, dict[str, Any]],
    source_count: int,
    holdout_count: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    verb_indices = [
        index for index, row in rows.items()
        if row["action_verb"] == verb and is_heterogeneous(row)
    ]
    distinct = one_state_per_episode(verb_indices, rows, rng)
    chosen = round_robin_tasks(distinct, rows, source_count + holdout_count, rng)
    sources = chosen[:source_count]
    same_verb_holdout = chosen[source_count:]
    used_episodes = {rows[index]["episode_key"] for index in chosen}
    same_task, unrelated = choose_matched_controls(
        source_indices=sources,
        rows=rows,
        verb=verb,
        count=holdout_count,
        used_episodes=used_episodes,
        seed=seed + 1,
    )
    return {
        "sources": sources,
        "same_action_holdout": same_verb_holdout,
        "same_task_different_action": same_task,
        "different_task_different_action": unrelated,
    }


def gradient_norm(gradient: dict[str, torch.Tensor]) -> float:
    tensors = list(gradient.values())
    total = torch.zeros((), dtype=torch.float64, device=tensors[0].device)
    for tensor in tensors:
        total.add_(torch.sum(tensor.double().square()))
    return float(torch.sqrt(total).item())


def gradient_dot(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    first = next(iter(left.values()))
    total = torch.zeros((), dtype=torch.float64, device=first.device)
    for name in left:
        total.add_(torch.sum(left[name].double() * right[name].double()))
    return float(total.item())


def mean_gradient(gradients: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    # Accumulate directly on CUDA.  Avoid torch.stack, whose temporary tensor is
    # source_count times larger than a single gradient for the last-four-blocks probe.
    result = {name: tensor.clone() for name, tensor in gradients[0].items()}
    for gradient in gradients[1:]:
        for name in result:
            result[name].add_(gradient[name])
    inverse_count = 1.0 / len(gradients)
    for tensor in result.values():
        tensor.mul_(inverse_count)
    return result


def capture_gradient(
    *,
    model: Any,
    parameters: dict[str, torch.nn.Parameter],
    names: list[str],
    tokenizer: Any,
    spec: dict[str, Any],
    baseline_scores: np.ndarray,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], float]:
    model.zero_grad(set_to_none=True)
    coefficients = objective_coefficients("value_expectation", baseline_scores, spec)
    recomputed = score_actions(
        model, tokenizer, spec["prompt"], spec["candidates"], device,
        batch_size, max_length, coefficients,
    )
    error = float(np.max(np.abs(recomputed - baseline_scores)))
    if error > 1e-5:
        raise AssertionError(f"Gradient score reproduction error {error}")
    gradient = {
        name: parameters[name].grad.detach().float().clone()
        for name in names
    }
    return gradient, error


def writeback_and_score(
    *,
    model: Any,
    parameters: dict[str, torch.nn.Parameter],
    names: list[str],
    gradient: dict[str, torch.Tensor],
    step_norm: float,
    specs: list[dict[str, Any]],
    baselines: list[np.ndarray],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    norm = gradient_norm(gradient)
    if not math.isfinite(norm) or norm <= 1e-20:
        raise RuntimeError(f"Invalid gradient norm {norm}")
    backups = [parameters[name].detach().clone() for name in names]
    scale = -step_norm / norm
    with torch.no_grad():
        for name in names:
            parameters[name].add_(gradient[name], alpha=scale)
    updated = [
        score_actions(
            model, tokenizer, spec["prompt"], spec["candidates"], device,
            batch_size, max_length,
        )
        for spec in specs
    ]
    with torch.no_grad():
        for name, backup in zip(names, backups, strict=True):
            parameters[name].copy_(backup)
    restore_error = max(
        float(torch.max(torch.abs(parameters[name].detach() - backup)).item())
        for name, backup in zip(names, backups, strict=True)
    )
    del backups
    states = []
    for spec, baseline, changed in zip(specs, baselines, updated, strict=True):
        states.append(
            {
                "global_decision_index": spec["global_decision_index"],
                "relationship": spec["relationship"],
                "matched_source_index": spec.get("matched_source_index"),
                "episode_key": spec["episode_key"],
                "task_type": spec["task_type"],
                "action_verb": spec["action_verb"],
                **candidate_distribution_metrics(
                    baseline, changed, spec["values"], spec["expert_index"]
                ),
            }
        )
    return {
        "raw_gradient_norm": norm,
        "parameter_delta_l2_norm": step_norm,
        "scale_on_raw_gradient": scale,
        "restore_max_absolute_error": restore_error,
        "states": states,
    }


def build_specs(selection, decisions, rows, history_window):
    specs = []
    for index in selection["sources"]:
        specs.append(state_spec(index, "source", decisions, rows, history_window))
    for index in selection["same_action_holdout"]:
        specs.append(state_spec(index, "same_action_holdout", decisions, rows, history_window))
    for relationship in ("same_task_different_action", "different_task_different_action"):
        for index, matched_source in selection[relationship]:
            spec = state_spec(index, relationship, decisions, rows, history_window)
            spec["matched_source_index"] = matched_source
            specs.append(spec)
    return specs


def main() -> None:
    args = parse_args()
    if args.source_count < 2 or args.holdout_count < 1 or args.step_norm <= 0:
        raise ValueError("Invalid source, holdout, or step size")
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

    verb_results = []
    for verb_position, verb in enumerate(args.verbs, start=1):
        selection = select_verb_states(
            verb, rows, args.source_count, args.holdout_count,
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
        repeat = [
            score_actions(
                model, tokenizer, spec["prompt"], spec["candidates"], device,
                args.batch_size, args.max_length,
            )
            for spec in specs
        ]
        repeat_error = max(
            float(np.max(np.abs(left - right)))
            for left, right in zip(baselines, repeat, strict=True)
        )

        gradients = []
        reproduction_errors = []
        for source_position in range(args.source_count):
            gradient, error = capture_gradient(
                model=model,
                parameters=parameters,
                names=names,
                tokenizer=tokenizer,
                spec=specs[source_position],
                baseline_scores=baselines[source_position],
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            gradients.append(gradient)
            reproduction_errors.append(error)

        norms = [gradient_norm(gradient) for gradient in gradients]
        pairwise = []
        for left in range(args.source_count):
            for right in range(left + 1, args.source_count):
                dot = gradient_dot(gradients[left], gradients[right])
                pairwise.append(
                    {
                        "left_source_index": selection["sources"][left],
                        "right_source_index": selection["sources"][right],
                        "gradient_dot": dot,
                        "gradient_cosine": dot / max(norms[left] * norms[right], 1e-30),
                    }
                )

        single_updates = []
        for source_position, gradient in enumerate(gradients):
            update = writeback_and_score(
                model=model,
                parameters=parameters,
                names=names,
                gradient=gradient,
                step_norm=args.step_norm,
                specs=specs,
                baselines=baselines,
                tokenizer=tokenizer,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            update["gradient_source_index"] = selection["sources"][source_position]
            single_updates.append(update)
            print(
                f"[{verb_position}/{len(args.verbs)}] {args.split}/{args.parameter_group}/{verb} "
                f"single={source_position + 1}/{args.source_count}", flush=True
            )

        mean = mean_gradient(gradients)
        mean_norm = gradient_norm(mean)
        mean_update = writeback_and_score(
            model=model,
            parameters=parameters,
            names=names,
            gradient=mean,
            step_norm=args.step_norm,
            specs=specs,
            baselines=baselines,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        mean_cosines = [
            gradient_dot(mean, gradient) / max(mean_norm * norm, 1e-30)
            for gradient, norm in zip(gradients, norms, strict=True)
        ]
        for pair in pairwise:
            left_position = selection["sources"].index(pair["left_source_index"])
            right_position = selection["sources"].index(pair["right_source_index"])
            pair["left_update_on_right_value_delta"] = single_updates[left_position]["states"][right_position]["expected_value_delta"]
            pair["right_update_on_left_value_delta"] = single_updates[right_position]["states"][left_position]["expected_value_delta"]

        verb_results.append(
            {
                "verb": verb,
                "selection": selection,
                "baseline_repeat_max_absolute_error": repeat_error,
                "gradient_score_reproduction_max_error": max(reproduction_errors),
                "individual_gradient_norms": norms,
                "mean_raw_gradient_norm": mean_norm,
                "mean_norm_over_mean_individual_norm": mean_norm / max(float(np.mean(norms)), 1e-30),
                "mean_direction_cosines_to_sources": mean_cosines,
                "pairwise_source_gradients": pairwise,
                "single_source_updates": single_updates,
                "mean_gradient_update": mean_update,
            }
        )
        del gradients, mean
        model.zero_grad(set_to_none=True)
        print(
            f"[{verb_position}/{len(args.verbs)}] {args.split}/{args.parameter_group}/{verb} complete",
            flush=True,
        )

    result = {
        "split": args.split,
        "parameter_group": args.parameter_group,
        "parameter_count": sum(parameters[name].numel() for name in names),
        "dtype": "float32",
        "objective": "value_expectation",
        "step_protocol": "equal_parameter_l2_norm_for_each_single_and_normalized_raw_mean_gradient",
        "parameter_delta_l2_norm": args.step_norm,
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
