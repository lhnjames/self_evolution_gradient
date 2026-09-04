#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_multisource_value_gradients import (
    ALLOWED_GROUPS,
    build_specs,
    capture_gradient,
    choose_matched_controls,
    gradient_dot,
    gradient_norm,
    is_heterogeneous,
    one_state_per_episode,
    round_robin_tasks,
    writeback_and_score,
)
from probe_value_gradient_writeback import load_value_rows, score_actions, selected_parameter_groups
from self_evolve.alfworld_data import load_decisions
from self_evolve.skill_gradient_purification import (
    cosine_consensus_weights,
    cosine_matrix_from_dots,
    normalized_weighted_coefficients,
)


DEFAULT_VERBS = ("go", "open", "close")
AGGREGATES = ("mean12", "purified12")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-file", required=True)
    parser.add_argument("--value-trace", required=True)
    parser.add_argument("--base-score-trace", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--delta-output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--split", choices=("valid_seen",), default="valid_seen")
    parser.add_argument("--parameter-group", choices=ALLOWED_GROUPS, default="last_four_blocks")
    parser.add_argument("--step-norm", type=float, default=0.18)
    parser.add_argument("--verbs", nargs="+", default=list(DEFAULT_VERBS))
    parser.add_argument("--source-count", type=int, default=12)
    parser.add_argument("--holdout-count", type=int, default=6)
    parser.add_argument("--sample-seed", type=int, default=20260904)
    parser.add_argument("--keep-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def base_failure_indices(
    score_trace: str, rows: dict[int, dict[str, Any]], split: str
) -> set[int]:
    failures = set()
    for scored in read_jsonl(score_trace):
        index = int(scored["global_decision_index"])
        if index not in rows:
            continue
        if scored.get("split", split) != split and "split" in scored:
            continue
        values = np.asarray(
            [float(action["discounted_success"]) for action in rows[index]["actions"]],
            dtype=np.float64,
        )
        scores = scored["normalized_scores"]
        if isinstance(scores, dict):
            scores = scores["plain"]
        top = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        if not np.isclose(values[top], values.max(), rtol=0.0, atol=1e-12):
            failures.add(index)
    return failures


def select_failure_skill_states(
    *,
    verb: str,
    rows: dict[int, dict[str, Any]],
    failure_indices: set[int],
    source_count: int,
    holdout_count: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    eligible = [
        index
        for index, row in rows.items()
        if index in failure_indices and row["action_verb"] == verb and is_heterogeneous(row)
    ]
    distinct = one_state_per_episode(eligible, rows, rng)
    chosen = round_robin_tasks(distinct, rows, source_count + holdout_count, rng)
    sources = chosen[:source_count]
    holdouts = chosen[source_count:]
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
        "same_action_holdout": holdouts,
        "same_task_different_action": same_task,
        "different_task_different_action": unrelated,
        "eligible_failure_states": len(eligible),
        "eligible_failure_episodes": len({rows[index]["episode_key"] for index in eligible}),
    }


def weighted_unit_gradient(
    gradients: list[dict[str, torch.Tensor]],
    norms: list[float],
    weights: np.ndarray,
) -> dict[str, torch.Tensor]:
    coefficients = normalized_weighted_coefficients(weights, norms)
    result = {
        name: tensor.clone().mul_(float(coefficients[0]))
        for name, tensor in gradients[0].items()
    }
    for coefficient, gradient in zip(coefficients[1:], gradients[1:], strict=True):
        for name in result:
            result[name].add_(gradient[name], alpha=float(coefficient))
    return result


def save_delta(
    *,
    output_dir: str,
    verb: str,
    aggregate: str,
    gradient: dict[str, torch.Tensor],
    step_norm: float,
    raw_norm: float,
    base_model: str,
    parameter_group: str,
    sample_seed: int,
    source_indices: list[int],
) -> str:
    destination = Path(output_dir) / aggregate / f"{verb}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    scale = -step_norm / raw_norm
    torch.save(
        {
            "format": "selected_parameter_delta_v1",
            "base_model": base_model,
            "split": "valid_seen",
            "parameter_group": parameter_group,
            "strategy": f"skill_conditioned_{aggregate}",
            "skill_route": verb,
            "sample_seed": sample_seed,
            "source_indices": source_indices,
            "parameter_delta_l2_norm": step_norm,
            "state_dict": {
                name: tensor.detach().mul(scale).to(device="cpu", dtype=torch.float16)
                for name, tensor in gradient.items()
            },
        },
        destination,
    )
    return str(destination)


def enrich_update(update: dict[str, Any], specs: list[dict[str, Any]]) -> None:
    if len(update["states"]) != len(specs):
        raise ValueError("update states and specs do not align")
    for state, spec in zip(update["states"], specs, strict=True):
        optimal = float(np.max(spec["values"]))
        state["optimal_value"] = optimal
        state["baseline_top_is_value_optimal"] = int(
            np.isclose(state["baseline_top_value"], optimal, rtol=0.0, atol=1e-12)
        )
        state["updated_top_is_value_optimal"] = int(
            np.isclose(state["updated_top_value"], optimal, rtol=0.0, atol=1e-12)
        )


def main() -> None:
    args = parse_args()
    if args.source_count < 2 or args.holdout_count < 1 or args.step_norm <= 0:
        raise ValueError("invalid source count, holdout count, or step norm")
    if len(set(args.verbs)) != len(args.verbs):
        raise ValueError("duplicate skill routes")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("CUDA is required")

    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    rows = load_value_rows(args.value_trace, args.split)
    failures = base_failure_indices(args.base_score_trace, rows, args.split)
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
    delta_manifest: dict[str, dict[str, str]] = {name: {} for name in AGGREGATES}
    for verb_position, verb in enumerate(args.verbs, start=1):
        selection = select_failure_skill_states(
            verb=verb,
            rows=rows,
            failure_indices=failures,
            source_count=args.source_count,
            holdout_count=args.holdout_count,
            seed=args.sample_seed + verb_position * 1_000_003,
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
        dots = np.empty((args.source_count, args.source_count), dtype=np.float64)
        for left in range(args.source_count):
            for right in range(left, args.source_count):
                dot = gradient_dot(gradients[left], gradients[right])
                dots[left, right] = dots[right, left] = dot
        cosines = cosine_matrix_from_dots(dots, norms)
        purified_weights = cosine_consensus_weights(
            cosines, keep_fraction=args.keep_fraction
        )
        uniform_weights = np.full(args.source_count, 1.0 / args.source_count)

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
            enrich_update(update, specs)
            update["gradient_source_index"] = selection["sources"][source_position]
            single_updates.append(update)
            print(
                f"[{verb_position}/{len(args.verbs)}] {verb} "
                f"single={source_position + 1}/{args.source_count}",
                flush=True,
            )

        aggregate_updates = {}
        for aggregate, weights in (
            ("mean12", uniform_weights),
            ("purified12", purified_weights),
        ):
            gradient = weighted_unit_gradient(gradients, norms, weights)
            raw_norm = gradient_norm(gradient)
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
            enrich_update(update, specs)
            delta_manifest[aggregate][verb] = save_delta(
                output_dir=args.delta_output_dir,
                verb=verb,
                aggregate=aggregate,
                gradient=gradient,
                step_norm=args.step_norm,
                raw_norm=raw_norm,
                base_model=args.base_model,
                parameter_group=args.parameter_group,
                sample_seed=args.sample_seed,
                source_indices=selection["sources"],
            )
            aggregate_updates[aggregate] = update
            holdout = [
                row["expected_value_delta"]
                for row in update["states"]
                if row["relationship"] == "same_action_holdout"
            ]
            print(
                f"[{verb_position}/{len(args.verbs)}] {verb} {aggregate} "
                f"holdout={np.mean(holdout):+.6f}",
                flush=True,
            )
            del gradient

        verb_results.append(
            {
                "verb": verb,
                "selection": selection,
                "baseline_repeat_max_absolute_error": repeat_error,
                "gradient_score_reproduction_max_error": max(reproduction_errors),
                "individual_gradient_norms": norms,
                "pairwise_cosine_matrix": cosines.tolist(),
                "purified_weights": purified_weights.tolist(),
                "purified_retained_source_indices": [
                    selection["sources"][index]
                    for index, weight in enumerate(purified_weights)
                    if weight > 0.0
                ],
                "single_source_updates": single_updates,
                "aggregate_updates": aggregate_updates,
            }
        )
        del gradients
        model.zero_grad(set_to_none=True)

    manifest = {
        "format": "skill_conditioned_parameter_delta_bank_v1",
        "base_model": args.base_model,
        "split": args.split,
        "parameter_group": args.parameter_group,
        "verbs": list(args.verbs),
        "aggregates": list(AGGREGATES),
        "sample_seed": args.sample_seed,
        "source_count_per_skill": args.source_count,
        "parameter_delta_l2_norm_per_skill": args.step_norm,
        "routes": delta_manifest,
    }
    manifest_path = Path(args.delta_output_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = {
        "split": args.split,
        "parameter_group": args.parameter_group,
        "parameter_count": sum(parameters[name].numel() for name in names),
        "dtype": "float32",
        "objective": "value_expectation",
        "protocol": "base-failure-triggered_skill_gradient_purification_300x",
        "failure_definition": "base_top_action_is_not_discounted_value_optimal",
        "step_protocol": "equal_parameter_l2_norm_for_single_mean_and_purified_directions",
        "parameter_delta_l2_norm": args.step_norm,
        "source_count_per_skill": args.source_count,
        "holdout_count_per_relationship_per_skill": args.holdout_count,
        "verbs": list(args.verbs),
        "sample_seed": args.sample_seed,
        "keep_fraction": args.keep_fraction,
        "delta_manifest": str(manifest_path),
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
