#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_multisource_value_gradients import (
    build_specs,
    gradient_norm,
    select_verb_states,
    writeback_and_score,
)
from probe_value_gradient_writeback import (
    load_value_rows,
    objective_coefficients,
    score_actions,
    selected_parameter_groups,
)
from self_evolve.alfworld_data import load_decisions


def accumulate_source_gradient_gpu(
    *, model, parameters, names, tokenizer, spec, baseline_scores, device,
    batch_size, max_length, gradient_sum,
):
    model.zero_grad(set_to_none=True)
    coefficients = objective_coefficients("value_expectation", baseline_scores, spec)
    recomputed = score_actions(
        model, tokenizer, spec["prompt"], spec["candidates"], device,
        batch_size, max_length, coefficients,
    )
    error = float(np.max(np.abs(recomputed - baseline_scores)))
    if error > 1e-5:
        raise AssertionError(f"Gradient score reproduction error {error}")
    norm = math.sqrt(math.fsum(
        float(torch.sum(parameters[name].grad.detach().float() ** 2).item())
        for name in names
    ))
    with torch.no_grad():
        if gradient_sum is None:
            gradient_sum = {
                name: parameters[name].grad.detach().clone() for name in names
            }
        else:
            for name in names:
                gradient_sum[name].add_(parameters[name].grad.detach())
    return gradient_sum, norm, error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-file", required=True)
    parser.add_argument("--value-trace", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--split", choices=("valid_seen", "valid_unseen"), required=True)
    parser.add_argument("--parameter-group", choices=("last_mlp", "last_four_blocks"), required=True)
    parser.add_argument("--step-norms", nargs="+", type=float, required=True)
    parser.add_argument("--verbs", nargs="+", default=["go", "open", "close", "take", "move"])
    parser.add_argument("--source-count", type=int, default=12)
    parser.add_argument("--holdout-count", type=int, default=6)
    parser.add_argument("--sample-seed", type=int, default=20260903)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    if any(step <= 0 for step in args.step_norms):
        raise ValueError("All step norms must be positive")
    device = torch.device(args.device)
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
        gradient_sum = None
        errors = []
        individual_norms = []
        for source_position in range(args.source_count):
            gradient_sum, norm, error = accumulate_source_gradient_gpu(
                model=model,
                parameters=parameters,
                names=names,
                tokenizer=tokenizer,
                spec=specs[source_position],
                baseline_scores=baselines[source_position],
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                gradient_sum=gradient_sum,
            )
            errors.append(error)
            individual_norms.append(norm)
            print(
                f"[{verb_position}/{len(args.verbs)}] {args.split}/{args.parameter_group}/{verb} "
                f"gradient={source_position + 1}/{args.source_count}", flush=True,
            )
        assert gradient_sum is not None
        with torch.no_grad():
            for tensor in gradient_sum.values():
                tensor.div_(float(args.source_count))
        mean_gradient = gradient_sum
        updates = []
        for step in args.step_norms:
            update = writeback_and_score(
                model=model,
                parameters=parameters,
                names=names,
                gradient=mean_gradient,
                step_norm=step,
                specs=specs,
                baselines=baselines,
                tokenizer=tokenizer,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            updates.append(update)
            source_states = update["states"][: args.source_count]
            holdout_states = update["states"][args.source_count :]
            print(
                f"[{verb_position}/{len(args.verbs)}] {args.split}/{args.parameter_group}/{verb} "
                f"step={step:g} source_dV={np.mean([x['expected_value_delta'] for x in source_states]):.6g} "
                f"holdout_dV={np.mean([x['expected_value_delta'] for x in holdout_states]):.6g} "
                f"top_flips={sum(x['baseline_top_index'] != x['updated_top_index'] for x in update['states'])}",
                flush=True,
            )
        verb_results.append(
            {
                "verb": verb,
                "selection": selection,
                "individual_gradient_norms": individual_norms,
                "mean_raw_gradient_norm": gradient_norm(mean_gradient),
                "gradient_score_reproduction_max_error": max(errors),
                "dose_updates": updates,
            }
        )
        del mean_gradient
        model.zero_grad(set_to_none=True)
        print(f"[{verb_position}/{len(args.verbs)}] {args.split}/{args.parameter_group}/{verb} complete", flush=True)

    result = {
        "split": args.split,
        "parameter_group": args.parameter_group,
        "parameter_count": sum(parameters[name].numel() for name in names),
        "dtype": "float32",
        "objective": "value_expectation",
        "direction": "normalized_raw_mean_of_source_gradients",
        "gradient_storage_and_accumulation_device": "cuda",
        "step_norms": args.step_norms,
        "source_count_per_verb": args.source_count,
        "holdout_count_per_category_per_verb": args.holdout_count,
        "verbs": args.verbs,
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
