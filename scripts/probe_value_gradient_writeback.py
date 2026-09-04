#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from self_evolve.alfworld_data import AlfworldDecision, load_decisions
from self_evolve.alfworld_skills import build_action_prompt
from self_evolve.token_value import (
    one_hot_nll_coefficients,
    optimal_set_loss_coefficients,
    value_expectation_loss_coefficients,
)
from self_evolve.value_writeback import (
    CONTROL_BUCKETS,
    candidate_distribution_metrics,
    group_gradient_norm,
    relationship_bucket,
)


ALL_OBJECTIVES = ("value_expectation", "value_optimal_set", "expert_nll_control")
DEFAULT_GROUPS = (
    "all_rmsnorm",
    "last_attention",
    "last_mlp",
    "last_four_blocks",
    "tied_embedding_output",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-file", required=True)
    parser.add_argument("--value-trace", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--split", required=True, choices=("valid_seen", "valid_unseen"))
    parser.add_argument("--sample-size", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=20260903)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--controls-per-bucket", type=int, default=2)
    parser.add_argument("--step-norm", type=float, default=0.01)
    parser.add_argument("--target-source-kl", type=float, default=1e-4)
    parser.add_argument("--calibration-steps", type=int, default=3)
    parser.add_argument("--calibration-tolerance", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    parser.add_argument("--objectives", nargs="+", choices=ALL_OBJECTIVES, default=list(ALL_OBJECTIVES))
    parser.add_argument("--groups", nargs="+", choices=DEFAULT_GROUPS, default=list(DEFAULT_GROUPS))
    return parser.parse_args()


def load_value_rows(path: str, split: str) -> dict[int, dict[str, Any]]:
    result = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == split:
                result[int(row["global_decision_index"])] = row
    return result


def source_indices(
    rows: dict[int, dict[str, Any]], size: int, seed: int, controls_per_bucket: int
) -> list[int]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in rows.items():
        values = [float(action["discounted_success"]) for action in row["actions"]]
        if max(values) - min(values) <= 1e-12:
            continue
        available: dict[str, int] = defaultdict(int)
        for target_index, target in rows.items():
            if target_index == index or target["episode_key"] == row["episode_key"]:
                continue
            target_values = [float(action["discounted_success"]) for action in target["actions"]]
            if max(target_values) - min(target_values) <= 1e-12:
                continue
            available[relationship_bucket(row, target)] += 1
        if any(available[bucket] < controls_per_bucket for bucket in CONTROL_BUCKETS):
            continue
        sign = "positive" if row["discounted_expected_value_delta"] > 0 else "nonpositive"
        groups[(row["task_type"], row["action_verb"], sign)].append(index)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for values in groups.values():
        rng.shuffle(values)
    result = []
    depth = 0
    while len(result) < min(size, sum(map(len, groups.values()))):
        for key in keys:
            if depth < len(groups[key]):
                result.append(groups[key][depth])
                if len(result) == size:
                    return result
        depth += 1
    return result


def select_controls(
    source_index: int,
    rows: dict[int, dict[str, Any]],
    per_bucket: int,
    seed: int,
) -> list[tuple[int, str]]:
    source = rows[source_index]
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, row in rows.items():
        if index == source_index or row["episode_key"] == source["episode_key"]:
            continue
        values = [float(action["discounted_success"]) for action in row["actions"]]
        if max(values) - min(values) <= 1e-12:
            continue
        buckets[relationship_bucket(source, row)].append(index)
    rng = random.Random(seed + source_index * 104729)
    result = []
    for bucket in CONTROL_BUCKETS:
        candidates = buckets[bucket]
        rng.shuffle(candidates)
        if len(candidates) < per_bucket:
            raise ValueError(f"Only {len(candidates)} controls available for {bucket}")
        result.extend((index, bucket) for index in candidates[:per_bucket])
    return result


def selected_parameter_groups(model: Any) -> tuple[dict[str, torch.nn.Parameter], dict[str, list[str]]]:
    last = int(model.config.num_hidden_layers) - 1
    first_of_last_four = last - 3
    selected: dict[str, torch.nn.Parameter] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        match = re.search(r"model\.layers\.(\d+)\.", name)
        layer = int(match.group(1)) if match else None
        is_norm = "layernorm.weight" in name or name == "model.norm.weight"
        is_embedding = name == "model.embed_tokens.weight"
        in_last_four = layer is not None and layer >= first_of_last_four
        if not (is_norm or is_embedding or in_last_four):
            continue
        selected[name] = parameter
        if is_norm:
            groups["all_rmsnorm"].append(name)
        if is_embedding:
            groups["tied_embedding_output"].append(name)
        if in_last_four:
            groups["last_four_blocks"].append(name)
        if layer == last and ".self_attn." in name:
            groups["last_attention"].append(name)
        if layer == last and ".mlp." in name:
            groups["last_mlp"].append(name)
    if set(groups) != set(DEFAULT_GROUPS):
        raise RuntimeError(f"Parameter group discovery failed: {sorted(groups)}")
    return selected, {name: sorted(values) for name, values in sorted(groups.items())}


def candidate_pairs(tokenizer: Any, prompt: str, candidates: Sequence[str], max_length: int):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    pairs = []
    for candidate in candidates:
        candidate_ids = tokenizer.encode(" " + candidate, add_special_tokens=False)
        max_prompt = max_length - len(candidate_ids)
        if max_prompt < 1:
            raise ValueError(f"Candidate exceeds max length: {candidate!r}")
        pairs.append((prompt_ids[-max_prompt:], candidate_ids))
    return pairs


def score_actions(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    coefficients: np.ndarray | None = None,
) -> np.ndarray:
    pairs = candidate_pairs(tokenizer, prompt, candidates, max_length)
    scores: list[float] = []
    pad_id = int(tokenizer.pad_token_id)
    context = torch.enable_grad() if coefficients is not None else torch.inference_mode()
    with context:
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            full_ids = [prompt_ids + candidate_ids for prompt_ids, candidate_ids in batch]
            width = max(map(len, full_ids))
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(input_ids)
            for row_index, ids in enumerate(full_ids):
                input_ids[row_index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                attention_mask[row_index, : len(ids)] = 1
            prompt_lengths = {len(prompt_ids) for prompt_ids, _ in batch}
            if len(prompt_lengths) != 1:
                raise AssertionError("Batch prompt lengths differ")
            prompt_length = next(iter(prompt_lengths))
            positions = torch.arange(prompt_length - 1, width - 1, device=device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=positions,
            ).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            batch_scores = []
            for row_index, (_, candidate_ids) in enumerate(batch):
                targets = torch.tensor(candidate_ids, dtype=torch.long, device=device)
                local_positions = torch.arange(len(candidate_ids), device=device)
                batch_scores.append(log_probs[row_index, local_positions, targets].mean())
            score_tensor = torch.stack(batch_scores)
            scores.extend(float(value) for value in score_tensor.detach().cpu())
            if coefficients is not None:
                local = torch.tensor(
                    coefficients[start : start + len(batch)],
                    dtype=score_tensor.dtype,
                    device=device,
                )
                torch.dot(score_tensor, local).backward()
    return np.asarray(scores, dtype=np.float64)


def state_spec(
    index: int,
    relationship: str,
    decisions: Sequence[AlfworldDecision],
    rows: dict[int, dict[str, Any]],
    history_window: int,
) -> dict[str, Any]:
    decision = decisions[index]
    row = rows[index]
    candidates = list(decision.admissible_actions)
    if candidates != [action["action"] for action in row["actions"]]:
        raise AssertionError(f"Candidate mismatch at state {index}")
    return {
        "global_decision_index": index,
        "relationship": relationship,
        "episode_key": row["episode_key"],
        "task_type": row["task_type"],
        "action_verb": row["action_verb"],
        "step_index": row["step_index"],
        "prompt": build_action_prompt(decision, history_window=history_window),
        "candidates": candidates,
        "values": np.asarray([action["discounted_success"] for action in row["actions"]], dtype=np.float64),
        "expert_index": candidates.index(decision.expert_action),
    }


def objective_coefficients(name: str, scores: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    if name == "value_expectation":
        return value_expectation_loss_coefficients(scores, spec["values"])
    if name == "value_optimal_set":
        return optimal_set_loss_coefficients(scores, spec["values"])
    if name == "expert_nll_control":
        return one_hot_nll_coefficients(scores, spec["expert_index"])
    raise ValueError(name)


def main() -> None:
    args = parse_args()
    if args.step_norm <= 0 or args.target_source_kl <= 0 or args.controls_per_bucket < 0:
        raise ValueError("step-norm must be positive and controls-per-bucket nonnegative")
    if args.calibration_steps < 1 or not 0 < args.calibration_tolerance < 1:
        raise ValueError("invalid calibration settings")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Writeback probing requires CUDA")

    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    rows = load_value_rows(args.value_trace, args.split)
    all_sources = source_indices(
        rows, args.sample_size, args.sample_seed, args.controls_per_bucket
    )
    sources = all_sources[args.shard_index :: args.num_shards]
    if not sources:
        raise ValueError("No source states selected")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, local_files_only=True
    ).to(device)
    parameters, groups = selected_parameter_groups(model)
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in parameters)

    output_rows = []
    for position, source_index in enumerate(sources, start=1):
        controls = select_controls(
            source_index, rows, args.controls_per_bucket, args.sample_seed
        )
        specs = [state_spec(source_index, "source", decisions, rows, args.history_window)]
        specs.extend(
            state_spec(index, relationship, decisions, rows, args.history_window)
            for index, relationship in controls
        )
        baseline_scores = [
            score_actions(
                model, tokenizer, spec["prompt"], spec["candidates"], device,
                args.batch_size, args.max_length,
            )
            for spec in specs
        ]
        repeat_scores = [
            score_actions(
                model, tokenizer, spec["prompt"], spec["candidates"], device,
                args.batch_size, args.max_length,
            )
            for spec in specs
        ]
        repeat_error = max(
            float(np.max(np.abs(first - second)))
            for first, second in zip(baseline_scores, repeat_scores, strict=True)
        )
        objective_rows = {}
        source_spec = specs[0]
        for objective in args.objectives:
            model.zero_grad(set_to_none=True)
            coefficients = objective_coefficients(objective, baseline_scores[0], source_spec)
            gradient_scores = score_actions(
                model, tokenizer, source_spec["prompt"], source_spec["candidates"], device,
                args.batch_size, args.max_length, coefficients,
            )
            reproduction_error = float(np.max(np.abs(gradient_scores - baseline_scores[0])))
            if reproduction_error > 1e-5:
                raise AssertionError(f"FP32 score reproduction error {reproduction_error}")
            group_rows = {}
            for group in args.groups:
                names = groups[group]
                group_parameters = [parameters[name] for name in names]
                gradient_norm = group_gradient_norm(group_parameters)
                if not math.isfinite(gradient_norm) or gradient_norm <= 1e-20:
                    raise RuntimeError(f"Invalid gradient norm for {objective}/{group}: {gradient_norm}")
                backups = [parameter.detach().clone() for parameter in group_parameters]
                calibrated_step = args.step_norm
                calibration_trace = []
                for _ in range(args.calibration_steps):
                    scale = -calibrated_step / gradient_norm
                    with torch.no_grad():
                        for parameter, backup in zip(group_parameters, backups, strict=True):
                            parameter.copy_(backup)
                            parameter.add_(parameter.grad, alpha=scale)
                    calibration_scores = score_actions(
                        model, tokenizer, source_spec["prompt"], source_spec["candidates"], device,
                        args.batch_size, args.max_length,
                    )
                    calibration_metrics = candidate_distribution_metrics(
                        baseline_scores[0], calibration_scores, source_spec["values"],
                        source_spec["expert_index"],
                    )
                    observed_kl = float(calibration_metrics["kl_baseline_to_updated"])
                    calibration_trace.append(
                        {"parameter_delta_norm": calibrated_step, "source_kl": observed_kl}
                    )
                    relative_error = abs(observed_kl - args.target_source_kl) / args.target_source_kl
                    if relative_error <= args.calibration_tolerance:
                        break
                    if observed_kl <= 1e-20:
                        factor = 10.0
                    else:
                        factor = math.sqrt(args.target_source_kl / observed_kl)
                        factor = min(10.0, max(0.1, factor))
                    calibrated_step *= factor

                scale = -calibrated_step / gradient_norm
                with torch.no_grad():
                    for parameter, backup in zip(group_parameters, backups, strict=True):
                        parameter.copy_(backup)
                        parameter.add_(parameter.grad, alpha=scale)
                actual_delta_norm = math.sqrt(
                    math.fsum(
                        float(torch.sum((parameter.detach() - backup).float() ** 2).item())
                        for parameter, backup in zip(group_parameters, backups, strict=True)
                    )
                )
                updated_scores = [
                    score_actions(
                        model, tokenizer, spec["prompt"], spec["candidates"], device,
                        args.batch_size, args.max_length,
                    )
                    for spec in specs
                ]
                with torch.no_grad():
                    for parameter, backup in zip(group_parameters, backups, strict=True):
                        parameter.copy_(backup)
                restore_error = max(
                    float(torch.max(torch.abs(parameter.detach() - backup)).item())
                    for parameter, backup in zip(group_parameters, backups, strict=True)
                )
                del backups
                state_rows = []
                for spec, baseline, updated in zip(specs, baseline_scores, updated_scores, strict=True):
                    state_rows.append(
                        {
                            "global_decision_index": spec["global_decision_index"],
                            "relationship": spec["relationship"],
                            "episode_key": spec["episode_key"],
                            "task_type": spec["task_type"],
                            "action_verb": spec["action_verb"],
                            "step_index": spec["step_index"],
                            **candidate_distribution_metrics(
                                baseline, updated, spec["values"], spec["expert_index"]
                            ),
                        }
                    )
                group_rows[group] = {
                    "parameter_count": sum(parameter.numel() for parameter in group_parameters),
                    "gradient_norm": gradient_norm,
                    "initial_parameter_delta_norm": args.step_norm,
                    "calibrated_parameter_delta_norm": calibrated_step,
                    "actual_parameter_delta_norm": actual_delta_norm,
                    "target_source_kl": args.target_source_kl,
                    "calibration_trace": calibration_trace,
                    "scale_on_raw_gradient": scale,
                    "restore_max_absolute_error": restore_error,
                    "states": state_rows,
                }
                print(
                    f"[{position}/{len(sources)}] split={args.split} source={source_index} "
                    f"objective={objective} group={group} "
                    f"step={calibrated_step:.4g} "
                    f"source_dV={state_rows[0]['expected_value_delta']:.6g} "
                    f"source_KL={state_rows[0]['kl_baseline_to_updated']:.6g}",
                    flush=True,
                )
            objective_rows[objective] = {
                "gradient_score_reproduction_error": reproduction_error,
                "groups": group_rows,
            }
        model.zero_grad(set_to_none=True)
        output_rows.append(
            {
                "source_global_decision_index": source_index,
                "baseline_repeat_max_absolute_error": repeat_error,
                "objectives": objective_rows,
            }
        )

    result = {
        "split": args.split,
        "dtype": "float32",
        "writeback_scope": "in_memory_only_and_exactly_restored_after_each_condition",
        "sample_size_requested_across_all_shards": args.sample_size,
        "selected_source_indices_all_shards": all_sources,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "controls_per_relationship_bucket": args.controls_per_bucket,
        "relationship_buckets": list(CONTROL_BUCKETS),
        "objectives": args.objectives,
        "parameter_groups": args.groups,
        "parameter_step_l2_norm": args.step_norm,
        "target_source_kl": args.target_source_kl,
        "calibration_steps": args.calibration_steps,
        "calibration_tolerance": args.calibration_tolerance,
        "sources": output_rows,
    }
    destination = Path(args.output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"split": args.split, "sources": len(output_rows), "output": str(destination)}))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
