#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import hashlib
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

from self_evolve.alfworld_data import load_decisions
from self_evolve.alfworld_skills import build_action_prompt
from self_evolve.token_value import (
    one_hot_nll_coefficients,
    optimal_set_loss_coefficients,
    value_expectation_loss_coefficients,
)


OBJECTIVES = ("value_expectation", "value_optimal_set", "expert_nll_control")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-file", required=True)
    parser.add_argument("--value-trace", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--seed-model", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--split", required=True, choices=("valid_seen", "valid_unseen"))
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=20260903)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    parser.add_argument("--sketch-size", type=int, default=4096)
    parser.add_argument("--max-states", type=int, default=-1)
    return parser.parse_args()


def load_value_rows(path: str, split: str) -> dict[int, dict[str, Any]]:
    result = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == split:
                result[int(row["global_decision_index"])] = row
    return result


def stratified_indices(
    rows: dict[int, dict[str, Any]], sample_size: int, seed: int
) -> list[int]:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in rows.items():
        values = [action["discounted_value"] if "discounted_value" in action else action["discounted_success"] for action in row["actions"]]
        if max(values) - min(values) <= 1e-12:
            continue
        value_shift = float(row["discounted_expected_value_delta"])
        sign = "positive" if value_shift > 1e-12 else "negative" if value_shift < -1e-12 else "zero"
        key = (
            row["task_type"],
            row["action_verb"],
            row["expert_transition"],
            sign,
            int(row["discounted_seed_top_is_value_optimal"]),
        )
        groups[key].append(index)
    rng = random.Random(seed)
    keys = sorted(groups, key=str)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    selected = []
    depth = 0
    while len(selected) < min(sample_size, sum(len(group) for group in groups.values())):
        added = False
        for key in keys:
            if depth < len(groups[key]):
                selected.append(groups[key][depth])
                added = True
                if len(selected) >= sample_size:
                    break
        if not added:
            break
        depth += 1
    return selected


def selected_parameter_groups(model: Any) -> tuple[dict[str, torch.nn.Parameter], dict[str, list[str]]]:
    layer_count = int(model.config.num_hidden_layers)
    last = layer_count - 1
    first_of_last_four = layer_count - 4
    selected: dict[str, torch.nn.Parameter] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        layer_match = re.search(r"model\.layers\.(\d+)\.", name)
        layer_index = int(layer_match.group(1)) if layer_match else None
        is_norm = "layernorm.weight" in name or name == "model.norm.weight"
        is_embedding = name == "model.embed_tokens.weight"
        in_last_four = layer_index is not None and layer_index >= first_of_last_four
        if not (is_embedding or is_norm or in_last_four):
            continue
        selected[name] = parameter
        groups["selected_union"].append(name)
        if is_embedding:
            groups["tied_embedding_output"].append(name)
        if is_norm:
            groups["all_rmsnorm"].append(name)
        if in_last_four:
            groups["last_four_blocks"].append(name)
        if layer_index == last:
            groups["last_block"].append(name)
            if ".self_attn." in name:
                groups["last_attention"].append(name)
            if ".mlp." in name:
                groups["last_mlp"].append(name)
    required = {
        "selected_union",
        "tied_embedding_output",
        "all_rmsnorm",
        "last_four_blocks",
        "last_block",
        "last_attention",
        "last_mlp",
    }
    if set(groups) != required or any(not groups[name] for name in required):
        raise RuntimeError(f"Parameter group discovery failed: {sorted(groups)}")
    return selected, {name: sorted(values) for name, values in sorted(groups.items())}


def configure_gradients(model: Any, selected_names: set[str]) -> None:
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected_names)


def candidate_pairs(
    tokenizer: Any, prompt: str, candidates: Sequence[str], max_length: int
) -> list[tuple[list[int], list[int]]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    pairs = []
    for candidate in candidates:
        candidate_ids = tokenizer.encode(" " + candidate, add_special_tokens=False)
        max_prompt = max_length - len(candidate_ids)
        if max_prompt < 1:
            raise ValueError(f"Candidate exceeds max length: {candidate!r}")
        pairs.append((prompt_ids[-max_prompt:], candidate_ids))
    return pairs


def compute_gradient(
    *,
    model: Any,
    selected: dict[str, torch.nn.Parameter],
    tokenizer: Any,
    prompt: str,
    candidates: Sequence[str],
    coefficients: np.ndarray,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    pairs = candidate_pairs(tokenizer, prompt, candidates, max_length)
    scores = []
    pad_id = int(tokenizer.pad_token_id)
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        full_ids = [prompt_ids + candidate_ids for prompt_ids, candidate_ids in batch]
        width = max(len(ids) for ids in full_ids)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        for row_index, ids in enumerate(full_ids):
            input_ids[row_index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[row_index, : len(ids)] = 1
        prompt_lengths = {len(prompt_ids) for prompt_ids, _ in batch}
        if len(prompt_lengths) != 1:
            raise AssertionError("Gradient batch prompt lengths differ")
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
        local_coefficients = torch.tensor(
            coefficients[start : start + len(batch)], dtype=score_tensor.dtype, device=device
        )
        objective = torch.dot(score_tensor, local_coefficients)
        objective.backward()
        del logits, log_probs, score_tensor, objective
    if any(parameter.grad is None for parameter in selected.values()):
        missing = [name for name, parameter in selected.items() if parameter.grad is None]
        raise RuntimeError(f"Missing selected gradients: {missing[:5]}")
    return np.asarray(scores, dtype=np.float64)


def build_sketch_specs(
    groups: dict[str, list[str]], parameters: dict[str, torch.nn.Parameter], size: int, seed: int
) -> dict[str, dict[str, list[tuple[int, int]]]]:
    specs = {}
    for group_index, (group, names) in enumerate(groups.items()):
        lengths = [parameters[name].numel() for name in names]
        cumulative = np.cumsum(lengths, dtype=np.int64)
        total = int(cumulative[-1])
        count = min(size, total)
        rng = random.Random(seed + group_index * 104729)
        positions: set[int] = set()
        while len(positions) < count:
            positions.add(rng.randrange(total))
        by_parameter: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for output_index, global_position in enumerate(sorted(positions)):
            parameter_index = bisect.bisect_right(cumulative, global_position)
            previous = int(cumulative[parameter_index - 1]) if parameter_index else 0
            by_parameter[names[parameter_index]].append(
                (output_index, global_position - previous)
            )
        specs[group] = dict(by_parameter)
    return specs


def extract_sketch(
    parameters: dict[str, torch.nn.Parameter], spec: dict[str, list[tuple[int, int]]]
) -> np.ndarray:
    size = sum(len(items) for items in spec.values())
    result = np.empty(size, dtype=np.float32)
    for name, items in spec.items():
        output_indices = np.asarray([item[0] for item in items], dtype=np.int64)
        local_indices = torch.tensor(
            [item[1] for item in items], dtype=torch.long, device=parameters[name].device
        )
        values = parameters[name].grad.detach().flatten()[local_indices].float().cpu().numpy()
        result[output_indices] = values
    norm = float(np.linalg.norm(result))
    return result / norm if norm > 1e-20 else result


def gradient_pair_metrics(
    base_parameters: dict[str, torch.nn.Parameter],
    seed_parameters: dict[str, torch.nn.Parameter],
    groups: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    per_parameter = {}
    for name in base_parameters:
        base_gradient = base_parameters[name].grad.detach().float()
        seed_gradient = seed_parameters[name].grad.detach().float()
        delta = seed_parameters[name].detach().float() - base_parameters[name].detach().float()
        per_parameter[name] = {
            "base_sq": float(torch.sum(base_gradient * base_gradient)),
            "seed_sq": float(torch.sum(seed_gradient * seed_gradient)),
            "delta_sq": float(torch.sum(delta * delta)),
            "base_seed_dot": float(torch.sum(base_gradient * seed_gradient)),
            "base_delta_dot": float(torch.sum(base_gradient * delta)),
            "seed_delta_dot": float(torch.sum(seed_gradient * delta)),
        }
    result = {}
    for group, names in groups.items():
        sums = {
            key: math.fsum(per_parameter[name][key] for name in names)
            for key in next(iter(per_parameter.values()))
        }
        base_norm = math.sqrt(max(sums["base_sq"], 0.0))
        seed_norm = math.sqrt(max(sums["seed_sq"], 0.0))
        delta_norm = math.sqrt(max(sums["delta_sq"], 0.0))
        result[group] = {
            "parameters": sum(base_parameters[name].numel() for name in names),
            "base_gradient_norm": base_norm,
            "seed_gradient_norm": seed_norm,
            "seed_to_base_gradient_norm_ratio": seed_norm / max(base_norm, 1e-30),
            "base_seed_gradient_cosine": sums["base_seed_dot"]
            / max(base_norm * seed_norm, 1e-30),
            "parameter_delta_norm": delta_norm,
            "base_gradient_dot_parameter_delta": sums["base_delta_dot"],
            "seed_gradient_dot_parameter_delta": sums["seed_delta_dot"],
            "base_descent_parameter_delta_cosine": -sums["base_delta_dot"]
            / max(base_norm * delta_norm, 1e-30),
            "seed_descent_parameter_delta_cosine": -sums["seed_delta_dot"]
            / max(seed_norm * delta_norm, 1e-30),
        }
    return result


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Gradient probing requires CUDA")
    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    value_rows = load_value_rows(args.value_trace, args.split)
    selected_all = stratified_indices(value_rows, args.sample_size, args.sample_seed)
    selected_indices = selected_all[args.shard_index :: args.num_shards]
    if args.max_states > 0:
        selected_indices = selected_indices[: args.max_states]
        selected_expected = sorted(
            index
            for shard in range(args.num_shards)
            for index in selected_all[shard :: args.num_shards][: args.max_states]
        )
    else:
        selected_expected = selected_all
    if not selected_indices:
        raise ValueError("No states selected")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    seed_model = AutoModelForCausalLM.from_pretrained(
        args.seed_model, dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    base_parameters, groups = selected_parameter_groups(base_model)
    seed_parameters, seed_groups = selected_parameter_groups(seed_model)
    if groups != seed_groups or set(base_parameters) != set(seed_parameters):
        raise RuntimeError("Base/SEED parameter structures differ")
    configure_gradients(base_model, set(base_parameters))
    configure_gradients(seed_model, set(seed_parameters))
    sketch_specs = build_sketch_specs(groups, base_parameters, args.sketch_size, args.sample_seed)

    states = []
    saved_sketches: dict[str, list[np.ndarray]] = defaultdict(list)
    for local_position, global_index in enumerate(selected_indices, start=1):
        decision = decisions[global_index]
        value_row = value_rows[global_index]
        candidates = list(decision.admissible_actions)
        if candidates != [action["action"] for action in value_row["actions"]]:
            raise AssertionError(f"Candidate mismatch at state {global_index}")
        values = np.asarray(
            [action["discounted_success"] for action in value_row["actions"]],
            dtype=np.float64,
        )
        base_scores = np.asarray(
            [action["base_score"] for action in value_row["actions"]], dtype=np.float64
        )
        seed_scores = np.asarray(
            [action["seed_score"] for action in value_row["actions"]], dtype=np.float64
        )
        expert_index = candidates.index(decision.expert_action)
        base_coefficients = {
            "value_expectation": value_expectation_loss_coefficients(base_scores, values),
            "value_optimal_set": optimal_set_loss_coefficients(base_scores, values),
            "expert_nll_control": one_hot_nll_coefficients(base_scores, expert_index),
        }
        seed_coefficients = {
            "value_expectation": value_expectation_loss_coefficients(seed_scores, values),
            "value_optimal_set": optimal_set_loss_coefficients(seed_scores, values),
            "expert_nll_control": one_hot_nll_coefficients(seed_scores, expert_index),
        }
        prompt = build_action_prompt(decision, history_window=args.history_window)
        objective_rows = {}
        max_base_error = 0.0
        max_seed_error = 0.0
        for objective in OBJECTIVES:
            recomputed_base = compute_gradient(
                model=base_model,
                selected=base_parameters,
                tokenizer=tokenizer,
                prompt=prompt,
                candidates=candidates,
                coefficients=base_coefficients[objective],
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            recomputed_seed = compute_gradient(
                model=seed_model,
                selected=seed_parameters,
                tokenizer=tokenizer,
                prompt=prompt,
                candidates=candidates,
                coefficients=seed_coefficients[objective],
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            max_base_error = max(max_base_error, float(np.max(np.abs(recomputed_base - base_scores))))
            max_seed_error = max(max_seed_error, float(np.max(np.abs(recomputed_seed - seed_scores))))
            objective_rows[objective] = gradient_pair_metrics(
                base_parameters, seed_parameters, groups
            )
            for group in groups:
                saved_sketches[f"{objective}__base__{group}"].append(
                    extract_sketch(base_parameters, sketch_specs[group]).astype(np.float16)
                )
                saved_sketches[f"{objective}__seed__{group}"].append(
                    extract_sketch(seed_parameters, sketch_specs[group]).astype(np.float16)
                )
            base_model.zero_grad(set_to_none=True)
            seed_model.zero_grad(set_to_none=True)
        if max(max_base_error, max_seed_error) > 1e-4:
            raise AssertionError(
                f"Score reproduction failed at {global_index}: {max_base_error}, {max_seed_error}"
            )
        states.append(
            {
                "global_decision_index": global_index,
                "episode_key": value_row["episode_key"],
                "task_type": value_row["task_type"],
                "action_verb": value_row["action_verb"],
                "step_index": value_row["step_index"],
                "expert_transition": value_row["expert_transition"],
                "candidate_count": len(candidates),
                "value_range": float(np.ptp(values)),
                "discounted_expected_value_delta": value_row["discounted_expected_value_delta"],
                "base_top_is_value_optimal": value_row["discounted_base_top_is_value_optimal"],
                "seed_top_is_value_optimal": value_row["discounted_seed_top_is_value_optimal"],
                "max_base_score_reproduction_error": max_base_error,
                "max_seed_score_reproduction_error": max_seed_error,
                "objectives": objective_rows,
            }
        )
        print(
            f"[{local_position}/{len(selected_indices)}] split={args.split} global={global_index} "
            f"candidates={len(candidates)} errors=({max_base_error:.3g},{max_seed_error:.3g})",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.stack(values) for name, values in saved_sketches.items()}
    arrays["global_decision_indices"] = np.asarray(
        [state["global_decision_index"] for state in states], dtype=np.int64
    )
    np.savez_compressed(output_dir / "gradient_sketches.npz", **arrays)
    result = {
        "split": args.split,
        "sample_size_requested": args.sample_size,
        "sample_seed": args.sample_seed,
        "selected_global_indices_all_shards": selected_expected,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "objectives": list(OBJECTIVES),
        "parameter_groups": {
            group: {
                "parameter_count": sum(base_parameters[name].numel() for name in names),
                "parameter_names": names,
                "sketch_size": sum(len(items) for items in sketch_specs[group].values()),
                "overlap_note": "Groups intentionally overlap; selected_union is the deduplicated union.",
            }
            for group, names in groups.items()
        },
        "states": states,
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "states": len(states),
                "objectives": list(OBJECTIVES),
                "groups": {name: item["parameter_count"] for name, item in result["parameter_groups"].items()},
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
