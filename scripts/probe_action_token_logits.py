#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from self_evolve.alfworld_data import load_decisions
from self_evolve.alfworld_skills import build_action_prompt
from self_evolve.token_value import (
    branch_value_summary,
    build_prefix_trie,
    conditional_probabilities,
    role_score_contributions,
    semantic_token_roles,
)


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
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    parser.add_argument("--top-k-movers", type=int, default=12)
    parser.add_argument("--max-decisions", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_value_rows(path: str, split: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == split:
                index = int(row["global_decision_index"])
                if index in result:
                    raise ValueError(f"Duplicate value row for {split} index {index}")
                result[index] = row
    return result


def load_completed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(int(json.loads(line)["global_decision_index"]))
    return completed


@torch.inference_mode()
def next_token_logits(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    attention_mask = torch.ones_like(input_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=1,
    )
    return outputs.logits[:, -1, :].float()


def token_description(tokenizer: Any, token_id: int) -> dict[str, Any]:
    return {
        "token_id": int(token_id),
        "token": tokenizer.convert_ids_to_tokens(int(token_id)),
        "text": tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False),
    }


def mover_rows(
    tokenizer: Any,
    base_logits: torch.Tensor,
    seed_logits: torch.Tensor,
    base_probabilities: torch.Tensor,
    seed_probabilities: torch.Tensor,
    k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    delta = seed_probabilities - base_probabilities
    k = min(k, int(delta.numel()))

    def rows(indices: torch.Tensor) -> list[dict[str, Any]]:
        result = []
        for raw_index in indices.tolist():
            index = int(raw_index)
            item = token_description(tokenizer, index)
            item.update(
                {
                    "base_logit": float(base_logits[index]),
                    "seed_logit": float(seed_logits[index]),
                    "logit_delta": float(seed_logits[index] - base_logits[index]),
                    "base_probability": float(base_probabilities[index]),
                    "seed_probability": float(seed_probabilities[index]),
                    "probability_delta": float(delta[index]),
                }
            )
            result.append(item)
        return result

    receivers = rows(torch.topk(delta, k=k).indices)
    donors = rows(torch.topk(-delta, k=k).indices)
    return receivers, donors


def score_one_decision(
    *,
    decision: Any,
    global_index: int,
    value_row: dict[str, Any],
    tokenizer: Any,
    base_model: Any,
    seed_model: Any,
    device: torch.device,
    batch_size: int,
    max_length: int,
    history_window: int,
    top_k_movers: int,
) -> dict[str, Any]:
    if list(decision.admissible_actions) != [item["action"] for item in value_row["actions"]]:
        raise AssertionError(f"Candidate mismatch at state {global_index}")
    values = [float(item["discounted_success"]) for item in value_row["actions"]]
    prompt = build_action_prompt(decision, history_window=history_window)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

    token_ids_by_action: list[list[int]] = []
    roles_by_action: list[list[str]] = []
    offsets_by_action: list[list[list[int]]] = []
    for action in decision.admissible_actions:
        encoded = tokenizer(
            " " + action,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = [int(value) for value in encoded["input_ids"]]
        offsets = [[int(a), int(b)] for a, b in encoded["offset_mapping"]]
        if not token_ids:
            raise ValueError(f"Candidate produced no tokens: {action!r}")
        token_ids_by_action.append(token_ids)
        offsets_by_action.append(offsets)
        roles_by_action.append(semantic_token_roles(action, offsets))

    # Preserve the exact batching and right-padding semantics used by
    # SequenceActionScorer. In BF16, changing batch/sequence shapes can move a
    # logit by one or two quantization bins, so unique-prefix forward passes do
    # not numerically reproduce the action scores used in phase one.
    pairs: list[tuple[list[int], list[int]]] = []
    prompt_was_truncated = False
    for token_ids in token_ids_by_action:
        local_prompt = prompt_ids
        max_prompt_tokens = max_length - len(token_ids)
        if max_prompt_tokens < 1:
            raise ValueError(f"Candidate exceeds max length at state {global_index}")
        if len(local_prompt) > max_prompt_tokens:
            local_prompt = local_prompt[-max_prompt_tokens:]
            prompt_was_truncated = True
        pairs.append((local_prompt, token_ids))

    trie = build_prefix_trie(token_ids_by_action)
    node_rows_by_prefix: dict[tuple[int, ...], dict[str, Any]] = {}
    action_rows: list[dict[str, Any]] = []
    max_base_score_error = 0.0
    max_seed_score_error = 0.0
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        full_ids = [local_prompt + token_ids for local_prompt, token_ids in batch]
        width = max(len(item) for item in full_ids)
        input_ids = torch.full(
            (len(batch), width), pad_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, ids in enumerate(full_ids):
            input_ids[row_index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[row_index, : len(ids)] = 1
        prompt_lengths = {len(local_prompt) for local_prompt, _ in batch}
        if len(prompt_lengths) != 1:
            raise AssertionError("Truncation changed prompt lengths inside a score batch")
        prompt_length = next(iter(prompt_lengths))
        logits_to_keep = torch.arange(prompt_length - 1, width - 1, device=device)
        with torch.inference_mode():
            base_logits_batch = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=logits_to_keep,
            ).logits.float()
            seed_logits_batch = seed_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=logits_to_keep,
            ).logits.float()
        base_log_probs_batch = torch.log_softmax(base_logits_batch, dim=-1)
        seed_log_probs_batch = torch.log_softmax(seed_logits_batch, dim=-1)

        for row_index, (_, token_ids) in enumerate(batch):
            action_index = start + row_index
            action = decision.admissible_actions[action_index]
            roles = roles_by_action[action_index]
            token_count = len(token_ids)
            local_positions = torch.arange(token_count, device=device)
            targets = torch.tensor(token_ids, dtype=torch.long, device=device)
            base_logits_for_tokens = base_logits_batch[row_index, local_positions, targets]
            seed_logits_for_tokens = seed_logits_batch[row_index, local_positions, targets]
            base_log_probs_for_tokens = base_log_probs_batch[row_index, local_positions, targets]
            seed_log_probs_for_tokens = seed_log_probs_batch[row_index, local_positions, targets]
            base_logits = [float(value) for value in base_logits_for_tokens]
            seed_logits = [float(value) for value in seed_logits_for_tokens]
            base_log_probs = [float(value) for value in base_log_probs_for_tokens]
            seed_log_probs = [float(value) for value in seed_log_probs_for_tokens]
            deltas = [
                seed - base for base, seed in zip(base_log_probs, seed_log_probs, strict=True)
            ]
            base_score = math.fsum(base_log_probs) / token_count
            seed_score = math.fsum(seed_log_probs) / token_count
            source = value_row["actions"][action_index]
            max_base_score_error = max(
                max_base_score_error, abs(base_score - float(source["base_score"]))
            )
            max_seed_score_error = max(
                max_seed_score_error, abs(seed_score - float(source["seed_score"]))
            )
            action_rows.append(
                {
                    "candidate_index": action_index,
                    "action": action,
                    "discounted_value": values[action_index],
                    "won": int(source["won"]),
                    "is_value_optimal": int(values[action_index] == max(values)),
                    "base_action_probability": float(source["base_probability"]),
                    "seed_action_probability": float(source["seed_probability"]),
                    "action_probability_delta": float(source["probability_delta"]),
                    "token_ids": token_ids,
                    "token_texts": [
                        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                        for token_id in token_ids
                    ],
                    "token_offsets": offsets_by_action[action_index],
                    "token_roles": roles,
                    "base_token_logits": base_logits,
                    "seed_token_logits": seed_logits,
                    "token_logit_deltas": [
                        seed - base for base, seed in zip(base_logits, seed_logits, strict=True)
                    ],
                    "base_token_log_probabilities": base_log_probs,
                    "seed_token_log_probabilities": seed_log_probs,
                    "token_log_probability_deltas": deltas,
                    "base_normalized_score": base_score,
                    "seed_normalized_score": seed_score,
                    "normalized_score_delta": seed_score - base_score,
                    "role_score_delta_contributions": role_score_contributions(deltas, roles),
                }
            )

            # Save one canonical full-vocabulary distribution for every unique
            # candidate prefix. All candidate-child logits are taken from that
            # same distribution, so branch comparisons remain internally exact.
            for depth, token_id in enumerate(token_ids):
                prefix = tuple(token_ids[:depth])
                if prefix in node_rows_by_prefix:
                    continue
                base_logits_node = base_logits_batch[row_index, depth]
                seed_logits_node = seed_logits_batch[row_index, depth]
                base_log_probs_node = base_log_probs_batch[row_index, depth]
                seed_log_probs_node = seed_log_probs_batch[row_index, depth]
                base_probabilities = base_log_probs_node.exp()
                seed_probabilities = seed_log_probs_node.exp()
                child_map = trie[prefix]
                child_ids = sorted(child_map)
                base_child = [float(base_probabilities[index]) for index in child_ids]
                seed_child = [float(seed_probabilities[index]) for index in child_ids]
                base_conditional = conditional_probabilities(base_child)
                seed_conditional = conditional_probabilities(seed_child)
                children = []
                for position, child_token_id in enumerate(child_ids):
                    action_indices = child_map[child_token_id]
                    child = token_description(tokenizer, child_token_id)
                    child.update(
                        {
                            "roles": sorted(
                                {
                                    roles_by_action[index][depth]
                                    for index in action_indices
                                }
                            ),
                            "action_indices": list(action_indices),
                            "branch_value": branch_value_summary(action_indices, values),
                            "base_logit": float(base_logits_node[child_token_id]),
                            "seed_logit": float(seed_logits_node[child_token_id]),
                            "logit_delta": float(
                                seed_logits_node[child_token_id]
                                - base_logits_node[child_token_id]
                            ),
                            "base_log_probability": float(base_log_probs_node[child_token_id]),
                            "seed_log_probability": float(seed_log_probs_node[child_token_id]),
                            "log_probability_delta": float(
                                seed_log_probs_node[child_token_id]
                                - base_log_probs_node[child_token_id]
                            ),
                            "base_probability": base_child[position],
                            "seed_probability": seed_child[position],
                            "probability_delta": seed_child[position] - base_child[position],
                            "base_candidate_conditional_probability": base_conditional[position],
                            "seed_candidate_conditional_probability": seed_conditional[position],
                            "candidate_conditional_probability_delta": (
                                seed_conditional[position] - base_conditional[position]
                            ),
                        }
                    )
                    children.append(child)
                receivers, donors = mover_rows(
                    tokenizer,
                    base_logits_node,
                    seed_logits_node,
                    base_probabilities,
                    seed_probabilities,
                    top_k_movers,
                )
                delta_probabilities = seed_probabilities - base_probabilities
                node_rows_by_prefix[prefix] = {
                    "depth": depth,
                    "prefix_token_ids": list(prefix),
                    "prefix_text": tokenizer.decode(
                        list(prefix), clean_up_tokenization_spaces=False
                    ),
                    "canonical_candidate_index": action_index,
                    "canonical_batch_width": width,
                    "candidate_children": children,
                    "candidate_child_count": len(children),
                    "base_candidate_child_mass": math.fsum(base_child),
                    "seed_candidate_child_mass": math.fsum(seed_child),
                    "candidate_child_mass_delta": math.fsum(seed_child) - math.fsum(base_child),
                    "vocabulary_total_variation": float(
                        0.5 * torch.abs(delta_probabilities).sum()
                    ),
                    "top_probability_receivers": receivers,
                    "top_probability_donors": donors,
                }

    node_rows = list(node_rows_by_prefix.values())

    return {
        "split": decision.split,
        "global_decision_index": global_index,
        "episode_id": decision.episode_id,
        "episode_key": value_row["episode_key"],
        "task_type": decision.task_type,
        "step_index": decision.step_index,
        "expert_action": decision.expert_action,
        "expert_transition": value_row["expert_transition"],
        "discounted_expected_value_delta": value_row["discounted_expected_value_delta"],
        "base_top_action": value_row["base_top_action"],
        "seed_top_action": value_row["seed_top_action"],
        "base_top_is_value_optimal": value_row["discounted_base_top_is_value_optimal"],
        "seed_top_is_value_optimal": value_row["discounted_seed_top_is_value_optimal"],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_tokens": len(prompt_ids),
        "prompt_was_truncated": int(prompt_was_truncated),
        "candidate_actions": len(action_rows),
        "candidate_tokens": sum(len(item) for item in token_ids_by_action),
        "unique_prefix_nodes": len(node_rows),
        "max_base_score_reproduction_error": max_base_score_error,
        "max_seed_score_reproduction_error": max_seed_score_error,
        "actions": action_rows,
        "nodes": sorted(node_rows, key=lambda item: (item["depth"], item["prefix_token_ids"])),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.batch_size < 1 or args.top_k_movers < 1:
        raise ValueError("batch-size and top-k-movers must be positive")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This full-vocabulary experiment requires a CUDA device")

    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    selected = [
        (index, decision)
        for index, decision in enumerate(decisions)
        if index % args.num_shards == args.shard_index
    ]
    if args.max_decisions > 0:
        selected = selected[: args.max_decisions]
    values = load_value_rows(args.value_trace, args.split)
    missing = [index for index, _ in selected if index not in values]
    if missing:
        raise ValueError(f"Missing {len(missing)} value rows; first indices: {missing[:10]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"
    if trace_path.exists() and not args.resume:
        trace_path.unlink()
    completed = load_completed(trace_path) if args.resume else set()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for semantic offset mapping")
    dtype = torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, local_files_only=True
    ).to(device)
    seed_model = AutoModelForCausalLM.from_pretrained(
        args.seed_model, dtype=dtype, local_files_only=True
    ).to(device)
    base_model.eval()
    seed_model.eval()
    for model in (base_model, seed_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    mode = "a" if args.resume else "w"
    newly_completed = 0
    with trace_path.open(mode, encoding="utf-8") as handle:
        for position, (global_index, decision) in enumerate(selected, start=1):
            if global_index in completed:
                continue
            row = score_one_decision(
                decision=decision,
                global_index=global_index,
                value_row=values[global_index],
                tokenizer=tokenizer,
                base_model=base_model,
                seed_model=seed_model,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                history_window=args.history_window,
                top_k_movers=args.top_k_movers,
            )
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            newly_completed += 1
            print(
                f"[{position}/{len(selected)}] split={args.split} global={global_index} "
                f"nodes={row['unique_prefix_nodes']} "
                f"score_error=({row['max_base_score_reproduction_error']:.3g},"
                f"{row['max_seed_score_reproduction_error']:.3g})",
                flush=True,
            )

    metadata = vars(args).copy()
    metadata.update(
        {
            "selected_decisions": len(selected),
            "previously_completed": len(completed),
            "newly_completed": newly_completed,
            "trace_rows": len(load_completed(trace_path)),
            "tokenizer_class": type(tokenizer).__name__,
            "vocabulary_size": len(tokenizer),
            "dtype": str(dtype),
        }
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
