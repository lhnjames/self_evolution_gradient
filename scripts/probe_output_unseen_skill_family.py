#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_output_gradient_evolution import (
    atom_dot,
    make_atom,
    softmax_weights,
    strategy_scores,
)
from probe_output_head_representation_sufficiency import cache_state_hidden, cached_scores_from_logits
from probe_output_only_h6 import read_trace
from probe_skill_gradient_purification import base_failure_indices
from probe_value_gradient_writeback import load_value_rows, state_spec
from self_evolve.alfworld_data import load_decisions
from self_evolve.output_head_oracle import (
    is_value_heterogeneous,
    one_failure_per_episode,
    task_round_robin,
)
from self_evolve.value_writeback import candidate_distribution_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-file", required=True)
    parser.add_argument("--value-trace", required=True)
    parser.add_argument("--base-score-trace", required=True)
    parser.add_argument("--seed-score-trace", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--sample-seed", type=int, required=True)
    parser.add_argument("--heldout-task-type", default="pick_cool_then_place_in_recep")
    parser.add_argument("--step-norm", type=float, default=0.72)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--harm-lambda", type=float, default=1.0)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    return parser.parse_args()


def cosine_matrix(atoms: Sequence[dict[str, Any]]) -> np.ndarray:
    count = len(atoms)
    matrix = np.empty((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left, count):
            value = atom_dot(atoms[left], atoms[right]) / (
                atoms[left]["norm"] * atoms[right]["norm"]
            )
            matrix[left, right] = matrix[right, left] = value
    np.fill_diagonal(matrix, 1.0)
    return np.clip(matrix, -1.0, 1.0)


def evaluate(
    atoms, cosine, weights, targets, base_logits, base_scores, trace_base, trace_seed, step_norm
):
    changed, aggregate_norm = strategy_scores(
        atoms=atoms, cosine=cosine, weights=weights, step_norm=step_norm,
        targets=targets, base_logits=base_logits,
    )
    rows = []
    for target, scores in zip(targets, changed, strict=True):
        spec = target["spec"]
        index = spec["global_decision_index"]
        metrics = candidate_distribution_metrics(
            base_scores[index], scores, spec["values"], spec["expert_index"]
        )
        seed_metrics = candidate_distribution_metrics(
            trace_base[index]["normalized_scores"], trace_seed[index]["normalized_scores"],
            spec["values"], spec["expert_index"],
        )
        rows.append(
            {
                "global_decision_index": index,
                "relationship": spec["relationship"],
                **metrics,
                "seed_expected_value_delta": seed_metrics["expected_value_delta"],
                "seed_top_value_delta": seed_metrics["top_value_delta"],
            }
        )
    return {"weights": weights.tolist(), "aggregate_unit_norm": aggregate_norm, "rows": rows}


def utility_for_atoms(
    atoms, cosine, validation, harm, base_logits, base_scores, step_norm
):
    transfer, damage = [], []
    for position in range(len(atoms)):
        weights = np.zeros(len(atoms)); weights[position] = 1.0
        targets = list(validation) + list(harm)
        changed, _ = strategy_scores(
            atoms=atoms, cosine=cosine, weights=weights, step_norm=step_norm,
            targets=targets, base_logits=base_logits,
        )
        local_transfer, local_harm = [], []
        for target, scores in zip(targets, changed, strict=True):
            spec = target["spec"]
            index = spec["global_decision_index"]
            metrics = candidate_distribution_metrics(
                base_scores[index], scores, spec["values"], spec["expert_index"]
            )
            if spec["relationship"] == "transfer_validation":
                local_transfer.append(metrics["expected_value_delta"])
            else:
                local_harm.append(max(0.0, -metrics["expected_value_delta"]))
        transfer.append(float(np.mean(local_transfer)))
        damage.append(float(np.mean(local_harm)))
    return np.asarray(transfer), np.asarray(damage)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Unseen-family test requires CUDA")
    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    value_rows = load_value_rows(args.value_trace, "valid_seen")
    failures = base_failure_indices(args.base_score_trace, value_rows, "valid_seen")
    rng = random.Random(args.sample_seed)
    eligible = [
        index for index, row in value_rows.items()
        if index in failures and row["action_verb"] == "close" and is_value_heterogeneous(row)
    ]
    distinct = one_failure_per_episode(eligible, value_rows, rng)
    unseen_indices = [
        index for index in distinct if value_rows[index]["task_type"] == args.heldout_task_type
    ]
    historical = [
        index for index in distinct if value_rows[index]["task_type"] != args.heldout_task_type
    ]
    chosen = task_round_robin(historical, value_rows, 12, rng)
    source_indices, validation_indices = chosen[:9], chosen[9:]
    used_episodes = {value_rows[index]["episode_key"] for index in distinct}
    protection_candidates = [
        index for index, row in value_rows.items()
        if row["action_verb"] != "close"
        and row["episode_key"] not in used_episodes
        and is_value_heterogeneous(row)
    ]
    protection_distinct = one_failure_per_episode(protection_candidates, value_rows, rng)
    protection_indices = protection_distinct[:6]
    if len(unseen_indices) < 4 or len(protection_indices) < 6:
        raise ValueError("Insufficient unseen-family or protection episodes")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, local_files_only=True
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base_weight = model.model.embed_tokens.weight.detach()
    trace_base, trace_seed = read_trace(args.base_score_trace), read_trace(args.seed_score_trace)
    groups = {
        "historical_source": source_indices,
        "transfer_validation": validation_indices,
        "unseen_family": unseen_indices,
        "harm_validation": protection_indices[:3],
        "protection_test": protection_indices[3:],
    }
    cached_groups = {}
    for relationship, indices in groups.items():
        cached_groups[relationship] = []
        for index in indices:
            spec = state_spec(index, relationship, decisions, value_rows, args.history_window)
            cached_groups[relationship].append(
                cache_state_hidden(
                    backbone=model.model, tokenizer=tokenizer, spec=spec, device=device,
                    candidate_batch_size=args.candidate_batch_size, max_length=args.max_length,
                )
            )
            print(f"cached {relationship} {index}", flush=True)
    all_cached = [item for values in cached_groups.values() for item in values]
    base_logits = {
        item["spec"]["global_decision_index"]: torch.nn.functional.linear(
            torch.cat(item["candidate_hidden"], dim=0), base_weight
        ).detach() for item in all_cached
    }
    base_scores = {
        item["spec"]["global_decision_index"]: cached_scores_from_logits(
            base_logits[item["spec"]["global_decision_index"]], item
        ).detach().cpu().double().numpy() for item in all_cached
    }
    historical_atoms = [make_atom(base_weight, item) for item in cached_groups["historical_source"]]
    historical_cosine = cosine_matrix(historical_atoms)
    transfer, harm = utility_for_atoms(
        historical_atoms, historical_cosine, cached_groups["transfer_validation"],
        cached_groups["harm_validation"], base_logits, base_scores, args.step_norm,
    )
    evolved_weights = softmax_weights(transfer - args.harm_lambda * harm, args.temperature)
    zero_shot = {
        "mean9": evaluate(
            historical_atoms, historical_cosine, np.full(9, 1/9),
            cached_groups["unseen_family"] + cached_groups["protection_test"],
            base_logits, base_scores, trace_base, trace_seed, args.step_norm,
        ),
        "evolved9": evaluate(
            historical_atoms, historical_cosine, evolved_weights,
            cached_groups["unseen_family"] + cached_groups["protection_test"],
            base_logits, base_scores, trace_base, trace_seed, args.step_norm,
        ),
    }

    feedback_state = cached_groups["unseen_family"][0]
    inherited_test = cached_groups["unseen_family"][1:]
    adapted_atoms = historical_atoms + [make_atom(base_weight, feedback_state)]
    adapted_cosine = cosine_matrix(adapted_atoms)
    adapted_transfer, adapted_harm = utility_for_atoms(
        adapted_atoms, adapted_cosine, cached_groups["transfer_validation"],
        cached_groups["harm_validation"], base_logits, base_scores, args.step_norm,
    )
    adapted_weights = softmax_weights(
        adapted_transfer - args.harm_lambda * adapted_harm, args.temperature
    )
    after_feedback = evaluate(
        adapted_atoms, adapted_cosine, adapted_weights,
        inherited_test + cached_groups["protection_test"],
        base_logits, base_scores, trace_base, trace_seed, args.step_norm,
    )
    output = {
        "experiment": "unseen_skill_family_output_gradient_transfer_v1",
        "status": "complete",
        "sample_seed": args.sample_seed,
        "heldout_task_type": args.heldout_task_type,
        "skill_action": "close",
        "parameter_delta_l2_norm": args.step_norm,
        "historical_source_indices": source_indices,
        "transfer_validation_indices": validation_indices,
        "unseen_family_indices": unseen_indices,
        "feedback_state_index": feedback_state["spec"]["global_decision_index"],
        "historical_transfer_utility": transfer.tolist(),
        "historical_harm_utility": harm.tolist(),
        "zero_shot": zero_shot,
        "after_first_unseen_failure_feedback": after_feedback,
        "adapted_transfer_utility": adapted_transfer.tolist(),
        "adapted_harm_utility": adapted_harm.tolist(),
        "final_holdout_used_for_weighting": False,
        "protection_test_used_for_weighting": False,
    }
    destination = Path(args.output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
