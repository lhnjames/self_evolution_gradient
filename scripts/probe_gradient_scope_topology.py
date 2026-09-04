#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_output_gradient_evolution import atom_dot, make_atom
from probe_output_gradient_scope import aggregate_unit_effect, make_atom_from_logits
from probe_output_head_representation_sufficiency import cache_state_hidden, cached_scores_from_logits
from probe_output_only_h6 import read_trace
from probe_value_gradient_writeback import load_value_rows, state_spec
from self_evolve.alfworld_data import load_decisions
from self_evolve.gradient_scope import residual_novelty_from_gram, scope_label
from self_evolve.value_writeback import candidate_distribution_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-file", required=True)
    parser.add_argument("--value-trace", required=True)
    parser.add_argument("--base-score-trace", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--scope-reference-file", required=True)
    parser.add_argument("--unseen-family-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--matrix-file", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--step-norms", type=float, nargs="+", default=(0.18, 0.72))
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    return parser.parse_args()


def gram(atoms: Sequence[dict[str, Any]]) -> np.ndarray:
    result = np.empty((len(atoms), len(atoms)), dtype=np.float64)
    for left in range(len(atoms)):
        for right in range(left, len(atoms)):
            value = atom_dot(atoms[left], atoms[right])
            result[left, right] = result[right, left] = value
    return result


def vector_gram(vectors: Sequence[torch.Tensor]) -> np.ndarray:
    stacked = torch.stack(vectors).double()
    return torch.matmul(stacked, stacked.transpose(0, 1)).cpu().numpy()


def cosine_from_gram(matrix: np.ndarray) -> np.ndarray:
    norms = np.sqrt(np.clip(np.diag(matrix), 1e-300, None))
    result = np.clip(matrix / np.outer(norms, norms), -1.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result


def scope_jaccard(left: Sequence[float], right: Sequence[float], epsilon: float) -> float:
    left_positive = {i for i, value in enumerate(left) if scope_label(value, epsilon) == "positive"}
    right_positive = {i for i, value in enumerate(right) if scope_label(value, epsilon) == "positive"}
    union = left_positive | right_positive
    return len(left_positive & right_positive) / len(union) if union else 1.0


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or min(args.step_norms) <= 0:
        raise ValueError("Topology probe requires CUDA and positive step norms")
    scope_reference = json.loads(Path(args.scope_reference_file).read_text(encoding="utf-8"))
    unseen = json.loads(Path(args.unseen_family_file).read_text(encoding="utf-8"))
    sample_seed = int(scope_reference["sample_seed"])
    if sample_seed != int(unseen["sample_seed"]):
        raise ValueError("Scope and unseen-family seeds differ")
    groups = {key: [int(index) for index in value] for key, value in scope_reference["groups"].items()}
    close_indices = groups["source"] + groups["transfer_validation"] + groups["same_skill_holdout"]
    panel_indices = [index for name in groups for index in groups[name]]
    historical_indices = [int(index) for index in unseen["historical_source_indices"]]
    feedback_index = int(unseen["feedback_state_index"])
    feedback_target_indices = sorted({
        int(row["global_decision_index"])
        for row in unseen["zero_shot"]["evolved9"]["rows"]
    })
    needed_indices = list(dict.fromkeys(
        panel_indices + historical_indices + [feedback_index] + feedback_target_indices
    ))

    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    value_rows = load_value_rows(args.value_trace, "valid_seen")
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
    trace_base = read_trace(args.base_score_trace)

    relationship_by_index = {
        index: relationship for relationship, indices in groups.items() for index in indices
    }
    cached: dict[int, dict[str, Any]] = {}
    for index in needed_indices:
        relationship = relationship_by_index.get(index, "heldout_feedback_auxiliary")
        cached[index] = cache_state_hidden(
            backbone=model.model,
            tokenizer=tokenizer,
            spec=state_spec(index, relationship, decisions, value_rows, args.history_window),
            device=device,
            candidate_batch_size=args.candidate_batch_size,
            max_length=args.max_length,
        )
        print(f"cached {relationship} {index}", flush=True)
    base_logits = {
        index: torch.nn.functional.linear(torch.cat(item["candidate_hidden"], dim=0), base_weight).detach()
        for index, item in cached.items()
    }
    base_scores = {
        index: cached_scores_from_logits(base_logits[index], item).detach().cpu().double().numpy()
        for index, item in cached.items()
    }
    trace_error = max(
        float(np.max(np.abs(base_scores[index] - np.asarray(trace_base[index]["normalized_scores"]))))
        for index in cached
    )
    if trace_error > 1e-4:
        raise AssertionError(f"Base trace reproduction error {trace_error}")

    atom_indices = list(dict.fromkeys(close_indices + historical_indices + [feedback_index] + panel_indices))
    atoms = {
        index: make_atom_from_logits(cached[index], base_logits[index]) for index in atom_indices
    }
    close_atoms = [atoms[index] for index in close_indices]
    gradient_gram = gram(close_atoms)
    gradient_cosine = cosine_from_gram(gradient_gram)
    delta_signatures = [atom["delta"].mean(dim=0) for atom in close_atoms]
    hidden_signatures = [atom["hidden"].mean(dim=0) for atom in close_atoms]
    delta_gram = vector_gram(delta_signatures)
    hidden_gram = vector_gram(hidden_signatures)
    delta_cosine = cosine_from_gram(delta_gram)
    hidden_cosine = cosine_from_gram(hidden_gram)

    transfer_cube = np.empty(
        (len(close_indices), len(args.step_norms), len(panel_indices)), dtype=np.float64
    )
    transfer_rows = []
    for source_position, source_index in enumerate(close_indices):
        atom = atoms[source_index]
        for target_position, target_index in enumerate(panel_indices):
            target = cached[target_index]
            effect = aggregate_unit_effect(
                torch.cat(target["candidate_hidden"], dim=0), [atom], np.ones(1), 1.0
            )
            target_atom = atoms[target_index]
            first_order_dot = atom_dot(atom, target_atom) / atom["norm"]
            gradient_pair_cosine = first_order_dot / target_atom["norm"]
            for dose_position, step_norm in enumerate(args.step_norms):
                changed = cached_scores_from_logits(
                    base_logits[target_index] - step_norm * effect, target
                ).detach().cpu().double().numpy()
                metrics = candidate_distribution_metrics(
                    base_scores[target_index], changed,
                    target["spec"]["values"], target["spec"]["expert_index"],
                )
                transfer_cube[source_position, dose_position, target_position] = metrics["expected_value_delta"]
                transfer_rows.append({
                    "source_index": source_index,
                    "target_index": target_index,
                    "source_panel": relationship_by_index[source_index],
                    "target_panel": relationship_by_index[target_index],
                    "step_norm": float(step_norm),
                    "strength_multiple": float(step_norm / 0.0006),
                    "gradient_cosine": float(gradient_pair_cosine),
                    "first_order_expected_value_delta": float(step_norm * first_order_dot),
                    "logit_influence_l2": float(step_norm * torch.linalg.vector_norm(effect).item()),
                    **metrics,
                })
        print(f"source {source_position + 1}/{len(close_indices)} complete", flush=True)

    source_sequence_positions = [close_indices.index(index) for index in groups["source"]]
    sequence_novelty = []
    for order in range(1, len(source_sequence_positions)):
        current_position = source_sequence_positions[order]
        previous = source_sequence_positions[:order]
        previous_gram = gradient_gram[np.ix_(previous, previous)]
        cross = gradient_gram[previous, current_position]
        residual = residual_novelty_from_gram(
            previous_gram, cross, gradient_gram[current_position, current_position]
        )
        max_cosine = float(np.max(gradient_cosine[previous, current_position]))
        sequence_novelty.append({
            "experience_count": order + 1,
            "new_source_index": close_indices[current_position],
            "span_residual_novelty": residual,
            "cosine_novelty": 1.0 - max_cosine,
            "maximum_previous_gradient_cosine": max_cosine,
        })

    feedback_group = historical_indices + [feedback_index]
    feedback_atoms = [atoms[index] for index in feedback_group]
    feedback_gradient_gram = gram(feedback_atoms)
    feedback_gradient_cosine = cosine_from_gram(feedback_gradient_gram)
    feedback_delta_gram = vector_gram([atom["delta"].mean(dim=0) for atom in feedback_atoms])
    feedback_hidden_gram = vector_gram([atom["hidden"].mean(dim=0) for atom in feedback_atoms])
    feedback_position = len(historical_indices)

    def novelty(matrix: np.ndarray) -> dict[str, float]:
        previous = matrix[:-1, :-1]
        cross = matrix[:-1, -1]
        residual = residual_novelty_from_gram(previous, cross, matrix[-1, -1])
        cosine = cosine_from_gram(matrix)
        return {
            "span_residual_novelty": residual,
            "maximum_historical_cosine": float(np.max(cosine[:-1, -1])),
            "cosine_novelty": float(1.0 - np.max(cosine[:-1, -1])),
        }

    historical_weights = np.asarray(unseen["zero_shot"]["evolved9"]["weights"], dtype=np.float64)
    historical_cosine = feedback_gradient_cosine[:-1, :-1]
    historical_norm = math.sqrt(float(historical_weights @ historical_cosine @ historical_weights))
    evolved_feedback_cosine = float(
        historical_weights @ feedback_gradient_cosine[:-1, -1] / historical_norm
    )
    before_rows = {
        int(row["global_decision_index"]): row
        for row in unseen["zero_shot"]["evolved9"]["rows"]
    }
    after_rows = {
        int(row["global_decision_index"]): row
        for row in unseen["after_first_unseen_failure_feedback"]["rows"]
    }
    common_feedback_targets = sorted(before_rows.keys() & after_rows.keys())
    feedback_atom = atoms[feedback_index]
    individual_feedback_rows = []
    for target_index in common_feedback_targets:
        target = cached[target_index]
        effect = aggregate_unit_effect(
            torch.cat(target["candidate_hidden"], dim=0), [feedback_atom], np.ones(1), 1.0
        )
        changed = cached_scores_from_logits(
            base_logits[target_index] - 0.72 * effect, target
        ).detach().cpu().double().numpy()
        metrics = candidate_distribution_metrics(
            base_scores[target_index], changed,
            target["spec"]["values"], target["spec"]["expert_index"],
        )
        individual_feedback_rows.append({
            "target_index": target_index,
            "relationship": before_rows[target_index]["relationship"],
            **metrics,
        })
    before_values = [before_rows[index]["expected_value_delta"] for index in common_feedback_targets]
    feedback_values = [
        next(row["expected_value_delta"] for row in individual_feedback_rows if row["target_index"] == index)
        for index in common_feedback_targets
    ]
    merged_change = [
        after_rows[index]["expected_value_delta"] - before_rows[index]["expected_value_delta"]
        for index in common_feedback_targets
    ]
    feedback_novelty = {
        "feedback_state_index": feedback_index,
        "historical_source_indices": historical_indices,
        "full_gradient": novelty(feedback_gradient_gram),
        "delta_signature": novelty(feedback_delta_gram),
        "hidden_signature": novelty(feedback_hidden_gram),
        "cosine_with_historical_evolved_direction": evolved_feedback_cosine,
        "common_target_indices": common_feedback_targets,
        "individual_feedback_rows": individual_feedback_rows,
        "scope_novelty_epsilon_0.01": 1.0 - scope_jaccard(before_values, feedback_values, 0.01),
        "mean_merged_feedback_effect_on_common_targets": float(np.mean(merged_change)),
    }

    output = {
        "experiment": "gradient_scope_topology_probe_v1",
        "status": "complete",
        "sample_seed": sample_seed,
        "step_norms": list(args.step_norms),
        "close_indices": close_indices,
        "panel_indices": panel_indices,
        "panel_groups": groups,
        "base_trace_max_absolute_error": trace_error,
        "gradient_gram": gradient_gram.tolist(),
        "gradient_cosine": gradient_cosine.tolist(),
        "delta_signature_definition": "mean over all candidate action token-position logit gradients",
        "delta_signature_gram": delta_gram.tolist(),
        "delta_signature_cosine": delta_cosine.tolist(),
        "hidden_signature_definition": "mean over all candidate action token-position hidden states",
        "hidden_signature_gram": hidden_gram.tolist(),
        "hidden_signature_cosine": hidden_cosine.tolist(),
        "sequence_novelty": sequence_novelty,
        "heldout_feedback_novelty": feedback_novelty,
        "transfer_rows": transfer_rows,
        "matrix_file": args.matrix_file,
    }
    destination = Path(args.output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    matrix_path = Path(args.matrix_file)
    np.savez_compressed(
        matrix_path,
        close_index=np.asarray(close_indices),
        panel_index=np.asarray(panel_indices),
        step_norm=np.asarray(args.step_norms),
        empirical_transfer=transfer_cube,
        gradient_gram=gradient_gram,
        gradient_cosine=gradient_cosine,
        delta_signature_gram=delta_gram,
        delta_signature_cosine=delta_cosine,
        hidden_signature_gram=hidden_gram,
        hidden_signature_cosine=hidden_cosine,
    )
    print(f"wrote {destination} and {matrix_path}", flush=True)


if __name__ == "__main__":
    main()
