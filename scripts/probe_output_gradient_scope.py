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

from probe_output_gradient_evolution import atom_dot, atoms_cosine, make_atom, softmax_weights
from probe_output_head_representation_sufficiency import cache_state_hidden, cached_scores_from_logits
from probe_output_only_h6 import read_trace
from probe_skill_gradient_purification import base_failure_indices
from probe_value_gradient_writeback import load_value_rows, state_spec
from self_evolve.alfworld_data import load_decisions
from self_evolve.gradient_scope import direction_cosine, scope_profile, scope_transition
from self_evolve.output_head_oracle import fixed_oracle_split
from self_evolve.value_writeback import candidate_distribution_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-file", required=True)
    parser.add_argument("--value-trace", required=True)
    parser.add_argument("--base-score-trace", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--matrix-file", required=True)
    parser.add_argument("--reference-evolution-file")
    parser.add_argument("--device", required=True)
    parser.add_argument("--sample-seed", type=int, required=True)
    parser.add_argument("--skill", default="close", choices=("go", "open", "close"))
    parser.add_argument("--selection-step-norm", type=float, default=0.72)
    parser.add_argument(
        "--step-norms", type=float, nargs="+", default=(0.18, 0.36, 0.54, 0.72, 1.8)
    )
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--harm-lambda", type=float, default=1.0)
    parser.add_argument("--protection-count", type=int, default=12)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    return parser.parse_args()


def make_atom_from_logits(cached: dict[str, Any], logits: torch.Tensor) -> dict[str, Any]:
    hidden = torch.cat(cached["candidate_hidden"], dim=0)
    variable_logits = logits.detach().requires_grad_(True)
    scores = cached_scores_from_logits(variable_logits, cached)
    values = torch.as_tensor(cached["spec"]["values"], dtype=scores.dtype, device=scores.device)
    loss = -torch.dot(torch.softmax(scores, dim=0), values)
    delta = torch.autograd.grad(loss, variable_logits)[0].detach()
    norm_squared = torch.sum(
        torch.matmul(delta, delta.transpose(0, 1))
        * torch.matmul(hidden, hidden.transpose(0, 1))
    )
    norm = math.sqrt(max(float(norm_squared.item()), 0.0))
    if not math.isfinite(norm) or norm <= 1e-20:
        raise RuntimeError("Invalid target atom norm")
    return {"hidden": hidden, "delta": delta, "norm": norm}


def aggregate_unit_effect(
    target_hidden: torch.Tensor,
    atoms: Sequence[dict[str, Any]],
    weights: np.ndarray,
    aggregate_norm: float,
) -> torch.Tensor:
    effect = None
    for weight, atom in zip(weights, atoms, strict=True):
        if weight == 0:
            continue
        local = torch.matmul(
            torch.matmul(target_hidden, atom["hidden"].transpose(0, 1)), atom["delta"]
        )
        local.mul_(float(weight / atom["norm"]))
        effect = local if effect is None else effect.add_(local)
    if effect is None:
        raise RuntimeError("Aggregate effect vanished")
    return effect / aggregate_norm


def source_utility(
    atoms: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    base_logits: dict[int, torch.Tensor],
    base_scores: dict[int, np.ndarray],
    step_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    transfer, harm = [], []
    for atom in atoms:
        benefits, damages = [], []
        for target in targets:
            spec = target["spec"]
            index = spec["global_decision_index"]
            effect = aggregate_unit_effect(
                torch.cat(target["candidate_hidden"], dim=0), [atom], np.ones(1), 1.0
            )
            changed = cached_scores_from_logits(
                base_logits[index] - step_norm * effect, target
            ).detach().cpu().double().numpy()
            metrics = candidate_distribution_metrics(
                base_scores[index], changed, spec["values"], spec["expert_index"]
            )
            if spec["relationship"] == "transfer_validation":
                benefits.append(float(metrics["expected_value_delta"]))
            else:
                damages.append(max(0.0, -float(metrics["expected_value_delta"])))
        transfer.append(float(np.mean(benefits)))
        harm.append(float(np.mean(damages)))
    return np.asarray(transfer), np.asarray(harm)


def base_features(scores: np.ndarray, values: Sequence[float]) -> dict[str, float]:
    centered = scores - scores.max()
    probabilities = np.exp(centered) / np.exp(centered).sum()
    sorted_scores = np.sort(scores)
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    return {
        "base_score_margin": float(sorted_scores[-1] - sorted_scores[-2]) if len(scores) > 1 else 0.0,
        "base_entropy": float(-np.sum(probabilities * np.log(probabilities.clip(1e-300)))),
        "value_gap": float(sorted_values[-1] - sorted_values[-2]) if len(values) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or min(args.step_norms) <= 0 or args.selection_step_norm <= 0:
        raise ValueError("Scope tomography requires CUDA and positive step norms")
    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    value_rows = load_value_rows(args.value_trace, "valid_seen")
    failures = base_failure_indices(args.base_score_trace, value_rows, "valid_seen")
    split = fixed_oracle_split(
        rows=value_rows,
        failure_indices=failures,
        verbs=(args.skill,),
        source_count=12,
        validation_count=3,
        test_count=3,
        protection_count_per_verb=args.protection_count,
        seed=args.sample_seed,
    )
    source_indices = split["by_skill"][args.skill]["source"]
    validation_indices = split["by_skill"][args.skill]["transfer_validation"]
    holdout_indices = split["by_skill"][args.skill]["final_holdout"]
    protection_indices = [
        int(item["global_decision_index"])
        for item in split["protection"]
        if item["matched_source_skill"] == args.skill
    ]
    groups = {
        "source": source_indices,
        "transfer_validation": validation_indices,
        "same_skill_holdout": holdout_indices,
        "harm_validation": protection_indices[:3],
        "protection_test": protection_indices[3:],
    }

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

    cached_groups: dict[str, list[dict[str, Any]]] = {}
    for relationship, indices in groups.items():
        cached_groups[relationship] = []
        for index in indices:
            spec = state_spec(index, relationship, decisions, value_rows, args.history_window)
            cached_groups[relationship].append(cache_state_hidden(
                backbone=model.model,
                tokenizer=tokenizer,
                spec=spec,
                device=device,
                candidate_batch_size=args.candidate_batch_size,
                max_length=args.max_length,
            ))
            print(f"cached {relationship} {index}", flush=True)

    panel = [item for name in groups for item in cached_groups[name]]
    base_logits = {
        item["spec"]["global_decision_index"]: torch.nn.functional.linear(
            torch.cat(item["candidate_hidden"], dim=0), base_weight
        ).detach()
        for item in panel
    }
    base_scores = {
        item["spec"]["global_decision_index"]: cached_scores_from_logits(
            base_logits[item["spec"]["global_decision_index"]], item
        ).detach().cpu().double().numpy()
        for item in panel
    }
    trace_error = max(
        float(np.max(np.abs(base_scores[index] - np.asarray(trace_base[index]["normalized_scores"]))))
        for index in base_scores
    )
    if trace_error > 1e-4:
        raise AssertionError(f"Base trace reproduction error {trace_error}")

    atoms = [make_atom(base_weight, item) for item in cached_groups["source"]]
    cosine = atoms_cosine(atoms)
    selection_targets = cached_groups["transfer_validation"] + cached_groups["harm_validation"]
    transfer, harm = source_utility(
        atoms, selection_targets, base_logits, base_scores, args.selection_step_norm
    )

    weights_by_k: list[np.ndarray] = []
    aggregate_norms = []
    for count in range(1, len(atoms) + 1):
        local_fitness = transfer[:count] - args.harm_lambda * harm[:count]
        weights = softmax_weights(local_fitness, args.temperature)
        padded = np.zeros(len(atoms), dtype=np.float64)
        padded[:count] = weights
        weights_by_k.append(padded)
        aggregate_norms.append(math.sqrt(max(float(padded @ cosine @ padded), 0.0)))

    reference_check: dict[str, Any] | None = None
    if args.reference_evolution_file:
        reference = json.loads(Path(args.reference_evolution_file).read_text(encoding="utf-8"))
        if reference["source_indices"] != source_indices:
            raise AssertionError("Reference evolution source indices differ")
        reference_weights = np.asarray([
            item["weights"] + [0.0] * (len(atoms) - len(item["weights"]))
            for item in reference["sequential_transfer_harm_weighted"]
        ])
        observed_weights = np.asarray(weights_by_k)
        reference_check = {
            "path": args.reference_evolution_file,
            "max_absolute_weight_error": float(np.max(np.abs(reference_weights - observed_weights))),
        }
        if reference_check["max_absolute_weight_error"] > 1e-10:
            raise AssertionError(f"Evolution direction mismatch: {reference_check}")

    target_atoms = [make_atom_from_logits(item, base_logits[item["spec"]["global_decision_index"]]) for item in panel]
    source_target_dot = np.asarray([
        [atom_dot(source, target) for target in target_atoms] for source in atoms
    ], dtype=np.float64)
    source_hidden_means = [atom["hidden"].mean(dim=0) for atom in atoms]
    target_hidden_means = [atom["hidden"].mean(dim=0) for atom in target_atoms]
    hidden_cosines = np.empty((len(atoms), len(panel)), dtype=np.float64)
    for source_position, source_hidden in enumerate(source_hidden_means):
        for target_position, target_hidden in enumerate(target_hidden_means):
            denominator = torch.linalg.vector_norm(source_hidden) * torch.linalg.vector_norm(target_hidden)
            hidden_cosines[source_position, target_position] = float(
                (torch.dot(source_hidden, target_hidden) / denominator.clamp_min(1e-30)).item()
            )

    rows = []
    value_cube = np.empty((len(atoms), len(args.step_norms), len(panel)), dtype=np.float64)
    compatibility_matrix = np.empty((len(atoms), len(panel)), dtype=np.float64)
    influence_matrix = np.empty((len(atoms), len(panel)), dtype=np.float64)
    for count, (weights, aggregate_norm) in enumerate(zip(weights_by_k, aggregate_norms, strict=True), 1):
        aggregate_dots = weights @ source_target_dot / aggregate_norm
        aggregate_cosines = aggregate_dots / np.asarray([atom["norm"] for atom in target_atoms])
        compatibility_matrix[count - 1] = aggregate_cosines
        for target_position, (target, target_atom) in enumerate(zip(panel, target_atoms, strict=True)):
            spec = target["spec"]
            index = spec["global_decision_index"]
            target_hidden = torch.cat(target["candidate_hidden"], dim=0)
            unit_effect = aggregate_unit_effect(target_hidden, atoms, weights, aggregate_norm)
            unit_influence = float(torch.linalg.vector_norm(unit_effect).item())
            influence_matrix[count - 1, target_position] = unit_influence
            features = base_features(base_scores[index], spec["values"])
            for dose_position, step_norm in enumerate(args.step_norms):
                changed_scores = cached_scores_from_logits(
                    base_logits[index] - step_norm * unit_effect, target
                ).detach().cpu().double().numpy()
                metrics = candidate_distribution_metrics(
                    base_scores[index], changed_scores, spec["values"], spec["expert_index"]
                )
                value_cube[count - 1, dose_position, target_position] = metrics["expected_value_delta"]
                rows.append({
                    "experience_count": count,
                    "step_norm": float(step_norm),
                    "strength_multiple": float(step_norm / 0.0006),
                    "global_decision_index": index,
                    "relationship": spec["relationship"],
                    "target_action_verb": spec["action_verb"],
                    "target_task_type": value_rows[index]["task_type"],
                    "target_episode_key": value_rows[index]["episode_key"],
                    "aggregate_target_gradient_dot": float(aggregate_dots[target_position]),
                    "aggregate_target_gradient_cosine": float(aggregate_cosines[target_position]),
                    "first_order_expected_value_delta": float(step_norm * aggregate_dots[target_position]),
                    "unit_logit_influence_l2": unit_influence,
                    "logit_influence_l2": float(step_norm * unit_influence),
                    "weighted_mean_hidden_cosine": float(weights @ hidden_cosines[:, target_position]),
                    **features,
                    **metrics,
                })
        print(f"tomography K={count}/12 complete", flush=True)

    reference_dose_position = int(np.argmin(np.abs(np.asarray(args.step_norms) - args.selection_step_norm)))
    if abs(args.step_norms[reference_dose_position] - args.selection_step_norm) > 1e-12:
        raise ValueError("selection-step-norm must be present in step-norms")
    scope_profiles: dict[str, Any] = {}
    scope_transitions: dict[str, Any] = {}
    panel_positions = {
        "all": list(range(len(panel))),
        "independent_close": [
            i for i, item in enumerate(panel)
            if item["spec"]["relationship"] in {"transfer_validation", "same_skill_holdout"}
        ],
        "same_skill_holdout": [
            i for i, item in enumerate(panel) if item["spec"]["relationship"] == "same_skill_holdout"
        ],
        "protection_test": [
            i for i, item in enumerate(panel) if item["spec"]["relationship"] == "protection_test"
        ],
    }
    for epsilon in (0.0, 0.01):
        epsilon_key = f"epsilon_{epsilon:g}"
        scope_profiles[epsilon_key] = {}
        scope_transitions[epsilon_key] = {}
        for panel_name, positions in panel_positions.items():
            profiles = []
            transitions = []
            for count in range(1, len(atoms) + 1):
                values = value_cube[count - 1, reference_dose_position, positions]
                profiles.append({"experience_count": count, **scope_profile(values, epsilon)})
                if count > 1:
                    previous = value_cube[count - 2, reference_dose_position, positions]
                    angle = math.degrees(math.acos(direction_cosine(
                        weights_by_k[count - 2], weights_by_k[count - 1], cosine
                    )))
                    transitions.append({
                        "from_experience_count": count - 1,
                        "to_experience_count": count,
                        "gradient_rotation_degrees": angle,
                        **scope_transition(previous.tolist(), values.tolist(), epsilon),
                    })
            scope_profiles[epsilon_key][panel_name] = profiles
            scope_transitions[epsilon_key][panel_name] = transitions

    output = {
        "experiment": "output_gradient_scope_tomography_v1",
        "status": "complete",
        "sample_seed": args.sample_seed,
        "skill": args.skill,
        "formula_boundary": {
            "exact": "output-head logit change under frozen backbone and bias",
            "empirical": "sign and magnitude of expected long-term value change after softmax/action scoring",
        },
        "selection_step_norm": args.selection_step_norm,
        "step_norms": list(args.step_norms),
        "strength_multiples": [value / 0.0006 for value in args.step_norms],
        "temperature": args.temperature,
        "harm_lambda": args.harm_lambda,
        "source_indices": source_indices,
        "groups": groups,
        "panel_state_count": len(panel),
        "panel_relationship_counts": {key: len(value) for key, value in groups.items()},
        "base_trace_max_absolute_error": trace_error,
        "reference_evolution_check": reference_check,
        "transfer_utility_at_selection_dose": transfer.tolist(),
        "harm_utility_at_selection_dose": harm.tolist(),
        "gradient_cosine_matrix": cosine.tolist(),
        "weights_by_experience_count": [value.tolist() for value in weights_by_k],
        "aggregate_unit_norms": aggregate_norms,
        "scope_profiles_at_selection_dose": scope_profiles,
        "scope_transitions_at_selection_dose": scope_transitions,
        "rows": rows,
        "matrix_file": args.matrix_file,
        "final_holdout_used_for_weighting": False,
        "protection_test_used_for_weighting": False,
    }
    destination = Path(args.output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    matrix_path = Path(args.matrix_file)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        matrix_path,
        value_delta=value_cube,
        experience_count=np.arange(1, len(atoms) + 1),
        step_norm=np.asarray(args.step_norms),
        state_index=np.asarray([item["spec"]["global_decision_index"] for item in panel]),
        relationship=np.asarray([item["spec"]["relationship"] for item in panel]),
        gradient_compatibility_cosine=compatibility_matrix,
        unit_logit_influence_l2=influence_matrix,
    )
    print(f"wrote {destination} and {matrix_path}", flush=True)


if __name__ == "__main__":
    main()
