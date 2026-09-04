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

from probe_output_gradient_evolution import atom_dot, atoms_cosine, make_atom
from probe_output_gradient_scope import aggregate_unit_effect, make_atom_from_logits
from probe_output_head_representation_sufficiency import cache_state_hidden, cached_scores_from_logits
from probe_output_only_h6 import read_trace
from probe_value_gradient_writeback import load_value_rows, state_spec
from self_evolve.alfworld_data import load_decisions
from self_evolve.repair_space import (
    atom_weights_from_coordinates,
    orthonormal_repair_basis,
    sample_unit_directions,
    target_coordinates,
)


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
    parser.add_argument("--sample-count", type=int, default=100_000)
    parser.add_argument("--max-rank", type=int, default=6)
    parser.add_argument(
        "--step-norms", type=float, nargs="+",
        default=(0.18, 0.36, 0.54, 0.72, 1.08, 1.44, 1.8),
    )
    parser.add_argument("--grid-angle-step-degrees", type=float, default=2.0)
    parser.add_argument("--grid-step-norms", type=float, nargs="+", default=(0.72, 1.8))
    parser.add_argument("--candidate-input")
    parser.add_argument("--candidate-eval-batch-size", type=int, default=8)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    return parser.parse_args()


def batched_cached_scores(logits: torch.Tensor, cached: dict[str, Any]) -> torch.Tensor:
    targets = torch.cat(cached["candidate_targets"], dim=0)
    token_scores = (
        logits.gather(2, targets[None, :, None].expand(logits.shape[0], -1, 1)).squeeze(2)
        - torch.logsumexp(logits, dim=-1)
    )
    result, cursor = [], 0
    for candidate_targets in cached["candidate_targets"]:
        length = len(candidate_targets)
        result.append(token_scores[:, cursor : cursor + length].mean(dim=1))
        cursor += length
    return torch.stack(result, dim=1)


def distribution_arrays(
    baseline_scores: np.ndarray,
    updated_scores: np.ndarray,
    values: Sequence[float],
) -> dict[str, np.ndarray]:
    values_array = np.asarray(values, dtype=np.float64)
    base_exp = np.exp(baseline_scores - np.max(baseline_scores))
    base_p = base_exp / base_exp.sum()
    centered = updated_scores - np.max(updated_scores, axis=1, keepdims=True)
    updated_exp = np.exp(centered)
    updated_p = updated_exp / updated_exp.sum(axis=1, keepdims=True)
    optimal = np.isclose(values_array, values_array.max(), rtol=0.0, atol=1e-12)
    base_expected = float(base_p @ values_array)
    expected = updated_p @ values_array
    base_top_value = float(values_array[int(np.argmax(baseline_scores))])
    updated_top_value = values_array[np.argmax(updated_scores, axis=1)]
    return {
        "expected_value_delta": expected - base_expected,
        "top_value_delta": updated_top_value - base_top_value,
        "kl": np.sum(base_p[None, :] * (np.log(base_p)[None, :] - np.log(updated_p)), axis=1),
        "total_variation": 0.5 * np.sum(np.abs(updated_p - base_p[None, :]), axis=1),
        "optimal_mass_delta": updated_p[:, optimal].sum(axis=1) - base_p[optimal].sum(),
    }


def direction_statistics(
    directions: np.ndarray,
    coordinates: np.ndarray,
    target_positions: Sequence[int],
    protection_positions: Sequence[int],
    step_norm: float,
) -> dict[str, np.ndarray]:
    response = step_norm * (directions @ coordinates)
    target = response[:, target_positions]
    protection = response[:, protection_positions]
    return {
        "gain": target.mean(axis=1),
        "min_target": target.min(axis=1),
        "target_positive_rate": (target > 0).mean(axis=1),
        "mean_downside": np.maximum(-protection, 0).mean(axis=1),
        "max_downside": np.maximum(-protection, 0).max(axis=1),
        "protection_negative_rate": (protection < 0).mean(axis=1),
    }


def selected_indices(statistics: dict[str, np.ndarray]) -> list[tuple[int, str]]:
    gain = statistics["gain"]
    mean_downside = statistics["mean_downside"]
    max_downside = statistics["max_downside"]
    selected: list[tuple[int, str]] = [
        (int(np.argmax(gain)), "unconstrained_max_mean_gain"),
        (int(np.argmax(statistics["min_target"])), "maximin_target_gain"),
    ]
    for threshold in (0.0, 0.0025, 0.005, 0.01, 0.02):
        feasible = np.flatnonzero(max_downside <= threshold + 1e-12)
        if len(feasible):
            local = feasible[np.argmax(gain[feasible])]
            selected.append((int(local), f"max_gain_max_downside_le_{threshold:g}"))
    for penalty in (0.5, 2.0, 8.0, 32.0):
        objective = gain - penalty * mean_downside
        selected.append((int(np.argmax(objective)), f"mean_downside_penalty_{penalty:g}"))
        objective = gain - penalty * max_downside
        selected.append((int(np.argmax(objective)), f"max_downside_penalty_{penalty:g}"))
    return selected


def add_candidate(
    candidates: dict[tuple[Any, ...], dict[str, Any]],
    coordinates: np.ndarray,
    rank: int,
    step_norm: float,
    origin: dict[str, Any],
    max_rank: int,
) -> None:
    padded = np.zeros(max_rank, dtype=np.float64)
    padded[:rank] = coordinates
    key = (round(float(step_norm), 10), *np.round(padded, 10).tolist())
    if key not in candidates:
        candidates[key] = {
            "step_norm": float(step_norm),
            "strength_multiple": float(step_norm / 0.0006),
            "active_rank": int(rank),
            "coordinates": padded.tolist(),
            "selection_origins": [],
        }
    candidates[key]["selection_origins"].append(origin)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or min(args.step_norms) <= 0:
        raise ValueError("Repair-space probe requires CUDA and positive step norms")
    scope = json.loads(Path(args.scope_reference_file).read_text(encoding="utf-8"))
    unseen = json.loads(Path(args.unseen_family_file).read_text(encoding="utf-8"))
    sample_seed = int(scope["sample_seed"])
    if sample_seed != int(unseen["sample_seed"]):
        raise ValueError("Scope and unseen-family seeds differ")
    groups = {key: [int(index) for index in value] for key, value in scope["groups"].items()}
    panel_indices = [index for name in groups for index in groups[name]]
    source_indices = groups["source"]
    feedback_index = int(unseen["feedback_state_index"])
    needed_indices = list(dict.fromkeys(panel_indices + [feedback_index]))
    relationship_by_index = {
        index: relationship for relationship, indices in groups.items() for index in indices
    }

    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    value_rows = load_value_rows(args.value_trace, "valid_seen")
    trace_base = read_trace(args.base_score_trace)
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

    cached: dict[int, dict[str, Any]] = {}
    for index in needed_indices:
        relationship = relationship_by_index.get(index, "heldout_feedback")
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
        for index in panel_indices
    )
    if trace_error > 1e-4:
        raise AssertionError(f"Base trace reproduction error {trace_error}")

    source_atoms = [make_atom(base_weight, cached[index]) for index in source_indices]
    source_cosine = atoms_cosine(source_atoms)
    basis = orthonormal_repair_basis(source_cosine)
    max_rank = min(args.max_rank, basis.atom_coefficients.shape[1])
    if max_rank < args.max_rank:
        raise RuntimeError(f"Only {max_rank} non-degenerate repair directions")
    panel_atoms = [make_atom_from_logits(cached[index], base_logits[index]) for index in panel_indices]
    source_target_dots = np.asarray([
        [atom_dot(source, target) / source["norm"] for target in panel_atoms]
        for source in source_atoms
    ], dtype=np.float64)
    coordinates = target_coordinates(basis, source_target_dots)[:max_rank]
    target_norms = np.asarray([atom["norm"] for atom in panel_atoms], dtype=np.float64)
    base_expected = []
    for index in panel_indices:
        values = np.asarray(cached[index]["spec"]["values"], dtype=np.float64)
        scores = base_scores[index]
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        base_expected.append(float(probabilities @ values))
    base_expected_array = np.asarray(base_expected)
    positions = {
        name: [panel_indices.index(index) for index in indices] for name, indices in groups.items()
    }
    protocols = {
        "validation_only": {
            "target": positions["transfer_validation"],
            "protection": positions["harm_validation"],
        },
        "full_panel_oracle": {
            "target": positions["transfer_validation"] + positions["same_skill_holdout"],
            "protection": positions["harm_validation"] + positions["protection_test"],
        },
    }

    if args.candidate_input:
        candidate_payload = json.loads(Path(args.candidate_input).read_text(encoding="utf-8"))
        candidates = candidate_payload["candidates"]
        for candidate in candidates:
            coordinate = np.asarray(candidate["coordinates"], dtype=np.float64)
            if coordinate.shape != (max_rank,) or abs(np.linalg.norm(coordinate) - 1.0) > 1e-7:
                raise ValueError("Candidate-input directions must be unit vectors of max-rank length")
    else:
        candidate_map: dict[tuple[Any, ...], dict[str, Any]] = {}
        for rank in range(1, max_rank + 1):
            directions = sample_unit_directions(rank, args.sample_count, sample_seed + 1009 * rank)
            for step_norm in args.step_norms:
                for protocol, split_positions in protocols.items():
                    statistics = direction_statistics(
                        directions,
                        coordinates[:rank],
                        split_positions["target"],
                        split_positions["protection"],
                        step_norm,
                    )
                    for selected, label in selected_indices(statistics):
                        add_candidate(
                            candidate_map,
                            directions[selected], rank, step_norm,
                            {
                                "kind": "first_order_search",
                                "protocol": protocol,
                                "criterion": label,
                                "predicted_mean_target_delta": float(statistics["gain"][selected]),
                                "predicted_max_protection_downside": float(statistics["max_downside"][selected]),
                                "predicted_mean_protection_downside": float(statistics["mean_downside"][selected]),
                            },
                            max_rank,
                        )
            print(f"candidate search rank={rank}/{max_rank}", flush=True)

        grid_angles = np.arange(0.0, 360.0, args.grid_angle_step_degrees)
        grid_directions = np.stack([
            np.cos(np.deg2rad(grid_angles)), np.sin(np.deg2rad(grid_angles))
        ], axis=1)
        for step_norm in args.grid_step_norms:
            for angle, direction in zip(grid_angles, grid_directions, strict=True):
                add_candidate(
                    candidate_map, direction, 2, step_norm,
                    {"kind": "rank2_connectivity_grid", "angle_degrees": float(angle)},
                    max_rank,
                )
        candidates = list(candidate_map.values())
    for candidate_id, candidate in enumerate(candidates):
        candidate["candidate_id"] = candidate_id
    print(f"evaluating {len(candidates)} unique directions/doses", flush=True)
    candidate_coordinates = np.asarray([item["coordinates"] for item in candidates])
    candidate_steps = np.asarray([item["step_norm"] for item in candidates])
    metric_names = ("expected_value_delta", "top_value_delta", "kl", "total_variation", "optimal_mass_delta")
    matrices = {
        name: np.empty((len(candidates), len(panel_indices)), dtype=np.float64)
        for name in metric_names
    }
    basis_atom_coefficients = basis.atom_coefficients[:, :max_rank]
    for panel_position, index in enumerate(panel_indices):
        target = cached[index]
        target_hidden = torch.cat(target["candidate_hidden"], dim=0)
        basis_effects = []
        for component in range(max_rank):
            atom_weights = basis_atom_coefficients[:, component]
            norm_check = math.sqrt(float(atom_weights @ source_cosine @ atom_weights))
            if abs(norm_check - 1.0) > 1e-7:
                raise AssertionError(f"Basis norm drift {norm_check}")
            basis_effects.append(aggregate_unit_effect(
                target_hidden, source_atoms, atom_weights, norm_check
            ))
        basis_effect = torch.stack(basis_effects)
        for start in range(0, len(candidates), args.candidate_eval_batch_size):
            end = min(start + args.candidate_eval_batch_size, len(candidates))
            coefficients = torch.as_tensor(
                candidate_coordinates[start:end], dtype=basis_effect.dtype, device=device
            )
            effect = torch.einsum("br,rtv->btv", coefficients, basis_effect)
            steps = torch.as_tensor(
                candidate_steps[start:end], dtype=basis_effect.dtype, device=device
            )
            changed_logits = base_logits[index][None, :, :] - steps[:, None, None] * effect
            changed_scores = batched_cached_scores(changed_logits, target).detach().cpu().double().numpy()
            local = distribution_arrays(base_scores[index], changed_scores, target["spec"]["values"])
            for name in metric_names:
                matrices[name][start:end, panel_position] = local[name]
        del basis_effect, basis_effects
        torch.cuda.empty_cache()
        print(f"evaluated panel {panel_position + 1}/{len(panel_indices)}", flush=True)

    weights_by_k = [np.asarray(value, dtype=np.float64) for value in scope["weights_by_experience_count"]]
    trajectory = []
    for experience_count, weights in enumerate(weights_by_k, 1):
        aggregate_norm = math.sqrt(float(weights @ source_cosine @ weights))
        atom_weights = weights / aggregate_norm
        trajectory_coordinates = basis.atom_coefficients[:, :max_rank].T @ source_cosine @ atom_weights
        trajectory.append({
            "experience_count": experience_count,
            "coordinates": trajectory_coordinates.tolist(),
            "captured_norm_squared": float(trajectory_coordinates @ trajectory_coordinates),
        })
    feedback_atom = make_atom_from_logits(cached[feedback_index], base_logits[feedback_index])
    source_feedback_cosine = np.asarray([
        atom_dot(source, feedback_atom) / source["norm"] / feedback_atom["norm"]
        for source in source_atoms
    ])
    feedback_coordinates = basis.atom_coefficients[:, :max_rank].T @ source_feedback_cosine
    feedback_projection = {
        "feedback_state_index": feedback_index,
        "coordinates": feedback_coordinates.tolist(),
        "captured_norm_squared": float(feedback_coordinates @ feedback_coordinates),
        "residual_novelty": float(math.sqrt(max(0.0, 1.0 - feedback_coordinates @ feedback_coordinates))),
    }
    per_state_optimal = []
    for panel_position, index in enumerate(panel_indices):
        local = coordinates[:max_rank, panel_position]
        projected_norm = float(np.linalg.norm(local))
        per_state_optimal.append({
            "global_decision_index": index,
            "relationship": relationship_by_index[index],
            "coordinates": (local / projected_norm).tolist() if projected_norm else local.tolist(),
            "projection_fraction": projected_norm / target_norms[panel_position],
        })

    output = {
        "experiment": "repair_space_characterization_probe_v1",
        "status": "complete",
        "sample_seed": sample_seed,
        "scope_reference_file": args.scope_reference_file,
        "unseen_family_file": args.unseen_family_file,
        "source_indices": source_indices,
        "panel_indices": panel_indices,
        "groups": groups,
        "positions": positions,
        "base_expected_value": base_expected_array.tolist(),
        "base_trace_max_absolute_error": trace_error,
        "basis": {
            "definition": "uncentered PCA-ordered orthonormal basis of 12 unit source gradients",
            "eigenvalues": basis.eigenvalues[:max_rank].tolist(),
            "explained_gram_trace_fraction": (
                np.cumsum(basis.eigenvalues[:max_rank]) / np.sum(basis.eigenvalues)
            ).tolist(),
            "target_coordinates": coordinates.tolist(),
        },
        "protocols": protocols,
        "search": {
            "sample_count_per_rank": args.sample_count,
            "step_norms": list(args.step_norms),
            "strength_multiples": [value / 0.0006 for value in args.step_norms],
            "rank2_grid_angle_step_degrees": args.grid_angle_step_degrees,
            "rank2_grid_step_norms": list(args.grid_step_norms),
            "candidate_input": args.candidate_input,
        },
        "candidates": candidates,
        "trajectory": trajectory,
        "heldout_feedback_projection": feedback_projection,
        "per_state_projected_optimal_directions": per_state_optimal,
        "matrix_file": args.matrix_file,
        "selection_boundary": {
            "validation_only": "direction candidates selected from transfer_validation and harm_validation only",
            "full_panel_oracle": "optimistic finite-panel existence upper bound; not a learnable evaluation",
            "nonlinear_evaluation": "all reported final values use exact multi-token softmax candidate scoring",
        },
    }
    destination = Path(args.output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    matrix_path = Path(args.matrix_file)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        matrix_path,
        candidate_id=np.arange(len(candidates)),
        candidate_coordinates=candidate_coordinates,
        candidate_step_norm=candidate_steps,
        panel_index=np.asarray(panel_indices),
        relationship=np.asarray([relationship_by_index[index] for index in panel_indices]),
        base_expected_value=base_expected_array,
        target_basis_coordinates=coordinates,
        target_gradient_norm=target_norms,
        **matrices,
    )
    print(f"wrote {destination} and {matrix_path}", flush=True)


if __name__ == "__main__":
    main()
