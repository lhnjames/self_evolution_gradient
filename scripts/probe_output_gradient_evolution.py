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

from probe_output_head_representation_sufficiency import (
    cache_state_hidden,
    cached_scores_from_logits,
    state_loss,
)
from probe_output_only_h6 import read_trace
from probe_skill_gradient_purification import base_failure_indices
from probe_value_gradient_writeback import load_value_rows, state_spec
from self_evolve.alfworld_data import load_decisions
from self_evolve.output_head_oracle import fixed_oracle_split
from self_evolve.skill_gradient_purification import cosine_consensus_weights
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
    parser.add_argument("--atom-output-dir")
    parser.add_argument("--device", required=True)
    parser.add_argument("--sample-seed", type=int, required=True)
    parser.add_argument("--skill", default="close", choices=("go", "open", "close"))
    parser.add_argument("--step-norm", type=float, default=0.18)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--harm-lambda", type=float, default=1.0)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    return parser.parse_args()


def make_atom(base_weight: torch.Tensor, cached: dict[str, Any]) -> dict[str, Any]:
    hidden = torch.cat(cached["candidate_hidden"], dim=0)
    logits = torch.nn.functional.linear(hidden, base_weight).detach().requires_grad_(True)
    scores = cached_scores_from_logits(logits, cached)
    loss = state_loss(scores, cached, "value_expectation")
    delta = torch.autograd.grad(loss, logits)[0].detach()
    norm_squared = torch.sum(
        torch.matmul(delta, delta.transpose(0, 1))
        * torch.matmul(hidden, hidden.transpose(0, 1))
    )
    norm = math.sqrt(max(float(norm_squared.item()), 0.0))
    if not math.isfinite(norm) or norm <= 1e-20:
        raise RuntimeError("Invalid atom norm")
    return {"hidden": hidden, "delta": delta, "norm": norm}


def atom_with_hidden(atom: dict[str, Any], hidden: torch.Tensor) -> dict[str, Any]:
    norm_squared = torch.sum(
        torch.matmul(atom["delta"], atom["delta"].transpose(0, 1))
        * torch.matmul(hidden, hidden.transpose(0, 1))
    )
    norm = math.sqrt(max(float(norm_squared.item()), 0.0))
    if not math.isfinite(norm) or norm <= 1e-20:
        raise RuntimeError("Projected atom has invalid norm")
    return {"hidden": hidden, "delta": atom["delta"], "norm": norm}


def atoms_cosine(atoms: Sequence[dict[str, Any]]) -> np.ndarray:
    count = len(atoms)
    dots = np.empty((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left, count):
            value = atom_dot(atoms[left], atoms[right])
            dots[left, right] = dots[right, left] = value
    norms = np.asarray([atom["norm"] for atom in atoms])
    result = np.clip(dots / np.outer(norms, norms), -1.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result


def atom_dot(left: dict[str, Any], right: dict[str, Any]) -> float:
    delta_cross = torch.matmul(left["delta"], right["delta"].transpose(0, 1))
    hidden_cross = torch.matmul(left["hidden"], right["hidden"].transpose(0, 1))
    return float(torch.sum(delta_cross * hidden_cross).item())


def softmax_weights(fitness: Sequence[float], temperature: float) -> np.ndarray:
    values = np.asarray(fitness, dtype=np.float64) / temperature
    values -= values.max()
    result = np.exp(values)
    return result / result.sum()


def summarize_changed_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    baseline = float(np.mean([row["baseline_expected_value"] for row in rows]))
    updated = float(np.mean([row["updated_expected_value"] for row in rows]))
    return {
        "count": len(rows),
        "absolute_expected_value_gain": updated - baseline,
        "relative_expected_value_gain": (updated - baseline) / baseline,
        "positive_transfer_rate": float(np.mean([row["expected_value_delta"] > 0 for row in rows])),
        "top_value_repair_rate": float(np.mean([row["top_value_delta"] > 0 for row in rows])),
        "top_value_harm_rate": float(np.mean([row["top_value_delta"] < 0 for row in rows])),
        "mean_kl": float(np.mean([row["kl_baseline_to_updated"] for row in rows])),
    }


def strategy_scores(
    *,
    atoms: Sequence[dict[str, Any]],
    cosine: np.ndarray,
    weights: np.ndarray,
    step_norm: float,
    targets: Sequence[dict[str, Any]],
    base_logits: dict[int, torch.Tensor],
) -> tuple[list[np.ndarray], float]:
    aggregate_unit_norm = math.sqrt(max(float(weights @ cosine @ weights), 0.0))
    if aggregate_unit_norm <= 1e-20:
        raise RuntimeError("Aggregate gradient vanished")
    scale = -step_norm / aggregate_unit_norm
    results = []
    for target in targets:
        target_hidden = torch.cat(target["candidate_hidden"], dim=0)
        total_delta = torch.zeros_like(base_logits[target["spec"]["global_decision_index"]])
        for weight, atom in zip(weights, atoms, strict=True):
            if weight == 0:
                continue
            total_delta.add_(
                torch.matmul(
                    torch.matmul(target_hidden, atom["hidden"].transpose(0, 1)),
                    atom["delta"],
                ),
                alpha=float(weight / atom["norm"]),
            )
        logits = base_logits[target["spec"]["global_decision_index"]] + scale * total_delta
        results.append(cached_scores_from_logits(logits, target).detach().cpu().double().numpy())
    return results, aggregate_unit_norm


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or min(args.step_norm, args.temperature) <= 0:
        raise ValueError("Evolution probe requires CUDA and positive scales")
    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    value_rows = load_value_rows(args.value_trace, "valid_seen")
    failures = base_failure_indices(args.base_score_trace, value_rows, "valid_seen")
    split = fixed_oracle_split(
        rows=value_rows, failure_indices=failures, verbs=(args.skill,),
        source_count=12, validation_count=3, test_count=3,
        protection_count_per_verb=6, seed=args.sample_seed,
    )
    source_indices = split["by_skill"][args.skill]["source"]
    validation_indices = split["by_skill"][args.skill]["transfer_validation"]
    test_indices = split["by_skill"][args.skill]["final_holdout"]
    all_protection = [
        int(item["global_decision_index"])
        for item in split["protection"] if item["matched_source_skill"] == args.skill
    ]
    harm_validation_indices = all_protection[:3]
    protection_test_indices = all_protection[3:]

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
    trace_seed = read_trace(args.seed_score_trace)

    groups = {
        "source": source_indices,
        "transfer_validation": validation_indices,
        "same_skill_holdout": test_indices,
        "harm_validation": harm_validation_indices,
        "protection_test": protection_test_indices,
    }
    cached_by_group: dict[str, list[dict[str, Any]]] = {}
    for relationship, indices in groups.items():
        cached_by_group[relationship] = []
        for index in indices:
            spec = state_spec(index, relationship, decisions, value_rows, args.history_window)
            cached_by_group[relationship].append(
                cache_state_hidden(
                    backbone=model.model, tokenizer=tokenizer, spec=spec, device=device,
                    candidate_batch_size=args.candidate_batch_size, max_length=args.max_length,
                )
            )
            print(f"cached {relationship} {index}", flush=True)
    all_cached = [item for group in cached_by_group.values() for item in group]
    base_logits = {
        item["spec"]["global_decision_index"]: torch.nn.functional.linear(
            torch.cat(item["candidate_hidden"], dim=0), base_weight
        ).detach()
        for item in all_cached
    }
    base_scores = {
        item["spec"]["global_decision_index"]: cached_scores_from_logits(
            base_logits[item["spec"]["global_decision_index"]], item
        ).detach().cpu().double().numpy()
        for item in all_cached
    }
    atoms = [make_atom(base_weight, item) for item in cached_by_group["source"]]
    atom_manifest = []
    if args.atom_output_dir:
        atom_dir = Path(args.atom_output_dir)
        atom_dir.mkdir(parents=True, exist_ok=True)
        for index, (source_index, atom, cached_source) in enumerate(
            zip(source_indices, atoms, cached_by_group["source"], strict=True)
        ):
            atom_path = atom_dir / f"close_source_{source_index}.pt"
            torch.save(
                {
                    "format": "output_gradient_atom_v1",
                    "base_model_anchor": args.base_model,
                    "sample_seed": args.sample_seed,
                    "skill": args.skill,
                    "source_index": source_index,
                    "hidden": atom["hidden"].detach().to(device="cpu", dtype=torch.float16),
                    "delta": atom["delta"].detach().to(device="cpu", dtype=torch.float16),
                    "gradient_l2_norm_fp32": atom["norm"],
                    "candidate_lengths": [
                        len(value) for value in cached_source["candidate_targets"]
                    ],
                    "target_token_ids": torch.cat(
                        cached_source["candidate_targets"]
                    ).detach().cpu(),
                },
                atom_path,
            )
            atom_manifest.append({
                "position": index, "source_index": source_index,
                "path": str(atom_path), "gradient_l2_norm_fp32": atom["norm"],
            })
        (atom_dir / "manifest.json").write_text(
            json.dumps({
                "format": "output_gradient_atom_bank_v1",
                "base_model_anchor": args.base_model,
                "sample_seed": args.sample_seed,
                "skill": args.skill,
                "atoms": atom_manifest,
            }, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    count = len(atoms)
    norms = np.asarray([atom["norm"] for atom in atoms])
    cosine = atoms_cosine(atoms)

    transfer_utility = []
    harm_utility = []
    source_validation_rows = []
    for source_position in range(count):
        weights = np.zeros(count, dtype=np.float64)
        weights[source_position] = 1.0
        targets = cached_by_group["transfer_validation"] + cached_by_group["harm_validation"]
        changed, _ = strategy_scores(
            atoms=atoms, cosine=cosine, weights=weights, step_norm=args.step_norm,
            targets=targets, base_logits=base_logits,
        )
        transfer_values = []
        harms = []
        local_rows = []
        for target, scores in zip(targets, changed, strict=True):
            index = target["spec"]["global_decision_index"]
            metrics = candidate_distribution_metrics(
                base_scores[index], scores, target["spec"]["values"], target["spec"]["expert_index"]
            )
            if target["spec"]["relationship"] == "transfer_validation":
                transfer_values.append(float(metrics["expected_value_delta"]))
            else:
                harms.append(max(0.0, -float(metrics["expected_value_delta"])))
            local_rows.append({"target_index": index, "relationship": target["spec"]["relationship"], **metrics})
        transfer_utility.append(float(np.mean(transfer_values)))
        harm_utility.append(float(np.mean(harms)))
        source_validation_rows.append(
            {
                "source_index": source_indices[source_position],
                "transfer_utility": transfer_utility[-1],
                "harm_utility": harm_utility[-1],
                "rows": local_rows,
            }
        )

    uniform = np.full(count, 1.0 / count)
    strategy_configs = {
        "mean12": (atoms, cosine, uniform),
        "cosine_purified": (atoms, cosine, cosine_consensus_weights(cosine)),
        "transfer_weighted": (
            atoms, cosine, softmax_weights(transfer_utility, args.temperature)
        ),
        "transfer_harm_weighted": (
            atoms, cosine, softmax_weights(
                np.asarray(transfer_utility) - args.harm_lambda * np.asarray(harm_utility),
                args.temperature,
            ),
        ),
    }

    protection_hidden = torch.cat([
        torch.cat(item["candidate_hidden"], dim=0)
        for item in cached_by_group["harm_validation"]
    ], dim=0)
    protection_basis = torch.linalg.qr(
        protection_hidden.transpose(0, 1), mode="reduced"
    ).Q
    projected_atoms = []
    for atom in atoms:
        projected_hidden = atom["hidden"] - torch.matmul(
            torch.matmul(atom["hidden"], protection_basis), protection_basis.transpose(0, 1)
        )
        projected_atoms.append(atom_with_hidden(atom, projected_hidden))
    projected_cosine = atoms_cosine(projected_atoms)
    projected_transfer, projected_harm = [], []
    projected_validation_targets = (
        cached_by_group["transfer_validation"] + cached_by_group["harm_validation"]
    )
    for source_position in range(count):
        local_weights = np.zeros(count, dtype=np.float64)
        local_weights[source_position] = 1.0
        changed, _ = strategy_scores(
            atoms=projected_atoms, cosine=projected_cosine, weights=local_weights,
            step_norm=args.step_norm, targets=projected_validation_targets,
            base_logits=base_logits,
        )
        benefits, damages = [], []
        for target, scores in zip(projected_validation_targets, changed, strict=True):
            spec = target["spec"]
            index = spec["global_decision_index"]
            metrics = candidate_distribution_metrics(
                base_scores[index], scores, spec["values"], spec["expert_index"]
            )
            if spec["relationship"] == "transfer_validation":
                benefits.append(metrics["expected_value_delta"])
            else:
                damages.append(max(0.0, -metrics["expected_value_delta"]))
        projected_transfer.append(float(np.mean(benefits)))
        projected_harm.append(float(np.mean(damages)))
    projected_weights = softmax_weights(
        np.asarray(projected_transfer) - args.harm_lambda * np.asarray(projected_harm),
        args.temperature,
    )
    strategy_configs["protection_nullspace_transfer_harm"] = (
        projected_atoms, projected_cosine, projected_weights
    )
    evaluation_targets = (
        cached_by_group["source"] + cached_by_group["transfer_validation"]
        + cached_by_group["same_skill_holdout"] + cached_by_group["harm_validation"]
        + cached_by_group["protection_test"]
    )
    strategy_results = {}
    for name, (strategy_atoms, strategy_cosine, weights) in strategy_configs.items():
        changed, aggregate_norm = strategy_scores(
            atoms=strategy_atoms, cosine=strategy_cosine, weights=weights, step_norm=args.step_norm,
            targets=evaluation_targets, base_logits=base_logits,
        )
        rows_out = []
        for target, scores in zip(evaluation_targets, changed, strict=True):
            spec = target["spec"]
            index = spec["global_decision_index"]
            metrics = candidate_distribution_metrics(
                base_scores[index], scores, spec["values"], spec["expert_index"]
            )
            seed_metrics = candidate_distribution_metrics(
                trace_base[index]["normalized_scores"], trace_seed[index]["normalized_scores"],
                spec["values"], spec["expert_index"],
            )
            rows_out.append(
                {
                    "global_decision_index": index,
                    "relationship": spec["relationship"],
                    **metrics,
                    "seed_expected_value_delta": seed_metrics["expected_value_delta"],
                    "seed_top_value_delta": seed_metrics["top_value_delta"],
                }
            )
        strategy_results[name] = {
            "weights": weights.tolist(),
            "effective_source_count": float(1.0 / np.sum(weights ** 2)),
            "pre_normalization_aggregate_unit_norm": aggregate_norm,
            "rows": rows_out,
        }
        print(f"strategy {name} complete", flush=True)

    sequential = []
    sequential_targets = cached_by_group["same_skill_holdout"] + cached_by_group["protection_test"]
    for experience_count in range(1, count + 1):
        local_fitness = (
            np.asarray(transfer_utility[:experience_count])
            - args.harm_lambda * np.asarray(harm_utility[:experience_count])
        )
        local_weights = softmax_weights(local_fitness, args.temperature)
        changed, aggregate_norm = strategy_scores(
            atoms=atoms[:experience_count],
            cosine=cosine[:experience_count, :experience_count],
            weights=local_weights,
            step_norm=args.step_norm,
            targets=sequential_targets,
            base_logits=base_logits,
        )
        relationship_rows: dict[str, list[dict[str, Any]]] = {
            "same_skill_holdout": [], "protection_test": []
        }
        for target, scores in zip(sequential_targets, changed, strict=True):
            spec = target["spec"]
            index = spec["global_decision_index"]
            relationship_rows[spec["relationship"]].append(
                candidate_distribution_metrics(
                    base_scores[index], scores, spec["values"], spec["expert_index"]
                )
            )
        sequential.append(
            {
                "experience_count": experience_count,
                "weights": local_weights.tolist(),
                "effective_source_count": float(1.0 / np.sum(local_weights ** 2)),
                "pre_normalization_aggregate_unit_norm": aggregate_norm,
                "same_skill_holdout": summarize_changed_rows(
                    relationship_rows["same_skill_holdout"]
                ),
                "protection_test": summarize_changed_rows(
                    relationship_rows["protection_test"]
                ),
            }
        )
        print(f"sequential K={experience_count} complete", flush=True)

    output = {
        "experiment": "output_gradient_evolution_v1",
        "status": "complete",
        "skill": args.skill,
        "sample_seed": args.sample_seed,
        "source_count": count,
        "parameter_delta_l2_norm": args.step_norm,
        "temperature": args.temperature,
        "harm_lambda": args.harm_lambda,
        "selection_policy": "weights use transfer-validation and harm-validation only",
        "final_holdout_used_for_weighting": False,
        "protection_test_used_for_weighting": False,
        "source_indices": source_indices,
        "gradient_norms": norms.tolist(),
        "gradient_atom_manifest": atom_manifest,
        "gradient_cosine_matrix": cosine.tolist(),
        "transfer_utility": transfer_utility,
        "harm_utility": harm_utility,
        "protection_nullspace_rank": int(protection_basis.shape[1]),
        "projected_transfer_utility": projected_transfer,
        "projected_harm_utility": projected_harm,
        "source_validation": source_validation_rows,
        "strategies": strategy_results,
        "sequential_transfer_harm_weighted": sequential,
        "split_manifest": split,
    }
    destination = Path(args.output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
