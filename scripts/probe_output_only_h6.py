#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_output_head_representation_sufficiency import (
    cache_state_hidden,
    cached_scores_from_logits,
    state_loss,
)
from probe_skill_gradient_purification import base_failure_indices
from probe_value_gradient_writeback import load_value_rows, state_spec
from self_evolve.alfworld_data import load_decisions
from self_evolve.output_head_oracle import fixed_oracle_split
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
    parser.add_argument("--skill", default="close", choices=("go", "open", "close"))
    parser.add_argument("--step-norm", type=float, default=0.18)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    return parser.parse_args()


def read_trace(path: str) -> dict[int, dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return {
            int(row["global_decision_index"]): row
            for line in handle
            if (row := json.loads(line))
        }


def output_atom(base_weight: torch.Tensor, source: dict[str, Any]):
    hidden = torch.cat(source["candidate_hidden"], dim=0)
    logits = torch.nn.functional.linear(hidden, base_weight).detach().requires_grad_(True)
    scores = cached_scores_from_logits(logits, source)
    loss = state_loss(scores, source, "value_expectation")
    delta = torch.autograd.grad(loss, logits)[0].detach()
    delta_gram = torch.matmul(delta, delta.transpose(0, 1))
    hidden_gram = torch.matmul(hidden, hidden.transpose(0, 1))
    squared_norm = float(torch.sum(delta_gram * hidden_gram).item())
    norm = math.sqrt(max(squared_norm, 0.0))
    if not math.isfinite(norm) or norm <= 1e-20:
        raise RuntimeError(f"Invalid low-rank output gradient norm {norm}")
    return hidden, delta, norm, scores.detach()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or args.step_norm <= 0:
        raise ValueError("H6 requires CUDA and a positive step norm")
    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    value_rows = load_value_rows(args.value_trace, "valid_seen")
    failures = base_failure_indices(args.base_score_trace, value_rows, "valid_seen")
    split = fixed_oracle_split(
        rows=value_rows,
        failure_indices=failures,
        verbs=("go", "open", "close"),
        source_count=12,
        validation_count=3,
        test_count=3,
        protection_count_per_verb=3,
        seed=args.sample_seed,
    )
    source_indices = split["by_skill"][args.skill]["source"]
    holdout_indices = split["by_skill"][args.skill]["final_holdout"]
    protection_indices = [
        int(item["global_decision_index"])
        for item in split["protection"]
        if item["matched_source_skill"] == args.skill
    ]
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

    cached: dict[int, dict[str, Any]] = {}
    relationships = {
        **{index: "source" for index in source_indices},
        **{index: "same_skill_holdout" for index in holdout_indices},
        **{index: "unrelated_protection" for index in protection_indices},
    }
    for index, relationship in relationships.items():
        spec = state_spec(index, relationship, decisions, value_rows, args.history_window)
        cached[index] = cache_state_hidden(
            backbone=model.model, tokenizer=tokenizer, spec=spec, device=device,
            candidate_batch_size=args.candidate_batch_size, max_length=args.max_length,
        )
        print(f"cached {relationship} {index}", flush=True)

    base_logits = {
        index: torch.nn.functional.linear(
            torch.cat(item["candidate_hidden"], dim=0), base_weight
        ).detach()
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

    source_results = []
    transfer_pairs = []
    for source_index in source_indices:
        source_hidden, delta, norm, _ = output_atom(base_weight, cached[source_index])
        scale = -args.step_norm / norm
        targets = [source_index] + holdout_indices + protection_indices
        target_rows = []
        for target_index in targets:
            target = cached[target_index]
            target_hidden = torch.cat(target["candidate_hidden"], dim=0)
            logit_delta = scale * torch.matmul(
                torch.matmul(target_hidden, source_hidden.transpose(0, 1)), delta
            )
            updated_logits = base_logits[target_index] + logit_delta
            updated_scores = (
                cached_scores_from_logits(updated_logits, target).detach().cpu().double().numpy()
            )
            spec = target["spec"]
            metrics = candidate_distribution_metrics(
                base_scores[target_index], updated_scores, spec["values"], spec["expert_index"]
            )
            seed_metrics = candidate_distribution_metrics(
                trace_base[target_index]["normalized_scores"],
                trace_seed[target_index]["normalized_scores"],
                spec["values"], spec["expert_index"],
            )
            row = {
                "source_index": source_index,
                "target_index": target_index,
                "relationship": spec["relationship"],
                "target_action_verb": spec["action_verb"],
                **metrics,
                "seed_expected_value_delta": seed_metrics["expected_value_delta"],
                "seed_top_value_delta": seed_metrics["top_value_delta"],
            }
            target_rows.append(row)
            transfer_pairs.append(row)
        source_results.append(
            {
                "source_index": source_index,
                "gradient_l2_norm": norm,
                "parameter_delta_l2_norm": args.step_norm,
                "targets": target_rows,
            }
        )
        print(f"source {source_index} complete norm={norm:.6g}", flush=True)

    output = {
        "experiment": "h6_output_only_strong_v1",
        "status": "complete",
        "skill": args.skill,
        "sample_seed": args.sample_seed,
        "source_count": len(source_indices),
        "same_skill_holdout_count": len(holdout_indices),
        "protection_count": len(protection_indices),
        "parameter_delta_l2_norm": args.step_norm,
        "strength_label": "300x protocol (same L2 budget as prior last-four-layer H6/H8)",
        "objective": "negative expected environment long-term value",
        "implementation": "exact virtual output-head writeback after parameter/logit equivalence",
        "base_trace_max_absolute_error": trace_error,
        "split_manifest": split,
        "sources": source_results,
        "transfer_pairs": transfer_pairs,
    }
    destination = Path(args.output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
