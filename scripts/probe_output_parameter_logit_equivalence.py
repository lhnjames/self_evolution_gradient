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
    cached_candidate_scores,
    cached_scores_from_logits,
    state_loss,
)
from probe_skill_gradient_purification import base_failure_indices
from probe_value_gradient_writeback import load_value_rows, state_spec
from self_evolve.alfworld_data import load_decisions
from self_evolve.output_head_oracle import fixed_oracle_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-file", required=True)
    parser.add_argument("--value-trace", required=True)
    parser.add_argument("--base-score-trace", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--sample-seed", type=int, default=20260904)
    parser.add_argument("--source-pairs", type=int, default=3)
    parser.add_argument("--step-norm", type=float, default=0.3)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Equivalence test requires CUDA")
    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    rows = load_value_rows(args.value_trace, "valid_seen")
    failures = base_failure_indices(args.base_score_trace, rows, "valid_seen")
    split = fixed_oracle_split(
        rows=rows,
        failure_indices=failures,
        verbs=("go", "open", "close"),
        source_count=12,
        validation_count=3,
        test_count=3,
        protection_count_per_verb=3,
        seed=args.sample_seed,
    )
    source_indices = split["by_skill"]["close"]["source"][: args.source_pairs]
    target_specs: list[dict[str, Any]] = []
    for index in split["by_skill"]["close"]["final_holdout"]:
        target_specs.append(state_spec(index, "same_skill_holdout", decisions, rows, args.history_window))
    for item in split["protection"]:
        if item["matched_source_skill"] == "close":
            target_specs.append(
                state_spec(
                    int(item["global_decision_index"]), "unrelated_protection",
                    decisions, rows, args.history_window,
                )
            )
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
    input_ptr = base_weight.data_ptr()

    cached_targets = [
        cache_state_hidden(
            backbone=model.model, tokenizer=tokenizer, spec=spec, device=device,
            candidate_batch_size=args.candidate_batch_size, max_length=args.max_length,
        )
        for spec in target_specs
    ]
    rows_out = []
    gradient_errors = []
    logit_errors = []
    factorization_errors = []
    score_errors = []
    for source_index in source_indices:
        source_spec = state_spec(
            source_index, "source", decisions, rows, args.history_window
        )
        source = cache_state_hidden(
            backbone=model.model, tokenizer=tokenizer, spec=source_spec, device=device,
            candidate_batch_size=args.candidate_batch_size, max_length=args.max_length,
        )
        source_hidden = torch.cat(source["candidate_hidden"], dim=0)
        source_logits = torch.nn.functional.linear(source_hidden, base_weight).detach()
        source_logits.requires_grad_(True)
        source_scores = cached_scores_from_logits(source_logits, source)
        loss = state_loss(source_scores, source, "value_expectation")
        delta = torch.autograd.grad(loss, source_logits)[0].detach()
        formula_gradient = torch.matmul(delta.transpose(0, 1), source_hidden)
        gradient_norm = float(torch.linalg.vector_norm(formula_gradient).item())
        if not math.isfinite(gradient_norm) or gradient_norm <= 1e-20:
            raise RuntimeError("Invalid output gradient norm")

        head = torch.nn.Linear(
            int(model.config.hidden_size), int(model.config.vocab_size), bias=False,
            dtype=torch.float32, device=device,
        )
        with torch.no_grad():
            head.weight.copy_(base_weight)
        direct_loss = state_loss(cached_candidate_scores(head.weight, source), source, "value_expectation")
        direct_loss.backward()
        direct_gradient = head.weight.grad.detach()
        gradient_error = float(torch.max(torch.abs(direct_gradient - formula_gradient)).item())
        gradient_relative = gradient_error / max(
            float(torch.max(torch.abs(direct_gradient)).item()), 1e-30
        )
        gradient_errors.append(gradient_error)
        scale = -args.step_norm / gradient_norm
        update = formula_gradient.mul(scale)
        with torch.no_grad():
            head.weight.copy_(base_weight)
            head.weight.add_(update)
        if head.weight.data_ptr() == input_ptr:
            raise AssertionError("Equivalence head is tied to input embedding")

        for target in cached_targets:
            target_hidden = torch.cat(target["candidate_hidden"], dim=0)
            base_logits = torch.nn.functional.linear(target_hidden, base_weight)
            real_logits = torch.nn.functional.linear(target_hidden, head.weight)
            matrix_virtual_logits = base_logits + torch.nn.functional.linear(target_hidden, update)
            factorized_delta = scale * torch.matmul(
                torch.matmul(target_hidden, source_hidden.transpose(0, 1)), delta
            )
            factorized_virtual_logits = base_logits + factorized_delta
            logit_error = float(torch.max(torch.abs(real_logits - matrix_virtual_logits)).item())
            factorization_error = float(
                torch.max(torch.abs(matrix_virtual_logits - factorized_virtual_logits)).item()
            )
            real_scores = cached_scores_from_logits(real_logits, target)
            virtual_scores = cached_scores_from_logits(factorized_virtual_logits, target)
            score_error = float(torch.max(torch.abs(real_scores - virtual_scores)).item())
            logit_errors.append(logit_error)
            factorization_errors.append(factorization_error)
            score_errors.append(score_error)
            rows_out.append(
                {
                    "source_index": source_index,
                    "target_index": target["spec"]["global_decision_index"],
                    "relationship": target["spec"]["relationship"],
                    "source_token_positions": len(source_hidden),
                    "target_token_positions": len(target_hidden),
                    "gradient_l2_norm": gradient_norm,
                    "parameter_delta_l2_norm": args.step_norm,
                    "direct_vs_factorized_gradient_max_abs_error": gradient_error,
                    "direct_vs_factorized_gradient_relative_max_error": gradient_relative,
                    "writeback_vs_matrix_virtual_logit_max_abs_error": logit_error,
                    "matrix_vs_atom_factorized_logit_max_abs_error": factorization_error,
                    "writeback_vs_virtual_action_score_max_abs_error": score_error,
                }
            )
        del head, direct_gradient, formula_gradient, update, delta, source_logits
        torch.cuda.empty_cache()

    output = {
        "experiment": "parameter_logit_equivalence_v1",
        "status": "complete",
        "precision": "FP32",
        "skill": "close",
        "source_count": len(source_indices),
        "target_count_per_source": len(cached_targets),
        "pair_count": len(rows_out),
        "step_norm": args.step_norm,
        "bias_updated": False,
        "backbone_frozen": True,
        "input_embedding_unchanged": model.model.embed_tokens.weight.data_ptr() == input_ptr,
        "max_errors": {
            "direct_vs_factorized_gradient": max(gradient_errors),
            "writeback_vs_matrix_virtual_logits": max(logit_errors),
            "matrix_vs_atom_factorized_logits": max(factorization_errors),
            "writeback_vs_virtual_action_scores": max(score_errors),
        },
        "pairs": rows_out,
    }
    destination = Path(args.output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["max_errors"], indent=2), flush=True)


if __name__ == "__main__":
    main()
