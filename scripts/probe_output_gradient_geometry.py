#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_output_gradient_evolution import atom_dot, make_atom
from probe_output_head_representation_sufficiency import cache_state_hidden, cached_scores_from_logits
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
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--sample-seed", type=int, required=True)
    parser.add_argument("--step-norm", type=float, default=0.72)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    return parser.parse_args()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float((torch.dot(left, right) / denominator.clamp_min(1e-30)).item())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda": raise ValueError("Geometry probe requires CUDA")
    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    value_rows = load_value_rows(args.value_trace, "valid_seen")
    failures = base_failure_indices(args.base_score_trace, value_rows, "valid_seen")
    split = fixed_oracle_split(
        rows=value_rows, failure_indices=failures, verbs=("close",),
        source_count=12, validation_count=3, test_count=3,
        protection_count_per_verb=3, seed=args.sample_seed,
    )
    panel_indices = {
        panel: split["by_skill"]["close"][panel]
        for panel in ("source", "transfer_validation", "final_holdout")
    }
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, local_files_only=True
    ).to(device)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    base_weight = model.model.embed_tokens.weight.detach()
    cached = []
    panels = []
    for panel, indices in panel_indices.items():
        for index in indices:
            spec = state_spec(index, panel, decisions, value_rows, args.history_window)
            cached.append(cache_state_hidden(
                backbone=model.model, tokenizer=tokenizer, spec=spec, device=device,
                candidate_batch_size=args.candidate_batch_size, max_length=args.max_length,
            ))
            panels.append(panel)
            print(f"cached {panel} {index}", flush=True)
    atoms = [make_atom(base_weight, item) for item in cached]
    logits = [
        torch.nn.functional.linear(torch.cat(item["candidate_hidden"], dim=0), base_weight).detach()
        for item in cached
    ]
    scores = [
        cached_scores_from_logits(value, item).detach().cpu().double().numpy()
        for value, item in zip(logits, cached, strict=True)
    ]
    hidden_means = [atom["hidden"].mean(dim=0) for atom in atoms]
    delta_means = [atom["delta"].mean(dim=0) for atom in atoms]
    pairs = []
    for source_position, source in enumerate(atoms):
        scale = -args.step_norm / source["norm"]
        for target_position, target in enumerate(atoms):
            if source_position == target_position: continue
            target_hidden = target["hidden"]
            changed_logits = logits[target_position] + scale * torch.matmul(
                torch.matmul(target_hidden, source["hidden"].transpose(0, 1)), source["delta"]
            )
            changed_scores = cached_scores_from_logits(
                changed_logits, cached[target_position]
            ).detach().cpu().double().numpy()
            spec = cached[target_position]["spec"]
            metrics = candidate_distribution_metrics(
                scores[target_position], changed_scores, spec["values"], spec["expert_index"]
            )
            gradient_cosine = atom_dot(source, target) / (source["norm"] * target["norm"])
            hidden_cosine = cosine(hidden_means[source_position], hidden_means[target_position])
            delta_cosine = cosine(delta_means[source_position], delta_means[target_position])
            pairs.append({
                "source_index": cached[source_position]["spec"]["global_decision_index"],
                "target_index": spec["global_decision_index"],
                "source_panel": panels[source_position],
                "target_panel": panels[target_position],
                "mean_hidden_cosine": hidden_cosine,
                "mean_delta_cosine": delta_cosine,
                "mean_factor_product": hidden_cosine * delta_cosine,
                "full_multitoken_gradient_cosine": gradient_cosine,
                "expected_value_delta": metrics["expected_value_delta"],
                "top_value_delta": metrics["top_value_delta"],
                "kl_baseline_to_updated": metrics["kl_baseline_to_updated"],
            })
        print(f"source position {source_position + 1}/18 complete", flush=True)
    output = {
        "experiment": "output_gradient_geometry_transfer_v1",
        "status": "complete",
        "sample_seed": args.sample_seed,
        "skill": "close",
        "parameter_delta_l2_norm": args.step_norm,
        "state_count": len(cached),
        "directed_pair_count": len(pairs),
        "multitoken_identity": "<g_i,g_j>=sum_rt <delta_ir,delta_jt><h_ir,h_jt>",
        "pairs": pairs,
    }
    destination = Path(args.output_file); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__": main()
