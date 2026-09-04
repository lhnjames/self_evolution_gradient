#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_skill_gradient_purification import base_failure_indices
from probe_value_gradient_writeback import candidate_pairs, load_value_rows, state_spec
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
    parser.add_argument("--split", default="valid_seen", choices=("valid_seen",))
    parser.add_argument("--verbs", nargs="+", default=["go", "open", "close"])
    parser.add_argument("--train-verb", choices=("go", "open", "close"))
    parser.add_argument("--source-count", type=int, default=12)
    parser.add_argument("--validation-count", type=int, default=3)
    parser.add_argument("--test-count", type=int, default=3)
    parser.add_argument("--protection-count-per-verb", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=20260904)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=3)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--history-window", type=int, default=3)
    parser.add_argument(
        "--objective", choices=("optimal_set", "value_expectation"), default="optimal_set"
    )
    parser.add_argument("--max-delta-norm", type=float, default=0.0)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cache_state_hidden(
    *,
    backbone: Any,
    tokenizer: Any,
    spec: dict[str, Any],
    device: torch.device,
    candidate_batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    pairs = candidate_pairs(tokenizer, spec["prompt"], spec["candidates"], max_length)
    candidate_hidden: list[torch.Tensor] = []
    candidate_targets: list[torch.Tensor] = []
    pad_id = int(tokenizer.pad_token_id)
    with torch.inference_mode():
        for start in range(0, len(pairs), candidate_batch_size):
            batch = pairs[start : start + candidate_batch_size]
            full_ids = [prompt_ids + candidate_ids for prompt_ids, candidate_ids in batch]
            width = max(map(len, full_ids))
            input_ids = torch.full(
                (len(batch), width), pad_id, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros_like(input_ids)
            for row_index, ids in enumerate(full_ids):
                input_ids[row_index, : len(ids)] = torch.tensor(
                    ids, dtype=torch.long, device=device
                )
                attention_mask[row_index, : len(ids)] = 1
            prompt_lengths = {len(prompt_ids) for prompt_ids, _ in batch}
            if len(prompt_lengths) != 1:
                raise AssertionError("Batch prompt lengths differ")
            prompt_length = next(iter(prompt_lengths))
            hidden = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).last_hidden_state.float()
            for row_index, (_, candidate_ids) in enumerate(batch):
                length = len(candidate_ids)
                candidate_hidden.append(
                    hidden[row_index, prompt_length - 1 : prompt_length - 1 + length].detach()
                )
                candidate_targets.append(
                    torch.tensor(candidate_ids, dtype=torch.long, device=device)
                )
    return {
        "spec": spec,
        "candidate_hidden": candidate_hidden,
        "candidate_targets": candidate_targets,
    }


def cached_candidate_scores(
    weight: torch.Tensor, cached: dict[str, Any], token_chunk_size: int = 256
) -> torch.Tensor:
    hidden = torch.cat(cached["candidate_hidden"], dim=0)
    targets = torch.cat(cached["candidate_targets"], dim=0)
    lengths = [len(item) for item in cached["candidate_targets"]]
    token_scores = []
    for start in range(0, len(targets), token_chunk_size):
        local_hidden = hidden[start : start + token_chunk_size]
        local_targets = targets[start : start + token_chunk_size]
        logits = torch.nn.functional.linear(local_hidden, weight)
        token_scores.append(
            logits.gather(1, local_targets[:, None]).squeeze(1)
            - torch.logsumexp(logits, dim=-1)
        )
    return cached_scores_from_token_log_probs(torch.cat(token_scores), cached)


def cached_scores_from_token_log_probs(
    token_scores_tensor: torch.Tensor, cached: dict[str, Any]
) -> torch.Tensor:
    result = []
    cursor = 0
    for length in [len(item) for item in cached["candidate_targets"]]:
        result.append(token_scores_tensor[cursor : cursor + length].mean())
        cursor += length
    return torch.stack(result)


def cached_scores_from_logits(logits: torch.Tensor, cached: dict[str, Any]) -> torch.Tensor:
    targets = torch.cat(cached["candidate_targets"], dim=0)
    token_scores = (
        logits.gather(1, targets[:, None]).squeeze(1)
        - torch.logsumexp(logits, dim=-1)
    )
    return cached_scores_from_token_log_probs(token_scores, cached)


def state_loss(scores: torch.Tensor, cached: dict[str, Any], objective: str) -> torch.Tensor:
    values = torch.as_tensor(
        cached["spec"]["values"], dtype=scores.dtype, device=scores.device
    )
    if objective == "value_expectation":
        return -torch.dot(torch.softmax(scores, dim=0), values)
    optimal = torch.isclose(values, values.max(), rtol=0.0, atol=1e-12)
    return torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[optimal], dim=0)


def score_panel(weight: torch.Tensor, panel: Sequence[dict[str, Any]]) -> list[np.ndarray]:
    result = []
    with torch.inference_mode():
        for cached in panel:
            result.append(
                cached_candidate_scores(weight, cached).detach().cpu().double().numpy()
            )
    return result


def summarize_panel(
    panel: Sequence[dict[str, Any]],
    baseline_scores: Sequence[np.ndarray],
    updated_scores: Sequence[np.ndarray],
) -> dict[str, Any]:
    states = []
    for cached, baseline, updated in zip(panel, baseline_scores, updated_scores, strict=True):
        spec = cached["spec"]
        metrics = candidate_distribution_metrics(
            baseline, updated, spec["values"], spec["expert_index"]
        )
        states.append(
            {
                "global_decision_index": spec["global_decision_index"],
                "episode_key": spec["episode_key"],
                "task_type": spec["task_type"],
                "action_verb": spec["action_verb"],
                "relationship": spec["relationship"],
                **metrics,
            }
        )
    base_value = float(np.mean([row["baseline_expected_value"] for row in states]))
    updated_value = float(np.mean([row["updated_expected_value"] for row in states]))
    base_optimal = float(
        np.mean([row["baseline_top_value"] >= max(cached["spec"]["values"]) - 1e-12
                 for row, cached in zip(states, panel, strict=True)])
    )
    updated_optimal = float(
        np.mean([row["updated_top_value"] >= max(cached["spec"]["values"]) - 1e-12
                 for row, cached in zip(states, panel, strict=True)])
    )
    harm = float(
        np.mean([row["updated_top_value"] < row["baseline_top_value"] - 1e-12 for row in states])
    )
    return {
        "count": len(states),
        "baseline_expected_value": base_value,
        "updated_expected_value": updated_value,
        "absolute_expected_value_gain": updated_value - base_value,
        "relative_expected_value_gain": (
            (updated_value - base_value) / base_value if base_value > 1e-12 else None
        ),
        "baseline_top_value_optimal_rate": base_optimal,
        "updated_top_value_optimal_rate": updated_optimal,
        "top_value_optimal_rate_gain_points": updated_optimal - base_optimal,
        "top_value_harm_rate": harm,
        "mean_kl": float(np.mean([row["kl_baseline_to_updated"] for row in states])),
        "states": states,
    }


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda"):
        raise ValueError("Phase 0 must run on CUDA")
    if args.learning_rate <= 0 or args.epochs < 1 or args.train_batch_size < 1:
        raise ValueError("Invalid optimization settings")
    started = time.time()
    device = torch.device(args.device)
    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    rows = load_value_rows(args.value_trace, args.split)
    failures = base_failure_indices(args.base_score_trace, rows, args.split)
    split = fixed_oracle_split(
        rows=rows,
        failure_indices=failures,
        verbs=args.verbs,
        source_count=args.source_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
        protection_count_per_verb=args.protection_count_per_verb,
        seed=args.sample_seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, local_files_only=True
    ).to(device)
    model.eval()
    if not bool(model.config.tie_word_embeddings):
        raise AssertionError("Expected the base checkpoint to use tied embeddings")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise AssertionError("Checkpoint config says tied, but pointers are not tied")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    panels: dict[str, list[dict[str, Any]]] = {
        "source": [], "transfer_validation": [], "final_holdout": [], "protection": []
    }
    specs: dict[int, dict[str, Any]] = {}
    active_verbs = [args.train_verb] if args.train_verb else list(args.verbs)
    for verb, skill_split in split["by_skill"].items():
        if verb not in active_verbs:
            continue
        for panel_name in ("source", "transfer_validation", "final_holdout"):
            for index in skill_split[panel_name]:
                spec = state_spec(index, panel_name, decisions, rows, args.history_window)
                spec["source_skill"] = verb
                specs[index] = spec
    for item in split["protection"]:
        index = int(item["global_decision_index"])
        spec = state_spec(index, "protection", decisions, rows, args.history_window)
        spec["source_skill"] = item["matched_source_skill"]
        specs[index] = spec

    ordered = []
    for panel_name in panels:
        if panel_name == "protection":
            indices = [
                int(item["global_decision_index"])
                for item in split["protection"]
                if item["matched_source_skill"] in active_verbs
            ]
        else:
            indices = [
                index
                for verb in active_verbs
                for index in split["by_skill"][verb][panel_name]
            ]
        for index in indices:
            print(f"cache {panel_name} state={index}", flush=True)
            cached = cache_state_hidden(
                backbone=model.model,
                tokenizer=tokenizer,
                spec=specs[index],
                device=device,
                candidate_batch_size=args.candidate_batch_size,
                max_length=args.max_length,
            )
            panels[panel_name].append(cached)
            ordered.append(cached)

    input_embedding_ptr = model.model.embed_tokens.weight.data_ptr()
    base_weight = model.model.embed_tokens.weight.detach()
    baseline_scores = {name: score_panel(base_weight, panel) for name, panel in panels.items()}
    base_trace = {int(row["global_decision_index"]): row for row in read_jsonl(args.base_score_trace)}
    trace_error = 0.0
    for panel_name, panel in panels.items():
        for cached, scores in zip(panel, baseline_scores[panel_name], strict=True):
            reference = np.asarray(
                base_trace[cached["spec"]["global_decision_index"]]["normalized_scores"],
                dtype=np.float64,
            )
            trace_error = max(trace_error, float(np.max(np.abs(scores - reference))))
    # The reference trace used a different candidate-batch GEMM shape.  FP32
    # Qwen logits can consequently differ by several 1e-5 even though the
    # hidden states and tied weight are identical.  Keep the observed error in
    # the artifact and reject anything above a conservative 1e-4.
    if trace_error > 1e-4:
        raise AssertionError(f"Cached hidden score reproduction error {trace_error}")

    head = torch.nn.Linear(
        int(model.config.hidden_size), int(model.config.vocab_size), bias=False,
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        head.weight.copy_(base_weight)
    if head.weight.data_ptr() in (input_embedding_ptr, model.lm_head.weight.data_ptr()):
        raise AssertionError("Independent output head is still tied")
    head.weight.requires_grad_(True)
    optimizer = torch.optim.SGD([head.weight], lr=args.learning_rate)
    rng = random.Random(args.sample_seed + 91)
    training = list(panels["source"])
    best: dict[str, Any] | None = None
    epochs_without_gain = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(training)
        losses = []
        for start in range(0, len(training), args.train_batch_size):
            batch = training[start : start + args.train_batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack(
                [state_loss(cached_candidate_scores(head.weight, item), item, args.objective)
                 for item in batch]
            ).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if args.max_delta_norm > 0:
                with torch.no_grad():
                    delta = head.weight - base_weight
                    norm = torch.linalg.vector_norm(delta)
                    if norm > args.max_delta_norm:
                        head.weight.copy_(base_weight + delta * (args.max_delta_norm / norm))
        validation_scores = score_panel(head.weight, panels["transfer_validation"])
        validation = summarize_panel(
            panels["transfer_validation"],
            baseline_scores["transfer_validation"],
            validation_scores,
        )
        source_scores = score_panel(head.weight, panels["source"])
        source = summarize_panel(panels["source"], baseline_scores["source"], source_scores)
        objective_value = float(validation["updated_expected_value"])
        delta_norm = float(torch.linalg.vector_norm(head.weight - base_weight).item())
        epoch_row = {
            "epoch": epoch,
            "mean_training_loss": float(np.mean(losses)),
            "validation_updated_expected_value": objective_value,
            "validation_relative_expected_value_gain": validation["relative_expected_value_gain"],
            "validation_top_value_optimal_rate": validation["updated_top_value_optimal_rate"],
            "source_relative_expected_value_gain": source["relative_expected_value_gain"],
            "parameter_delta_l2_norm": delta_norm,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row), flush=True)
        if best is None or objective_value > best["objective_value"] + 1e-12:
            best = {
                "epoch": epoch,
                "objective_value": objective_value,
                "weight": head.weight.detach().to(device="cpu", dtype=torch.float32).clone(),
            }
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= args.patience:
                break
    assert best is not None
    with torch.no_grad():
        head.weight.copy_(best.pop("weight").to(device=device, dtype=torch.float32))
    final_scores = {name: score_panel(head.weight, panel) for name, panel in panels.items()}
    summaries = {
        name: summarize_panel(panel, baseline_scores[name], final_scores[name])
        for name, panel in panels.items()
    }
    delta_norm = float(torch.linalg.vector_norm(head.weight - base_weight).item())
    input_embedding_unchanged = bool(
        model.model.embed_tokens.weight.data_ptr() == input_embedding_ptr
        and model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    )
    output = {
        "experiment": "output_head_representation_sufficiency_v1",
        "status": "complete",
        "configuration": vars(args),
        "split_manifest": split,
        "model_checks": {
            "base_tie_word_embeddings": bool(model.config.tie_word_embeddings),
            "base_lm_head_is_input_embedding": True,
            "oracle_head_is_independent": head.weight.data_ptr() != input_embedding_ptr,
            "input_embedding_and_original_head_unchanged": input_embedding_unchanged,
            "bias_trained": False,
            "backbone_trainable_parameter_count": 0,
            "oracle_head_trainable_parameter_count": head.weight.numel(),
            "cached_score_max_absolute_error_vs_fp32_trace": trace_error,
        },
        "selection": {
            "criterion": "maximum transfer-validation mean expected long-term value",
            "best_epoch": best["epoch"],
            "best_validation_expected_value": best["objective_value"],
            "final_holdout_was_used_for_selection": False,
        },
        "parameter_delta_l2_norm": delta_norm,
        "history": history,
        "panels": summaries,
        "gates": {
            "holdout_relative_value_at_least_30_percent": bool(
                summaries["final_holdout"]["relative_expected_value_gain"] >= 0.30
            ),
            "holdout_failure_repair_at_least_30_points": bool(
                summaries["final_holdout"]["top_value_optimal_rate_gain_points"] >= 0.30
            ),
            "protection_top_value_harm_at_most_2_percent": bool(
                summaries["protection"]["top_value_harm_rate"] <= 0.02
            ),
        },
        "wall_time_seconds": time.time() - started,
    }
    destination = Path(args.output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
