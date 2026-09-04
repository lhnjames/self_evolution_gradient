#!/usr/bin/env python3
"""Measure skill-teacher gradients in the backbone RMSNorm parameter subspace."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from self_evolve.alfworld_data import load_decisions
from self_evolve.alfworld_skills import build_action_prompt


TEACHERS = (
    "evolved_skill",
    "mismatched_skill",
    "reformatted_skill",
    "anti_skill",
    "length_matched_placebo",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def softmax(scores: Sequence[float]) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64)
    values = np.exp(array - array.max())
    return values / values.sum()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(left.norm() * right.norm())
    return float(torch.dot(left, right) / denominator) if denominator > 1e-20 else float("nan")


def load_aligned_rows(
    decision_path: str, baseline_path: str, control_paths: list[str]
) -> list[dict[str, Any]]:
    decisions = [item for item in load_decisions(decision_path) if not item.is_trivial]
    baseline = read_jsonl(baseline_path)
    controls = [row for path in control_paths for row in read_jsonl(path)]
    controls.sort(key=lambda row: row["global_decision_index"])
    if not (len(decisions) == len(baseline) == len(controls)):
        raise ValueError(
            f"alignment lengths differ: {len(decisions)}, {len(baseline)}, {len(controls)}"
        )
    rows = []
    for index, (decision, base, control) in enumerate(zip(decisions, baseline, controls)):
        if decision.expert_action != base["expert_action"] or base["expert_action"] != control["expert_action"]:
            raise ValueError(f"row {index} expert action mismatch")
        scores = {
            "plain": base["normalized_scores"]["plain"],
            "evolved_skill": base["normalized_scores"]["evolved_skill"],
            "mismatched_skill": base["normalized_scores"]["mismatched_skill"],
            "reformatted_skill": control["normalized_scores"]["reformatted_skill"],
            "anti_skill": control["normalized_scores"]["anti_skill"],
            "length_matched_placebo": control["normalized_scores"]["length_matched_placebo"],
        }
        rows.append({"global_index": index, "decision": decision, "baseline": base, "scores": scores})
    return rows


def stratified_indices(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[int]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["decision"].task_type, row["decision"].action_verb)].append(index)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected = []
    depth = 0
    while len(selected) < min(sample_size, len(rows)):
        added = False
        for key in keys:
            if depth < len(groups[key]):
                selected.append(groups[key][depth])
                added = True
                if len(selected) >= sample_size:
                    break
        if not added:
            break
        depth += 1
    return selected


class RMSNormGradientProbe:
    def __init__(self, model_path: str, device: str, max_length: int, batch_size: int):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, local_files_only=True
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        selected = []
        for name, parameter in self.model.named_parameters():
            if "layernorm.weight" in name or name.endswith("model.norm.weight"):
                parameter.requires_grad_(True)
                selected.append((name, parameter))
        if not selected:
            raise RuntimeError("No RMSNorm parameters selected")
        self.parameter_names = [name for name, _ in selected]
        self.parameters = [parameter for _, parameter in selected]
        self.parameter_count = sum(parameter.numel() for parameter in self.parameters)
        self.max_length = max_length
        self.batch_size = batch_size

    def _pairs(self, prompt: str, candidates: Sequence[str]) -> list[tuple[list[int], list[int]]]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        pairs = []
        for candidate in candidates:
            candidate_ids = self.tokenizer.encode(" " + candidate, add_special_tokens=False)
            max_prompt = self.max_length - len(candidate_ids)
            pairs.append((prompt_ids[-max_prompt:], candidate_ids))
        return pairs

    def gradients(
        self, prompt: str, candidates: Sequence[str], coefficients: dict[str, np.ndarray]
    ) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        pairs = self._pairs(prompt, candidates)
        accumulated = {
            name: [torch.zeros_like(parameter, dtype=torch.float32) for parameter in self.parameters]
            for name in coefficients
        }
        scored = []
        pad_id = int(self.tokenizer.pad_token_id)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            full_ids = [prompt_ids + candidate_ids for prompt_ids, candidate_ids in batch]
            width = max(len(ids) for ids in full_ids)
            input_ids = torch.full(
                (len(batch), width), pad_id, dtype=torch.long, device=self.device
            )
            attention_mask = torch.zeros_like(input_ids)
            for batch_index, ids in enumerate(full_ids):
                input_ids[batch_index, : len(ids)] = torch.tensor(ids, device=self.device)
                attention_mask[batch_index, : len(ids)] = 1
            prompt_lengths = {len(prompt_ids) for prompt_ids, _ in batch}
            if len(prompt_lengths) != 1:
                raise AssertionError("Gradient batch prompt lengths differ")
            prompt_length = next(iter(prompt_lengths))
            positions = torch.arange(prompt_length - 1, width - 1, device=self.device)
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=positions,
            ).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            batch_scores = []
            for batch_index, (_, candidate_ids) in enumerate(batch):
                targets = torch.tensor(candidate_ids, dtype=torch.long, device=self.device)
                local_positions = torch.arange(len(candidate_ids), device=self.device)
                token_log_probs = log_probs[batch_index, local_positions, targets]
                batch_scores.append(token_log_probs.mean())
            score_tensor = torch.stack(batch_scores)
            scored.extend(float(value) for value in score_tensor.detach().cpu())
            # The six probe objectives are different linear combinations of the
            # same candidate scores.  Compute each candidate-score VJP once and
            # form all objectives from it, instead of traversing the model once
            # per objective.
            names = list(coefficients)
            for batch_index in range(len(batch)):
                gradients = torch.autograd.grad(
                    score_tensor[batch_index],
                    self.parameters,
                    retain_graph=batch_index < len(batch) - 1,
                    allow_unused=False,
                )
                for name in names:
                    coefficient = float(coefficients[name][start + batch_index])
                    for parameter_index, gradient in enumerate(gradients):
                        accumulated[name][parameter_index].add_(
                            gradient.detach().float(), alpha=coefficient
                        )
            del logits, log_probs, score_tensor
        vectors = {
            name: torch.cat([gradient.flatten() for gradient in gradients])
            for name, gradients in accumulated.items()
        }
        return vectors, np.asarray(scored, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-file", required=True)
    parser.add_argument("--baseline-trace", required=True)
    parser.add_argument("--control-traces", nargs="+", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260901)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--history-window", type=int, default=3)
    args = parser.parse_args()

    rows = load_aligned_rows(args.decision_file, args.baseline_trace, args.control_traces)
    selected = stratified_indices(rows, args.sample_size, args.sample_seed)
    selected = selected[args.shard_index :: args.num_shards]
    probe = RMSNormGradientProbe(
        args.model_path, args.device, args.max_length, args.batch_size
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    saved_gradients: dict[str, list[np.ndarray]] = defaultdict(list)
    for local_index, row_index in enumerate(selected):
        row = rows[row_index]
        decision = row["decision"]
        distributions = {name: softmax(scores) for name, scores in row["scores"].items()}
        p0 = distributions["plain"]
        verified_target = np.zeros_like(p0)
        gold = decision.admissible_actions.index(decision.expert_action)
        verified_target[gold] = 1.0
        coefficients = {"verified": p0 - verified_target}
        coefficients.update({name: p0 - distributions[name] for name in TEACHERS})
        prompt = build_action_prompt(decision, history_window=args.history_window)
        vectors, recomputed_scores = probe.gradients(
            prompt, decision.admissible_actions, coefficients
        )
        cached = np.asarray(row["scores"]["plain"], dtype=np.float64)
        max_error = float(np.max(np.abs(cached - recomputed_scores)))
        verified = vectors["verified"]
        condition_metrics = {}
        for name in TEACHERS:
            vector = vectors[name]
            norm = float(vector.norm())
            condition_metrics[name] = {
                "cosine_with_verified": cosine(vector, verified),
                "gradient_norm": norm,
                "norm_ratio_to_verified": norm / max(float(verified.norm()), 1e-20),
            }
            normalized = vector / max(norm, 1e-20)
            saved_gradients[name].append(normalized.cpu().numpy().astype(np.float16))
        metadata.append(
            {
                "global_index": row["global_index"],
                "episode_key": decision.gamefile,
                "task_type": decision.task_type,
                "action_verb": decision.action_verb,
                "step_index": decision.step_index,
                "candidate_count": len(decision.admissible_actions),
                "plain_score_max_abs_error": max_error,
                "verified_gradient_norm": float(verified.norm()),
                "conditions": condition_metrics,
            }
        )
        print(
            f"[{local_index + 1}/{len(selected)}] global={row['global_index']} "
            + " ".join(
                f"{name}={condition_metrics[name]['cosine_with_verified']:+.3f}"
                for name in TEACHERS
            ),
            flush=True,
        )
    arrays = {name: np.stack(values) for name, values in saved_gradients.items()}
    arrays["global_indices"] = np.asarray(
        [item["global_index"] for item in metadata], dtype=np.int64
    )
    np.savez(output_dir / "normalized_gradients.npz", **arrays)
    result = {
        "sample_size_requested": args.sample_size,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "parameter_subspace": "all_qwen_rmsnorm_scale_parameters",
        "parameter_count": probe.parameter_count,
        "parameter_names": probe.parameter_names,
        "states": metadata,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "states"}, indent=2))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
