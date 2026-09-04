from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class CandidateScores:
    candidates: tuple[str, ...]
    sum_logprobs: torch.Tensor
    normalized_scores: torch.Tensor
    token_lengths: torch.Tensor

    def probabilities(self, temperature: float = 1.0) -> torch.Tensor:
        return torch.softmax(self.normalized_scores / temperature, dim=-1)


class SequenceActionScorer:
    """Scores arbitrary multi-token commands with a frozen causal LM."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        batch_size: int = 8,
        max_length: int = 1536,
        length_penalty: float = 1.0,
        tokenizer_name_or_path: str | None = None,
        parameter_delta_path: str | None = None,
        force_float32: bool = False,
    ):
        self.device = torch.device(device)
        # Value-gradient edits are computed in FP32.  Keep that precision when a
        # sparse parameter delta is applied so the tiny update is not rounded
        # away by loading the base model directly as BF16.
        dtype = torch.float32 if parameter_delta_path or force_float32 else (
            torch.bfloat16 if self.device.type == "cuda" else torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path or model_name_or_path, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            local_files_only=True,
        ).to(self.device)
        self.parameter_delta_path = parameter_delta_path
        if parameter_delta_path:
            payload = torch.load(parameter_delta_path, map_location="cpu", weights_only=True)
            if payload.get("format") != "selected_parameter_delta_v1":
                raise ValueError(f"Unsupported parameter delta: {payload.get('format')!r}")
            parameters = dict(self.model.named_parameters())
            unexpected = sorted(set(payload["state_dict"]) - set(parameters))
            if unexpected:
                raise ValueError(f"Delta contains unknown parameters: {unexpected[:3]}")
            with torch.no_grad():
                for name, delta in payload["state_dict"].items():
                    parameters[name].add_(delta.to(device=self.device, dtype=parameters[name].dtype))
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.batch_size = batch_size
        self.max_length = max_length
        self.length_penalty = length_penalty

    def _tokenize_pair(self, prompt: str, candidate: str) -> tuple[list[int], list[int]]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        # The explicit leading space is the completion boundary after "Command:".
        candidate_ids = self.tokenizer.encode(" " + candidate, add_special_tokens=False)
        if not candidate_ids:
            raise ValueError(f"Candidate produced no tokens: {candidate!r}")
        max_prompt = self.max_length - len(candidate_ids)
        if max_prompt < 1:
            raise ValueError(f"Candidate is longer than max_length: {candidate!r}")
        # Keep the end of the prompt: it contains current observation, candidates,
        # and output instruction. Prompt builders bound history/skill size.
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]
        return prompt_ids, candidate_ids

    @torch.inference_mode()
    def score(self, prompt: str, candidates: Sequence[str]) -> CandidateScores:
        if not candidates:
            raise ValueError("At least one candidate is required")
        pairs = [self._tokenize_pair(prompt, candidate) for candidate in candidates]
        sums: list[torch.Tensor] = []
        lengths: list[int] = []
        pad_id = int(self.tokenizer.pad_token_id)

        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            full_ids = [prompt_ids + candidate_ids for prompt_ids, candidate_ids in batch]
            width = max(len(ids) for ids in full_ids)
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=self.device)
            attention_mask = torch.zeros_like(input_ids)
            for row, ids in enumerate(full_ids):
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
                attention_mask[row, : len(ids)] = 1
            prompt_lengths = {len(prompt_ids) for prompt_ids, _ in batch}
            if len(prompt_lengths) != 1:
                raise AssertionError("All candidates in a score batch must share one prompt")
            prompt_length = next(iter(prompt_lengths))
            # Only completion-prediction positions are needed. For Qwen this
            # avoids materializing [batch, full_prompt, vocab] logits.
            logits_to_keep = torch.arange(prompt_length - 1, width - 1, device=self.device)
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=logits_to_keep,
            ).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            for row, (prompt_ids, candidate_ids) in enumerate(batch):
                targets = torch.tensor(candidate_ids, dtype=torch.long, device=self.device)
                local_positions = torch.arange(len(candidate_ids), device=self.device)
                token_log_probs = log_probs[row, local_positions, targets]
                sums.append(token_log_probs.sum().cpu())
                lengths.append(len(candidate_ids))

        sum_tensor = torch.stack(sums)
        length_tensor = torch.tensor(lengths, dtype=torch.float32)
        normalized = sum_tensor / length_tensor.pow(self.length_penalty)
        return CandidateScores(tuple(candidates), sum_tensor, normalized, length_tensor)
