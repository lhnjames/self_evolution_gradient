from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .benchmark import ACTION_LABELS, LogicRouteTask, TOOL_NAMES


@dataclass
class StateEncoding:
    prompt: str
    hidden: torch.Tensor
    base_logits: torch.Tensor


@dataclass
class EncodedTask:
    task: LogicRouteTask
    route_state: StateEncoding
    answer_states: dict[int, StateEncoding]


class FrozenLLMScorer:
    def __init__(self, model_name_or_path: str, device: str, batch_size: int = 32):
        self.device = torch.device(device)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.batch_size = batch_size
        self.action_token_ids = []
        for label in ACTION_LABELS:
            token_ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Action label {label!r} is not a single token: {token_ids}")
            self.action_token_ids.append(token_ids[0])

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode_prompts(self, prompts: Iterable[str]) -> list[StateEncoding]:
        prompt_list = list(prompts)
        encoded: list[StateEncoding] = []
        action_ids = torch.tensor(self.action_token_ids, device=self.device)
        for start in range(0, len(prompt_list), self.batch_size):
            batch_prompts = prompt_list[start : start + self.batch_size]
            tokens = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            outputs = self.model(**tokens, output_hidden_states=True, use_cache=False)
            last_positions = tokens["attention_mask"].sum(dim=1) - 1
            if self.tokenizer.padding_side == "left":
                last_positions = torch.full_like(last_positions, tokens["input_ids"].shape[1] - 1)
            rows = torch.arange(len(batch_prompts), device=self.device)
            hidden = outputs.hidden_states[-1][rows, last_positions].float().cpu()
            vocab_logits = outputs.logits[rows, last_positions]
            base_logits = vocab_logits.index_select(-1, action_ids).float().cpu()
            for offset, prompt in enumerate(batch_prompts):
                encoded.append(StateEncoding(prompt, hidden[offset], base_logits[offset]))
        return encoded

    def encode_tasks(self, tasks: list[LogicRouteTask]) -> list[EncodedTask]:
        prompts: list[str] = []
        for task in tasks:
            prompts.append(task.step1_prompt())
            prompts.extend(task.step2_prompt(tool_label) for tool_label in range(len(TOOL_NAMES)))
        states = self.encode_prompts(prompts)
        result: list[EncodedTask] = []
        cursor = 0
        for task in tasks:
            route_state = states[cursor]
            cursor += 1
            answer_states = {}
            for tool_label in range(len(TOOL_NAMES)):
                answer_states[tool_label] = states[cursor]
                cursor += 1
            result.append(EncodedTask(task, route_state, answer_states))
        return result


def resolve_model_snapshot(cache_root: str, model_id: str = "Qwen--Qwen2.5-0.5B-Instruct") -> str:
    snapshots = Path(cache_root, "hub", f"models--{model_id}", "snapshots")
    candidates = sorted(path for path in snapshots.glob("*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No model snapshot found under {snapshots}")
    return str(candidates[-1])
