from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch.nn import functional as F

from .benchmark import CHOICE_LABELS, TOOL_NAMES
from .controller import DistributionRepairHead
from .model import EncodedTask, StateEncoding


@dataclass
class SkillEncodedTask:
    plain: EncodedTask
    skill_route: StateEncoding
    skill_answers: dict[int, StateEncoding]


@dataclass
class DistillationBatch:
    hidden: torch.Tensor
    base_logits: torch.Tensor
    masks: torch.Tensor
    targets: torch.Tensor
    plain_prob: torch.Tensor


def _mask(action_count: int, indices: Iterable[int], device: torch.device) -> torch.Tensor:
    result = torch.zeros(action_count, dtype=torch.bool, device=device)
    result[list(indices)] = True
    return result


def masked_prob(logits: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    output = torch.zeros_like(logits, dtype=torch.float32)
    output[mask] = torch.softmax(logits[mask].float() / temperature, dim=-1)
    return output


def categorical_kl(q: torch.Tensor, p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    q_active = q[mask].clamp_min(1e-12)
    p_active = p[mask].clamp_min(1e-12)
    return (q_active * (q_active.log() - p_active.log())).sum()


def trust_region_fusion(
    plain_logits: torch.Tensor,
    skill_logits: torch.Tensor,
    verifier_values: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    skill_weight: float,
    verifier_weight: float,
    residual_clip: float,
    max_kl: float,
    require_positive_skill_gain: bool,
) -> tuple[torch.Tensor, dict]:
    """Fuse skill likelihood-ratio and verifier value under KL(q || p) <= delta."""

    p = masked_prob(plain_logits, mask, temperature)
    p_skill = masked_prob(skill_logits, mask, temperature)
    log_ratio = torch.zeros_like(p)
    log_ratio[mask] = (
        p_skill[mask].clamp_min(1e-12).log() - p[mask].clamp_min(1e-12).log()
    ).clamp(-residual_clip, residual_clip)
    expected_plain = (p * verifier_values).sum()
    expected_skill = (p_skill * verifier_values).sum()
    skill_gain = expected_skill - expected_plain
    skill_gate = float((not require_positive_skill_gain) or skill_gain.item() > 0.0)
    residual = skill_weight * skill_gate * log_ratio + verifier_weight * verifier_values

    def candidate(scale: float) -> torch.Tensor:
        q = torch.zeros_like(p)
        q[mask] = torch.softmax(
            p[mask].clamp_min(1e-12).log() + scale * residual[mask], dim=-1
        )
        return q

    scale = 1.0
    q = candidate(scale)
    if max_kl > 0 and categorical_kl(q, p, mask).item() > max_kl:
        low, high = 0.0, 1.0
        for _ in range(32):
            middle = (low + high) / 2
            trial = candidate(middle)
            if categorical_kl(trial, p, mask).item() <= max_kl:
                low = middle
            else:
                high = middle
        scale = low
        q = candidate(scale)
    return q, {
        "skill_gain": float(skill_gain.item()),
        "skill_gate": skill_gate,
        "projection_scale": scale,
        "kl_to_plain": float(categorical_kl(q, p, mask).item()),
    }


def _head_prob(
    head: DistributionRepairHead,
    state: StateEncoding,
    mask: torch.Tensor,
    device: torch.device,
    temperature: float,
) -> torch.Tensor:
    logits = head(
        state.hidden.to(device), state.base_logits.to(device), mask,
        base_temperature=temperature,
    )[0]
    output = torch.zeros_like(logits)
    output[mask] = torch.softmax(logits[mask], dim=-1)
    return output


def skill_teacher_target(
    item: SkillEncodedTask,
    step: str,
    selected_tool: int,
    device: torch.device,
    temperature: float,
    fusion: dict,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    task = item.plain.task
    if step == "route":
        plain_state, skill_state = item.plain.route_state, item.skill_route
        indices = range(len(TOOL_NAMES))
        values = torch.zeros(9, device=device)
        values[task.correct_tool_label] = 1.0
    elif step == "answer":
        plain_state = item.plain.answer_states[selected_tool]
        skill_state = item.skill_answers[selected_tool]
        indices = CHOICE_LABELS
        values = torch.zeros(9, device=device)
        values[task.correct_choice_label] = 1.0
    else:
        raise ValueError(step)
    mask = _mask(9, indices, device)
    target, diagnostics = trust_region_fusion(
        plain_state.base_logits.to(device),
        skill_state.base_logits.to(device),
        values,
        mask,
        temperature,
        **fusion,
    )
    return target, mask, diagnostics


@torch.no_grad()
def build_distillation_batch(
    items: list[SkillEncodedTask],
    device: torch.device,
    temperature: float,
    fusion: dict,
) -> DistillationBatch:
    hidden, base_logits, masks, targets, plain_probs = [], [], [], [], []
    for item in items:
        task = item.plain.task
        for step, tool in (("route", task.correct_tool_label), ("answer", task.correct_tool_label)):
            target, mask, _ = skill_teacher_target(
                item, step, tool, device, temperature, fusion
            )
            state = (
                item.plain.route_state if step == "route" else item.plain.answer_states[tool]
            )
            hidden.append(state.hidden.to(device))
            base_logits.append(state.base_logits.to(device))
            masks.append(mask)
            targets.append(target)
            plain_probs.append(masked_prob(state.base_logits.to(device), mask, temperature))
    return DistillationBatch(
        hidden=torch.stack(hidden),
        base_logits=torch.stack(base_logits),
        masks=torch.stack(masks),
        targets=torch.stack(targets),
        plain_prob=torch.stack(plain_probs),
    )


def distillation_objective(
    head: DistributionRepairHead,
    batch: DistillationBatch,
    temperature: float,
    anchor_weight: float,
) -> torch.Tensor:
    logits = head(
        batch.hidden, batch.base_logits, batch.masks, base_temperature=temperature
    )
    log_prob = torch.log_softmax(logits, dim=-1)
    learned = torch.softmax(logits, dim=-1)
    cross_entropy = -(batch.targets * log_prob.masked_fill(~batch.masks, 0.0)).sum(dim=-1)
    if not anchor_weight:
        return cross_entropy.mean()
    safe_learned = learned.clamp_min(1e-12)
    safe_plain = batch.plain_prob.clamp_min(1e-12)
    anchor = (
        safe_learned * (safe_learned.log() - safe_plain.log())
    ).masked_fill(~batch.masks, 0.0).sum(dim=-1)
    return (cross_entropy + anchor_weight * anchor).mean()


def _ece(confidences: list[float], correct: list[bool], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    total = len(confidences)
    value = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, c in enumerate(confidences) if low <= c < high or (index == bins - 1 and c == 1)]
        if members:
            acc = sum(correct[i] for i in members) / len(members)
            conf = sum(confidences[i] for i in members) / len(members)
            value += len(members) / total * abs(acc - conf)
    return value


@torch.no_grad()
def evaluate_skill_method(
    items: list[SkillEncodedTask],
    method: str,
    device: torch.device,
    temperature: float,
    fusion: dict | None = None,
    head: DistributionRepairHead | None = None,
) -> tuple[dict, list[dict]]:
    traces: list[dict] = []
    confidences: list[float] = []
    top1_correct: list[bool] = []
    correct_probs: list[float] = []
    briers: list[float] = []
    kls: list[float] = []
    skill_gains: list[float] = []
    skill_gates: list[float] = []
    strict = tool_correct_count = choice_correct_count = 0

    def distribution(item: SkillEncodedTask, step: str, tool: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        task = item.plain.task
        if step == "route":
            plain_state, skill_state = item.plain.route_state, item.skill_route
            indices = range(len(TOOL_NAMES))
            correct_action = task.correct_tool_label
        else:
            plain_state = item.plain.answer_states[tool]
            skill_state = item.skill_answers[tool]
            indices = CHOICE_LABELS
            correct_action = task.correct_choice_label
        mask = _mask(9, indices, device)
        plain = masked_prob(plain_state.base_logits.to(device), mask, temperature)
        diagnostics: dict = {}
        if method == "plain":
            prob = plain
        elif method == "skill_prompt":
            prob = masked_prob(skill_state.base_logits.to(device), mask, temperature)
        elif method == "safe_skill":
            values = torch.zeros(9, device=device)
            values[correct_action] = 1.0
            local_fusion = dict(fusion or {})
            local_fusion["verifier_weight"] = 0.0
            prob, diagnostics = trust_region_fusion(
                plain_state.base_logits.to(device), skill_state.base_logits.to(device),
                values, mask, temperature, **local_fusion,
            )
        elif method == "skill_verifier_projection":
            values = torch.zeros(9, device=device)
            values[correct_action] = 1.0
            prob, diagnostics = trust_region_fusion(
                plain_state.base_logits.to(device), skill_state.base_logits.to(device),
                values, mask, temperature, **(fusion or {}),
            )
        elif method == "internalized":
            if head is None:
                raise ValueError("head is required for internalized evaluation")
            prob = _head_prob(head, plain_state, mask, device, temperature)
        else:
            raise ValueError(method)
        return prob, plain, diagnostics

    for item in items:
        task = item.plain.task
        route_prob, route_plain, route_diag = distribution(item, "route", 0)
        selected_tool = int(torch.argmax(route_prob).item())
        answer_prob, answer_plain, answer_diag = distribution(item, "answer", selected_tool)
        selected_choice = int(torch.argmax(answer_prob).item())
        tool_ok = selected_tool == task.correct_tool_label
        choice_ok = selected_choice == task.correct_choice_label
        tool_correct_count += int(tool_ok)
        choice_correct_count += int(choice_ok)
        strict += int(tool_ok and choice_ok)

        records = []
        for name, prob, plain, correct_action, diag in (
            ("route", route_prob, route_plain, task.correct_tool_label, route_diag),
            ("answer", answer_prob, answer_plain, task.correct_choice_label, answer_diag),
        ):
            active = prob[prob > 0]
            prediction = int(torch.argmax(prob).item())
            confidence = float(prob[prediction].item())
            correct_prob = float(prob[correct_action].item())
            target = torch.zeros_like(prob)
            target[correct_action] = 1.0
            confidences.append(confidence)
            top1_correct.append(prediction == correct_action)
            correct_probs.append(correct_prob)
            briers.append(float(((prob - target) ** 2).sum().item()))
            active_mask = prob > 0
            kls.append(float(categorical_kl(prob, plain, active_mask).item()))
            if "skill_gain" in diag:
                skill_gains.append(float(diag["skill_gain"]))
                skill_gates.append(float(diag["skill_gate"]))
            records.append({
                "step": name,
                "plain": [round(float(x), 7) for x in plain.cpu()],
                "adjusted": [round(float(x), 7) for x in prob.cpu()],
                "correct_action": correct_action,
                "prediction": prediction,
                "correct_probability": correct_prob,
                "entropy": float(-(active * active.clamp_min(1e-12).log()).sum().item()),
                **diag,
            })
        traces.append({
            "task_id": task.task_id,
            "operation": task.operation,
            "method": method,
            "selected_tool": selected_tool,
            "selected_choice": selected_choice,
            "strict_success": tool_ok and choice_ok,
            "steps": records,
        })

    count = max(len(items), 1)
    metrics = {
        "count": len(items),
        "tool_accuracy": tool_correct_count / count,
        "choice_accuracy": choice_correct_count / count,
        "strict_accuracy": strict / count,
        "mean_correct_action_probability": sum(correct_probs) / max(len(correct_probs), 1),
        "mean_brier": sum(briers) / max(len(briers), 1),
        "ece": _ece(confidences, top1_correct),
        "mean_kl_to_plain": sum(kls) / max(len(kls), 1),
        "mean_skill_gain": sum(skill_gains) / max(len(skill_gains), 1),
        "skill_gate_rate": sum(skill_gates) / max(len(skill_gates), 1),
    }
    return metrics, traces


def compare_skill_shift(
    items: list[SkillEncodedTask], temperature: float, device: torch.device
) -> dict:
    deltas: list[float] = []
    route_deltas: list[float] = []
    answer_deltas: list[float] = []
    for item in items:
        task = item.plain.task
        for name, plain_state, skill_state, indices, correct in (
            ("route", item.plain.route_state, item.skill_route, range(len(TOOL_NAMES)), task.correct_tool_label),
            ("answer", item.plain.answer_states[task.correct_tool_label], item.skill_answers[task.correct_tool_label], CHOICE_LABELS, task.correct_choice_label),
        ):
            mask = _mask(9, indices, device)
            plain = masked_prob(plain_state.base_logits.to(device), mask, temperature)
            skilled = masked_prob(skill_state.base_logits.to(device), mask, temperature)
            delta = float((skilled[correct] - plain[correct]).item())
            deltas.append(delta)
            (route_deltas if name == "route" else answer_deltas).append(delta)
    return {
        "mean_correct_probability_delta": sum(deltas) / max(len(deltas), 1),
        "mean_route_delta": sum(route_deltas) / max(len(route_deltas), 1),
        "mean_answer_delta": sum(answer_deltas) / max(len(answer_deltas), 1),
        "positive_shift_rate": sum(x > 0 for x in deltas) / max(len(deltas), 1),
        "harmful_shift_rate": sum(x < 0 for x in deltas) / max(len(deltas), 1),
    }
