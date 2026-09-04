from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch.nn import functional as F

from .benchmark import ACTION_LABELS, CHOICE_LABELS, TOOL_NAMES
from .controller import DistributionRepairHead
from .model import EncodedTask, StateEncoding
from .repair import RepairResult, repair_distribution


@dataclass
class RolloutResult:
    task_id: str
    proposed_tool: int
    selected_tool: int
    proposed_choice: int
    selected_choice: int
    tool_correct: bool
    choice_correct: bool
    strict_success: bool
    final_answer_nll: torch.Tensor
    route_nll: torch.Tensor
    repair_kl: torch.Tensor
    trace: dict


def _mask(action_count: int, indices: Iterable[int], device: torch.device) -> torch.Tensor:
    result = torch.zeros(action_count, dtype=torch.bool, device=device)
    result[list(indices)] = True
    return result


def _policy_logits(
    head: DistributionRepairHead,
    state: StateEncoding,
    indices: Iterable[int],
    device: torch.device,
    base_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_mask = _mask(head.action_count, indices, device)
    logits = head(
        state.hidden.to(device),
        state.base_logits.to(device),
        action_mask,
        base_temperature=base_temperature,
    )[0]
    return logits, action_mask


def _subset_kl(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_prob = torch.log_softmax(logits[mask], dim=-1)
    return F.kl_div(log_prob, target[mask].detach(), reduction="sum")


def _distribution_record(repair: RepairResult, mask: torch.Tensor) -> dict:
    p = repair.proposal_prob.detach().cpu()
    q = repair.repaired_prob.detach().cpu()
    active_p = p[mask.cpu()]
    sorted_p = torch.sort(active_p, descending=True).values
    entropy = float((-(active_p * active_p.clamp_min(1e-12).log()).sum()).item())
    margin = float((sorted_p[0] - sorted_p[1]).item()) if len(sorted_p) > 1 else 1.0
    return {
        "proposal": [round(float(value), 7) for value in p],
        "repaired": [round(float(value), 7) for value in q],
        "proposed_action": repair.proposed_action,
        "repaired_action": repair.repaired_action,
        "accepted": repair.accepted,
        "entropy": entropy,
        "top1_margin": margin,
        "verified_mask": repair.verified_mask.detach().cpu().tolist(),
    }


def rollout(
    head: DistributionRepairHead,
    encoded: EncodedTask,
    device: torch.device,
    base_temperature: float,
    repair_beta: float,
    repair_top_k: int,
    use_online_repair: bool,
) -> RolloutResult:
    task = encoded.task
    route_logits, route_mask = _policy_logits(
        head, encoded.route_state, range(len(TOOL_NAMES)), device, base_temperature
    )
    route_scores = torch.zeros(head.action_count, device=device)
    route_scores[: len(TOOL_NAMES)] = torch.tensor(task.tool_scores(), device=device)
    route_repair = repair_distribution(
        route_logits, route_scores, route_mask, beta=repair_beta, top_k=repair_top_k
    )
    selected_tool = (
        route_repair.repaired_action if use_online_repair else route_repair.proposed_action
    )

    answer_logits, answer_mask = _policy_logits(
        head, encoded.answer_states[selected_tool], CHOICE_LABELS, device, base_temperature
    )
    answer_scores = torch.zeros(head.action_count, device=device)
    answer_scores[list(CHOICE_LABELS)] = torch.tensor(task.choice_scores(), device=device)
    answer_repair = repair_distribution(
        answer_logits, answer_scores, answer_mask, beta=repair_beta, top_k=repair_top_k
    )
    selected_choice = (
        answer_repair.repaired_action if use_online_repair else answer_repair.proposed_action
    )
    correct_choice = task.correct_choice_label
    route_log_prob = torch.log_softmax(route_logits[route_mask], dim=-1)
    answer_log_prob = torch.log_softmax(answer_logits[answer_mask], dim=-1)
    route_nll = -route_log_prob[task.correct_tool_label]
    final_answer_nll = -answer_log_prob[correct_choice - CHOICE_LABELS[0]]
    repair_kl = _subset_kl(route_logits, route_repair.repaired_prob, route_mask)
    repair_kl = repair_kl + _subset_kl(answer_logits, answer_repair.repaired_prob, answer_mask)
    tool_correct = selected_tool == task.correct_tool_label
    choice_correct = selected_choice == correct_choice
    trace = {
        "task_id": task.task_id,
        "operation": task.operation,
        "question": task.question,
        "answer": task.answer,
        "route": _distribution_record(route_repair, route_mask),
        "answer_step": _distribution_record(answer_repair, answer_mask),
        "selected_tool": selected_tool,
        "selected_choice": selected_choice,
        "tool_correct": tool_correct,
        "choice_correct": choice_correct,
        "strict_success": tool_correct and choice_correct,
        "final_answer_nll": float(final_answer_nll.detach().cpu().item()),
    }
    return RolloutResult(
        task_id=task.task_id,
        proposed_tool=route_repair.proposed_action,
        selected_tool=selected_tool,
        proposed_choice=answer_repair.proposed_action,
        selected_choice=selected_choice,
        tool_correct=tool_correct,
        choice_correct=choice_correct,
        strict_success=tool_correct and choice_correct,
        final_answer_nll=final_answer_nll,
        route_nll=route_nll,
        repair_kl=repair_kl,
        trace=trace,
    )


def objective(
    head: DistributionRepairHead,
    encoded_tasks: list[EncodedTask],
    device: torch.device,
    base_temperature: float,
    repair_beta: float,
    repair_top_k: int,
    mode: str,
    outcome_weight: float,
) -> torch.Tensor:
    losses = []
    for encoded in encoded_tasks:
        result = rollout(
            head,
            encoded,
            device,
            base_temperature,
            repair_beta,
            repair_top_k,
            use_online_repair=False,
        )
        failure = result.final_answer_nll.new_tensor(float(not result.strict_success))
        if mode == "answer_nll":
            loss = result.final_answer_nll + outcome_weight * failure
        elif mode == "repair_kl":
            loss = result.repair_kl + outcome_weight * result.final_answer_nll
        elif mode == "route_and_answer_nll":
            loss = result.route_nll + result.final_answer_nll + outcome_weight * failure
        elif mode == "outcome":
            loss = failure
        else:
            raise ValueError(f"Unknown objective mode: {mode}")
        losses.append(loss)
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(
    head: DistributionRepairHead,
    encoded_tasks: list[EncodedTask],
    device: torch.device,
    base_temperature: float,
    repair_beta: float,
    repair_top_k: int,
    use_online_repair: bool,
) -> tuple[dict, list[dict]]:
    results = [
        rollout(
            head,
            encoded,
            device,
            base_temperature,
            repair_beta,
            repair_top_k,
            use_online_repair,
        )
        for encoded in encoded_tasks
    ]
    count = max(len(results), 1)
    traces = [result.trace for result in results]
    route_repairs = sum(trace["route"]["accepted"] is False for trace in traces)
    answer_repairs = sum(trace["answer_step"]["accepted"] is False for trace in traces)
    metrics = {
        "count": len(results),
        "tool_accuracy": sum(result.tool_correct for result in results) / count,
        "choice_accuracy": sum(result.choice_correct for result in results) / count,
        "strict_accuracy": sum(result.strict_success for result in results) / count,
        "mean_final_answer_nll": sum(
            float(result.final_answer_nll.cpu().item()) for result in results
        )
        / count,
        "mean_route_entropy": sum(trace["route"]["entropy"] for trace in traces) / count,
        "mean_answer_entropy": sum(trace["answer_step"]["entropy"] for trace in traces) / count,
        "route_repair_rate": route_repairs / count,
        "answer_repair_rate": answer_repairs / count,
        "online_repair": use_online_repair,
    }
    return metrics, traces

