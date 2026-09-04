from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F
import yaml

from .alfworld_runner import categorical_kl, summarize_condition


@dataclass
class RepairState:
    episode_id: str
    task_type: str
    action_verb: str
    candidates: list[str]
    gold_index: int
    plain_scores: torch.Tensor
    skill_scores: torch.Tensor
    mismatched_scores: torch.Tensor | None = None
    episode_key: str | None = None


def load_trace(path: str | Path) -> list[RepairState]:
    states = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            states.append(
                RepairState(
                    episode_id=row["episode_id"],
                    task_type=row["task_type"],
                    action_verb=row["action_verb"],
                    candidates=row["admissible_actions"],
                    gold_index=int(row["gold_index"]),
                    plain_scores=torch.tensor(row["normalized_scores"]["plain"], dtype=torch.float32),
                    skill_scores=torch.tensor(
                        row["normalized_scores"]["evolved_skill"], dtype=torch.float32
                    ),
                    mismatched_scores=torch.tensor(
                        row["normalized_scores"]["mismatched_skill"], dtype=torch.float32
                    ) if "mismatched_skill" in row["normalized_scores"] else None,
                    episode_key=row.get("episode_key", row.get("gamefile", row["episode_id"])),
                )
            )
    return states


def candidate_verb(command: str) -> str:
    return command.split(maxsplit=1)[0] if command.strip() else "<empty>"


class VerbVocabulary:
    def __init__(self, states: Sequence[RepairState]):
        verbs = sorted({candidate_verb(candidate) for state in states for candidate in state.candidates})
        self.names = ["<unk>"] + verbs
        self.index = {name: idx for idx, name in enumerate(self.names)}

    def encode(self, candidates: Sequence[str]) -> torch.Tensor:
        return torch.tensor(
            [self.index.get(candidate_verb(candidate), 0) for candidate in candidates],
            dtype=torch.long,
        )


def distribution_features(plain_scores: torch.Tensor, skill_scores: torch.Tensor) -> torch.Tensor:
    plain = torch.softmax(plain_scores, dim=-1)
    skill = torch.softmax(skill_scores, dim=-1)
    delta = skill_scores - plain_scores
    entropy = -(plain * plain.clamp_min(1e-12).log()).sum()
    skill_entropy = -(skill * skill.clamp_min(1e-12).log()).sum()
    return torch.stack(
        [
            entropy,
            skill_entropy,
            plain.max(),
            skill.max(),
            torch.tensor(categorical_kl(skill, plain)),
            delta.abs().mean(),
            torch.tensor(math.log1p(len(plain_scores)) / 4.0),
        ]
    )


class ConvexSkillGate(nn.Module):
    """A deployable gate constrained to accept or reject the skill delta."""

    def __init__(self, hidden_size: int = 12):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(7, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.log_base_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        plain_scores: torch.Tensor,
        skill_scores: torch.Tensor,
        _verb_ids: torch.Tensor,
        _lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        plain = plain_scores - plain_scores.mean()
        delta = (skill_scores - plain_scores)
        delta = delta - delta.mean()
        alpha = torch.sigmoid(self.gate(distribution_features(plain_scores, skill_scores))).squeeze()
        logits = self.log_base_scale.exp() * plain + alpha * delta
        return logits, alpha


class SignedLogicRepairHead(nn.Module):
    """Tiny stage-aware residual head over frozen command distributions."""

    def __init__(self, verb_count: int, hidden_size: int = 16, alpha_limit: float = 2.0):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(7, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.log_base_scale = nn.Parameter(torch.zeros(()))
        self.verb_bias = nn.Embedding(verb_count, 1)
        self.verb_delta_scale = nn.Embedding(verb_count, 1)
        self.length_weight = nn.Parameter(torch.zeros(()))
        self.alpha_limit = alpha_limit
        nn.init.zeros_(self.verb_bias.weight)
        nn.init.zeros_(self.verb_delta_scale.weight)

    def forward(
        self,
        plain_scores: torch.Tensor,
        skill_scores: torch.Tensor,
        verb_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        plain = plain_scores - plain_scores.mean()
        delta = skill_scores - plain_scores
        delta = delta - delta.mean()
        alpha = self.alpha_limit * torch.tanh(
            self.gate(distribution_features(plain_scores, skill_scores))
        ).squeeze()
        centered_length = lengths - lengths.mean()
        logits = (
            self.log_base_scale.exp() * plain
            + alpha * delta
            + self.verb_bias(verb_ids).squeeze(-1)
            + self.verb_delta_scale(verb_ids).squeeze(-1) * delta
            + self.length_weight * centered_length
        )
        return logits, alpha


class StageCalibrationHead(nn.Module):
    """Ablation: calibrates action stages without reading the skill distribution."""

    def __init__(self, verb_count: int):
        super().__init__()
        self.log_base_scale = nn.Parameter(torch.zeros(()))
        self.verb_bias = nn.Embedding(verb_count, 1)
        self.length_weight = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.verb_bias.weight)

    def forward(
        self,
        plain_scores: torch.Tensor,
        _skill_scores: torch.Tensor,
        verb_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        plain = plain_scores - plain_scores.mean()
        centered_length = lengths - lengths.mean()
        logits = (
            self.log_base_scale.exp() * plain
            + self.verb_bias(verb_ids).squeeze(-1)
            + self.length_weight * centered_length
        )
        return logits, torch.zeros((), dtype=plain_scores.dtype)


def _inputs(state: RepairState, vocabulary: VerbVocabulary):
    verb_ids = vocabulary.encode(state.candidates)
    lengths = torch.tensor([len(candidate.split()) for candidate in state.candidates], dtype=torch.float32)
    return state.plain_scores, state.skill_scores, verb_ids, lengths


def split_by_episode(
    states: Sequence[RepairState], dev_fraction: float, seed: int
) -> tuple[list[RepairState], list[RepairState]]:
    episodes = sorted({state.episode_key or state.episode_id for state in states})
    random.Random(seed).shuffle(episodes)
    dev_count = max(1, round(len(episodes) * dev_fraction))
    dev_episodes = set(episodes[:dev_count])
    return (
        [state for state in states if (state.episode_key or state.episode_id) not in dev_episodes],
        [state for state in states if (state.episode_key or state.episode_id) in dev_episodes],
    )


@torch.no_grad()
def mean_nll(model: nn.Module, states: Sequence[RepairState], vocabulary: VerbVocabulary) -> float:
    if not states:
        return float("inf")
    losses = []
    model.eval()
    for state in states:
        logits, _ = model(*_inputs(state, vocabulary))
        losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([state.gold_index])))
    return float(torch.stack(losses).mean())


@torch.no_grad()
def mean_kl(model: nn.Module, states: Sequence[RepairState], vocabulary: VerbVocabulary) -> float:
    if not states:
        return float("inf")
    values = []
    model.eval()
    for state in states:
        logits, _ = model(*_inputs(state, vocabulary))
        repaired = torch.softmax(logits, dim=-1)
        plain = torch.softmax(state.plain_scores, dim=-1)
        values.append(categorical_kl(repaired, plain))
    return sum(values) / len(values)


def train_head(
    model: nn.Module,
    train_states: Sequence[RepairState],
    dev_states: Sequence[RepairState],
    vocabulary: VerbVocabulary,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    kl_lambda: float = 0.0,
) -> tuple[nn.Module, dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_dev_nll = mean_nll(model, dev_states, vocabulary)
    best_dev_kl = mean_kl(model, dev_states, vocabulary)
    best_objective = best_dev_nll + kl_lambda * best_dev_kl
    stale = 0
    history = []
    for epoch in range(epochs):
        model.train()
        order = list(train_states)
        random.shuffle(order)
        optimizer.zero_grad()
        losses = []
        for state in order:
            logits, _ = model(*_inputs(state, vocabulary))
            cross_entropy = F.cross_entropy(logits.unsqueeze(0), torch.tensor([state.gold_index]))
            repaired = torch.softmax(logits, dim=-1)
            plain = torch.softmax(state.plain_scores, dim=-1)
            kl = (repaired * (repaired.clamp_min(1e-12).log() - plain.clamp_min(1e-12).log())).sum()
            losses.append(cross_entropy + kl_lambda * kl)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        dev_nll = mean_nll(model, dev_states, vocabulary)
        dev_kl = mean_kl(model, dev_states, vocabulary)
        objective = dev_nll + kl_lambda * dev_kl
        history.append(
            {
                "epoch": epoch,
                "train_objective": float(loss.detach()),
                "dev_nll": dev_nll,
                "dev_kl": dev_kl,
                "dev_objective": objective,
            }
        )
        if objective < best_objective - 1e-5:
            best_objective = objective
            best_dev_nll = dev_nll
            best_dev_kl = dev_kl
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, {
        "kl_lambda": kl_lambda,
        "best_dev_nll": best_dev_nll,
        "best_dev_kl": best_dev_kl,
        "best_dev_objective": best_objective,
        "epochs_run": len(history),
        "history": history,
    }


def _metric_row(probabilities: torch.Tensor, plain: torch.Tensor, gold: int, alpha: float) -> dict[str, Any]:
    predicted = int(probabilities.argmax())
    expert_probability = float(probabilities[gold])
    rank = int((torch.argsort(probabilities, descending=True) == gold).nonzero()[0, 0]) + 1
    return {
        "correct": int(predicted == gold),
        "confidence": float(probabilities.max()),
        "expert_probability": expert_probability,
        "nll": -math.log(max(expert_probability, 1e-12)),
        "rank": rank,
        "entropy": float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()),
        "kl_from_plain": categorical_kl(probabilities, plain),
        "skill_selected": float(alpha > 0.5),
        "rho": alpha,
    }


@torch.no_grad()
def evaluate_heads(
    states: Sequence[RepairState],
    vocabulary: VerbVocabulary,
    heads: dict[str, nn.Module],
    kl_budget: float,
) -> dict[str, Any]:
    condition_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_verb: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    by_task: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    alphas: dict[str, list[float]] = defaultdict(list)
    for state in states:
        plain = torch.softmax(state.plain_scores, dim=-1)
        skill = torch.softmax(state.skill_scores, dim=-1)
        base_conditions = {"plain": (plain, 0.0), "evolved_skill": (skill, 1.0)}
        if state.mismatched_scores is not None:
            base_conditions["mismatched_skill"] = (
                torch.softmax(state.mismatched_scores, dim=-1),
                1.0,
            )
        for name, (distribution, alpha) in base_conditions.items():
            row = _metric_row(distribution, plain, state.gold_index, alpha)
            condition_rows[name].append(row)
            by_verb[state.action_verb][name].append(row["correct"])
            by_task[state.task_type][name].append(row["correct"])
        for name, head in heads.items():
            head.eval()
            logits, alpha_tensor = head(*_inputs(state, vocabulary))
            distribution = torch.softmax(logits, dim=-1)
            alpha = float(alpha_tensor)
            row = _metric_row(distribution, plain, state.gold_index, alpha)
            condition_rows[name].append(row)
            by_verb[state.action_verb][name].append(row["correct"])
            by_task[state.task_type][name].append(row["correct"])
            alphas[name].append(alpha)
            if name == "signed_logic_repair_head":
                projected, rho = _project_distribution(plain, distribution, kl_budget)
                projected_name = "signed_logic_repair_trust_region"
                projected_row = _metric_row(projected, plain, state.gold_index, rho)
                condition_rows[projected_name].append(projected_row)
                by_verb[state.action_verb][projected_name].append(projected_row["correct"])
                by_task[state.task_type][projected_name].append(projected_row["correct"])
                alphas[projected_name].append(rho)
    return {
        "conditions": {name: summarize_condition(rows) for name, rows in condition_rows.items()},
        "alpha_statistics": {
            name: {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "negative_rate": sum(value < 0 for value in values) / len(values),
            }
            for name, values in alphas.items()
        },
        "by_action_verb": {
            verb: {
                name: {"count": len(values), "accuracy": sum(values) / len(values)}
                for name, values in conditions.items()
            }
            for verb, conditions in sorted(by_verb.items())
        },
        "by_task_type": {
            task: {
                name: {"count": len(values), "accuracy": sum(values) / len(values)}
                for name, values in conditions.items()
            }
            for task, conditions in sorted(by_task.items())
        },
    }


def _project_distribution(
    plain: torch.Tensor,
    target: torch.Tensor,
    kl_budget: float,
    iterations: int = 30,
) -> tuple[torch.Tensor, float]:
    if categorical_kl(target, plain) <= kl_budget:
        return target, 1.0
    log_plain = plain.clamp_min(1e-12).log()
    direction = target.clamp_min(1e-12).log() - log_plain
    low, high = 0.0, 1.0
    for _ in range(iterations):
        rho = (low + high) / 2
        candidate = torch.softmax(log_plain + rho * direction, dim=-1)
        if categorical_kl(candidate, plain) <= kl_budget:
            low = rho
        else:
            high = rho
    return torch.softmax(log_plain + low * direction, dim=-1), low


def run(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    train_states = load_trace(config["train_trace"])
    eval_states = load_trace(config["eval_trace"])
    fit_states, dev_states = split_by_episode(train_states, float(config["dev_fraction"]), seed)
    vocabulary = VerbVocabulary(fit_states)
    train_kwargs = {
        "train_states": fit_states,
        "dev_states": dev_states,
        "vocabulary": vocabulary,
        "epochs": int(config["epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "patience": int(config["patience"]),
    }
    convex, convex_train = train_head(ConvexSkillGate(), **train_kwargs)
    stage, stage_train = train_head(StageCalibrationHead(len(vocabulary.names)), **train_kwargs)
    signed, signed_train = train_head(SignedLogicRepairHead(len(vocabulary.names)), **train_kwargs)
    constrained_candidates = []
    constrained_stage_candidates = []
    for candidate_index, kl_lambda in enumerate(config.get("constrained_kl_lambdas", [0.1, 0.5, 1.0])):
        torch.manual_seed(seed + 100 + candidate_index)
        candidate, candidate_train = train_head(
            SignedLogicRepairHead(len(vocabulary.names)),
            kl_lambda=float(kl_lambda),
            **train_kwargs,
        )
        constrained_candidates.append((candidate, candidate_train))
        torch.manual_seed(seed + 200 + candidate_index)
        stage_candidate, stage_candidate_train = train_head(
            StageCalibrationHead(len(vocabulary.names)),
            kl_lambda=float(kl_lambda),
            **train_kwargs,
        )
        constrained_stage_candidates.append((stage_candidate, stage_candidate_train))
    kl_budget = float(config.get("kl_budget", 0.05))
    feasible = [item for item in constrained_candidates if item[1]["best_dev_kl"] <= kl_budget]
    if feasible:
        constrained, constrained_train = min(feasible, key=lambda item: item[1]["best_dev_nll"])
    else:
        constrained, constrained_train = min(
            constrained_candidates, key=lambda item: item[1]["best_dev_kl"]
        )
    feasible_stage = [
        item for item in constrained_stage_candidates if item[1]["best_dev_kl"] <= kl_budget
    ]
    if feasible_stage:
        constrained_stage, constrained_stage_train = min(
            feasible_stage, key=lambda item: item[1]["best_dev_nll"]
        )
    else:
        constrained_stage, constrained_stage_train = min(
            constrained_stage_candidates, key=lambda item: item[1]["best_dev_kl"]
        )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": convex.state_dict(), "verbs": vocabulary.names},
        output_dir / "convex_skill_gate.pt",
    )
    torch.save(
        {"state_dict": stage.state_dict(), "verbs": vocabulary.names},
        output_dir / "stage_calibration_head.pt",
    )
    torch.save(
        {"state_dict": constrained_stage.state_dict(), "verbs": vocabulary.names},
        output_dir / "constrained_stage_calibration_head.pt",
    )
    torch.save(
        {"state_dict": constrained.state_dict(), "verbs": vocabulary.names},
        output_dir / "constrained_logic_repair_head.pt",
    )
    torch.save(
        {"state_dict": signed.state_dict(), "verbs": vocabulary.names},
        output_dir / "signed_logic_repair_head.pt",
    )
    heads = {
        "convex_skill_gate": convex,
        "stage_calibration_head": stage,
        "signed_logic_repair_head": signed,
        "constrained_logic_repair_head": constrained,
        "constrained_stage_calibration_head": constrained_stage,
    }
    results = {
        "seed": seed,
        "parameter_counts": {
            "convex_skill_gate": sum(parameter.numel() for parameter in convex.parameters()),
            "stage_calibration_head": sum(parameter.numel() for parameter in stage.parameters()),
            "signed_logic_repair_head": sum(parameter.numel() for parameter in signed.parameters()),
            "constrained_logic_repair_head": sum(
                parameter.numel() for parameter in constrained.parameters()
            ),
            "constrained_stage_calibration_head": sum(
                parameter.numel() for parameter in constrained_stage.parameters()
            ),
        },
        "data": {
            "train_decisions": len(fit_states),
            "dev_decisions": len(dev_states),
            "eval_decisions": len(eval_states),
            "train_episodes": len({state.episode_key or state.episode_id for state in fit_states}),
            "dev_episodes": len({state.episode_key or state.episode_id for state in dev_states}),
            "eval_episodes": len({state.episode_key or state.episode_id for state in eval_states}),
            "train_gold_verbs": dict(Counter(state.action_verb for state in fit_states)),
        },
        "training": {
            "convex_skill_gate": convex_train,
            "stage_calibration_head": stage_train,
            "signed_logic_repair_head": signed_train,
            "constrained_logic_repair_head": constrained_train,
            "constrained_candidates": [item[1] for item in constrained_candidates],
            "constrained_stage_calibration_head": constrained_stage_train,
            "constrained_stage_candidates": [
                item[1] for item in constrained_stage_candidates
            ],
        },
        "evaluation": evaluate_heads(
            eval_states,
            vocabulary,
            heads,
            kl_budget=kl_budget,
        ),
        "additional_evaluations": {
            name: evaluate_heads(load_trace(path), vocabulary, heads, kl_budget=kl_budget)
            for name, path in config.get("additional_eval_traces", {}).items()
        },
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
