from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from .alfworld_data import (
    AlfworldDecision,
    _goal_from_observation,
    _task_type_from_gamefile,
    load_decisions,
)
from .alfworld_skills import AlfworldSkillBank, build_action_prompt
from .logic_repair_head import (
    RepairState,
    SignedLogicRepairHead,
    StageCalibrationHead,
    VerbVocabulary,
    _inputs,
)
from .sequence_scorer import SequenceActionScorer


POLICIES = (
    "plain",
    "evolved_skill",
    "constrained_stage",
    "constrained_skill",
    "constrained_skill_cycle_repair",
)


def _load_head(path: str, policy: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    verbs = checkpoint["verbs"]
    if policy in {"constrained_skill", "constrained_skill_cycle_repair"}:
        head = SignedLogicRepairHead(len(verbs))
    else:
        head = StageCalibrationHead(len(verbs))
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    vocabulary = object.__new__(VerbVocabulary)
    vocabulary.names = verbs
    vocabulary.index = {name: index for index, name in enumerate(verbs)}
    return head, vocabulary


def _unique_gamefiles(decision_file: str, episodes: int) -> list[str]:
    result = []
    seen = set()
    for decision in load_decisions(decision_file):
        if decision.gamefile not in seen:
            seen.add(decision.gamefile)
            result.append(decision.gamefile)
        if len(result) == episodes:
            break
    return result


@torch.no_grad()
def _select_action(
    policy: str,
    decision: AlfworldDecision,
    scorer: SequenceActionScorer,
    bank: AlfworldSkillBank,
    head,
    vocabulary,
    history_window: int,
    state_action_counts: Counter | None = None,
    repeat_limit: int = 2,
) -> tuple[str, dict[str, Any]]:
    plain_prompt = build_action_prompt(decision, history_window=history_window)
    plain_scores = scorer.score(plain_prompt, decision.admissible_actions).normalized_scores
    if policy == "plain":
        selected_scores = plain_scores
        alpha = 0.0
    else:
        skill_context = bank.render(decision, general_top_k=3, task_top_k=3, mistakes_top_k=2)
        skill_prompt = build_action_prompt(decision, skill_context, history_window)
        skill_scores = scorer.score(skill_prompt, decision.admissible_actions).normalized_scores
        if policy == "evolved_skill":
            selected_scores = skill_scores
            alpha = 1.0
        else:
            state = RepairState(
                episode_id=decision.episode_id,
                task_type=decision.task_type,
                action_verb="unknown",
                candidates=list(decision.admissible_actions),
                gold_index=0,
                plain_scores=plain_scores,
                skill_scores=skill_scores,
            )
            selected_scores, alpha_tensor = head(*_inputs(state, vocabulary))
            alpha = float(alpha_tensor.detach())
    masked_count = 0
    if policy == "constrained_skill_cycle_repair" and state_action_counts is not None:
        blocked = torch.tensor(
            [
                state_action_counts[(decision.observation, action)] >= repeat_limit
                for action in decision.admissible_actions
            ],
            dtype=torch.bool,
        )
        # Preserve a fallback if an episode has exhausted every command in an
        # identical textual state.
        if blocked.any() and not blocked.all():
            selected_scores = selected_scores.clone()
            selected_scores[blocked] = -torch.inf
            masked_count = int(blocked.sum())
    index = int(selected_scores.argmax())
    probabilities = torch.softmax(selected_scores, dim=-1)
    return decision.admissible_actions[index], {
        "selected_index": index,
        "confidence": float(probabilities[index].detach()),
        "alpha": alpha,
        "cycle_masked_actions": masked_count,
    }


def run(
    config_path: str,
    policy: str,
    split: str,
    device: str,
    output_dir: str,
    parameter_delta: str | None = None,
    force_float32: bool = False,
    model_path: str | None = None,
) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}")
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    os.environ["ALFWORLD_DATA"] = str(Path(config["data_root"]).resolve())
    gamefiles = _unique_gamefiles(config["decision_files"][split], int(config["episodes"]))
    model = SequenceActionScorer(
        model_path or config["model_path"],
        device=device,
        batch_size=int(config["batch_size"]),
        max_length=int(config["max_length"]),
        length_penalty=float(config["length_penalty"]),
        parameter_delta_path=parameter_delta,
        force_float32=force_float32,
    )
    bank = AlfworldSkillBank(config["skill_path"])
    head = vocabulary = None
    if policy in {"constrained_stage", "constrained_skill", "constrained_skill_cycle_repair"}:
        filename = (
            "constrained_logic_repair_head.pt"
            if policy in {"constrained_skill", "constrained_skill_cycle_repair"}
            else "constrained_stage_calibration_head.pt"
        )
        head, vocabulary = _load_head(str(Path(config["checkpoint_dir"], filename)), policy)

    with open(config["alfworld_config"], encoding="utf-8") as handle:
        env_config = yaml.safe_load(handle)
    env_config["general"]["use_cuda"] = False
    env_config["env"]["domain_randomization"] = False
    env_config["dataset"]["num_eval_games"] = -1
    mode = "eval_in_distribution" if split == "valid_seen" else "eval_out_of_distribution"
    from alfworld.agents.environment import get_environment

    base_env = get_environment(env_config["env"]["type"])(env_config, train_eval=mode)
    trajectories = []
    for episode_index, gamefile in enumerate(gamefiles):
        base_env.game_files = [gamefile]
        base_env.num_games = 1
        env = base_env.init_env(batch_size=1)
        try:
            env.seed(int(config["seed"]) + episode_index)
            observations, infos = env.reset()
            observation = str(observations[0])
            goal = _goal_from_observation(observation)
            history: list[tuple[str, str]] = []
            steps = []
            state_action_counts: Counter = Counter()
            won = False
            for step_index in range(int(config["max_steps"])):
                candidates = tuple(str(item) for item in infos["admissible_commands"][0])
                decision = AlfworldDecision(
                    split=split,
                    episode_id=Path(gamefile).parent.parent.name,
                    gamefile=gamefile,
                    task_type=_task_type_from_gamefile(gamefile),
                    goal=goal,
                    step_index=step_index,
                    observation=observation,
                    history=tuple(history),
                    admissible_actions=candidates,
                    expert_action=candidates[0],  # placeholder; never exposed to the prompt/head
                )
                action, diagnostics = _select_action(
                    policy,
                    decision,
                    model,
                    bank,
                    head,
                    vocabulary,
                    int(config["history_window"]),
                    state_action_counts=state_action_counts,
                    repeat_limit=int(config.get("repeat_limit", 2)),
                )
                state_action_counts[(observation, action)] += 1
                next_observations, _, dones, infos = env.step([action])
                next_observation = str(next_observations[0])
                steps.append(
                    {
                        "step": step_index,
                        "observation": observation,
                        "action": action,
                        **diagnostics,
                    }
                )
                history.append((action, next_observation))
                observation = next_observation
                won = bool(infos["won"][0])
                if won or bool(dones[0]):
                    break
            trajectories.append(
                {
                    "episode_id": Path(gamefile).parent.parent.name,
                    "gamefile": gamefile,
                    "task_type": _task_type_from_gamefile(gamefile),
                    "goal": goal,
                    "won": won,
                    "steps": steps,
                }
            )
            print(f"[{episode_index + 1}/{len(gamefiles)}] won={won} steps={len(steps)}", flush=True)
        finally:
            env.close()

    by_task = defaultdict(list)
    for trajectory in trajectories:
        by_task[trajectory["task_type"]].append(trajectory["won"])
    results = {
        "policy": policy,
        "parameter_delta": parameter_delta,
        "model_dtype": str(model.model.dtype),
        "model_path": model_path or config["model_path"],
        "split": split,
        "episodes": len(trajectories),
        "success_rate": sum(item["won"] for item in trajectories) / len(trajectories),
        "mean_steps": sum(len(item["steps"]) for item in trajectories) / len(trajectories),
        "by_task_type": {
            task: {"episodes": len(values), "success_rate": sum(values) / len(values)}
            for task, values in sorted(by_task.items())
        },
        "trajectories": trajectories,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--split", choices=("valid_seen", "valid_unseen"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parameter-delta")
    parser.add_argument("--force-float32", action="store_true")
    parser.add_argument("--model-path")
    args = parser.parse_args()
    print(json.dumps(run(
        args.config, args.policy, args.split, args.device, args.output_dir,
        args.parameter_delta, args.force_float32, args.model_path
    ), indent=2))


if __name__ == "__main__":
    main()
