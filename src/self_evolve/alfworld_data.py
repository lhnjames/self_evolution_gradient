from __future__ import annotations

import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


GOAL_RE = re.compile(r"Your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE)
TRIVIAL_ACTIONS = {"look", "inventory", "help"}


@dataclass(frozen=True)
class AlfworldDecision:
    split: str
    episode_id: str
    gamefile: str
    task_type: str
    goal: str
    step_index: int
    observation: str
    history: tuple[tuple[str, str], ...]
    admissible_actions: tuple[str, ...]
    expert_action: str

    @property
    def action_verb(self) -> str:
        return self.expert_action.split(maxsplit=1)[0]

    @property
    def is_trivial(self) -> bool:
        return self.expert_action in TRIVIAL_ACTIONS

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["history"] = [list(item) for item in self.history]
        row["admissible_actions"] = list(self.admissible_actions)
        row["action_verb"] = self.action_verb
        row["is_trivial"] = self.is_trivial
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "AlfworldDecision":
        fields = {
            key: value
            for key, value in row.items()
            if key not in {"action_verb", "is_trivial"}
        }
        fields["history"] = tuple(tuple(item) for item in fields["history"])
        fields["admissible_actions"] = tuple(fields["admissible_actions"])
        return cls(**fields)


def _task_type_from_gamefile(gamefile: str) -> str:
    episode_dir = Path(gamefile).parent.parent.name
    return episode_dir.split("-")[0]


def _goal_from_observation(observation: str) -> str:
    match = GOAL_RE.search(observation)
    if not match:
        raise ValueError(f"Could not find ALFWorld task goal in observation: {observation[:200]!r}")
    return match.group(1).strip()


def _config_for_split(config_path: str, data_root: str, split: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    split_keys = {
        "train": "data_path",
        "valid_seen": "eval_id_data_path",
        "valid_unseen": "eval_ood_data_path",
    }
    if split not in split_keys:
        raise ValueError(f"Unknown split {split!r}; choose one of {sorted(split_keys)}")
    os.environ["ALFWORLD_DATA"] = str(Path(data_root).resolve())
    # Expert plans are exposed only in train mode. Point its data path at the
    # desired split; this does not modify the data or mix split examples.
    config["dataset"]["data_path"] = config["dataset"][split_keys[split]]
    config["dataset"]["num_train_games"] = -1
    config["general"]["use_cuda"] = False
    config["env"]["domain_randomization"] = False
    return config


def collect_expert_decisions(
    config_path: str,
    data_root: str,
    split: str,
    episodes: int,
    seed: int,
    max_steps: int = 50,
) -> tuple[list[AlfworldDecision], dict[str, Any]]:
    """Collect successful official-expert trajectories from explicit games.

    One TextWorld environment is created per selected game. This is slower than
    a randomized batch environment but guarantees that every requested episode
    is unique and that the reported seed fully determines the subset.
    """
    from alfworld.agents.environment import get_environment

    config = _config_for_split(config_path, data_root, split)
    base_env = get_environment(config["env"]["type"])(config, train_eval="train")
    game_files = sorted(base_env.game_files)
    rng = random.Random(seed)
    rng.shuffle(game_files)
    selected = game_files[: min(episodes, len(game_files))]

    all_decisions: list[AlfworldDecision] = []
    successful = 0
    skipped = []
    for episode_number, gamefile in enumerate(selected):
        # Register only this game so reset cannot silently select a duplicate.
        base_env.game_files = [gamefile]
        base_env.num_games = 1
        env = base_env.init_env(batch_size=1)
        try:
            env.seed(seed + episode_number)
            observations, infos = env.reset()
            initial_observation = str(observations[0])
            goal = _goal_from_observation(initial_observation)
            task_type = _task_type_from_gamefile(gamefile)
            history: list[tuple[str, str]] = []
            episode_rows: list[AlfworldDecision] = []
            observation = initial_observation
            won = False
            failure_reason = "timeout"

            for step_index in range(max_steps):
                expert_plan = infos.get("extra.expert_plan", [[]])[0]
                candidates = tuple(str(x) for x in infos["admissible_commands"][0])
                if not expert_plan:
                    failure_reason = "empty_expert_plan"
                    break
                expert_action = str(expert_plan[0])
                if expert_action not in candidates:
                    failure_reason = f"expert_not_admissible:{expert_action}"
                    break
                episode_rows.append(
                    AlfworldDecision(
                        split=split,
                        episode_id=Path(gamefile).parent.parent.name,
                        gamefile=gamefile,
                        task_type=task_type,
                        goal=goal,
                        step_index=step_index,
                        observation=observation,
                        history=tuple(history),
                        admissible_actions=candidates,
                        expert_action=expert_action,
                    )
                )
                next_observations, _, dones, infos = env.step([expert_action])
                next_observation = str(next_observations[0])
                history.append((expert_action, next_observation))
                observation = next_observation
                won = bool(infos["won"][0])
                if won:
                    successful += 1
                    all_decisions.extend(episode_rows)
                    break
                if bool(dones[0]):
                    failure_reason = "environment_done_without_win"
                    break
            if not won:
                skipped.append({"gamefile": gamefile, "reason": failure_reason})
        finally:
            env.close()

    metadata = {
        "split": split,
        "seed": seed,
        "requested_episodes": episodes,
        "selected_episodes": len(selected),
        "successful_episodes": successful,
        "skipped_episodes": skipped,
        "decisions": len(all_decisions),
        "nontrivial_decisions": sum(not row.is_trivial for row in all_decisions),
    }
    return all_decisions, metadata


def save_decisions(path: str | Path, rows: Iterable[AlfworldDecision]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def load_decisions(path: str | Path) -> list[AlfworldDecision]:
    with Path(path).open(encoding="utf-8") as handle:
        return [AlfworldDecision.from_dict(json.loads(line)) for line in handle if line.strip()]
