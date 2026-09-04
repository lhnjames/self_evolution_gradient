#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from self_evolve.action_value import discounted_success
from self_evolve.alfworld_data import _config_for_split, load_decisions


def resolve_gamefile(original: str, data_root: str | Path) -> Path:
    parts = Path(original).parts
    try:
        marker = parts.index("json_2.1.1")
    except ValueError as error:
        raise ValueError(f"gamefile has no json_2.1.1 component: {original}") from error
    resolved = Path(data_root, *parts[marker:]).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _single(infos: dict[str, Any], key: str, default: Any = None) -> Any:
    value = infos.get(key, [default])
    return value[0] if isinstance(value, (list, tuple)) and len(value) else default


def replay_target(env, decision, seed: int) -> tuple[str, dict[str, Any]]:
    # ALFWorld's hand-coded expert uses Python's module-level RNG when several
    # objects or fallback commands are possible. Reset it for common-random-
    # number comparisons across candidate actions and exact reruns.
    random.seed(seed)
    env.seed(seed)
    observations, infos = env.reset()
    observation = str(observations[0])
    for history_index, (action, expected_observation) in enumerate(decision.history):
        admissible = tuple(str(item) for item in _single(infos, "admissible_commands", []))
        if action not in admissible:
            raise RuntimeError(
                f"history action {history_index} is not admissible at decision {decision.step_index}: {action}"
            )
        observations, _, dones, infos = env.step([action])
        observation = str(observations[0])
        if observation != expected_observation:
            raise RuntimeError(
                f"history observation mismatch at item {history_index} for {decision.gamefile}"
            )
        if bool(dones[0]) and history_index + 1 < len(decision.history):
            raise RuntimeError("environment terminated before target history was replayed")
    if observation != decision.observation:
        raise RuntimeError(f"target observation mismatch for {decision.gamefile} step {decision.step_index}")
    actual_actions = tuple(str(item) for item in _single(infos, "admissible_commands", []))
    if actual_actions != decision.admissible_actions:
        raise RuntimeError(
            f"admissible actions mismatch for {decision.gamefile} step {decision.step_index}"
        )
    return observation, infos


def evaluate_candidate(
    env,
    decision,
    candidate_index: int,
    candidate: str,
    seed: int,
    max_steps: int,
    gamma: float,
) -> dict[str, Any]:
    _, infos = replay_target(env, decision, seed)
    started = time.monotonic()
    recovery_actions: list[str] = []
    failure_reason = None
    recovery_budget = max_steps - len(decision.history)
    if recovery_budget < 1:
        raise ValueError("target state has no remaining action budget")
    try:
        observations, scores, dones, infos = env.step([candidate])
        forced_feedback = str(observations[0])
        won = bool(_single(infos, "won", False))
        done = bool(dones[0])
        while not won and not done and 1 + len(recovery_actions) < recovery_budget:
            plan = _single(infos, "extra.expert_plan", []) or []
            if not plan:
                failure_reason = "empty_expert_plan"
                break
            next_action = str(plan[0])
            admissible = tuple(str(item) for item in _single(infos, "admissible_commands", []))
            if next_action not in admissible:
                failure_reason = f"expert_action_not_admissible:{next_action}"
                break
            recovery_actions.append(next_action)
            observations, scores, dones, infos = env.step([next_action])
            won = bool(_single(infos, "won", False))
            done = bool(dones[0])
        if not won and failure_reason is None:
            failure_reason = "environment_done" if done else "recovery_step_limit"
    except Exception as error:  # ALFWorld raises a generic Exception on expert timeout.
        forced_feedback = ""
        scores = [0.0]
        won = False
        done = False
        failure_reason = f"{type(error).__name__}:{error}"

    recovery_steps = 1 + len(recovery_actions)
    return {
        "candidate_index": candidate_index,
        "action": candidate,
        "is_recorded_expert_action": int(candidate == decision.expert_action),
        "won": int(won),
        "done": int(done),
        "recovery_steps": recovery_steps,
        "recovery_budget": recovery_budget,
        "total_steps_from_episode_start": len(decision.history) + recovery_steps,
        "expert_followup_steps": len(recovery_actions),
        "discounted_success": discounted_success(won, recovery_steps, gamma),
        "final_score": float(scores[0]) if scores else 0.0,
        "failure_reason": failure_reason,
        "forced_feedback": forced_feedback,
        "expert_recovery_actions": recovery_actions,
        "wall_seconds": time.monotonic() - started,
    }


def read_completed(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(int(json.loads(line)["global_decision_index"]))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--decision-file", required=True)
    parser.add_argument("--split", choices=("valid_seen", "valid_unseen"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expert-type", choices=("handcoded", "planner"), default="handcoded")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--decision-offset", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=-1)
    parser.add_argument("--max-actions", type=int, default=-1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    decisions = [item for item in load_decisions(args.decision_file) if not item.is_trivial]
    indexed = list(enumerate(decisions))[args.decision_offset :]
    if args.max_decisions > 0:
        indexed = indexed[: args.max_decisions]
    indexed = [item for item in indexed if item[0] % args.num_shards == args.shard_index]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"
    completed = read_completed(trace_path) if args.resume else set()
    indexed = [item for item in indexed if item[0] not in completed]

    env_config = _config_for_split(args.config, args.data_root, args.split)
    env_config["env"]["expert_type"] = args.expert_type
    from alfworld.agents.environment import get_environment

    base_env = get_environment(env_config["env"]["type"])(env_config, train_eval="train")
    grouped: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    for global_index, decision in indexed:
        grouped[decision.gamefile].append((global_index, decision))

    mode = "a" if args.resume and trace_path.exists() else "w"
    started = time.monotonic()
    written = 0
    with trace_path.open(mode, encoding="utf-8") as trace_file:
        for game_number, (original_gamefile, members) in enumerate(sorted(grouped.items())):
            gamefile = resolve_gamefile(original_gamefile, args.data_root)
            base_env.game_files = [str(gamefile)]
            base_env.num_games = 1
            env = base_env.init_env(batch_size=1)
            try:
                for global_index, decision in members:
                    candidates = list(decision.admissible_actions)
                    if args.max_actions > 0:
                        candidates = candidates[: args.max_actions]
                    candidate_values = [
                        evaluate_candidate(
                            env=env,
                            decision=decision,
                            candidate_index=candidate_index,
                            candidate=candidate,
                            seed=args.seed + global_index,
                            max_steps=args.max_steps,
                            gamma=args.gamma,
                        )
                        for candidate_index, candidate in enumerate(candidates)
                    ]
                    row = {
                        "split": args.split,
                        "global_decision_index": global_index,
                        "episode_key": str(gamefile.relative_to(Path(args.data_root).resolve())),
                        "episode_id": decision.episode_id,
                        "task_type": decision.task_type,
                        "step_index": decision.step_index,
                        "action_verb": decision.action_verb,
                        "goal": decision.goal,
                        "expert_action": decision.expert_action,
                        "admissible_actions": candidates,
                        "prefix_steps": len(decision.history),
                        "expert_type": args.expert_type,
                        "gamma": args.gamma,
                        "candidate_values": candidate_values,
                    }
                    trace_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    trace_file.flush()
                    written += 1
                    wins = sum(item["won"] for item in candidate_values)
                    elapsed = time.monotonic() - started
                    print(
                        f"[{written}/{len(indexed)} shard={args.shard_index}] "
                        f"game={game_number + 1}/{len(grouped)} index={global_index} "
                        f"actions={len(candidates)} wins={wins} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
            finally:
                env.close()

    rows = []
    if trace_path.exists():
        with trace_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    metadata = {
        "split": args.split,
        "expert_type": args.expert_type,
        "gamma": args.gamma,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "states": len(rows),
        "candidate_actions": sum(len(row["candidate_values"]) for row in rows),
        "successful_candidates": sum(
            item["won"] for row in rows for item in row["candidate_values"]
        ),
        "wall_seconds": time.monotonic() - started,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
