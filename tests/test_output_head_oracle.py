import random

import pytest

from self_evolve.output_head_oracle import (
    fixed_oracle_split,
    one_failure_per_episode,
    task_round_robin,
)


def _row(index: int, verb: str, episode: str, task: str = "task"):
    return {
        "action_verb": verb,
        "episode_key": episode,
        "task_type": task,
        "actions": [
            {"discounted_success": 0.0},
            {"discounted_success": 1.0},
        ],
    }


def test_one_failure_per_episode_is_deterministic_and_disjoint():
    rows = {index: _row(index, "open", f"e{index // 2}") for index in range(8)}
    first = one_failure_per_episode(rows, rows, random.Random(7))
    second = one_failure_per_episode(rows, rows, random.Random(7))
    assert first == second
    assert len(first) == 4
    assert len({rows[index]["episode_key"] for index in first}) == 4


def test_task_round_robin_requires_enough_examples():
    rows = {index: _row(index, "open", f"e{index}", f"t{index % 2}") for index in range(4)}
    selected = task_round_robin(rows, rows, 4, random.Random(1))
    assert len(selected) == 4
    with pytest.raises(ValueError):
        task_round_robin(rows, rows, 5, random.Random(1))


def test_fixed_split_keeps_every_episode_disjoint():
    rows = {}
    failures = set()
    cursor = 0
    for verb in ("go", "open"):
        for local in range(6):
            rows[cursor] = _row(cursor, verb, f"{verb}-{local}", f"t{local % 2}")
            failures.add(cursor)
            cursor += 1
    for local in range(8):
        verb = "open" if local % 2 == 0 else "go"
        rows[cursor] = _row(cursor, verb, f"p-{local}", "t0")
        cursor += 1
    split = fixed_oracle_split(
        rows=rows,
        failure_indices=failures,
        verbs=("go", "open"),
        source_count=2,
        validation_count=1,
        test_count=1,
        protection_count_per_verb=1,
        seed=11,
    )
    indices = []
    for skill in split["by_skill"].values():
        indices.extend(skill["source"])
        indices.extend(skill["transfer_validation"])
        indices.extend(skill["final_holdout"])
    indices.extend(item["global_decision_index"] for item in split["protection"])
    panel_by_episode = {}
    for skill in split["by_skill"].values():
        for panel in ("source", "transfer_validation", "final_holdout"):
            for index in skill[panel]:
                episode = rows[index]["episode_key"]
                assert panel_by_episode.get(episode, panel) == panel
                panel_by_episode[episode] = panel
    protection_episodes = {
        rows[item["global_decision_index"]]["episode_key"] for item in split["protection"]
    }
    assert protection_episodes.isdisjoint(panel_by_episode)
