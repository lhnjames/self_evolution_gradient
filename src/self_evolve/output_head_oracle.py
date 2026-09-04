from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable


def is_value_heterogeneous(row: dict[str, Any]) -> bool:
    values = [float(action["discounted_success"]) for action in row["actions"]]
    return max(values) - min(values) > 1e-12


def one_failure_per_episode(
    indices: Iterable[int], rows: dict[int, dict[str, Any]], rng: random.Random
) -> list[int]:
    shuffled = list(indices)
    rng.shuffle(shuffled)
    result: list[int] = []
    used: set[str] = set()
    for index in shuffled:
        episode = str(rows[index]["episode_key"])
        if episode not in used:
            result.append(index)
            used.add(episode)
    return result


def task_round_robin(
    indices: Iterable[int],
    rows: dict[int, dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[int]:
    by_task: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_task[str(rows[index]["task_type"])].append(index)
    tasks = sorted(by_task)
    rng.shuffle(tasks)
    for values in by_task.values():
        rng.shuffle(values)
    result: list[int] = []
    depth = 0
    while len(result) < count:
        added = False
        for task in tasks:
            if depth < len(by_task[task]):
                result.append(by_task[task][depth])
                added = True
                if len(result) == count:
                    return result
        if not added:
            break
        depth += 1
    raise ValueError(f"Only {len(result)} distinct-episode states available; need {count}")


def fixed_oracle_split(
    *,
    rows: dict[int, dict[str, Any]],
    failure_indices: set[int],
    verbs: Iterable[str],
    source_count: int,
    validation_count: int,
    test_count: int,
    protection_count_per_verb: int,
    seed: int,
) -> dict[str, Any]:
    """Create a deterministic, episode-disjoint split before fitting the head."""
    if min(source_count, validation_count, test_count) < 1:
        raise ValueError("source/validation/test counts must be positive")
    result: dict[str, Any] = {"by_skill": {}}
    verb_list = list(verbs)
    eligible_by_verb: dict[str, list[int]] = {}
    for position, verb in enumerate(verb_list):
        rng = random.Random(seed + (position + 1) * 1_000_003)
        eligible = [
            index
            for index, row in rows.items()
            if index in failure_indices
            and row["action_verb"] == verb
            and is_value_heterogeneous(row)
        ]
        eligible_by_verb[verb] = one_failure_per_episode(eligible, rows, rng)
        result["by_skill"][verb] = {
            "source": [],
            "transfer_validation": [],
            "final_holdout": [],
            "eligible_failure_states": len(eligible),
            "eligible_failure_episodes": len(
                {str(rows[index]["episode_key"]) for index in eligible}
            ),
        }

    # An episode may contain several action verbs. Assign every episode to one
    # panel globally, so (for example) an open-source episode can never become
    # a close-test episode. Rare skills are allocated first.
    episode_panel: dict[str, str] = {}
    panel_counts = {
        "source": source_count,
        "transfer_validation": validation_count,
        "final_holdout": test_count,
    }
    allocation_order = sorted(verb_list, key=lambda verb: (len(eligible_by_verb[verb]), verb))
    for position, verb in enumerate(allocation_order):
        rng = random.Random(seed + (position + 1) * 3_000_017)
        distinct = eligible_by_verb[verb]
        for panel_name, count in panel_counts.items():
            compatible = [
                index
                for index in distinct
                if episode_panel.get(str(rows[index]["episode_key"])) == panel_name
            ]
            selected = task_round_robin(compatible, rows, min(count, len(compatible)), rng) \
                if compatible else []
            selected_episodes = {str(rows[index]["episode_key"]) for index in selected}
            if len(selected) < count:
                unassigned = [
                    index
                    for index in distinct
                    if str(rows[index]["episode_key"]) not in episode_panel
                    and str(rows[index]["episode_key"]) not in selected_episodes
                ]
                fill = task_round_robin(unassigned, rows, count - len(selected), rng)
                selected.extend(fill)
            for index in selected:
                episode_panel[str(rows[index]["episode_key"])] = panel_name
            result["by_skill"][verb][panel_name] = selected

    used_episodes = set(episode_panel)

    protection: list[dict[str, Any]] = []
    for position, verb in enumerate(verb_list):
        rng = random.Random(seed + (position + 1) * 2_000_003)
        source_tasks = {
            str(rows[index]["task_type"])
            for index in result["by_skill"][verb]["source"]
        }
        candidates = [
            index
            for index, row in rows.items()
            if row["action_verb"] != verb
            and str(row["task_type"]) in source_tasks
            and str(row["episode_key"]) not in used_episodes
            and is_value_heterogeneous(row)
        ]
        rng.shuffle(candidates)
        selected: list[int] = []
        for index in candidates:
            episode = str(rows[index]["episode_key"])
            if episode in used_episodes:
                continue
            selected.append(index)
            used_episodes.add(episode)
            if len(selected) == protection_count_per_verb:
                break
        if len(selected) < protection_count_per_verb:
            raise ValueError(
                f"Only {len(selected)} episode-disjoint protection states for {verb}; "
                f"need {protection_count_per_verb}"
            )
        protection.extend(
            {"global_decision_index": index, "matched_source_skill": verb}
            for index in selected
        )
    result["protection"] = protection
    result["seed"] = seed
    result["counts"] = {
        "source_per_skill": source_count,
        "validation_per_skill": validation_count,
        "test_per_skill": test_count,
        "protection_per_skill": protection_count_per_verb,
    }
    return result
