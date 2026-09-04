from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .alfworld_data import AlfworldDecision


TASK_TYPE_MAP = {
    "pick_and_place_simple": "pick_and_place",
    "pick_two_obj_and_place": "pick_and_place",
    "look_at_obj_in_light": "look_at_obj_in_light",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
}


class AlfworldSkillBank:
    def __init__(self, path: str | Path):
        with Path(path).open(encoding="utf-8") as handle:
            self.data: dict[str, Any] = json.load(handle)

    def category(self, decision: AlfworldDecision) -> str:
        if decision.task_type in TASK_TYPE_MAP:
            return TASK_TYPE_MAP[decision.task_type]
        goal = decision.goal.lower()
        if "look at" in goal and "under" in goal:
            return "look_at_obj_in_light"
        for name in ("clean", "heat", "cool", "examine"):
            if name in goal:
                return name
        return "pick_and_place"

    def render(
        self,
        decision: AlfworldDecision,
        general_top_k: int = 3,
        task_top_k: int = 3,
        mistakes_top_k: int = 2,
        category_override: str | None = None,
    ) -> str:
        category = category_override or self.category(decision)
        sections: list[str] = []
        general = self.data.get("general_skills", [])[:general_top_k]
        task = self.data.get("task_specific_skills", {}).get(category, [])[:task_top_k]
        mistakes = self.data.get("common_mistakes", [])[:mistakes_top_k]
        if general:
            sections.append("General evolved skills:\n" + "\n".join(_format_skill(x) for x in general))
        if task:
            sections.append(
                f"Evolved skills for {category}:\n" + "\n".join(_format_skill(x) for x in task)
            )
        if mistakes:
            lines = []
            for mistake in mistakes:
                description = mistake.get("description", "")
                remedy = mistake.get("how_to_avoid", "")
                lines.append(f"- Avoid: {description} Remedy: {remedy}".strip())
            sections.append("Mistakes learned from failures:\n" + "\n".join(lines))
        return "\n\n".join(sections)

    def wrong_category(self, decision: AlfworldDecision) -> str:
        correct = self.category(decision)
        categories = sorted(self.data.get("task_specific_skills", {}))
        if len(categories) < 2:
            raise ValueError("Need at least two task skill categories for a mismatch control")
        index = categories.index(correct) if correct in categories else 0
        return categories[(index + 1) % len(categories)]


def _format_skill(skill: dict[str, Any]) -> str:
    title = skill.get("title", "")
    principle = skill.get("principle", "")
    when = skill.get("when_to_apply", "")
    suffix = f" Apply when: {when}" if when else ""
    return f"- {title}: {principle}{suffix}".strip()


def build_action_prompt(
    decision: AlfworldDecision,
    skill_context: str = "",
    history_window: int = 3,
) -> str:
    history = decision.history[-history_window:]
    history_text = "None."
    if history:
        history_text = "\n".join(
            f"Action: {action}\nObservation: {observation}" for action, observation in history
        )
    skill_text = f"\nLearned strategy:\n{skill_context}\n" if skill_context else ""
    candidates = "\n".join(f"- {action}" for action in decision.admissible_actions)
    return (
        "You control an agent in ALFWorld. Select exactly one currently admissible command.\n"
        f"Task: {decision.goal}\n"
        f"{skill_text}"
        f"Recent history:\n{history_text}\n"
        f"Current observation: {decision.observation}\n"
        f"Admissible commands:\n{candidates}\n"
        "Return only the command, with no reasoning.\nCommand:"
    )


def render_conditions(
    decision: AlfworldDecision,
    bank: AlfworldSkillBank,
    general_top_k: int,
    task_top_k: int,
    mistakes_top_k: int,
    history_window: int,
) -> dict[str, str]:
    correct = bank.render(decision, general_top_k, task_top_k, mistakes_top_k)
    wrong = bank.render(
        decision,
        general_top_k,
        task_top_k,
        mistakes_top_k,
        category_override=bank.wrong_category(decision),
    )
    return {
        "plain": build_action_prompt(decision, history_window=history_window),
        "evolved_skill": build_action_prompt(decision, correct, history_window),
        "mismatched_skill": build_action_prompt(decision, wrong, history_window),
    }
