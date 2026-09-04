from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .benchmark import LogicRouteTask


@dataclass(frozen=True)
class SkillVersion:
    skill_id: str
    version: str
    name: str
    body: str
    provenance: tuple[str, ...]
    promoted_by: str

    def to_dict(self) -> dict:
        value = asdict(self)
        value["provenance"] = list(self.provenance)
        return value


def build_controlled_skill_lineage(train_tasks: Iterable[LogicRouteTask]) -> list[SkillVersion]:
    """Build a controlled before/after skill lineage from verified train traces.

    The content is deliberately deterministic: this experiment isolates the effect of
    skill text on action distributions, rather than confounding it with analyzer-model
    quality.  The v1 mutation is the kind of explicit decision rule produced after
    replaying route failures and promoting on held-out traces.
    """

    task_ids = tuple(task.task_id for task in train_tasks)
    v0 = SkillVersion(
        skill_id="logic-route",
        version="0.1.0",
        name="Arithmetic tool routing",
        body=(
            "# Goal\n"
            "Choose the arithmetic tool that matches the requested relation.\n"
            "# Workflow\n"
            "First identify the operation expressed by the question. Then choose the "
            "semantically matching tool. After observing the tool result, select the "
            "candidate answer whose value exactly equals that observation. Do not guess."
        ),
        provenance=task_ids[: min(8, len(task_ids))],
        promoted_by="initial extraction from successful and repaired traces",
    )
    v1 = SkillVersion(
        skill_id="logic-route",
        version="0.2.0",
        name="Verified arithmetic label routing",
        body=(
            "# Goal\n"
            "Translate the requested arithmetic logic into the exact action label.\n"
            "# Verified decision table\n"
            "- sum, add, or combine quantities -> action 0 (add)\n"
            "- starting value take away another, subtract -> action 1 (subtract)\n"
            "- product or multiply -> action 2 (multiply)\n"
            "- larger, greatest, or maximum -> action 3 (maximum)\n"
            "- smaller, least, or minimum -> action 4 (minimum)\n"
            "# Answer selection\n"
            "Treat answer labels as pointers, not values. After the tool observation, "
            "choose only the label whose displayed value exactly equals the observation.\n"
            "# Guardrail\n"
            "Use exactly one action label and never replace the requested operation with "
            "a numerically convenient one."
        ),
        provenance=task_ids,
        promoted_by="failure replay + held-out promotion rule",
    )
    return [v0, v1]


def inject_skill(prompt: str, skill: SkillVersion) -> str:
    block = (
        "\n## Retrieved evolved skill\n"
        "Use this skill only when it matches the current decision.\n"
        f"Skill: {skill.name} (v{skill.version})\n"
        f"{skill.body}\n"
    )
    anchor = "Action label:"
    if anchor in prompt:
        return prompt.replace(anchor, block + anchor, 1)
    return prompt + block
