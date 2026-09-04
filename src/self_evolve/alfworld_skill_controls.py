from __future__ import annotations

from typing import Any

from .alfworld_data import AlfworldDecision
from .alfworld_skills import AlfworldSkillBank, _format_skill


def _selected(
    bank: AlfworldSkillBank,
    decision: AlfworldDecision,
    general_top_k: int,
    task_top_k: int,
    mistakes_top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    category = bank.category(decision)
    return (
        bank.data.get("general_skills", [])[:general_top_k],
        bank.data.get("task_specific_skills", {}).get(category, [])[:task_top_k],
        bank.data.get("common_mistakes", [])[:mistakes_top_k],
    )


def render_reformatted(
    bank: AlfworldSkillBank,
    decision: AlfworldDecision,
    general_top_k: int,
    task_top_k: int,
    mistakes_top_k: int,
) -> str:
    """Same selected semantic fields under a different surface template."""
    general, task, mistakes = _selected(
        bank, decision, general_top_k, task_top_k, mistakes_top_k
    )

    def rule(item: dict[str, Any]) -> str:
        return (
            f"- In the situation '{item.get('when_to_apply', '')}', use the method "
            f"'{item.get('title', '')}': {item.get('principle', '')}"
        )

    sections = []
    if general:
        sections.append("Reusable guidance:\n" + "\n".join(rule(item) for item in general))
    if task:
        sections.append("Task guidance:\n" + "\n".join(rule(item) for item in task))
    if mistakes:
        sections.append(
            "Failure-prevention guidance:\n"
            + "\n".join(
                f"- To prevent '{item.get('description', '')}', do this: "
                f"{item.get('how_to_avoid', '')}"
                for item in mistakes
            )
        )
    return "\n\n".join(sections)


def render_anti_skill(
    bank: AlfworldSkillBank,
    decision: AlfworldDecision,
    general_top_k: int,
    task_top_k: int,
    mistakes_top_k: int,
) -> str:
    """Explicitly negate the same selected rules; diagnostic only."""
    general, task, mistakes = _selected(
        bank, decision, general_top_k, task_top_k, mistakes_top_k
    )
    lines = [
        "Adversarial anti-strategy for a controlled experiment.",
        "Deliberately reject every rule below and choose behavior that contradicts it:",
    ]
    for item in general + task:
        lines.append(
            f"- Do NOT follow '{item.get('title', '')}' when {item.get('when_to_apply', '')}. "
            f"Reject this behavior: {item.get('principle', '')}"
        )
    for item in mistakes:
        lines.append(
            f"- Do NOT use this remedy: {item.get('how_to_avoid', '')} "
            f"Instead permit the failure: {item.get('description', '')}"
        )
    return "\n".join(lines)


def render_task_only(
    bank: AlfworldSkillBank,
    decision: AlfworldDecision,
    task_top_k: int,
) -> str:
    category = bank.category(decision)
    task = bank.data.get("task_specific_skills", {}).get(category, [])[:task_top_k]
    return f"Evolved skills for {category}:\n" + "\n".join(_format_skill(item) for item in task)


def render_general_only(
    bank: AlfworldSkillBank,
    general_top_k: int,
    mistakes_top_k: int,
) -> str:
    general = bank.data.get("general_skills", [])[:general_top_k]
    mistakes = bank.data.get("common_mistakes", [])[:mistakes_top_k]
    sections = ["General evolved skills:\n" + "\n".join(_format_skill(item) for item in general)]
    if mistakes:
        sections.append(
            "Mistakes learned from failures:\n"
            + "\n".join(
                f"- Avoid: {item.get('description', '')} Remedy: {item.get('how_to_avoid', '')}"
                for item in mistakes
            )
        )
    return "\n\n".join(sections)


def token_length_matched_placebo(reference: str, tokenizer) -> str:
    """Non-actionable filler with the same tokenizer length as the reference."""
    target = len(tokenizer.encode(reference, add_special_tokens=False))
    sentence = (
        " This paragraph is neutral filler for a controlled comparison."
        " It gives no recommendation about actions, objects, locations, order, or strategy."
    )
    candidate = sentence
    while len(tokenizer.encode(candidate, add_special_tokens=False)) < target + 16:
        candidate += sentence
    ids = tokenizer.encode(candidate, add_special_tokens=False)[:target]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    current = tokenizer.encode(text, add_special_tokens=False)
    if len(current) != target:
        # Tokenizer decode/encode is normally stable. Preserve an explicit
        # failure instead of silently weakening the length-matched control.
        raise ValueError(f"Could not round-trip placebo token length: {len(current)} != {target}")
    return text


def render_control_contexts(
    bank: AlfworldSkillBank,
    decision: AlfworldDecision,
    tokenizer,
    general_top_k: int,
    task_top_k: int,
    mistakes_top_k: int,
) -> dict[str, str]:
    reference = bank.render(decision, general_top_k, task_top_k, mistakes_top_k)
    return {
        "reformatted_skill": render_reformatted(
            bank, decision, general_top_k, task_top_k, mistakes_top_k
        ),
        "anti_skill": render_anti_skill(
            bank, decision, general_top_k, task_top_k, mistakes_top_k
        ),
        "task_only_skill": render_task_only(bank, decision, task_top_k),
        "general_only_skill": render_general_only(bank, general_top_k, mistakes_top_k),
        "length_matched_placebo": token_length_matched_placebo(reference, tokenizer),
    }
