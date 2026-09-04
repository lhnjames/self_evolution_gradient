from self_evolve.alfworld_data import AlfworldDecision
from self_evolve.alfworld_skill_controls import (
    render_anti_skill,
    render_reformatted,
    render_task_only,
)
from self_evolve.alfworld_skills import AlfworldSkillBank


def decision():
    return AlfworldDecision(
        split="valid_seen",
        episode_id="episode",
        gamefile="game",
        task_type="pick_clean_then_place_in_recep",
        goal="clean some mug and put it in cabinet",
        step_index=1,
        observation="obs",
        history=(),
        admissible_actions=("go to sinkbasin 1", "take mug 1 from table 1"),
        expert_action="take mug 1 from table 1",
    )


def test_semantic_controls_select_the_correct_task_category(tmp_path):
    path = tmp_path / "skills.json"
    path.write_text(
        '{"general_skills":[{"title":"G","principle":"general","when_to_apply":"always"}],'
        '"task_specific_skills":{"clean":[{"title":"C","principle":"clean it","when_to_apply":"dirty"}]},'
        '"common_mistakes":[{"description":"loop","how_to_avoid":"move"}]}',
        encoding="utf-8",
    )
    bank = AlfworldSkillBank(path)
    item = decision()
    assert "clean it" in render_task_only(bank, item, 1)
    assert "clean it" in render_reformatted(bank, item, 1, 1, 1)
    anti = render_anti_skill(bank, item, 1, 1, 1)
    assert "Do NOT" in anti
    assert "clean it" in anti
