import torch

from self_evolve.alfworld_data import AlfworldDecision
from self_evolve.alfworld_runner import categorical_kl, trust_region_skill_distribution
from self_evolve.alfworld_skills import build_action_prompt


def test_trust_region_projection_respects_budget():
    plain = torch.tensor([0.8, 0.1, 0.1])
    skill = torch.tensor([0.1, 0.1, 0.8])
    repaired, rho = trust_region_skill_distribution(plain, skill, kl_budget=0.05)
    assert 0.0 < rho < 1.0
    assert categorical_kl(repaired, plain) <= 0.050001
    assert repaired[2] > plain[2]


def test_prompt_contains_no_expert_label_outside_candidate_list():
    decision = AlfworldDecision(
        split="valid_seen",
        episode_id="episode",
        gamefile="game.tw-pddl",
        task_type="pick_and_place_simple",
        goal="put a vase in coffeetable.",
        step_index=1,
        observation="You see a shelf.",
        history=(("look", "You see a shelf."),),
        admissible_actions=("go to shelf 1", "inventory"),
        expert_action="go to shelf 1",
    )
    prompt = build_action_prompt(decision)
    assert "Expert action" not in prompt
    assert prompt.count("go to shelf 1") == 1
