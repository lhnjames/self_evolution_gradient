from self_evolve.benchmark import CHOICE_LABELS, TOOL_NAMES, generate_tasks


def test_generated_tasks_have_unique_options_and_correct_execution():
    tasks = generate_tasks(25, "train", seed=3)
    assert len(tasks) == 25
    for task in tasks:
        assert len(set(task.options)) == 4
        assert task.execute(task.correct_tool_label) == task.answer
        assert task.correct_choice_label in CHOICE_LABELS
        assert 0 <= task.correct_tool_label < len(TOOL_NAMES)


def test_eval_uses_held_out_paraphrases():
    train = generate_tasks(5, "train", seed=1)
    evaluation = generate_tasks(5, "eval", seed=1)
    assert [task.question for task in train] != [task.question for task in evaluation]

