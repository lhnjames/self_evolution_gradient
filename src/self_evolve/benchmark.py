from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Callable


TOOL_NAMES = ("add", "subtract", "multiply", "maximum", "minimum")
TOOL_LABELS = {name: index for index, name in enumerate(TOOL_NAMES)}
CHOICE_LABELS = tuple(range(len(TOOL_NAMES), len(TOOL_NAMES) + 4))
ACTION_LABELS = tuple(str(index) for index in range(len(TOOL_NAMES) + 4))

_OPERATIONS: dict[str, Callable[[int, int], int]] = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "maximum": max,
    "minimum": min,
}

_TRAIN_TEMPLATES = {
    "add": "Add {a} and {b}.",
    "subtract": "Subtract {b} from {a}.",
    "multiply": "Multiply {a} by {b}.",
    "maximum": "Return the larger of {a} and {b}.",
    "minimum": "Return the smaller of {a} and {b}.",
}

_EVAL_TEMPLATES = {
    "add": "What is the sum obtained from {a} together with {b}?",
    "subtract": "Starting at {a}, take away {b}. What remains?",
    "multiply": "Find the product of the pair {a}, {b}.",
    "maximum": "Select the greatest value among {a} and {b}.",
    "minimum": "Select the least value among {a} and {b}.",
}


@dataclass(frozen=True)
class LogicRouteTask:
    task_id: str
    split: str
    operation: str
    a: int
    b: int
    question: str
    options: tuple[int, int, int, int]
    answer_index: int

    @property
    def correct_tool_label(self) -> int:
        return TOOL_LABELS[self.operation]

    @property
    def correct_choice_label(self) -> int:
        return CHOICE_LABELS[self.answer_index]

    @property
    def answer(self) -> int:
        return self.options[self.answer_index]

    def execute(self, tool_label: int) -> int:
        tool_name = TOOL_NAMES[tool_label]
        return _OPERATIONS[tool_name](self.a, self.b)

    def tool_scores(self) -> list[float]:
        return [float(index == self.correct_tool_label) for index in range(len(TOOL_NAMES))]

    def choice_scores(self) -> list[float]:
        return [float(index == self.answer_index) for index in range(4)]

    def step1_prompt(self) -> str:
        options = ", ".join(f"{chr(65 + i)}={value}" for i, value in enumerate(self.options))
        mapping = ", ".join(f"{TOOL_LABELS[name]}={name}" for name in TOOL_NAMES)
        return (
            "You are the routing policy of a tool-using agent. Choose one action label only.\n"
            f"Tool labels: {mapping}.\n"
            f"Question: {self.question}\n"
            f"Candidate final answers: {options}.\n"
            "Action label:"
        )

    def step2_prompt(self, selected_tool_label: int) -> str:
        observation = self.execute(selected_tool_label)
        options = ", ".join(
            f"{CHOICE_LABELS[i]}={value}" for i, value in enumerate(self.options)
        )
        return (
            "You are the answer-selection policy of a tool-using agent. Choose one action label only.\n"
            f"The previous policy selected tool '{TOOL_NAMES[selected_tool_label]}'.\n"
            f"Tool observation: {observation}.\n"
            f"Answer labels: {options}.\n"
            "Select the label whose value is supported by the tool observation.\n"
            "Action label:"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _make_options(operation: str, a: int, b: int, rng: random.Random) -> tuple[tuple[int, ...], int]:
    target = _OPERATIONS[operation](a, b)
    candidates = [target]
    for name in TOOL_NAMES:
        value = _OPERATIONS[name](a, b)
        if value not in candidates:
            candidates.append(value)
    radius = 1
    while len(candidates) < 7:
        for value in (target - radius, target + radius):
            if value not in candidates:
                candidates.append(value)
        radius += 1
    distractors = candidates[1:]
    rng.shuffle(distractors)
    options = [target, *distractors[:3]]
    rng.shuffle(options)
    return tuple(options), options.index(target)


def generate_tasks(count: int, split: str, seed: int) -> list[LogicRouteTask]:
    if split not in {"train", "eval"}:
        raise ValueError(f"split must be train or eval, got {split!r}")
    rng = random.Random(seed + (0 if split == "train" else 100_003))
    templates = _TRAIN_TEMPLATES if split == "train" else _EVAL_TEMPLATES
    tasks: list[LogicRouteTask] = []
    for index in range(count):
        operation = TOOL_NAMES[index % len(TOOL_NAMES)]
        a = rng.randint(4, 29)
        b = rng.randint(2, 13)
        if operation == "subtract" and b > a:
            a, b = b, a
        if a == b and operation in {"maximum", "minimum"}:
            a += 1
        options, answer_index = _make_options(operation, a, b, rng)
        question = templates[operation].format(a=a, b=b)
        tasks.append(
            LogicRouteTask(
                task_id=f"{split}-{index:04d}",
                split=split,
                operation=operation,
                a=a,
                b=b,
                question=question,
                options=options,
                answer_index=answer_index,
            )
        )
    return tasks

