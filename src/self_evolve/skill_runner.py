from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import torch
import yaml

from .benchmark import ACTION_LABELS, TOOL_NAMES, generate_tasks
from .controller import DistributionRepairHead
from .model import EncodedTask, FrozenLLMScorer, resolve_model_snapshot
from .runner import _seed_everything
from .skill_evolution import (
    SkillEncodedTask,
    build_distillation_batch,
    compare_skill_shift,
    distillation_objective,
    evaluate_skill_method,
)
from .skills import SkillVersion, build_controlled_skill_lineage, inject_skill
from .zo import ZerothOrderAdam


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _encode_skill_tasks(
    scorer: FrozenLLMScorer, plain: list[EncodedTask], skill: SkillVersion
) -> list[SkillEncodedTask]:
    prompts: list[str] = []
    for item in plain:
        prompts.append(inject_skill(item.task.step1_prompt(), skill))
        prompts.extend(
            inject_skill(item.task.step2_prompt(tool), skill)
            for tool in range(len(TOOL_NAMES))
        )
    states = scorer.encode_prompts(prompts)
    result: list[SkillEncodedTask] = []
    cursor = 0
    for item in plain:
        route = states[cursor]
        cursor += 1
        answers = {}
        for tool in range(len(TOOL_NAMES)):
            answers[tool] = states[cursor]
            cursor += 1
        result.append(SkillEncodedTask(item, route, answers))
    return result


def _print(name: str, metrics: dict) -> None:
    print(
        f"{name:25s} strict={metrics['strict_accuracy']:.3f} "
        f"tool={metrics['tool_accuracy']:.3f} choice={metrics['choice_accuracy']:.3f} "
        f"p(correct)={metrics['mean_correct_action_probability']:.3f} "
        f"KL={metrics['mean_kl_to_plain']:.4f}"
    )


def run(config: dict, output_dir: Path, device_override: str | None, seed_override: int | None) -> dict:
    seed = int(config["experiment"]["seed"] if seed_override is None else seed_override)
    _seed_everything(seed)
    device = torch.device(device_override or config["experiment"]["device"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = config["model"].get("path") or resolve_model_snapshot(config["model"]["cache_root"])
    scorer = FrozenLLMScorer(model_path, str(device), int(config["model"]["batch_size"]))
    hidden_size = scorer.hidden_size
    train_tasks = generate_tasks(int(config["benchmark"]["train_tasks"]), "train", seed)
    eval_tasks = generate_tasks(int(config["benchmark"]["eval_tasks"]), "eval", seed)
    plain_train = scorer.encode_tasks(train_tasks)
    plain_eval = scorer.encode_tasks(eval_tasks)
    lineage = build_controlled_skill_lineage(train_tasks)
    temperature = float(config["head"]["base_temperature"])
    fusion = dict(config["fusion"])

    metrics: dict[str, dict] = {}
    traces: dict[str, list[dict]] = {}
    skill_items: dict[str, tuple[list[SkillEncodedTask], list[SkillEncodedTask]]] = {}
    for skill in lineage:
        print(f"Encoding skill version {skill.version}")
        skill_items[skill.version] = (
            _encode_skill_tasks(scorer, plain_train, skill),
            _encode_skill_tasks(scorer, plain_eval, skill),
        )
    del scorer.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    newest_train, newest_eval = skill_items[lineage[-1].version]
    distill_batch = build_distillation_batch(
        newest_train, device, temperature, fusion
    )
    plain_metrics, traces["plain"] = evaluate_skill_method(
        newest_eval, "plain", device, temperature
    )
    metrics["plain"] = plain_metrics
    _print("plain", plain_metrics)
    for skill in lineage:
        _, eval_items = skill_items[skill.version]
        name = f"skill_v{skill.version}"
        value, traces[name] = evaluate_skill_method(
            eval_items, "skill_prompt", device, temperature
        )
        value["distribution_shift"] = compare_skill_shift(eval_items, temperature, device)
        metrics[name] = value
        _print(name, value)

    for method in ("safe_skill", "skill_verifier_projection"):
        output_name = "verifier_gated_skill" if method == "safe_skill" else method
        value, traces[output_name] = evaluate_skill_method(
            newest_eval, method, device, temperature, fusion=fusion
        )
        metrics[output_name] = value
        _print(output_name, value)

    head_cfg = config["head"]
    initial = DistributionRepairHead(
        hidden_size, len(ACTION_LABELS), rank=int(head_cfg["rank"]), seed=seed
    ).to(device)
    initial_state = deepcopy(initial.state_dict())

    fo_head = DistributionRepairHead(
        hidden_size, len(ACTION_LABELS), rank=int(head_cfg["rank"]), seed=seed
    ).to(device)
    fo_head.load_state_dict(initial_state)
    fo_cfg = config["first_order"]
    optimizer = torch.optim.Adam(fo_head.parameters(), lr=float(fo_cfg["learning_rate"]))
    fo_history = []
    for step in range(int(fo_cfg["steps"])):
        optimizer.zero_grad(set_to_none=True)
        loss = distillation_objective(
            fo_head, distill_batch, temperature,
            anchor_weight=float(fo_cfg["anchor_weight"]),
        )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(fo_head.parameters(), 5.0)
        optimizer.step()
        fo_history.append({"step": step + 1, "loss": float(loss.item()), "gradient_norm": float(norm)})
    value, traces["fo_internalized"] = evaluate_skill_method(
        newest_eval, "internalized", device, temperature, head=fo_head
    )
    metrics["fo_internalized"] = value
    _print("FO internalized", value)

    zo_head = DistributionRepairHead(
        hidden_size, len(ACTION_LABELS), rank=int(head_cfg["rank"]), seed=seed
    ).to(device)
    zo_head.load_state_dict(initial_state)
    zo_cfg = config["zero_order"]
    zo = ZerothOrderAdam(
        zo_head,
        learning_rate=float(zo_cfg["learning_rate"]),
        sigma=float(zo_cfg["sigma"]),
        directions=int(zo_cfg["directions"]),
        two_sided=bool(zo_cfg["two_sided"]),
    )
    zo_history = []
    for step in range(int(zo_cfg["steps"])):
        stats = zo.step(lambda: distillation_objective(
            zo_head, distill_batch, temperature,
            anchor_weight=float(zo_cfg["anchor_weight"]),
        ))
        zo_history.append({"step": step + 1, **stats.__dict__})
    value, traces["zo_internalized"] = evaluate_skill_method(
        newest_eval, "internalized", device, temperature, head=zo_head
    )
    metrics["zo_internalized"] = value
    _print("ZO internalized", value)

    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "device": str(device),
            "model_path": model_path,
            "trainable_head_parameters": initial.trainable_parameter_count,
        },
        "skill_lineage": [skill.to_dict() for skill in lineage],
        "metrics": metrics,
        "fo_history": fo_history,
        "zo_history": zo_history,
    }
    _write_json(output_dir / "results.json", result)
    _write_json(output_dir / "config.resolved.json", config)
    _write_json(output_dir / "skillbank.json", [skill.to_dict() for skill in lineage])
    for name, rows in traces.items():
        _write_jsonl(output_dir / f"trace.{name}.jsonl", rows)
    torch.save({"state_dict": fo_head.state_dict()}, output_dir / "head.fo_skill.pt")
    torch.save({"state_dict": zo_head.state_dict()}, output_dir / "head.zo_skill.pt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/skill_experiment.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = Path(args.output or f"outputs/skill_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run(config, output, args.device, args.seed)


if __name__ == "__main__":
    main()
