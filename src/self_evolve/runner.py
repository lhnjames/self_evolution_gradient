from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from .benchmark import ACTION_LABELS, generate_tasks
from .controller import DistributionRepairHead
from .evolution import evaluate, objective
from .model import FrozenLLMScorer, resolve_model_snapshot
from .zo import ZerothOrderAdam


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_metrics(name: str, metrics: dict) -> None:
    print(
        f"{name:24s} strict={metrics['strict_accuracy']:.3f} "
        f"tool={metrics['tool_accuracy']:.3f} choice={metrics['choice_accuracy']:.3f} "
        f"answer_nll={metrics['mean_final_answer_nll']:.3f}"
    )


def run(config: dict, output_dir: Path, device_override: str | None = None, seed_override: int | None = None) -> dict:
    seed = int(config["experiment"]["seed"] if seed_override is None else seed_override)
    _seed_everything(seed)
    device_name = device_override or config["experiment"]["device"]
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = config["model"].get("path")
    if not model_path:
        model_path = resolve_model_snapshot(config["model"]["cache_root"])
    scorer = FrozenLLMScorer(model_path, str(device), int(config["model"]["batch_size"]))
    train_tasks = generate_tasks(int(config["benchmark"]["train_tasks"]), "train", seed)
    eval_tasks = generate_tasks(int(config["benchmark"]["eval_tasks"]), "eval", seed)
    print(f"Encoding {len(train_tasks) + len(eval_tasks)} tasks with frozen model {model_path}")
    encoded_train = scorer.encode_tasks(train_tasks)
    encoded_eval = scorer.encode_tasks(eval_tasks)

    head_config = config["head"]
    initial_head = DistributionRepairHead(
        scorer.hidden_size,
        len(ACTION_LABELS),
        rank=int(head_config["rank"]),
        seed=seed,
    ).to(device)
    initial_state = deepcopy(initial_head.state_dict())
    common = {
        "device": device,
        "base_temperature": float(head_config["base_temperature"]),
        "repair_beta": float(config["repair"]["beta"]),
        "repair_top_k": int(config["repair"]["top_k"]),
    }

    all_metrics: dict[str, dict] = {}
    base_metrics, base_traces = evaluate(initial_head, encoded_eval, use_online_repair=False, **common)
    repaired_metrics, repaired_traces = evaluate(initial_head, encoded_eval, use_online_repair=True, **common)
    all_metrics["base"] = base_metrics
    all_metrics["base_with_online_repair"] = repaired_metrics
    _print_metrics("base", base_metrics)
    _print_metrics("base + online repair", repaired_metrics)

    zo_head = DistributionRepairHead(
        scorer.hidden_size, len(ACTION_LABELS), rank=int(head_config["rank"]), seed=seed
    ).to(device)
    zo_head.load_state_dict(initial_state)
    zo_config = config["zero_order"]
    zo_optimizer = ZerothOrderAdam(
        zo_head,
        learning_rate=float(zo_config["learning_rate"]),
        sigma=float(zo_config["sigma"]),
        directions=int(zo_config["directions"]),
        two_sided=bool(zo_config["two_sided"]),
    )
    zo_history = []
    for step in range(int(zo_config["steps"])):
        stats = zo_optimizer.step(
            lambda: objective(
                zo_head,
                encoded_train,
                mode=str(zo_config["objective"]),
                outcome_weight=float(zo_config["outcome_weight"]),
                **common,
            )
        )
        record = {"step": step + 1, **stats.__dict__}
        zo_history.append(record)
        print(
            f"ZO step {step + 1:02d}: loss={stats.baseline_loss:.4f} "
            f"probe={stats.mean_probe_loss:.4f} |g|={stats.gradient_norm:.4f}"
        )
    zo_metrics, zo_traces = evaluate(zo_head, encoded_eval, use_online_repair=False, **common)
    zo_repaired_metrics, _ = evaluate(zo_head, encoded_eval, use_online_repair=True, **common)
    all_metrics["zero_order_internalized"] = zo_metrics
    all_metrics["zero_order_with_online_repair"] = zo_repaired_metrics
    _print_metrics("ZO internalized", zo_metrics)

    fo_head = DistributionRepairHead(
        scorer.hidden_size, len(ACTION_LABELS), rank=int(head_config["rank"]), seed=seed
    ).to(device)
    fo_head.load_state_dict(initial_state)
    fo_config = config["first_order_baseline"]
    fo_optimizer = torch.optim.Adam(fo_head.parameters(), lr=float(fo_config["learning_rate"]))
    fo_history = []
    for step in range(int(fo_config["steps"])):
        fo_optimizer.zero_grad(set_to_none=True)
        loss = objective(
            fo_head,
            encoded_train,
            mode="repair_kl",
            outcome_weight=float(fo_config["answer_nll_weight"]),
            **common,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(fo_head.parameters(), max_norm=10.0)
        fo_optimizer.step()
        fo_history.append(
            {"step": step + 1, "loss": float(loss.item()), "gradient_norm": float(grad_norm)}
        )
        print(f"FO step {step + 1:02d}: loss={loss.item():.4f} |g|={float(grad_norm):.4f}")
    fo_metrics, fo_traces = evaluate(fo_head, encoded_eval, use_online_repair=False, **common)
    all_metrics["first_order_repair_distilled"] = fo_metrics
    _print_metrics("FO repair distilled", fo_metrics)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "device": str(device),
        "model_path": model_path,
        "hidden_size": scorer.hidden_size,
        "trainable_head_parameters": initial_head.trainable_parameter_count,
        "action_count": len(ACTION_LABELS),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
    }
    result = {"metadata": metadata, "metrics": all_metrics, "zo_history": zo_history, "fo_history": fo_history}
    _write_json(output_dir / "results.json", result)
    _write_json(output_dir / "config.resolved.json", config)
    _write_jsonl(output_dir / "tasks.train.jsonl", [task.to_dict() for task in train_tasks])
    _write_jsonl(output_dir / "tasks.eval.jsonl", [task.to_dict() for task in eval_tasks])
    _write_jsonl(output_dir / "trace.base.jsonl", base_traces)
    _write_jsonl(output_dir / "trace.online_repair.jsonl", repaired_traces)
    _write_jsonl(output_dir / "trace.zo.jsonl", zo_traces)
    _write_jsonl(output_dir / "trace.fo.jsonl", fo_traces)
    torch.save({"state_dict": zo_head.state_dict(), "metadata": metadata}, output_dir / "head.zo.pt")
    torch.save({"state_dict": fo_head.state_dict(), "metadata": metadata}, output_dir / "head.fo.pt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/smoke.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = args.output or f"outputs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run(config, Path(output), args.device, args.seed)


if __name__ == "__main__":
    main()

