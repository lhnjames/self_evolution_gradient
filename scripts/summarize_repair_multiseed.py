#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def condition_metrics(evaluation: dict[str, Any], name: str) -> dict[str, float]:
    row = evaluation["conditions"][name]
    return {
        "top1": float(row["top1_accuracy"]),
        "nll": float(row["mean_nll"]),
        "kl": float(row["mean_kl_from_plain"]),
    }


def run(paths: list[str]) -> dict[str, Any]:
    rows = []
    for path in paths:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "seed": result["seed"],
            "train_decisions": result["data"]["train_decisions"],
            "dev_decisions": result["data"]["dev_decisions"],
            "selected_skill_kl_lambda": result["training"]["constrained_logic_repair_head"]["kl_lambda"],
            "selected_stage_kl_lambda": result["training"]["constrained_stage_calibration_head"]["kl_lambda"],
        }
        for split, evaluation in (
            ("valid_seen", result["evaluation"]),
            ("valid_unseen", result["additional_evaluations"]["valid_unseen"]),
        ):
            plain = condition_metrics(evaluation, "plain")
            stage = condition_metrics(evaluation, "constrained_stage_calibration_head")
            skill = condition_metrics(evaluation, "constrained_logic_repair_head")
            row[split] = {
                "plain": plain,
                "stage": stage,
                "skill": skill,
                "stage_minus_plain_top1": stage["top1"] - plain["top1"],
                "skill_minus_plain_top1": skill["top1"] - plain["top1"],
                "skill_minus_stage_top1": skill["top1"] - stage["top1"],
                "skill_minus_stage_nll": skill["nll"] - stage["nll"],
            }
        rows.append(row)
    rows.sort(key=lambda row: row["seed"])
    aggregate = {}
    for split in ("valid_seen", "valid_unseen"):
        aggregate[split] = {}
        for metric in (
            "stage_minus_plain_top1",
            "skill_minus_plain_top1",
            "skill_minus_stage_top1",
            "skill_minus_stage_nll",
        ):
            values = np.asarray([row[split][metric] for row in rows], dtype=np.float64)
            aggregate[split][metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "positive_seeds": int(np.sum(values > 0)),
                "total_seeds": len(values),
            }
    return {"seeds": rows, "aggregate": aggregate}


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Unique-episode repair head multi-seed summary",
        "",
        "| seed | seen stage-plain | seen skill-plain | seen skill-stage | unseen stage-plain | unseen skill-plain | unseen skill-stage |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["seeds"]:
        seen, unseen = row["valid_seen"], row["valid_unseen"]
        lines.append(
            f"| {row['seed']} | {seen['stage_minus_plain_top1']:+.4f} | "
            f"{seen['skill_minus_plain_top1']:+.4f} | {seen['skill_minus_stage_top1']:+.4f} | "
            f"{unseen['stage_minus_plain_top1']:+.4f} | {unseen['skill_minus_plain_top1']:+.4f} | "
            f"{unseen['skill_minus_stage_top1']:+.4f} |"
        )
    lines.extend(["", "## Aggregate", ""])
    for split, metrics in summary["aggregate"].items():
        lines.append(f"### {split}")
        lines.append("")
        for metric, value in metrics.items():
            lines.append(
                f"- {metric}: {value['mean']:+.4f} ± {value['sample_std']:.4f}; "
                f"range [{value['min']:+.4f}, {value['max']:+.4f}], "
                f"positive {value['positive_seeds']}/{value['total_seeds']} seeds."
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run(args.results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(render(summary), encoding="utf-8")
    print(render(summary))


if __name__ == "__main__":
    main()
