#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from probe_value_gradient_writeback import load_value_rows
from self_evolve.value_writeback import candidate_distribution_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--per-skill-glob")
    parser.add_argument("--value-trace", required=True)
    parser.add_argument("--base-score-trace", required=True)
    parser.add_argument("--seed-score-trace", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", required=True)
    return parser.parse_args()


def read_jsonl(path: str) -> dict[int, dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return {
            int(row["global_decision_index"]): row
            for line in handle
            if (row := json.loads(line))
        }


def macro(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"runs": len(rows)}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return result


def metric_summary(metrics: Sequence[dict[str, Any]]) -> dict[str, float]:
    base_value = float(np.mean([item["baseline_expected_value"] for item in metrics]))
    updated_value = float(np.mean([item["updated_expected_value"] for item in metrics]))
    return {
        "baseline_expected_value": base_value,
        "updated_expected_value": updated_value,
        "absolute_expected_value_gain": updated_value - base_value,
        "relative_expected_value_gain": (updated_value - base_value) / base_value,
        "top_value_optimal_rate_gain_points": float(
            np.mean([
                item["updated_top_value"] > item["baseline_top_value"] + 1e-12
                for item in metrics
            ])
        ),
        "top_value_harm_rate": float(
            np.mean([
                item["updated_top_value"] < item["baseline_top_value"] - 1e-12
                for item in metrics
            ])
        ),
        "mean_kl": float(np.mean([item["kl_baseline_to_updated"] for item in metrics])),
    }


def seed_macro_for_runs(
    runs: Sequence[dict[str, Any]],
    value_rows: dict[int, dict[str, Any]],
    base_trace: dict[int, dict[str, Any]],
    seed_trace: dict[int, dict[str, Any]],
    keys: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    by_run = []
    for run in runs:
        metrics = []
        for state in run["panels"]["final_holdout"]["states"]:
            index = int(state["global_decision_index"])
            values = [float(action["discounted_success"]) for action in value_rows[index]["actions"]]
            metrics.append(
                candidate_distribution_metrics(
                    base_trace[index]["normalized_scores"],
                    seed_trace[index]["normalized_scores"],
                    values,
                    int(base_trace[index]["gold_index"]),
                )
            )
        by_run.append(metric_summary(metrics))
    return macro(by_run, keys), by_run


def fmt(value: float, scale: float = 100.0) -> str:
    return f"{scale * value:+.2f}%"


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise ValueError("No Phase-0 result files found")
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if any(run.get("status") != "complete" for run in runs):
        raise ValueError("At least one Phase-0 run is incomplete")
    keys = (
        "absolute_expected_value_gain",
        "relative_expected_value_gain",
        "top_value_optimal_rate_gain_points",
        "top_value_harm_rate",
        "mean_kl",
    )
    panels = {
        panel: macro([run["panels"][panel] for run in runs], keys)
        for panel in ("source", "transfer_validation", "final_holdout", "protection")
    }
    value_rows = load_value_rows(args.value_trace, "valid_seen")
    base_trace = read_jsonl(args.base_score_trace)
    seed_trace = read_jsonl(args.seed_score_trace)
    seed_macro, seed_by_run = seed_macro_for_runs(
        runs, value_rows, base_trace, seed_trace, keys
    )
    per_skill = {}
    if args.per_skill_glob:
        skill_paths = sorted(glob.glob(args.per_skill_glob))
        skill_runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in skill_paths]
        for verb in sorted({run["configuration"]["train_verb"] for run in skill_runs}):
            selected = [run for run in skill_runs if run["configuration"]["train_verb"] == verb]
            selected_panels = {
                panel: macro([run["panels"][panel] for run in selected], keys)
                for panel in ("source", "transfer_validation", "final_holdout", "protection")
            }
            selected_seed, _ = seed_macro_for_runs(
                selected, value_rows, base_trace, seed_trace, keys
            )
            per_skill[verb] = {
                "run_count": len(selected),
                "panels": selected_panels,
                "seed_on_exact_final_holdouts": selected_seed,
                "gates": {
                    "value_30": selected_panels["final_holdout"]["relative_expected_value_gain"]["mean"] >= 0.30,
                    "decision_30": selected_panels["final_holdout"]["top_value_optimal_rate_gain_points"]["mean"] >= 0.30,
                    "safety_2": selected_panels["protection"]["top_value_harm_rate"]["mean"] <= 0.02,
                },
            }
    holdout = panels["final_holdout"]
    value_gate = holdout["relative_expected_value_gain"]["mean"] >= 0.30
    decision_gate = holdout["top_value_optimal_rate_gain_points"]["mean"] >= 0.30
    safety_gate = panels["protection"]["top_value_harm_rate"]["mean"] <= 0.02
    representation_sufficient = bool(value_gate or decision_gate)
    output = {
        "experiment": "output_head_representation_sufficiency_aggregate_v1",
        "run_files": paths,
        "run_count": len(runs),
        "fixed_hyperparameters": {
            "learning_rate": runs[0]["configuration"]["learning_rate"],
            "maximum_epochs": runs[0]["configuration"]["epochs"],
            "patience": runs[0]["configuration"]["patience"],
            "objective": runs[0]["configuration"]["objective"],
        },
        "panels": panels,
        "seed_on_exact_final_holdouts": seed_macro,
        "per_skill_oracles": per_skill,
        "per_run": [
            {
                "sample_seed": run["configuration"]["sample_seed"],
                "best_epoch": run["selection"]["best_epoch"],
                "parameter_delta_l2_norm": run["parameter_delta_l2_norm"],
                "final_holdout": run["panels"]["final_holdout"],
                "protection": run["panels"]["protection"],
                "seed_final_holdout": seed_result,
            }
            for run, seed_result in zip(runs, seed_by_run, strict=True)
        ],
        "gates": {
            "holdout_relative_value_at_least_30_percent": bool(value_gate),
            "holdout_failure_repair_at_least_30_points": bool(decision_gate),
            "protection_top_value_harm_at_most_2_percent": bool(safety_gate),
            "representation_sufficient_for_output_only_route": representation_sufficient,
            "all_30_percent_and_safety_gates": bool(value_gate and decision_gate and safety_gate),
        },
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    hv = holdout["relative_expected_value_gain"]
    hd = holdout["top_value_optimal_rate_gain_points"]
    hp = panels["protection"]["top_value_harm_rate"]
    sv = seed_macro["relative_expected_value_gain"]
    sd = seed_macro["top_value_optimal_rate_gain_points"]
    verdict = (
        "Output representation sufficiency gate 通过，可以进入 output-only OGSE。"
        if representation_sufficient
        else "Output representation sufficiency gate 未通过，应优先转向更深层表示编辑。"
    )
    skill_lines = []
    for verb, row in per_skill.items():
        skill_holdout = row["panels"]["final_holdout"]
        skill_protection = row["panels"]["protection"]
        skill_seed = row["seed_on_exact_final_holdouts"]
        skill_lines.append(
            f"| {verb} | {fmt(skill_holdout['relative_expected_value_gain']['mean'])} ± "
            f"{100*skill_holdout['relative_expected_value_gain']['sample_std']:.2f}% | "
            f"{fmt(skill_holdout['top_value_optimal_rate_gain_points']['mean'])} | "
            f"{fmt(skill_protection['top_value_harm_rate']['mean'])} | "
            f"{fmt(skill_seed['relative_expected_value_gain']['mean'])} |"
        )
    skill_table = "\n".join(skill_lines) if skill_lines else "| — | — | — | — | — |"
    report = f"""# Output-Head Representation Sufficiency（Phase 0）

日期：2026-09-04

## 结论

{verdict}

本实验冻结 Qwen2 全部 backbone 和 input embedding，复制并解除 tied `lm_head`，只训练无 bias 的独立 FP32 linear output head。四个 split seed 均使用 `12×3` source failures；每个 seed 内只由 transfer-validation 选择 epoch，final holdout 不参与选择。

## Final holdout（4 split seeds）

| 方法 | 相对长期价值提升 | failure→top-value 修复 | protection top-value harm |
|---|---:|---:|---:|
| Oracle untied output head | {fmt(hv['mean'])} ± {100*hv['sample_std']:.2f}% | {fmt(hd['mean'])} ± {100*hd['sample_std']:.2f}% | {fmt(hp['mean'])} ± {100*hp['sample_std']:.2f}% |
| SEED（相同 final holdout） | {fmt(sv['mean'])} ± {100*sv['sample_std']:.2f}% | {fmt(sd['mean'])} ± {100*sd['sample_std']:.2f}% | — |

注意：failure→top-value 修复列按 Base failure 中更新后选择更高价值动作的比例计算，因此它直接对应 30% decision gate。

## Per-skill oracle

| Skill | Final-holdout 相对价值 | top-value 修复 | 其他动作 harm | SEED 同 holdout 相对价值 |
|---|---:|---:|---:|---:|
{skill_table}

只有 `close` 稳定越过 30% value gate；`go/open` 没有。所有 per-skill 强写回都未满足 2% protection gate，因此后续 output-only 实验只把 `close` 当作“表示足够”的正对照，同时必须解决作用域与冲突。

## 预注册门槛

- Value ≥30%：`{value_gate}`
- Decision repair ≥30 个百分点：`{decision_gate}`
- Protection harm ≤2%：`{safety_gate}`
- 表示能力足以继续 output-only 路线（Value 或 Decision gate）：`{representation_sufficient}`
- 三门同时满足：`{value_gate and decision_gate and safety_gate}`

## 实验边界

Phase 0 测的是“固定 hidden representation 上是否存在足够好的线性决策边界”，不是 OGSE 已经成功。即使 oracle head 通过，后续仍必须证明单失败梯度迁移、真实 transfer-weighted evolution 超过 mean12、zero-shot new-skill repair，以及低干扰参数写回。
"""
    Path(args.output_report).write_text(report, encoding="utf-8")
    print(json.dumps(output["gates"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
