#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--required-absolute-gain", type=float, default=0.30)
    return parser.parse_args()


def paired_summary(base: dict, comparison: dict, samples: int, seed: int) -> dict:
    base_rows = {row["gamefile"]: row for row in base["trajectories"]}
    comp_rows = {row["gamefile"]: row for row in comparison["trajectories"]}
    if set(base_rows) != set(comp_rows):
        raise ValueError("Full-episode gamefiles are not paired")
    keys = sorted(base_rows)
    base_won = np.asarray([base_rows[key]["won"] for key in keys], dtype=np.float64)
    comp_won = np.asarray([comp_rows[key]["won"] for key in keys], dtype=np.float64)
    delta = comp_won - base_won
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(keys), size=(samples, len(keys)))
    draws = np.mean(delta[selected], axis=1)
    base_steps = np.asarray([len(base_rows[key]["steps"]) for key in keys], dtype=np.float64)
    comp_steps = np.asarray([len(comp_rows[key]["steps"]) for key in keys], dtype=np.float64)
    return {
        "episodes": len(keys),
        "base_success_rate": float(np.mean(base_won)),
        "comparison_success_rate": float(np.mean(comp_won)),
        "absolute_success_rate_gain": float(np.mean(delta)),
        "relative_success_rate_gain": (
            float(np.mean(delta) / np.mean(base_won)) if np.mean(base_won) > 0 else None
        ),
        "gain_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_gain_positive": float(np.mean(draws > 0)),
        "rescued_episodes": int(np.sum((base_won == 0) & (comp_won == 1))),
        "harmed_episodes": int(np.sum((base_won == 1) & (comp_won == 0))),
        "base_mean_steps": float(np.mean(base_steps)),
        "comparison_mean_steps": float(np.mean(comp_steps)),
        "mean_step_delta": float(np.mean(comp_steps - base_steps)),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    results = {
        "candidate_label": args.candidate_label,
        "required_absolute_success_rate_gain": args.required_absolute_gain,
        "splits": {},
    }
    lines = [
        f"# 完整 ALFWorld 回合：Base / SEED / {args.candidate_label}",
        "",
        f"硬门槛：相对同一批 Base，完整回合成功率绝对提高至少 {100 * args.required_absolute_gain:.0f} 个百分点。所有比较使用同一 gamefile 顺序和配对 bootstrap。",
        "",
        "| split | 条件 | 成功率 Base→条件 | 绝对变化 [95% CI] | rescue / harm | 平均步数变化 | 30 点门槛 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split_position, split in enumerate(("valid_seen", "valid_unseen")):
        base = json.loads((root / f"base_{split}" / "results.json").read_text())
        seed_result = json.loads((root / f"seed_{split}" / "results.json").read_text())
        candidate = json.loads(
            (root / f"{args.candidate_label}_{split}" / "results.json").read_text()
        )
        seed_summary = paired_summary(
            base, seed_result, args.bootstrap_samples, args.seed + split_position * 10
        )
        candidate_summary = paired_summary(
            base, candidate, args.bootstrap_samples, args.seed + split_position * 10 + 1
        )
        candidate_summary["meets_required_absolute_gain"] = bool(
            candidate_summary["absolute_success_rate_gain"] >= args.required_absolute_gain
        )
        results["splits"][split] = {"seed_vs_base": seed_summary, "candidate_vs_base": candidate_summary}
        for label, row in (("SEED", seed_summary), (args.candidate_label, candidate_summary)):
            passed = row["absolute_success_rate_gain"] >= args.required_absolute_gain
            lines.append(
                f"| {split} | {label} | {row['base_success_rate']:.1%}→{row['comparison_success_rate']:.1%} | "
                f"{row['absolute_success_rate_gain']:+.1%} [{row['gain_ci95'][0]:+.1%},{row['gain_ci95'][1]:+.1%}] | "
                f"{row['rescued_episodes']} / {row['harmed_episodes']} | {row['mean_step_delta']:+.2f} | "
                f"{'通过' if passed else '未通过'} |"
            )
    lines.extend([
        "",
        "这里的成功率是模型自身闭环 rollout，不是 expert-recovery 条件价值；两者必须分开报告。",
        "",
    ])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = "\n".join(lines)
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "ANALYSIS_COMPLETE").touch()
    print(report)


if __name__ == "__main__":
    main()
