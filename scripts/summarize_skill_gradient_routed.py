#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--seed-analysis", action="append", required=True)
    parser.add_argument("--seed-analysis-label", action="append", required=True)
    parser.add_argument("--seed-baseline-analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--required-relative-improvement", type=float, default=0.30)
    args = parser.parse_args()
    if len(args.seed_analysis) != len(args.seed_analysis_label):
        raise ValueError("seed analyses and labels must align")
    seed_reference = json.loads(Path(args.seed_baseline_analysis).read_text(encoding="utf-8"))
    rows = []
    for label, path in zip(args.seed_analysis_label, args.seed_analysis, strict=True):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        route_mode, seed_text = label.rsplit("_seed_", 1)
        for split, summary in payload["splits"].items():
            rows.append(
                {
                    "route_mode": route_mode,
                    "seed": int(seed_text),
                    "split": split,
                    "base_expected_value": summary["base_expected_discounted_value"],
                    "expected_value": summary["comparison_expected_discounted_value"],
                    "relative_expected_value_gain": summary[
                        "relative_expected_discounted_value_gain"
                    ],
                    "base_top_optimal_rate": summary["base_top_value_optimal_rate"],
                    "top_optimal_rate": summary["seed_top_value_optimal_rate"],
                    "relative_top_optimal_gain": summary[
                        "relative_top_value_optimal_rate_gain"
                    ],
                }
            )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["route_mode"], row["split"])].append(row)
    summary_rows = []
    for (route_mode, split), members in sorted(grouped.items()):
        expected = np.asarray([row["relative_expected_value_gain"] for row in members])
        top = np.asarray([row["relative_top_optimal_gain"] for row in members])
        seed_split = seed_reference["splits"][split]
        summary_rows.append(
            {
                "route_mode": route_mode,
                "split": split,
                "seeds": len(members),
                "relative_expected_value_gain_mean": float(expected.mean()),
                "relative_expected_value_gain_std": float(expected.std()),
                "relative_top_optimal_gain_mean": float(top.mean()),
                "relative_top_optimal_gain_std": float(top.std()),
                "all_seeds_expected_value_meet_30pct": bool(
                    np.all(expected >= args.required_relative_improvement)
                ),
                "all_seeds_top_optimal_meet_30pct": bool(
                    np.all(top >= args.required_relative_improvement)
                ),
                "seed_relative_expected_value_gain": seed_split[
                    "relative_expected_discounted_value_gain"
                ],
                "seed_relative_top_optimal_gain": seed_split[
                    "relative_top_value_optimal_rate_gain"
                ],
            }
        )
    result = {
        "required_relative_improvement": args.required_relative_improvement,
        "rows": rows,
        "summary": summary_rows,
    }
    lines = [
        "# 300× 技能条件梯度 bank × SEED 横向比较",
        "",
        "所有值均来自相同 971 状态、FP32 Base、相同 prompt/tokenizer/candidates/value。",
        "",
        "| 路由 | split | 梯度 bank 相对价值提升 | 梯度 bank top-value 相对提升 | SEED 相对价值提升 | SEED top-value 相对提升 | 30% 全种子通过 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        passed = row["all_seeds_expected_value_meet_30pct"] and row[
            "all_seeds_top_optimal_meet_30pct"
        ]
        lines.append(
            f"| {row['route_mode']} | {row['split']} | "
            f"{100 * row['relative_expected_value_gain_mean']:+.2f}% ± "
            f"{100 * row['relative_expected_value_gain_std']:.2f}% | "
            f"{100 * row['relative_top_optimal_gain_mean']:+.2f}% ± "
            f"{100 * row['relative_top_optimal_gain_std']:.2f}% | "
            f"{100 * row['seed_relative_expected_value_gain']:+.2f}% | "
            f"{100 * row['seed_relative_top_optimal_gain']:+.2f}% | "
            f"{'是' if passed else '否'} |"
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
