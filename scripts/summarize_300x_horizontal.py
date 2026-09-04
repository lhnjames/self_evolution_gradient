#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


CONDITIONS = ("seed_fp32", "direct300_last4", "project300_last4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--required-relative-improvement", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.analysis_root)
    loaded = {
        condition: json.loads((root / condition / "results.json").read_text(encoding="utf-8"))
        for condition in CONDITIONS
    }
    result = {
        "required_relative_improvement": args.required_relative_improvement,
        "splits": {},
    }
    lines = [
        "# 300× 价值梯度与 SEED 的 971 状态横向比较",
        "",
        "四个模型均以 FP32、相同 plain prompt、相同候选动作和相同 expert-recovery 长期价值评测。300× 参数增量只由 valid_seen 的 60 条交错动作经验产生；valid_unseen 未参与该候选模型写回。",
        "",
    ]
    for split in ("valid_seen", "valid_unseen"):
        rows = {}
        for condition in CONDITIONS:
            summary = loaded[condition]["splits"][split]
            rows[condition] = {
                "states": summary["states"],
                "base_expected_value": summary["base_expected_discounted_value"],
                "comparison_expected_value": summary["comparison_expected_discounted_value"],
                "absolute_expected_value_gain": summary["descriptive"]["discounted_expected_value_delta"]["mean"],
                "relative_expected_value_gain": summary["relative_expected_discounted_value_gain"],
                "expected_value_gain_ci95": summary["episode_cluster_bootstrap"]["discounted_expected_value_delta"]["ci95"],
                "base_top_value_optimal_rate": summary["base_top_value_optimal_rate"],
                "comparison_top_value_optimal_rate": summary["seed_top_value_optimal_rate"],
                "relative_top_value_optimal_rate_gain": summary["relative_top_value_optimal_rate_gain"],
                "meets_30pct_expected_value_gain": bool(
                    summary["relative_expected_discounted_value_gain"] >= args.required_relative_improvement
                ),
                "meets_30pct_top_rate_gain": bool(
                    summary["relative_top_value_optimal_rate_gain"] >= args.required_relative_improvement
                ),
            }
        seed_value = rows["seed_fp32"]["comparison_expected_value"]
        seed_top = rows["seed_fp32"]["comparison_top_value_optimal_rate"]
        for condition in ("direct300_last4", "project300_last4"):
            rows[condition]["expected_value_minus_seed"] = (
                rows[condition]["comparison_expected_value"] - seed_value
            )
            rows[condition]["top_value_optimal_rate_minus_seed"] = (
                rows[condition]["comparison_top_value_optimal_rate"] - seed_top
            )
        result["splits"][split] = rows
        lines.extend([
            f"## {split}",
            "",
            "| 条件 | 概率加权长期价值 Base→条件 | 绝对 ΔV [episode 95% CI] | 相对提升 | 最高价值动作命中率 Base→条件 | 命中率相对提升 | 价值/命中 30% 门槛 | 相对 SEED 价值 / 命中率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for condition in CONDITIONS:
            row = rows[condition]
            ci = row["expected_value_gain_ci95"]
            vs_seed = "—" if condition == "seed_fp32" else (
                f"{row['expected_value_minus_seed']:+.6f} / {row['top_value_optimal_rate_minus_seed']:+.2%}"
            )
            lines.append(
                f"| {condition} | {row['base_expected_value']:.6f}→{row['comparison_expected_value']:.6f} | "
                f"{row['absolute_expected_value_gain']:+.6f} [{ci[0]:+.6f},{ci[1]:+.6f}] | "
                f"{row['relative_expected_value_gain']:+.2%} | "
                f"{row['base_top_value_optimal_rate']:.2%}→{row['comparison_top_value_optimal_rate']:.2%} | "
                f"{row['relative_top_value_optimal_rate_gain']:+.2%} | "
                f"{'通过' if row['meets_30pct_expected_value_gain'] else '未过'}/"
                f"{'通过' if row['meets_30pct_top_rate_gain'] else '未过'} | {vs_seed} |"
            )
        lines.append("")
    lines.extend([
        "## 判定规则",
        "",
        "概率加权长期价值和最高价值动作命中率分别判定，不能用后者的离散翻转替代前者；只有来自 seen 经验的同一个增量在 unseen 上仍通过，才能称为跨任务性能提升。",
        "",
    ])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = "\n".join(lines)
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "SUMMARY_COMPLETE").touch()
    print(report)


if __name__ == "__main__":
    main()
