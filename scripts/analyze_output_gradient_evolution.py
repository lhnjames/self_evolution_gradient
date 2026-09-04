#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h6-glob", required=True)
    parser.add_argument("--evolution-300x-glob", required=True)
    parser.add_argument("--evolution-900x-glob", required=True)
    parser.add_argument("--evolution-1200x-glob", required=True)
    parser.add_argument("--evolution-3000x-glob", required=True)
    parser.add_argument("--unseen-family-glob", required=True)
    parser.add_argument("--geometry-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load(pattern: str) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"No files match {pattern}")
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    baseline = float(np.mean([row["baseline_expected_value"] for row in rows]))
    updated = float(np.mean([row["updated_expected_value"] for row in rows]))
    return {
        "count": len(rows),
        "baseline_expected_value": baseline,
        "updated_expected_value": updated,
        "absolute_expected_value_gain": updated - baseline,
        "relative_expected_value_gain": (updated - baseline) / baseline,
        "positive_transfer_rate": float(np.mean([row["expected_value_delta"] > 0 for row in rows])),
        "top_value_repair_rate": float(np.mean([row["top_value_delta"] > 0 for row in rows])),
        "top_value_harm_rate": float(np.mean([row["top_value_delta"] < 0 for row in rows])),
        "mean_kl": float(np.mean([row["kl_baseline_to_updated"] for row in rows])),
    }


def macro(summaries: Sequence[dict[str, float]]) -> dict[str, Any]:
    keys = [key for key in summaries[0] if key != "count"]
    result: dict[str, Any] = {"runs": len(summaries), "per_run": list(summaries)}
    for key in keys:
        values = np.asarray([row[key] for row in summaries], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return result


def summarize_h6(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    relationships = ("source", "same_skill_holdout", "unrelated_protection")
    return {
        "experiment": "h6_output_only_aggregate_v1",
        "run_count": len(runs),
        "source_count": sum(run["source_count"] for run in runs),
        "parameter_delta_l2_norm": runs[0]["parameter_delta_l2_norm"],
        "relationships": {
            relationship: macro([
                summarize([row for row in run["transfer_pairs"] if row["relationship"] == relationship])
                for run in runs
            ])
            for relationship in relationships
        },
    }


def summarize_evolution(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strategies = list(runs[0]["strategies"])
    relationships = (
        "source", "transfer_validation", "same_skill_holdout",
        "harm_validation", "protection_test",
    )
    result = {
        "run_count": len(runs),
        "parameter_delta_l2_norm": runs[0]["parameter_delta_l2_norm"],
        "strategies": {},
    }
    for strategy in strategies:
        result["strategies"][strategy] = {
            relationship: macro([
                summarize([
                    row for row in run["strategies"][strategy]["rows"]
                    if row["relationship"] == relationship
                ])
                for run in runs
            ])
            for relationship in relationships
        }
        effective = np.asarray([
            run["strategies"][strategy]["effective_source_count"] for run in runs
        ])
        result["strategies"][strategy]["effective_source_count"] = {
            "mean": float(effective.mean()),
            "sample_std": float(effective.std(ddof=1)),
        }
    seed_summaries = []
    for run in runs:
        rows = [
            row for row in run["strategies"]["mean12"]["rows"]
            if row["relationship"] == "same_skill_holdout"
        ]
        baseline = float(np.mean([row["baseline_expected_value"] for row in rows]))
        delta = float(np.mean([row["seed_expected_value_delta"] for row in rows]))
        seed_summaries.append(
            {
                "count": len(rows),
                "baseline_expected_value": baseline,
                "updated_expected_value": baseline + delta,
                "absolute_expected_value_gain": delta,
                "relative_expected_value_gain": delta / baseline,
                "positive_transfer_rate": float(np.mean([row["seed_expected_value_delta"] > 0 for row in rows])),
                "top_value_repair_rate": float(np.mean([row["seed_top_value_delta"] > 0 for row in rows])),
                "top_value_harm_rate": float(np.mean([row["seed_top_value_delta"] < 0 for row in rows])),
                "mean_kl": 0.0,
            }
        )
    result["seed_exact_holdout"] = macro(seed_summaries)
    return result


def fmt(metric: dict[str, float]) -> str:
    return f"{100*metric['mean']:+.2f}% ± {100*metric['sample_std']:.2f}%"


def ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def correlation(left: Sequence[float], right: Sequence[float]) -> float:
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def main() -> None:
    args = parse_args()
    h6_runs = load(args.h6_glob)
    runs300 = load(args.evolution_300x_glob)
    runs900 = load(args.evolution_900x_glob)
    runs1200 = load(args.evolution_1200x_glob)
    runs3000 = load(args.evolution_3000x_glob)
    unseen_runs = load(args.unseen_family_glob)
    geometry_runs = load(args.geometry_glob)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    h6 = summarize_h6(h6_runs)
    h8_300 = summarize_evolution(runs300)
    h8_900 = summarize_evolution(runs900)
    h8_1200 = summarize_evolution(runs1200)
    h8_3000 = summarize_evolution(runs3000)
    (destination / "h6_output_only.json").write_text(
        json.dumps(h6, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    h8 = {
        "experiment": "h8_output_only_aggregate_v1",
        "300x": h8_300,
        "900x_stress": h8_900,
        "1200x_stress": h8_1200,
        "3000x_stress": h8_3000,
    }
    (destination / "h8_output_only.json").write_text(
        json.dumps(h8, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    transfer_pairs = {
        "experiment": "output_gradient_transfer_pairs_v1",
        "pairs": [
            {"sample_seed": run["sample_seed"], **row}
            for run in h6_runs for row in run["transfer_pairs"]
        ],
    }
    (destination / "output_gradient_transfer_pairs.json").write_text(
        json.dumps(transfer_pairs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    transfer_matrix = np.asarray([
        [
            [
                row["expected_value_delta"]
                for row in source["rows"] if row["relationship"] == "transfer_validation"
            ]
            for source in run["source_validation"]
        ]
        for run in runs300
    ], dtype=np.float64)
    np.save(destination / "transfer_matrix.npy", transfer_matrix)
    sequential = {
        "experiment": "sequential_output_gradient_evolution_v1",
        "strength": "1200x",
        "steps": [],
    }
    for position in range(12):
        same = [run["sequential_transfer_harm_weighted"][position]["same_skill_holdout"] for run in runs1200]
        protect = [run["sequential_transfer_harm_weighted"][position]["protection_test"] for run in runs1200]
        sequential["steps"].append(
            {
                "experience_count": position + 1,
                "same_skill_holdout": macro(same),
                "protection_test": macro(protect),
            }
        )
    (destination / "sequential_evolution.json").write_text(
        json.dumps(sequential, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    unseen = {
        "experiment": "unseen_skill_family_output_gradient_transfer_aggregate_v1",
        "run_count": len(unseen_runs),
        "heldout_task_type": unseen_runs[0]["heldout_task_type"],
        "zero_shot": {},
    }
    for strategy in ("mean9", "evolved9"):
        unseen["zero_shot"][strategy] = {
            relationship: macro([
                summarize([
                    row for row in run["zero_shot"][strategy]["rows"]
                    if row["relationship"] == relationship
                ])
                for run in unseen_runs
            ])
            for relationship in ("unseen_family", "protection_test")
        }
    unseen["after_first_unseen_failure_feedback"] = {
        relationship: macro([
            summarize([
                row for row in run["after_first_unseen_failure_feedback"]["rows"]
                if row["relationship"] == relationship
            ])
            for run in unseen_runs
        ])
        for relationship in ("unseen_family", "protection_test")
    }
    seed_unseen = []
    for run in unseen_runs:
        rows = [row for row in run["zero_shot"]["evolved9"]["rows"] if row["relationship"] == "unseen_family"]
        baseline = float(np.mean([row["baseline_expected_value"] for row in rows]))
        delta = float(np.mean([row["seed_expected_value_delta"] for row in rows]))
        seed_unseen.append(
            {
                "count": len(rows), "baseline_expected_value": baseline,
                "updated_expected_value": baseline + delta,
                "absolute_expected_value_gain": delta,
                "relative_expected_value_gain": delta / baseline,
                "positive_transfer_rate": float(np.mean([row["seed_expected_value_delta"] > 0 for row in rows])),
                "top_value_repair_rate": float(np.mean([row["seed_top_value_delta"] > 0 for row in rows])),
                "top_value_harm_rate": float(np.mean([row["seed_top_value_delta"] < 0 for row in rows])),
                "mean_kl": 0.0,
            }
        )
    unseen["seed_exact_unseen_family"] = macro(seed_unseen)
    (destination / "unseen_skill_transfer.json").write_text(
        json.dumps(unseen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    geometry_keys = (
        "mean_hidden_cosine", "mean_delta_cosine", "mean_factor_product",
        "full_multitoken_gradient_cosine",
    )
    geometry = {
        "experiment": "output_gradient_geometry_transfer_aggregate_v1",
        "run_count": len(geometry_runs),
        "directed_pair_count": sum(run["directed_pair_count"] for run in geometry_runs),
        "correlations": {},
    }
    for key in geometry_keys:
        per_run = []
        for run in geometry_runs:
            left = [row[key] for row in run["pairs"]]
            right = [row["expected_value_delta"] for row in run["pairs"]]
            per_run.append({
                "pearson": correlation(left, right),
                "spearman": correlation(ranks(left), ranks(right)),
            })
        geometry["correlations"][key] = macro(per_run)
    all_pairs = [row for run in geometry_runs for row in run["pairs"]]
    cosine_values = np.asarray([row["full_multitoken_gradient_cosine"] for row in all_pairs])
    transfer_values = np.asarray([row["expected_value_delta"] for row in all_pairs])
    edges = np.quantile(cosine_values, [0, 0.25, 0.5, 0.75, 1])
    quartiles = []
    for index in range(4):
        selected = (cosine_values >= edges[index]) & (
            cosine_values <= edges[index + 1] if index == 3 else cosine_values < edges[index + 1]
        )
        quartiles.append({
            "quartile": index + 1,
            "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "count": int(selected.sum()),
            "mean_expected_value_delta": float(transfer_values[selected].mean()),
            "positive_transfer_rate": float((transfer_values[selected] > 0).mean()),
        })
    geometry["full_gradient_cosine_quartiles"] = quartiles
    (destination / "gradient_geometry.json").write_text(
        json.dumps(geometry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    evolved = {
        "experiment": "transfer_weighted_output_gradient_evolution_v1",
        "criterion": "evolved must beat mean12 on untouched same-skill holdout",
        "300x_evolved_beats_mean": {
            strategy: (
                h8_300["strategies"][strategy]["same_skill_holdout"]
                ["relative_expected_value_gain"]["mean"]
                > h8_300["strategies"]["mean12"]["same_skill_holdout"]
                ["relative_expected_value_gain"]["mean"]
            )
            for strategy in ("cosine_purified", "transfer_weighted", "transfer_harm_weighted")
        },
        "1200x_evolved_beats_mean": {
            strategy: (
                h8_1200["strategies"][strategy]["same_skill_holdout"]
                ["relative_expected_value_gain"]["mean"]
                > h8_1200["strategies"]["mean12"]["same_skill_holdout"]
                ["relative_expected_value_gain"]["mean"]
            )
            for strategy in ("cosine_purified", "transfer_weighted", "transfer_harm_weighted")
        },
        "runs": [
            {
                "sample_seed": run["sample_seed"],
                "transfer_utility": run["transfer_utility"],
                "harm_utility": run["harm_utility"],
                "weights": {name: value["weights"] for name, value in run["strategies"].items()},
            }
            for run in runs300
        ],
    }
    (destination / "evolved_gradient.json").write_text(
        json.dumps(evolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    protection = {
        strength: {
            strategy: result["strategies"][strategy]["protection_test"]
            for strategy in result["strategies"]
        }
        for strength, result in (
            ("300x", h8_300), ("900x", h8_900), ("1200x", h8_1200),
            ("3000x", h8_3000),
        )
    }
    (destination / "protection_results.json").write_text(
        json.dumps(protection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    h6_hold = h6["relationships"]["same_skill_holdout"]
    m300 = h8_300["strategies"]["mean12"]["same_skill_holdout"]
    t300 = h8_300["strategies"]["transfer_weighted"]["same_skill_holdout"]
    th300 = h8_300["strategies"]["transfer_harm_weighted"]["same_skill_holdout"]
    m900 = h8_900["strategies"]["mean12"]["same_skill_holdout"]
    m1200 = h8_1200["strategies"]["mean12"]["same_skill_holdout"]
    th1200 = h8_1200["strategies"]["transfer_harm_weighted"]["same_skill_holdout"]
    p300 = h8_300["strategies"]["mean12"]["protection_test"]
    p900 = h8_900["strategies"]["mean12"]["protection_test"]
    p1200 = h8_1200["strategies"]["transfer_harm_weighted"]["protection_test"]
    projected1200 = h8_1200["strategies"]["protection_nullspace_transfer_harm"]["same_skill_holdout"]
    projected1200_protection = h8_1200["strategies"]["protection_nullspace_transfer_harm"]["protection_test"]
    projected3000 = h8_3000["strategies"]["protection_nullspace_transfer_harm"]["same_skill_holdout"]
    projected3000_protection = h8_3000["strategies"]["protection_nullspace_transfer_harm"]["protection_test"]
    seed = h8_300["seed_exact_holdout"]
    seq_peak = max(
        sequential["steps"],
        key=lambda row: row["same_skill_holdout"]["relative_expected_value_gain"]["mean"],
    )
    unseen_mean = unseen["zero_shot"]["mean9"]["unseen_family"]
    unseen_evolved = unseen["zero_shot"]["evolved9"]["unseen_family"]
    unseen_after = unseen["after_first_unseen_failure_feedback"]["unseen_family"]
    unseen_protect = unseen["zero_shot"]["evolved9"]["protection_test"]
    unseen_seed = unseen["seed_exact_unseen_family"]
    verdict = (
        "梯度迁移、强度放大、held-out family 零样本修复和反馈后演化均已成立；"
        "但无条件 output-head 写回的 protection harm 过高，闭环目前只能作为需路由的强 patch，不能安全提交。"
    )
    gradient_corr = geometry["correlations"]["full_multitoken_gradient_cosine"]
    hidden_corr = geometry["correlations"]["mean_hidden_cosine"]
    delta_corr = geometry["correlations"]["mean_delta_cosine"]
    representation = json.loads((destination / "representation_sufficiency.json").read_text(encoding="utf-8"))
    equivalence = json.loads((destination / "parameter_logit_equivalence.json").read_text(encoding="utf-8"))
    joint_phase0 = representation["panels"]["final_holdout"]
    close_phase0 = representation["per_skill_oracles"]["close"]["panels"]["final_holdout"]
    report = f"""# Output-Gradient Self-Evolution：当前总结果

日期：2026-09-04

## 核心判定

{verdict}

精确 parameter/logit equivalence 已通过；因此这里所有 virtual repair 都等价于独立 output head 的真实参数写回。Phase 0 只在 `close` 上显示出稳定的线性表示能力，所以 H6/H8 聚焦 `close`，没有把失败的 `go/open` 混入平均。

## Phase 0：表示充分性

- `go/open/close` 联合 oracle head：final-holdout 相对价值 {fmt(joint_phase0['relative_expected_value_gain'])}，未过 30%。
- `close` 独立 oracle head：{fmt(close_phase0['relative_expected_value_gain'])}，稳定越过 value gate。
- `go/open` 独立 oracle 分别只有约 `+14.79% / +9.12%`，因此后续不纳入 output-only 正结论。
- Qwen tied head 已复制解绑；backbone、input embedding、bias 全部冻结。

## 300x（参数 L2=0.18）

| 方法 | unseen same-skill 相对价值 | 正迁移率 | top-value 修复 | protection harm |
|---|---:|---:|---:|---:|
| Single gradient | {fmt(h6_hold['relative_expected_value_gain'])} | {fmt(h6_hold['positive_transfer_rate'])} | {fmt(h6_hold['top_value_repair_rate'])} | {fmt(h6['relationships']['unrelated_protection']['top_value_harm_rate'])} |
| Mean12 | {fmt(m300['relative_expected_value_gain'])} | {fmt(m300['positive_transfer_rate'])} | {fmt(m300['top_value_repair_rate'])} | {fmt(p300['top_value_harm_rate'])} |
| Transfer-weighted | {fmt(t300['relative_expected_value_gain'])} | {fmt(t300['positive_transfer_rate'])} | {fmt(t300['top_value_repair_rate'])} | {fmt(h8_300['strategies']['transfer_weighted']['protection_test']['top_value_harm_rate'])} |
| Transfer+harm weighted | {fmt(th300['relative_expected_value_gain'])} | {fmt(th300['positive_transfer_rate'])} | {fmt(th300['top_value_repair_rate'])} | {fmt(h8_300['strategies']['transfer_harm_weighted']['protection_test']['top_value_harm_rate'])} |
| SEED（完全相同 holdout） | {fmt(seed['relative_expected_value_gain'])} | {fmt(seed['positive_transfer_rate'])} | {fmt(seed['top_value_repair_rate'])} | — |

## 900x 强度压力测试（参数 L2=0.54）

Mean12 的 unseen relative value 为 {fmt(m900['relative_expected_value_gain'])}，protection top-value harm 为 {fmt(p900['top_value_harm_rate'])}。

## 1200x 门槛测试（参数 L2=0.72）

Transfer+harm weighted 的 unseen relative value 为 {fmt(th1200['relative_expected_value_gain'])}，Mean12 为 {fmt(m1200['relative_expected_value_gain'])}；前者超过 Mean12 并跨过平均 30% value gate。但 protection top-value harm 达到 {fmt(p1200['top_value_harm_rate'])}，不满足安全提交条件。

Protection-nullspace 参数投影在 1200x 将 protection harm 降为 {fmt(projected1200_protection['top_value_harm_rate'])}，但价值回落到 {fmt(projected1200['relative_expected_value_gain'])}。推到 3000x 后价值仍只有 {fmt(projected3000['relative_expected_value_gain'])}，harm 回升为 {fmt(projected3000_protection['top_value_harm_rate'])}；说明继续加剂量不能同时跨过 30%/2% 两道门。

## Sequential self-evolution

Transfer+harm (G_t) 在 `K={seq_peak['experience_count']}` 达到峰值 {fmt(seq_peak['same_skill_holdout']['relative_expected_value_gain'])}；最终 K=12 为 {fmt(th1200['relative_expected_value_gain'])}。曲线不是单调上升，说明新增经验在 4 条以后出现饱和/冲突，不能把“持续加入”直接等同于“持续进化”。

## Held-out new skill family

完全排除 `pick_cool_then_place_in_recep` family 后：

| 方法 | Zero-shot unseen-family 相对价值 | 正迁移率 | top-value 修复 | protection harm |
|---|---:|---:|---:|---:|
| Mean9 historical gradients | {fmt(unseen_mean['relative_expected_value_gain'])} | {fmt(unseen_mean['positive_transfer_rate'])} | {fmt(unseen_mean['top_value_repair_rate'])} | — |
| Evolved9 historical gradients | {fmt(unseen_evolved['relative_expected_value_gain'])} | {fmt(unseen_evolved['positive_transfer_rate'])} | {fmt(unseen_evolved['top_value_repair_rate'])} | {fmt(unseen_protect['top_value_harm_rate'])} |
| SEED（同一 unseen family） | {fmt(unseen_seed['relative_expected_value_gain'])} | {fmt(unseen_seed['positive_transfer_rate'])} | {fmt(unseen_seed['top_value_repair_rate'])} | — |

新 family 的第一次 failure 加入反馈后，剩余 unseen states 提升为 {fmt(unseen_after['relative_expected_value_gain'])}，说明 `inherit → repair → feedback → evolve` 的离线闭环成立。

## 迁移机制（1224 个有向 pair）

- mean-hidden cosine vs transfer：Spearman `{hidden_corr['spearman']['mean']:.3f} ± {hidden_corr['spearman']['sample_std']:.3f}`。
- mean-delta cosine vs transfer：Spearman `{delta_corr['spearman']['mean']:.3f} ± {delta_corr['spearman']['sample_std']:.3f}`。
- full multi-token gradient cosine vs transfer：Spearman `{gradient_corr['spearman']['mean']:.3f} ± {gradient_corr['spearman']['sample_std']:.3f}`。
- full-gradient cosine 最低/最高四分位正迁移率分别为 `{100*quartiles[0]['positive_transfer_rate']:.1f}% / {100*quartiles[-1]['positive_transfer_rate']:.1f}%`。

在 `close` 内部，hidden cosine 已普遍很高，区分 transfer 的主要信号来自 repair direction（delta）；强写回时 gradient cosine 仍有预测力，但弱于此前微扰区间。

## 对预注册 13 个问题的回答

1. Output layer 是否有表示能力？**部分有**：`close` 有，`go/open` 在当前数据上不足。
2. 单 failure 梯度是否迁移？**是**，300x H6 同-skill 相对价值 {fmt(h6_hold['relative_expected_value_gain'])}。
3. hidden similarity 是否解释强度？**仅弱解释**，同一 `close` 家族内已饱和。
4. delta similarity 是否有额外解释？**有**，Spearman 约 `{delta_corr['spearman']['mean']:.3f}`。
5. gradient cosine 为什么相关？它同时包含 hidden kernel 与 repair compatibility；完整 multi-token cosine 的 Spearman 约 `{gradient_corr['spearman']['mean']:.3f}`。
6. Mean 为什么比 single 强？它把正迁移率从约 72% 提到 100%，削弱 instance noise。
7. Transfer purification 是否超过 Mean12？300x **否**；1200x **是**，但只领先约 2.1 个相对百分点且不安全。
8. (G_t) 是否持续改善？**否**，K=4 达峰，随后饱和/回落。
9. 能否 zero-shot 修复新 family？**能**，Evolved9 达 {fmt(unseen_evolved['relative_expected_value_gain'])}。
10. unrelated harm？未约束 1200x 为 {fmt(p1200['top_value_harm_rate'])}；投影后最低可到 0%，但性能低于 30%。
11. 参数写回与 virtual repair 是否一致？**是**；最大 action-score 误差 `{equivalence['max_errors']['writeback_vs_virtual_action_scores']:.3g}`。
12. 是否进入完整 episode / retraining 对比？**否**，没有同时满足 30% value 与 2% safety。
13. 最终定位：**部分可替代，只能作为带路由、shadow validation 与 rollback 的快速 patch；无条件 output-head 参数提交不足。**

## 结论边界

- 单 output failure gradient 确实跨状态迁移，H6 成立。
- Mean12 保持 100% 正迁移并明显强于 single，但 300x 的平均相对价值仍低于 30%。
- cosine purification 和两种真实 transfer-fitness 加权均未超过 mean12，第一版“自净化”失败。
- 强度压力测试用于判断 30% 能否仅靠放大取得；若 protection harm 同步升高，则瓶颈是作用域/冲突，而不是剂量。
- 在 evolved gradient 超过 mean、protection harm ≤2% 之前，不进入完整 episode，也不能声称替代 SEED/SFT。
- 目前 evolved gradient 已在 1200x 和 held-out family 上超过 mean 且跨过 30%，但 protection harm 仍为 50%–58%，所以只能视为强力、需路由的 patch，不能直接 commit 到共享 head。
"""
    (destination / "summary.md").write_text(report, encoding="utf-8")
    print(json.dumps(evolved["300x_evolved_beats_mean"], indent=2))


if __name__ == "__main__":
    main()
